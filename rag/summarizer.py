"""
RAG Summarizer — GPT-4o with Zero-Tolerance Citation Guardrails
────────────────────────────────────────────────────────────────
Retrieves relevant chunks from ChromaDB, assembles a grounded prompt,
calls OpenAI GPT-4o, then validates every citation against source text.
Hallucinated references are flagged and stripped before output.
"""
from __future__ import annotations

import json
import logging 
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# pyrefly: ignore [missing-import]
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
_DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))

# ── OpenAI client (lazy init) ─────────────────────────────────────────────────
_client: Optional[AsyncOpenAI] = None


def _get_openai() -> AsyncOpenAI:
    global _client
    if _client is None:
        if not _OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        _client = AsyncOpenAI(api_key=_OPENAI_API_KEY)
    return _client


# ═══════════════════════════════════════════════════════════════════════════════
# Output Schema
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CitationCheck:
    citation: str
    found_in_source: bool
    source_snippet: str = ""


@dataclass
class SummaryResult:
    doc_id: str
    title: str
    high_level_summary: str = ""
    key_issues: list[str] = field(default_factory=list)
    ratio_decidendi: str = ""
    statutes_involved: list[str] = field(default_factory=list)
    procedural_posture: str = ""
    outcome: str = ""
    validated_citations: list[CitationCheck] = field(default_factory=list)
    flagged_hallucinations: list[str] = field(default_factory=list)
    source_chunks_used: int = 0
    model_used: str = _OPENAI_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are LexAI, an expert Indian legal analyst. Your task is to produce a structured multi-level summary of a legal judgment.

GOLDEN RULE — ZERO HALLUCINATION:
- You MUST reason ONLY over the retrieved passages provided in the user message.
- Do NOT invent citations, section numbers, case names, or dates from memory.
- Every citation you mention MUST appear verbatim or near-verbatim in the source passages.
- If a detail is not in the source passages, write "Not determinable from available text."

OUTPUT FORMAT:
Respond with ONLY a valid JSON object (no markdown fences) with this exact structure:
{
  "high_level_summary": "2-3 sentence plain-language summary.",
  "key_issues": ["Issue 1", "Issue 2", ...],
  "ratio_decidendi": "The court's core legal reasoning and rule of law applied.",
  "statutes_involved": ["Section X of Act Y", ...],
  "procedural_posture": "How the case reached this court.",
  "outcome": "What was ordered / decided.",
  "citations_used": ["(YYYY) Vol Court Page", ...]
}

Keep language precise, neutral, and legally accurate."""


# ═══════════════════════════════════════════════════════════════════════════════
# Citation Validation
# ═══════════════════════════════════════════════════════════════════════════════

# Matches standard Indian citation formats
_CITATION_RE = re.compile(
    r"\(?\d{4}\)?\s*\d*\s*"
    r"(?:SCC|SCR|AIR|Bom(?:CR)?|Mad(?:LJ)?|Cal(?:LJ)?|All(?:LJ)?|"
    r"Del(?:LJ)?|Ker(?:LJ)?|Guj(?:LR)?|Raj(?:LW)?|HP|P&H|MP|Ori|"
    r"JT|SCALE|RCR|SLT|GLH|ILR|MLJ|PLR|UPLBEC)\s*\d+",
    re.I,
)

# Matches references to statutory sections
_STATUTE_RE = re.compile(
    r"[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?(?:\s+of\s+(?:the\s+)?"
    r"[A-Z][a-zA-Z\s]+(?:Act|Code|Rules?|Ordinance|Regulation)[,\s\d]*)?",
)


def _extract_all_citations(text: str) -> list[str]:
    """Extract all citation-like strings from LLM output."""
    return list({m.strip() for m in _CITATION_RE.findall(text)})


def _validate_citations(
    citations_claimed: list[str],
    source_texts: list[str],
) -> tuple[list[CitationCheck], list[str]]:
    """
    Cross-check each citation against the raw source passages.

    Returns:
        validated: List of CitationCheck objects (found=True/False).
        hallucinated: List of citations NOT found in any source.
    """
    source_blob = " ".join(source_texts).lower()
    validated: list[CitationCheck] = []
    hallucinated: list[str] = []

    for cite in citations_claimed:
        cite_lower = cite.lower().strip()
        found = cite_lower in source_blob

        # Find a nearby snippet for context
        snippet = ""
        if found:
            idx = source_blob.find(cite_lower)
            start = max(0, idx - 80)
            end = min(len(source_blob), idx + len(cite) + 80)
            snippet = source_blob[start:end].strip()

        check = CitationCheck(
            citation=cite,
            found_in_source=found,
            source_snippet=snippet[:200],
        )
        validated.append(check)
        if not found:
            hallucinated.append(cite)

    return validated, hallucinated


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Assembly
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_prompt(title: str, chunks: list[dict]) -> str:
    """Assemble the grounded user prompt from retrieved source chunks."""
    passages = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.get("section", "unknown")
        text = chunk.get("text", "").strip()
        passages.append(f"[PASSAGE {i} | Section: {section}]\n{text}")

    passages_text = "\n\n---\n\n".join(passages)

    return f"""Judgment: {title}

RETRIEVED SOURCE PASSAGES (ground your answer ONLY in these):
{passages_text}

Now produce the JSON summary following the format in your system prompt.
Remember: cite ONLY what appears in the passages above."""


# ═══════════════════════════════════════════════════════════════════════════════
# Main Summarizer
# ═══════════════════════════════════════════════════════════════════════════════

async def summarize_document(
    doc_id: str,
    title: str,
    top_k: int = _DEFAULT_TOP_K,
    query_override: Optional[str] = None,
) -> SummaryResult:
    """
    Generate a grounded multi-level summary for a document.

    Args:
        doc_id: The document's ID (used to fetch chunks from ChromaDB).
        title: Case title (used in prompt).
        top_k: Number of chunks to retrieve for context.
        query_override: Optional custom query for retrieval (defaults to title).

    Returns:
        SummaryResult with validated summary and citation checks.
    """
    from vector_store.indexer import get_document_chunks, search

    # ── Step 1: Retrieve relevant chunks ─────────────────────────────────────
    chunks = get_document_chunks(doc_id)

    if not chunks:
        # Fallback: semantic search using title as query
        q = query_override or title
        chunks = search(q, top_k=top_k)

    # Trim to top_k
    chunks = chunks[:top_k]

    if not chunks:
        return SummaryResult(
            doc_id=doc_id,
            title=title,
            high_level_summary="No source text available for summarization.",
            source_chunks_used=0,
        )

    source_texts = [c.get("text", "") for c in chunks]

    # ── Step 2: Call GPT-4o ───────────────────────────────────────────────────
    client = _get_openai()
    user_prompt = _build_user_prompt(title, chunks)

    logger.info(f"Calling {_OPENAI_MODEL} for doc: {doc_id} ({len(chunks)} chunks)")

    try:
        response = await client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temp for factual grounded output
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content or "{}"
    except Exception as exc:
        logger.error(f"OpenAI API error: {exc}")
        return SummaryResult(
            doc_id=doc_id,
            title=title,
            high_level_summary=f"Summarization failed: {exc}",
            source_chunks_used=len(chunks),
        )

    # ── Step 3: Parse JSON output ─────────────────────────────────────────────
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        logger.warning("LLM output was not valid JSON. Attempting extraction.")
        # Fallback: extract JSON block
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        parsed = json.loads(json_match.group(0)) if json_match else {}

    # ── Step 4: Citation Validation ───────────────────────────────────────────
    citations_claimed = parsed.get("citations_used", [])
    # Also scan free text for any extra citations
    extra = _extract_all_citations(raw_output)
    all_citations = list({*citations_claimed, *extra})

    validated, hallucinated = _validate_citations(all_citations, source_texts)

    if hallucinated:
        logger.warning(
            f"⚠️  {len(hallucinated)} hallucinated citation(s) detected and flagged: "
            + ", ".join(hallucinated)
        )

    return SummaryResult(
        doc_id=doc_id,
        title=title,
        high_level_summary=parsed.get("high_level_summary", ""),
        key_issues=parsed.get("key_issues", []),
        ratio_decidendi=parsed.get("ratio_decidendi", ""),
        statutes_involved=parsed.get("statutes_involved", []),
        procedural_posture=parsed.get("procedural_posture", ""),
        outcome=parsed.get("outcome", ""),
        validated_citations=validated,
        flagged_hallucinations=hallucinated,
        source_chunks_used=len(chunks),
        model_used=_OPENAI_MODEL,
    )


async def summarize_query(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
) -> SummaryResult:
    """
    Retrieve and summarize the most relevant judgment chunks for a query.
    Useful for question-answering mode (not bound to a single document).
    """
    from vector_store.indexer import search

    chunks = search(query, top_k=top_k)
    if not chunks:
        return SummaryResult(
            doc_id="query",
            title=f"Query: {query}",
            high_level_summary="No relevant passages found in the index.",
        )

    source_texts = [c.get("text", "") for c in chunks]
    client = _get_openai()
    user_prompt = _build_user_prompt(f"Query: {query}", chunks)

    try:
        response = await client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw_output = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_output)
    except Exception as exc:
        logger.error(f"OpenAI error (query mode): {exc}")
        return SummaryResult(
            doc_id="query",
            title=query,
            high_level_summary=f"Error: {exc}",
            source_chunks_used=len(chunks),
        )

    citations_claimed = parsed.get("citations_used", [])
    extra = _extract_all_citations(raw_output)
    validated, hallucinated = _validate_citations(
        list({*citations_claimed, *extra}), source_texts
    )

    return SummaryResult(
        doc_id="query",
        title=f"Query: {query}",
        high_level_summary=parsed.get("high_level_summary", ""),
        key_issues=parsed.get("key_issues", []),
        ratio_decidendi=parsed.get("ratio_decidendi", ""),
        statutes_involved=parsed.get("statutes_involved", []),
        procedural_posture=parsed.get("procedural_posture", ""),
        outcome=parsed.get("outcome", ""),
        validated_citations=validated,
        flagged_hallucinations=hallucinated,
        source_chunks_used=len(chunks),
        model_used=_OPENAI_MODEL,
    )
