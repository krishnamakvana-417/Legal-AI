"""
Legal Judgment PDF Parser
─────────────────────────
Extracts raw text from PDFs using pdfplumber, cleans it, segments into logical
legal sections, and extracts structured metadata per document.

Output: ParsedDocument dataclasses, saved as JSON to ./data/parsed/
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)

PARSED_DIR = Path(os.getenv("PARSED_DIR", "./data/parsed"))


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegalSections:
    facts: str = ""
    procedural_history: str = ""
    key_issues: str = ""
    ratio_decidendi: str = ""
    statutes_cited: str = ""
    disposition: str = ""
    full_text: str = ""


@dataclass
class ParsedDocument:
    doc_id: str = ""
    title: str = ""
    court: str = ""
    decision_date: str = ""
    bench: list[str] = field(default_factory=list)
    case_number: str = ""
    cite_list: list[str] = field(default_factory=list)
    cited_by_list: list[str] = field(default_factory=list)
    num_cites: int = 0
    num_cited_by: int = 0
    sections: LegalSections = field(default_factory=LegalSections)
    source_file: str = ""
    total_pages: int = 0
    word_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Text Cleaning
# ═══════════════════════════════════════════════════════════════════════════════

# Patterns that indicate header/footer lines to discard
_HEADER_FOOTER_PATTERNS = re.compile(
    r"(page\s+\d+\s+(of|\/)\s+\d+|scc\s*online|manupatra|westlaw\b|"
    r"www\.|http[s]?://|©\s*\d{4}|all\s+rights\s+reserved)",
    re.I,
)

# Leading standalone line numbers (e.g., "  15 The court held…")
_LINE_NUMBER_PREFIX = re.compile(r"^\s{0,4}\d{1,4}\s{2,}")

# Form-feed / vertical tab artifacts
_FORM_FEED = re.compile(r"[\f\v]")

# Collapse 3+ blank lines into 2
_MULTI_BLANK = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Strip OCR artifacts, page numbers, headers, footers, and normalise whitespace."""
    lines: list[str] = []
    for line in raw.split("\n"):
        # Drop pure numeric lines (standalone page numbers)
        if re.fullmatch(r"\s*\d{1,4}\s*", line):
            continue
        # Drop header/footer lines
        if _HEADER_FOOTER_PATTERNS.search(line):
            continue
        # Remove leading line-number prefixes
        line = _LINE_NUMBER_PREFIX.sub("", line)
        # Remove form-feed characters
        line = _FORM_FEED.sub(" ", line)
        # Collapse inner whitespace
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        lines.append(line)

    text = "\n".join(lines)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata Extraction
# ═══════════════════════════════════════════════════════════════════════════════

_COURT_PATTERNS = [
    r"(Supreme Court of India)",
    r"(High Court of [A-Za-z\s,]+)",
    r"(National Company Law (?:Appellate )?Tribunal[,\s]*[A-Za-z\s]*)",
    r"(National Consumer Disputes Redressal Commission)",
    r"(Central Administrative Tribunal[,\s]*[A-Za-z\s]*)",
    r"(Income Tax Appellate Tribunal[,\s]*[A-Za-z\s]*)",
    r"(Debt Recovery (?:Appellate )?Tribunal[,\s]*[A-Za-z\s]*)",
    r"(District (?:&|and)? Sessions Court[,\s]*[A-Za-z\s]*)",
    r"(District Court[,\s]+[A-Za-z\s]+)",
]

_DATE_RE = re.compile(
    r"(?:decided\s+on|date\s+of\s+(?:the\s+)?judgment|judgment\s+dated|"
    r"order\s+dated|pronounced\s+on)[:\s]*"
    r"(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\,?\s+\d{4})",
    re.I,
)

_JUDGE_RE = re.compile(
    r"(?:coram|bench|before|j\.|justice|hon['']?ble\s+(?:mr\.?\s*)?justice)[:\s]+"
    r"([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+){1,4})",
    re.I,
)

_CASE_NO_RE = re.compile(
    r"(?:civil\s+appeal|criminal\s+appeal|writ\s+petition|slp|"
    r"special\s+leave\s+petition|w\.?\s*p\.?|c\.?\s*a\.?|cr\.?\s*a\.?|"
    r"o\.?\s*s\.?|r\.?\s*s\.?\s*a\.?|crl\.?\s*petition)\s*"
    r"(?:no\.?)?\s*[\d\/\(\)\-]+(?:\s*of\s+\d{4})?",
    re.I,
)

# Standard Indian legal citation format: (YYYY) Vol Court Page
_CITATION_RE = re.compile(
    r"\(?\d{4}\)?\s*\d*\s*"
    r"(?:SCC|SCR|AIR|Bom(?:CR)?|Mad(?:LJ)?|Cal(?:LJ)?|All(?:LJ)?|"
    r"Del(?:LJ)?|Ker(?:LJ)?|Guj(?:LR)?|Raj(?:LW)?|HP|P&H|MP|Ori|"
    r"JT|SCALE|RCR|SLT|GLH|ILR|MLJ|PLR|UPLBEC)\s*\d+",
    re.I,
)

# Case title: "X v. Y" or "X vs Y"
_TITLE_RE = re.compile(
    r"^(.{5,120}?(?:\bv(?:s\.?|ersus)\b).{5,120}?)$",
    re.MULTILINE | re.I,
)


def _extract_court(text: str) -> str:
    for pattern in _COURT_PATTERNS:
        m = re.search(pattern, text[:1500], re.I)
        if m:
            return m.group(1).strip()
    return ""


def extract_metadata(text: str, filename: str) -> dict:
    """Extract structured metadata from cleaned judgment text."""
    header = text[:800]
    body_start = text[:3000]

    # Title
    title_match = _TITLE_RE.search(header)
    title = (
        title_match.group(1).strip()
        if title_match
        else Path(filename).stem.replace("_", " ").replace("-", " ").title()
    )

    # Court
    court = _extract_court(body_start)

    # Date
    date_match = _DATE_RE.search(body_start)
    decision_date = date_match.group(1).strip() if date_match else ""

    # Judges
    judges_raw = _JUDGE_RE.findall(body_start)
    bench = list({j.strip() for j in judges_raw if len(j.strip()) > 4})[:6]

    # Case number
    case_match = _CASE_NO_RE.search(header)
    case_number = case_match.group(0).strip() if case_match else ""

    # Citations referenced in judgment body
    cite_list = list({c.strip() for c in _CITATION_RE.findall(text)})

    return {
        "title": title,
        "court": court,
        "decision_date": decision_date,
        "bench": bench,
        "case_number": case_number,
        "cite_list": cite_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

_SECTION_HEADERS: dict[str, list[str]] = {
    "facts": [
        r"(?:brief\s+)?facts?\s*(?:of\s+the\s+case)?",
        r"background\s*(?:facts?)?",
        r"statement\s+of\s+facts",
        r"factual\s+(?:background|matrix)",
    ],
    "procedural_history": [
        r"procedural\s+(?:history|background)",
        r"proceedings?\s+(?:before\s+the\s+(?:lower|trial)\s+court|below)",
        r"earlier\s+proceedings?",
        r"impugned\s+(?:order|judgment)",
        r"genesis\s+of\s+(?:the\s+)?litigation",
    ],
    "key_issues": [
        r"(?:key\s+)?issues?\s+(?:raised|involved|for\s+(?:our\s+)?consideration|framed)",
        r"points?\s+(?:of\s+)?(?:law\s+)?(?:raised|for\s+determination)",
        r"questions?\s+of\s+law",
        r"issues?\s+arise(?:s)?\s+for\s+consideration",
    ],
    "ratio_decidendi": [
        r"ratio\s+decidendi",
        r"(?:our\s+)?(?:analysis\s+and\s+)?(?:findings?\s+and\s+)?(?:reasoning|discussion|analysis)",
        r"(?:our\s+)?(?:conclusion|finding)s?",
        r"held[:\s]",
        r"(?:the\s+)?court\s+(?:observe[ds]|held|find[s]?|conclude[ds])",
    ],
    "statutes_cited": [
        r"(?:statutes?|provisions?|sections?)\s+(?:referred\s+to|cited|relied\s+upon)",
        r"relevant\s+(?:statutory\s+)?(?:provisions?|sections?|enactments?)",
        r"applicable\s+law",
        r"legislative\s+framework",
    ],
    "disposition": [
        r"(?:in\s+the\s+)?(?:result|conclusion|premises|view\s+of\s+the\s+foregoing)",
        r"(?:the\s+)?(?:order|operative\s+part|relief)",
        r"appeal\s+(?:is\s+)?(?:allowed|dismissed|disposed\s+of)",
        r"petition\s+(?:is\s+)?(?:allowed|dismissed|disposed\s+of)",
        r"writ\s+petition\s+(?:is\s+)?(?:allowed|dismissed)",
    ],
}

# Build one compiled pattern per section for efficiency
_COMPILED_SECTIONS: dict[str, re.Pattern] = {
    section: re.compile(
        r"(?:(?:^|\n)\s*(?:\d+[\.\)]\s*)?(?:" + "|".join(pats) + r")\s*[:\-]?\s*\n)",
        re.I | re.MULTILINE,
    )
    for section, pats in _SECTION_HEADERS.items()
}


def segment_sections(text: str) -> LegalSections:
    """Segment judgment text into logical legal sections using regex heuristics."""
    sections = LegalSections(full_text=text)

    # Collect all detected boundaries: (char_offset, section_name, content_start)
    boundaries: list[tuple[int, str, int]] = []
    for section_name, pattern in _COMPILED_SECTIONS.items():
        for m in pattern.finditer(text):
            boundaries.append((m.start(), section_name, m.end()))

    if not boundaries:
        # Graceful fallback: put first 4 KB in facts
        sections.facts = text[:4000]
        return sections

    boundaries.sort(key=lambda x: x[0])

    # Remove duplicate sections (keep first occurrence)
    seen: set[str] = set()
    unique_boundaries: list[tuple[int, str, int]] = []
    for start, name, end in boundaries:
        if name not in seen:
            seen.add(name)
            unique_boundaries.append((start, name, end))

    # Extract text slices between consecutive headers
    section_texts: dict[str, str] = {}
    for i, (_, name, content_start) in enumerate(unique_boundaries):
        next_start = unique_boundaries[i + 1][0] if i + 1 < len(unique_boundaries) else len(text)
        section_texts[name] = text[content_start:next_start].strip()

    sections.facts = section_texts.get("facts", "")
    sections.procedural_history = section_texts.get("procedural_history", "")
    sections.key_issues = section_texts.get("key_issues", "")
    sections.ratio_decidendi = section_texts.get("ratio_decidendi", "")
    sections.statutes_cited = section_texts.get("statutes_cited", "")
    sections.disposition = section_texts.get("disposition", "")

    return sections


# ═══════════════════════════════════════════════════════════════════════════════
# PDF Text Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """Extract raw text from all pages of a PDF using pdfplumber."""
    pages: list[str] = []
    total_pages = 0

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                try:
                    txt = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if txt:
                        pages.append(txt)
                except Exception as exc:
                    logger.debug(f"Page extraction error [{pdf_path.name}]: {exc}")
    except Exception as exc:
        logger.error(f"Cannot open PDF [{pdf_path.name}]: {exc}")
        return "", 0

    return "\n\n".join(pages), total_pages


# ═══════════════════════════════════════════════════════════════════════════════
# Main Parse Function
# ═══════════════════════════════════════════════════════════════════════════════

def _make_doc_id(filename: str) -> str:
    """Generate a stable 12-char hex ID from the filename."""
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:12]


def parse_pdf(pdf_path: Path) -> Optional[ParsedDocument]:
    """
    Parse a single PDF into a structured ParsedDocument.

    Returns None if the file cannot be read or yields no text.
    """
    raw_text, total_pages = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        logger.warning(f"No text extracted: {pdf_path.name}")
        return None

    clean = clean_text(raw_text)
    meta = extract_metadata(clean, pdf_path.name)
    sections = segment_sections(clean)

    return ParsedDocument(
        doc_id=_make_doc_id(pdf_path.name),
        title=meta["title"],
        court=meta["court"],
        decision_date=meta["decision_date"],
        bench=meta["bench"],
        case_number=meta["case_number"],
        cite_list=meta["cite_list"],
        cited_by_list=[],  # Populated by graph builder
        num_cites=len(meta["cite_list"]),
        num_cited_by=0,
        sections=sections,
        source_file=str(pdf_path.resolve()),
        total_pages=total_pages,
        word_count=len(clean.split()),
    )


def parse_all(
    pdf_paths: list[Path],
    save: bool = True,
    skip_existing: bool = True,
) -> list[ParsedDocument]:
    """
    Parse a batch of PDFs, optionally saving JSON outputs to PARSED_DIR.

    Args:
        pdf_paths: List of PDF Paths to parse.
        save: Persist parsed JSON files to disk.
        skip_existing: Skip PDFs whose JSON output already exists.

    Returns:
        List of successfully parsed ParsedDocument objects.
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ParsedDocument] = []

    for pdf_path in tqdm(pdf_paths, desc="📄 Parsing PDFs", unit="file"):
        doc_id = _make_doc_id(pdf_path.name)
        out_path = PARSED_DIR / f"{doc_id}.json"

        if skip_existing and out_path.exists():
            # Load from cache
            try:
                with open(out_path, encoding="utf-8") as f:
                    data = json.load(f)
                doc = ParsedDocument(**{
                    k: (LegalSections(**v) if k == "sections" else v)
                    for k, v in data.items()
                })
                results.append(doc)
                continue
            except Exception:
                pass  # Re-parse if cache is corrupt

        doc = parse_pdf(pdf_path)
        if doc:
            results.append(doc)
            if save:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(doc), f, indent=2, ensure_ascii=False)

    success = len(results)
    total = len(pdf_paths)
    logger.info(f"Parsing complete: {success}/{total} documents processed.")
    return results


def load_all_parsed() -> list[ParsedDocument]:
    """Load all previously parsed documents from PARSED_DIR."""
    docs: list[ParsedDocument] = []
    for json_path in sorted(PARSED_DIR.glob("*.json")):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            doc = ParsedDocument(**{
                k: (LegalSections(**v) if k == "sections" else v)
                for k, v in data.items()
            })
            docs.append(doc)
        except Exception as exc:
            logger.warning(f"Failed to load {json_path.name}: {exc}")
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Parse legal judgment PDFs")
    ap.add_argument("--input", help="Single PDF file path")
    ap.add_argument("--all", action="store_true", help="Parse all PDFs in data/judgments/")
    ap.add_argument("--no-save", action="store_true", help="Do not write JSON files")
    args = ap.parse_args()

    if args.input:
        doc = parse_pdf(Path(args.input))
        if doc:
            print(f"✅ Parsed: {doc.title}")
            print(f"   Court : {doc.court}")
            print(f"   Date  : {doc.decision_date}")
            print(f"   Pages : {doc.total_pages}")
            print(f"   Cites : {doc.num_cites}")
    elif args.all:
        pdfs = sorted(Path("./data/judgments").glob("*.pdf"))
        docs = parse_all(pdfs, save=not args.no_save)
        print(f"\n✅ {len(docs)} documents parsed.")
    else:
        ap.print_help()
