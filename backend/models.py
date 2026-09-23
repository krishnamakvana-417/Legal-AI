"""
Pydantic v2 response models for the FastAPI backend.
"""
from __future__ import annotations
from typing import Optional, Any
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field


# ─── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionRequest(BaseModel):
    folder_id: Optional[str] = Field(None, description="Override Google Drive folder ID")
    reset_index: bool = Field(False, description="Wipe and rebuild the vector index")


class IngestionStatus(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "done" | "error"
    message: str = ""
    pdfs_found: int = 0
    pdfs_parsed: int = 0
    chunks_indexed: int = 0
    error: Optional[str] = None


# ─── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=50)
    court_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SearchResult(BaseModel):
    doc_id: str
    title: str
    court: str
    date: str
    section: str
    snippet: str
    score: float
    source_file: str


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]


# ─── Document ──────────────────────────────────────────────────────────────────

class LegalSectionsOut(BaseModel):
    facts: str = ""
    procedural_history: str = ""
    key_issues: str = ""
    ratio_decidendi: str = ""
    statutes_cited: str = ""
    disposition: str = ""


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    court: str
    decision_date: str
    bench: list[str]
    case_number: str
    cite_list: list[str]
    cited_by_list: list[str]
    num_cites: int
    num_cited_by: int
    sections: LegalSectionsOut
    source_file: str
    total_pages: int
    word_count: int


# ─── Summary ───────────────────────────────────────────────────────────────────

class CitationValidation(BaseModel):
    citation: str
    found_in_source: bool
    source_snippet: Optional[str] = None


class SummaryOut(BaseModel):
    doc_id: str
    title: str
    high_level_summary: str
    key_issues: list[str]
    ratio_decidendi: str
    statutes_involved: list[str]
    procedural_posture: str
    outcome: str
    validated_citations: list[CitationValidation]
    flagged_hallucinations: list[str]
    source_chunks_used: int
    model_used: str


# ─── Citation Graph ────────────────────────────────────────────────────────────

class GraphNode(BaseModel):
    id: str
    title: str
    court: str
    date: str
    num_cites: int
    num_cited_by: int
    authority_score: float = 0.0
    group: int = 0  # For frontend color grouping


class GraphEdge(BaseModel):
    source: str
    target: str
    citation_text: Optional[str] = None


class GraphData(BaseModel):
    nodes: list[GraphNode]
    links: list[GraphEdge]
    center_node: Optional[str] = None


# ─── External APIs ─────────────────────────────────────────────────────────────

class KanoonSearchResult(BaseModel):
    doc_id: str
    title: str
    headline: str
    publishdate: str
    docsource: str
    numcites: int


class KanoonSearchResponse(BaseModel):
    query: str
    total: int
    results: list[KanoonSearchResult]


class AnrakCaseOut(BaseModel):
    cnr: str
    case_title: str
    court: str
    filing_date: str
    status: str
    next_hearing: Optional[str] = None
    raw: dict[str, Any] = {}


class AnrakStatuteResult(BaseModel):
    section_id: str
    act_name: str
    section_number: str
    section_title: str
    text_preview: str


# ─── Stats ─────────────────────────────────────────────────────────────────────

class IndexStats(BaseModel):
    total_documents: int
    total_chunks: int
    graph_nodes: int
    graph_edges: int
    collection_name: str
    persist_dir: str
