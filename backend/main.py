"""
FastAPI Backend — AI Legal Platform
─────────────────────────────────────
Exposes REST endpoints for:
  - Ingestion pipeline (Drive → Parse → Index)
  - Semantic search
  - Document retrieval
  - RAG summarization
  - Citation graph queries
  - External API proxies (Indian Kanoon, AnrakLegal)
  - Index statistics
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

# pyrefly: ignore [missing-import]
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.models import (
    AnrakCaseOut,
    AnrakStatuteResult,
    DocumentOut,
    GraphData,
    IngestionRequest,
    IngestionStatus,
    IndexStats,
    KanoonSearchResponse,
    KanoonSearchResult,
    LegalSectionsOut,
    SearchResponse,
    SearchResult,
    SummaryOut,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── In-memory ingestion job store ─────────────────────────────────────────────
_ingestion_jobs: dict[str, IngestionStatus] = {}

# ── Cached graph (loaded once on startup) ────────────────────────────────────
_citation_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure directories and pre-load graph if available."""
    global _citation_graph
    settings.ensure_dirs()

    from graph.citation_network import load_graph, GRAPH_PATH
    _citation_graph = load_graph(GRAPH_PATH)
    if _citation_graph:
        logger.info(
            f"Citation graph loaded: {_citation_graph.number_of_nodes()} nodes"
        )
    else:
        logger.info("No citation graph found — will be built after ingestion.")

    yield

    logger.info("Backend shutting down.")


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="LexAI — Legal Judgment Intelligence Platform",
    description="AI-powered legal document retrieval, summarization, and citation analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Background Ingestion Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _run_ingestion_pipeline(job_id: str, folder_id: Optional[str], reset_index: bool):
    """
    Full ingestion pipeline:
      1. Download PDFs from Google Drive
      2. Parse PDFs → ParsedDocuments
      3. Index in ChromaDB
      4. Build citation graph
    """
    global _citation_graph
    job = _ingestion_jobs[job_id]
    job.status = "running"
    job.message = "Downloading PDFs from Google Drive..."

    try:
        # Step 1: Download
        from ingestion.drive_fetcher import fetch_all

        pdf_paths = fetch_all(
            folder_id=folder_id or settings.GOOGLE_DRIVE_FOLDER_ID
        )
        job.pdfs_found = len(pdf_paths)
        job.message = f"Downloaded {len(pdf_paths)} PDFs. Parsing..."

        if not pdf_paths:
            job.status = "error"
            job.error = "No PDFs found in the specified Google Drive folder."
            return

        # Step 2: Parse
        from ingestion.parser import parse_all

        docs = parse_all(pdf_paths, save=True, skip_existing=not reset_index)
        job.pdfs_parsed = len(docs)
        job.message = f"Parsed {len(docs)} documents. Indexing..."

        # Step 3: Index
        from vector_store.indexer import index_documents

        chunks_indexed = index_documents(docs, reset=reset_index)
        job.chunks_indexed = chunks_indexed
        job.message = f"Indexed {chunks_indexed} chunks. Building citation graph..."

        # Step 4: Citation graph
        from graph.citation_network import build_and_save

        _citation_graph = build_and_save(docs)
        job.message = (
            f"Complete! {len(docs)} docs | {chunks_indexed} chunks | "
            f"{_citation_graph.number_of_nodes()} graph nodes."
        )
        job.status = "done"
        logger.info(f"Ingestion job {job_id} complete.")

    except Exception as exc:
        logger.error(f"Ingestion job {job_id} failed: {exc}", exc_info=True)
        job.status = "error"
        job.error = str(exc)
        job.message = f"Pipeline failed: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Health
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "LexAI Legal Platform", "version": "1.0.0"}


@app.get("/api/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Ingestion
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/ingest", response_model=IngestionStatus, tags=["Ingestion"])
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Trigger the full ingestion pipeline in the background.
    Returns a job_id to poll /api/ingest/status/{job_id}.
    """
    job_id = str(uuid.uuid4())[:8]
    job = IngestionStatus(job_id=job_id, status="pending", message="Queued.")
    _ingestion_jobs[job_id] = job

    background_tasks.add_task(
        _run_ingestion_pipeline,
        job_id=job_id,
        folder_id=request.folder_id,
        reset_index=request.reset_index,
    )
    logger.info(f"Ingestion job queued: {job_id}")
    return job


@app.get("/api/ingest/status/{job_id}", response_model=IngestionStatus, tags=["Ingestion"])
async def get_ingestion_status(job_id: str):
    """Poll the status of a running ingestion job."""
    if job_id not in _ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _ingestion_jobs[job_id]


@app.get("/api/ingest/jobs", tags=["Ingestion"])
async def list_ingestion_jobs():
    """List all ingestion jobs (recent history)."""
    return list(_ingestion_jobs.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Search
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(
    q: str = Query(..., description="Legal search query", min_length=2),
    top_k: int = Query(10, ge=1, le=50),
    court: Optional[str] = Query(None, description="Filter by court name"),
    section: Optional[str] = Query(None, description="Filter by document section"),
):
    """
    Semantic similarity search over indexed judgments.
    Returns ranked results with relevance scores.
    """
    from vector_store.indexer import search

    try:
        raw_results = search(
            query=q,
            top_k=top_k,
            court_filter=court,
            section_filter=section,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    results = [
        SearchResult(
            doc_id=r["doc_id"],
            title=r["title"],
            court=r["court"],
            date=r["date"],
            section=r["section"],
            snippet=r["text"][:400],
            score=r["score"],
            source_file=r["source_file"],
        )
        for r in raw_results
    ]

    return SearchResponse(query=q, total=len(results), results=results)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Documents
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/document/{doc_id}", response_model=DocumentOut, tags=["Documents"])
async def get_document(doc_id: str):
    """Fetch the full parsed document and its metadata."""
    import json

    parsed_dir = Path(settings.PARSED_DIR)
    doc_file = parsed_dir / f"{doc_id}.json"

    if not doc_file.exists():
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    try:
        with open(doc_file, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading document: {exc}")

    sections_data = data.get("sections", {})
    sections = LegalSectionsOut(**{
        k: v for k, v in sections_data.items()
        if k in LegalSectionsOut.model_fields
    })

    return DocumentOut(
        doc_id=data.get("doc_id", doc_id),
        title=data.get("title", ""),
        court=data.get("court", ""),
        decision_date=data.get("decision_date", ""),
        bench=data.get("bench", []),
        case_number=data.get("case_number", ""),
        cite_list=data.get("cite_list", []),
        cited_by_list=data.get("cited_by_list", []),
        num_cites=data.get("num_cites", 0),
        num_cited_by=data.get("num_cited_by", 0),
        sections=sections,
        source_file=data.get("source_file", ""),
        total_pages=data.get("total_pages", 0),
        word_count=data.get("word_count", 0),
    )


@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """List all parsed documents with basic metadata."""
    import json

    parsed_dir = Path(settings.PARSED_DIR)
    doc_files = sorted(parsed_dir.glob("*.json"))
    total = len(doc_files)
    page_files = doc_files[skip: skip + limit]

    docs = []
    for f in page_files:
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            docs.append({
                "doc_id": data.get("doc_id"),
                "title": data.get("title"),
                "court": data.get("court"),
                "decision_date": data.get("decision_date"),
                "num_cites": data.get("num_cites", 0),
                "word_count": data.get("word_count", 0),
            })
        except Exception:
            continue

    return {"total": total, "skip": skip, "limit": limit, "documents": docs}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Summarization
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/summarize/{doc_id}", response_model=SummaryOut, tags=["Summarization"])
async def summarize_document(
    doc_id: str,
    top_k: int = Query(10, ge=3, le=20),
    force: bool = Query(False, description="Bypass cache and regenerate"),
):
    """
    Generate a GPT-4o multi-level summary for a document.
    Citations are validated against source text before output.
    """
    import json
    from dataclasses import asdict

    # Check cache
    cache_path = Path(settings.PARSED_DIR) / f"{doc_id}_summary.json"
    if cache_path.exists() and not force:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            return SummaryOut(**cached)
        except Exception:
            pass

    # Get document title
    doc_path = Path(settings.PARSED_DIR) / f"{doc_id}.json"
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    with open(doc_path, encoding="utf-8") as f:
        doc_data = json.load(f)

    from rag.summarizer import summarize_document as _summarize
    from dataclasses import fields

    try:
        result = await _summarize(
            doc_id=doc_id,
            title=doc_data.get("title", ""),
            top_k=top_k,
        )
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {exc}")

    # Build response
    validated = [
        {"citation": c.citation, "found_in_source": c.found_in_source, "source_snippet": c.source_snippet}
        for c in result.validated_citations
    ]

    summary_out = SummaryOut(
        doc_id=result.doc_id,
        title=result.title,
        high_level_summary=result.high_level_summary,
        key_issues=result.key_issues,
        ratio_decidendi=result.ratio_decidendi,
        statutes_involved=result.statutes_involved,
        procedural_posture=result.procedural_posture,
        outcome=result.outcome,
        validated_citations=[
            {"citation": c.citation, "found_in_source": c.found_in_source, "source_snippet": c.source_snippet}
            for c in result.validated_citations
        ],
        flagged_hallucinations=result.flagged_hallucinations,
        source_chunks_used=result.source_chunks_used,
        model_used=result.model_used,
    )

    # Cache to disk
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(summary_out.model_dump(), f, indent=2)
    except Exception:
        pass

    return summary_out


@app.post("/api/summarize/query", tags=["Summarization"])
async def summarize_query(
    q: str = Query(..., description="Legal question to answer from the corpus"),
    top_k: int = Query(10, ge=3, le=20),
):
    """RAG Q&A: answer a legal question using retrieved judgment passages."""
    from rag.summarizer import summarize_query as _sq

    try:
        result = await _sq(query=q, top_k=top_k)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "query": q,
        "high_level_summary": result.high_level_summary,
        "key_issues": result.key_issues,
        "ratio_decidendi": result.ratio_decidendi,
        "statutes_involved": result.statutes_involved,
        "outcome": result.outcome,
        "validated_citations": [asdict(c) for c in result.validated_citations],
        "flagged_hallucinations": result.flagged_hallucinations,
        "source_chunks_used": result.source_chunks_used,
    }


def asdict(obj):
    from dataclasses import asdict as _asdict
    try:
        return _asdict(obj)
    except TypeError:
        return obj.__dict__


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Citation Graph
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_graph():
    global _citation_graph
    if _citation_graph is None:
        from graph.citation_network import load_graph, GRAPH_PATH
        _citation_graph = load_graph(GRAPH_PATH)
    if _citation_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Citation graph not built yet. Run ingestion first.",
        )
    return _citation_graph


@app.get("/api/graph/all", tags=["Citation Graph"])
async def get_full_graph():
    """Export the full citation network as D3-compatible JSON."""
    from graph.citation_network import export_full_graph
    G = _ensure_graph()
    return export_full_graph(G)


@app.get("/api/graph/{doc_id}", tags=["Citation Graph"])
async def get_document_graph(
    doc_id: str,
    depth: int = Query(2, ge=1, le=4),
    direction: str = Query("precedents", pattern="^(precedents|citing)$"),
):
    """
    Return the citation subgraph for a specific document.
    - precedents: cases that doc_id cites (outbound)
    - citing: cases that cite doc_id (inbound)
    """
    from graph.citation_network import get_precedent_tree, get_citing_tree
    G = _ensure_graph()

    if direction == "precedents":
        data = get_precedent_tree(G, doc_id, depth=depth)
    else:
        data = get_citing_tree(G, doc_id, depth=depth)

    if not data["nodes"]:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not in citation graph.")

    return data


@app.get("/api/graph/{doc_id}/stats", tags=["Citation Graph"])
async def get_graph_stats(doc_id: str):
    """Return citation statistics for a document."""
    from graph.citation_network import get_citation_stats
    G = _ensure_graph()
    stats = get_citation_stats(G, doc_id)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found in graph.")
    return stats


@app.get("/api/graph/authorities/top", tags=["Citation Graph"])
async def get_top_authorities(n: int = Query(20, ge=5, le=50)):
    """Return top-N most authoritative judgments by PageRank score."""
    from graph.citation_network import get_top_authorities
    G = _ensure_graph()
    return get_top_authorities(G, n=n)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — External APIs
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/external/kanoon/search", tags=["External APIs"])
async def kanoon_search(
    q: str = Query(..., description="Search query"),
    page: int = Query(0, ge=0),
    doc_type: Optional[str] = Query(None),
):
    """Proxy a search to Indian Kanoon API."""
    from api_connectors.kanoon_client import KanoonClient
    client = KanoonClient()
    try:
        data = await client.search(
            query=q,
            pagenum=page,
            doc_types=[doc_type] if doc_type else None,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kanoon API error: {exc}")

    docs = data.get("docs", [])
    results = [
        KanoonSearchResult(
            doc_id=str(d.get("tid", "")),
            title=d.get("title", ""),
            headline=d.get("headline", "")[:300],
            publishdate=d.get("publishdate", ""),
            docsource=d.get("docsource", ""),
            numcites=d.get("numcites", 0),
        )
        for d in docs
    ]
    return KanoonSearchResponse(query=q, total=len(results), results=results)


@app.get("/api/external/kanoon/doc/{docid}", tags=["External APIs"])
async def kanoon_get_doc(docid: str):
    """Fetch a full document from Indian Kanoon."""
    from api_connectors.kanoon_client import KanoonClient
    client = KanoonClient()
    try:
        return await client.get_document(docid)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Kanoon API error: {exc}")


@app.get("/api/external/anrak/case/{cnr}", tags=["External APIs"])
async def anrak_case(cnr: str):
    """Fetch live eCourts case status via AnrakLegal API."""
    from api_connectors.anrak_client import AnrakClient
    client = AnrakClient()
    try:
        data = await client.get_case_by_cnr(cnr)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anrak API error: {exc}")

    if data.get("error") == "not_found":
        raise HTTPException(status_code=404, detail=f"Case '{cnr}' not found.")

    return AnrakCaseOut(
        cnr=cnr,
        case_title=data.get("case_title", ""),
        court=data.get("court", ""),
        filing_date=data.get("filing_date", ""),
        status=data.get("status", ""),
        next_hearing=data.get("next_hearing"),
        raw=data,
    )


@app.get("/api/external/anrak/statute", tags=["External APIs"])
async def anrak_statute_search(
    q: str = Query(...),
    act: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
):
    """Search statutes via AnrakLegal API."""
    from api_connectors.anrak_client import AnrakClient
    client = AnrakClient()
    try:
        data = await client.search_statutes(q, act=act, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anrak API error: {exc}")

    results = [
        AnrakStatuteResult(
            section_id=r.get("id", ""),
            act_name=r.get("act_name", ""),
            section_number=r.get("section_number", ""),
            section_title=r.get("section_title", ""),
            text_preview=r.get("text", "")[:300],
        )
        for r in data.get("results", [])
    ]
    return {"query": q, "total": len(results), "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# Routes — Stats
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats", response_model=IndexStats, tags=["Stats"])
async def get_stats():
    """Return overall platform statistics."""
    from vector_store.indexer import get_stats as vs_stats

    vec_stats = vs_stats()

    # Count parsed documents
    parsed_dir = Path(settings.PARSED_DIR)
    doc_count = len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
    # Exclude summary caches from count
    doc_count = len([f for f in parsed_dir.glob("*.json") if "_summary" not in f.name]) if parsed_dir.exists() else 0

    # Graph stats
    graph_nodes = 0
    graph_edges = 0
    if _citation_graph is not None:
        graph_nodes = _citation_graph.number_of_nodes()
        graph_edges = _citation_graph.number_of_edges()

    return IndexStats(
        total_documents=doc_count,
        total_chunks=vec_stats.get("total_chunks", 0),
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        collection_name=vec_stats.get("collection_name", ""),
        persist_dir=vec_stats.get("persist_dir", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True,
        log_level="info",
    )
