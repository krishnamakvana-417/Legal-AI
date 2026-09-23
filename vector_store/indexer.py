"""
Vector Store Indexer — ChromaDB + Legal-BERT
─────────────────────────────────────────────
Chunks ParsedDocuments, generates embeddings with Legal-BERT (or MiniLM fallback),
and stores them in a persistent ChromaDB collection with cosine-similarity search.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "legal_judgments")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nlpaueb/legal-bert-base-uncased")
EMBEDDING_FALLBACK = os.getenv("EMBEDDING_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")
USE_FALLBACK = os.getenv("USE_FALLBACK_EMBEDDING", "false").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Sections to index (prioritised; full_text used only if section-level missing)
SECTION_PRIORITY = [
    "ratio_decidendi",
    "key_issues",
    "facts",
    "disposition",
    "statutes_cited",
    "procedural_history",
    "full_text",
]

# ── Embedding Function ────────────────────────────────────────────────────────

def _load_embedding_function():
    """Load SentenceTransformer embedding function with Legal-BERT (or fallback)."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    model = EMBEDDING_FALLBACK if USE_FALLBACK else EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model}")

    try:
        ef = SentenceTransformerEmbeddingFunction(
            model_name=model,
            device="cpu",
            normalize_embeddings=True,
        )
        # Warm-up call to catch loading errors early
        ef(["legal judgment test"])
        logger.info(f"✅ Embedding model ready: {model}")
        return ef
    except Exception as exc:
        if model != EMBEDDING_FALLBACK:
            logger.warning(f"Primary model failed ({exc}). Falling back to {EMBEDDING_FALLBACK}")
            fallback_ef = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_FALLBACK,
                device="cpu",
                normalize_embeddings=True,
            )
            return fallback_ef
        raise


# ── ChromaDB Client ───────────────────────────────────────────────────────────

def _get_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(client=None):
    """Get or create the legal judgments ChromaDB collection."""
    if client is None:
        client = _get_client()
    ef = _load_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={
            "hnsw:space": "cosine",
            "description": "Legal judgment chunks — Legal-BERT embeddings",
        },
    )


# ── Chunking ─────────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


def _chunk_document(doc) -> list[dict]:
    """Split a ParsedDocument into embedding-ready chunk dicts."""
    chunks: list[dict] = []

    for section_name in SECTION_PRIORITY:
        if section_name == "full_text":
            text = doc.sections.full_text
        else:
            text = getattr(doc.sections, section_name, "")

        if not text or len(text.strip()) < 80:
            continue

        # Avoid re-indexing the full_text if section content was already indexed
        if section_name == "full_text" and len(chunks) > 0:
            continue

        splits = _splitter.split_text(text)
        for i, chunk_text in enumerate(splits):
            chunk_id = f"{doc.doc_id}_{section_name}_{i}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc.doc_id,
                    "title": doc.title[:200],
                    "court": doc.court[:100],
                    "date": doc.decision_date,
                    "section": section_name,
                    "chunk_index": i,
                    "source_file": doc.source_file,
                    "num_cites": doc.num_cites,
                    "word_count": doc.word_count,
                },
            })

    return chunks


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_documents(
    docs: list,
    reset: bool = False,
    batch_size: int = 250,
) -> int:
    """
    Embed and index a list of ParsedDocuments into ChromaDB.

    Args:
        docs: List of ParsedDocument objects to index.
        reset: If True, wipe collection before indexing.
        batch_size: ChromaDB upsert batch size.

    Returns:
        Number of new chunks indexed.
    """
    from tqdm import tqdm

    client = _get_client()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("🗑️  Cleared existing collection.")
        except Exception:
            pass

    collection = get_collection(client)

    # Get already-indexed doc IDs
    existing_doc_ids: set[str] = set()
    try:
        existing = collection.get(include=["metadatas"])
        existing_doc_ids = {m.get("doc_id", "") for m in existing["metadatas"]}
    except Exception:
        pass

    new_chunks = 0

    for doc in tqdm(docs, desc="🔢 Indexing", unit="doc"):
        if doc.doc_id in existing_doc_ids:
            logger.debug(f"Skipping (indexed): {doc.doc_id}")
            continue

        chunks = _chunk_document(doc)
        if not chunks:
            logger.warning(f"No chunks for document: {doc.title[:60]}")
            continue

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        for i in range(0, len(chunks), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        new_chunks += len(chunks)
        logger.debug(f"Indexed {len(chunks)} chunks — {doc.title[:60]}")

    total = collection.count()
    logger.info(f"✅ Indexing done. New: {new_chunks} chunks | Total in store: {total}")
    return new_chunks


# ── Search ────────────────────────────────────────────────────────────────────

def search(
    query: str,
    top_k: int = 10,
    court_filter: Optional[str] = None,
    section_filter: Optional[str] = None,
) -> list[dict]:
    """
    Semantic similarity search over the indexed judgments.

    Args:
        query: Natural language legal query.
        top_k: Maximum results to return.
        court_filter: Restrict to a specific court name.
        section_filter: Restrict to a specific section (e.g. "ratio_decidendi").

    Returns:
        List of result dicts sorted by descending similarity score.
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    where: Optional[dict] = None
    if court_filter and section_filter:
        where = {"$and": [{"court": court_filter}, {"section": section_filter}]}
    elif court_filter:
        where = {"court": {"$eq": court_filter}}
    elif section_filter:
        where = {"section": {"$eq": section_filter}}

    query_result = collection.query(
        query_texts=[query],
        n_results=min(top_k, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = query_result.get("documents", [[]])[0]
    metadatas = query_result.get("metadatas", [[]])[0]
    distances = query_result.get("distances", [[]])[0]

    results: list[dict] = []
    for text, meta, dist in zip(documents, metadatas, distances):
        results.append({
            "text": text,
            "score": round(1.0 - float(dist), 4),  # cosine distance → similarity
            "doc_id": meta.get("doc_id", ""),
            "title": meta.get("title", ""),
            "court": meta.get("court", ""),
            "date": meta.get("date", ""),
            "section": meta.get("section", ""),
            "source_file": meta.get("source_file", ""),
            "num_cites": meta.get("num_cites", 0),
        })

    return results


def get_document_chunks(doc_id: str) -> list[dict]:
    """Retrieve all chunks stored for a specific document ID."""
    collection = get_collection()
    result = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=["documents", "metadatas"],
    )
    chunks: list[dict] = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        chunks.append({"text": text, **meta})
    return chunks


def get_stats() -> dict:
    """Return collection statistics."""
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        count = 0

    return {
        "total_chunks": count,
        "collection_name": COLLECTION_NAME,
        "persist_dir": CHROMA_PERSIST_DIR,
        "embedding_model": EMBEDDING_FALLBACK if USE_FALLBACK else EMBEDDING_MODEL,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Vector Store Indexer")
    ap.add_argument("--query", help="Run a test search query")
    ap.add_argument("--stats", action="store_true", help="Show index stats")
    ap.add_argument("--index-all", action="store_true", help="Index all parsed documents")
    ap.add_argument("--reset", action="store_true", help="Reset collection before indexing")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.stats:
        import json
        print(json.dumps(get_stats(), indent=2))

    elif args.query:
        results = search(args.query, top_k=args.top_k)
        for r in results:
            print(f"\n[{r['score']:.4f}] {r['title']} | {r['section']}")
            print(f"  {r['text'][:300]}...")

    elif args.index_all:
        from ingestion.parser import load_all_parsed
        docs = load_all_parsed()
        if not docs:
            print("No parsed documents found. Run parser.py --all first.")
            sys.exit(1)
        indexed = index_documents(docs, reset=args.reset)
        print(f"✅ {indexed} chunks indexed.")
