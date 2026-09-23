"""
Citation Network Graph Builder
────────────────────────────────
Builds and persists a directed citation graph (NetworkX DiGraph) from parsed
judgment metadata. Computes PageRank authority scores and exposes traversal
queries (precedent trees, citation stats) + D3-compatible JSON export.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

GRAPH_PATH = Path(os.getenv("GRAPH_PERSIST_PATH", "./data/citation_graph.gpickle"))


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph(docs: list) -> nx.DiGraph:
    """
    Build a directed citation graph from a list of ParsedDocuments.

    Nodes: each judgment (keyed by doc_id).
    Edges: doc_A → doc_B means doc_A cites doc_B.

    Node attributes stored:
        title, court, date, num_cites, num_cited_by, case_number, source_file

    Edge attributes stored:
        citation_text (the citation string as found in the judgment text)
    """
    G = nx.DiGraph()

    # ── Pass 1: Add all nodes ─────────────────────────────────────────────────
    id_to_doc: dict[str, object] = {}
    for doc in docs:
        G.add_node(
            doc.doc_id,
            title=doc.title,
            court=doc.court,
            date=doc.decision_date,
            num_cites=doc.num_cites,
            num_cited_by=doc.num_cited_by,
            case_number=doc.case_number,
            source_file=doc.source_file,
            authority_score=0.0,
        )
        id_to_doc[doc.doc_id] = doc

    # ── Build title → doc_id index for citation matching ────────────────────
    title_index: dict[str, str] = {
        doc.title.lower().strip(): doc.doc_id
        for doc in docs if doc.title
    }

    # ── Pass 2: Add citation edges ────────────────────────────────────────────
    for doc in docs:
        for cite_str in doc.cite_list:
            # Try to match citation string to a known document by title substring
            matched_id: Optional[str] = None
            cite_lower = cite_str.lower().strip()
            for known_title, known_id in title_index.items():
                if cite_lower in known_title or known_title in cite_lower:
                    matched_id = known_id
                    break

            if matched_id and matched_id != doc.doc_id:
                G.add_edge(doc.doc_id, matched_id, citation_text=cite_str)
            else:
                # External citation (not in our corpus) — add as external node
                ext_id = f"ext::{cite_str[:50]}"
                if not G.has_node(ext_id):
                    G.add_node(
                        ext_id,
                        title=cite_str,
                        court="External",
                        date="",
                        num_cites=0,
                        num_cited_by=0,
                        case_number="",
                        source_file="",
                        authority_score=0.0,
                        is_external=True,
                    )
                G.add_edge(doc.doc_id, ext_id, citation_text=cite_str)

    # ── Pass 3: Update num_cited_by from in-degree ───────────────────────────
    for node_id in G.nodes:
        G.nodes[node_id]["num_cited_by"] = G.in_degree(node_id)

    # ── Pass 4: Compute PageRank authority scores ─────────────────────────────
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)
        for node_id, score in pagerank.items():
            G.nodes[node_id]["authority_score"] = round(score, 6)
    except Exception as exc:
        logger.warning(f"PageRank computation failed: {exc}")

    logger.info(
        f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges."
    )
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Graph saved → {path}")


def load_graph(path: Path = GRAPH_PATH) -> Optional[nx.DiGraph]:
    if not path.exists():
        logger.warning(f"Graph file not found: {path}")
        return None
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# Query Functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_precedent_tree(G: nx.DiGraph, doc_id: str, depth: int = 2) -> dict:
    """
    Return the subgraph of all cases cited BY doc_id, up to `depth` hops.
    Represents the precedent authority tree.
    """
    if doc_id not in G:
        return {"nodes": [], "links": [], "center_node": doc_id}

    # BFS outward (following citation edges)
    reachable = {doc_id}
    frontier = {doc_id}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(G.successors(node))
        reachable.update(next_frontier)
        frontier = next_frontier

    subgraph = G.subgraph(reachable)
    return _graph_to_d3(subgraph, center_node=doc_id)


def get_citing_tree(G: nx.DiGraph, doc_id: str, depth: int = 2) -> dict:
    """
    Return the subgraph of cases that CITE doc_id (reverse direction), up to `depth` hops.
    Represents cases that treat this judgment as precedent.
    """
    if doc_id not in G:
        return {"nodes": [], "links": [], "center_node": doc_id}

    reachable = {doc_id}
    frontier = {doc_id}
    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            next_frontier.update(G.predecessors(node))
        reachable.update(next_frontier)
        frontier = next_frontier

    subgraph = G.subgraph(reachable)
    return _graph_to_d3(subgraph, center_node=doc_id)


def get_citation_stats(G: nx.DiGraph, doc_id: str) -> dict:
    """Return citation statistics for a single document."""
    if doc_id not in G:
        return {}

    attrs = G.nodes[doc_id]
    return {
        "doc_id": doc_id,
        "title": attrs.get("title", ""),
        "num_cites": G.out_degree(doc_id),      # cites others
        "num_cited_by": G.in_degree(doc_id),    # cited by others
        "authority_score": attrs.get("authority_score", 0.0),
        "cites": [
            {"doc_id": t, "citation_text": d.get("citation_text", "")}
            for _, t, d in G.out_edges(doc_id, data=True)
        ],
        "cited_by": [
            {"doc_id": s, "citation_text": d.get("citation_text", "")}
            for s, _, d in G.in_edges(doc_id, data=True)
        ],
    }


def get_top_authorities(G: nx.DiGraph, n: int = 20) -> list[dict]:
    """Return top-N most authoritative nodes by PageRank score."""
    nodes = [
        {
            "doc_id": node_id,
            **{k: v for k, v in G.nodes[node_id].items()},
        }
        for node_id in G.nodes
        if not G.nodes[node_id].get("is_external", False)
    ]
    return sorted(nodes, key=lambda x: x.get("authority_score", 0), reverse=True)[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# D3 Export
# ═══════════════════════════════════════════════════════════════════════════════

def _graph_to_d3(G: nx.DiGraph, center_node: Optional[str] = None) -> dict:
    """Convert a NetworkX graph to a D3-compatible JSON structure."""
    # Assign group IDs by court (for frontend colour coding)
    courts = list({G.nodes[n].get("court", "") for n in G.nodes})
    court_group: dict[str, int] = {c: i for i, c in enumerate(courts)}

    nodes = []
    for node_id in G.nodes:
        attrs = G.nodes[node_id]
        nodes.append({
            "id": node_id,
            "title": attrs.get("title", node_id)[:120],
            "court": attrs.get("court", ""),
            "date": attrs.get("date", ""),
            "num_cites": attrs.get("num_cites", 0),
            "num_cited_by": G.in_degree(node_id),
            "authority_score": round(attrs.get("authority_score", 0.0), 6),
            "group": court_group.get(attrs.get("court", ""), 0),
            "is_external": attrs.get("is_external", False),
            "is_center": node_id == center_node,
        })

    links = [
        {
            "source": u,
            "target": v,
            "citation_text": d.get("citation_text", ""),
        }
        for u, v, d in G.edges(data=True)
    ]

    return {
        "nodes": nodes,
        "links": links,
        "center_node": center_node,
        "total_nodes": len(nodes),
        "total_edges": len(links),
    }


def export_full_graph(G: nx.DiGraph) -> dict:
    """Export the entire graph as D3 JSON."""
    return _graph_to_d3(G)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Helper
# ═══════════════════════════════════════════════════════════════════════════════

def build_and_save(docs: list) -> nx.DiGraph:
    """Build citation graph from docs, compute PageRank, and persist."""
    G = build_graph(docs)
    save_graph(G)
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Citation Network Graph Builder")
    ap.add_argument("--build", action="store_true", help="Build graph from parsed docs")
    ap.add_argument("--export", action="store_true", help="Export graph as D3 JSON")
    ap.add_argument("--stats", default=None, help="Show stats for doc_id")
    ap.add_argument("--top", type=int, default=10, help="Top authorities")
    args = ap.parse_args()

    if args.build:
        from ingestion.parser import load_all_parsed
        docs = load_all_parsed()
        if not docs:
            print("No parsed documents found. Run parser.py --all first.")
            sys.exit(1)
        G = build_and_save(docs)
        print(f"✅ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    elif args.export:
        G = load_graph()
        if G:
            data = export_full_graph(G)
            out = Path("./data/citation_graph.json")
            out.write_text(json.dumps(data, indent=2))
            print(f"✅ Exported to {out}")

    elif args.stats:
        G = load_graph()
        if G:
            stats = get_citation_stats(G, args.stats)
            print(json.dumps(stats, indent=2))

    else:
        G = load_graph()
        if G:
            top = get_top_authorities(G, n=args.top)
            print(f"\nTop {args.top} authorities:")
            for doc in top:
                print(f"  [{doc['authority_score']:.5f}] {doc['title'][:80]}")
