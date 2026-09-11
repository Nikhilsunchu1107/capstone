"""One-shot CLI script to ingest all insurance policy PDFs into Graph RAG indices.

Usage:
    python scripts/ingest_policy_docs.py
    python scripts/ingest_policy_docs.py --policy-dir data/Policy_Documents
    python scripts/ingest_policy_docs.py --force-rebuild   # ignore graph cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # capstone/
GRAPH_RAG_DIR = Path(__file__).resolve().parents[1]  # capstone/graph_rag/
for p in [str(PROJECT_ROOT), str(GRAPH_RAG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline import GraphRAGPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the policy ingestion script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "Policy_Documents"),
        help="Directory containing the policy PDF files.",
    )
    parser.add_argument(
        "--graph-cache",
        type=str,
        default=str(GRAPH_RAG_DIR / "outputs" / "policy_graph.pkl"),
        help="Path to persist/load the pickled NetworkX graph.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=str(GRAPH_RAG_DIR / "outputs" / "policy_chromadb"),
        help="ChromaDB persistence directory for policy chunks.",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        default="policy_chunks",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore graph cache and rebuild from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    """Build graph and vector indices from all policy PDFs."""
    args = parse_args()
    load_dotenv()

    print("=" * 70)
    print("Graph RAG — Insurance Policy Document Ingestion")
    print("=" * 70)
    print(f"  Policy dir    : {args.policy_dir}")
    print(f"  Graph cache   : {args.graph_cache}")
    print(f"  ChromaDB dir  : {args.chroma_dir}")
    print(f"  Force rebuild : {args.force_rebuild}")
    print()

    t0 = time.perf_counter()

    pipeline = GraphRAGPipeline(policy_mode=True)
    pipeline.build_indices_from_policy_docs(
        policy_dir=args.policy_dir,
        graph_cache_path=args.graph_cache,
        chroma_persist_dir=args.chroma_dir,
        chroma_collection=args.chroma_collection,
        force_rebuild=args.force_rebuild,
    )

    elapsed = time.perf_counter() - t0

    # --- Print ingestion summary ---
    graph = pipeline.graph_index.graph
    n_docs = sum(
        1 for _, d in graph.nodes(data=True) if d.get("node_type") == "document"
    )
    n_chunks = len(pipeline.graph_index.chunk_lookup)
    n_entities = sum(
        1 for _, d in graph.nodes(data=True) if d.get("node_type") == "entity"
    )
    n_edges = graph.number_of_edges()

    print()
    print("=" * 70)
    print("Ingestion complete!")
    print(f"  Documents (graph nodes) : {n_docs}")
    print(f"  Chunks indexed          : {n_chunks}")
    print(f"  Entity nodes            : {n_entities}")
    print(f"  Graph edges             : {n_edges}")
    print(f"  Total time              : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
