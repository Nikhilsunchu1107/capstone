"""CLI entrypoint for running a quick Graph RAG MVP query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_RAG_DIR = Path(__file__).resolve().parent
for p in [str(PROJECT_ROOT), str(GRAPH_RAG_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from graph_rag.config import RANDOM_SEED
from src.pipeline import GraphRAGPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the MVP query run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        type=str,
        default="To what team was the 2014 NBA Rookie of the Year traded in October 2016?",
        help="Question to run against the Graph RAG pipeline.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/ragbench_50",
        help="Path to local sampled RAGBench dataset.",
    )
    return parser.parse_args()


def main() -> None:
    """Build MVP indices and execute one Graph RAG query."""
    args = parse_args()
    load_dotenv()

    pipeline = GraphRAGPipeline()
    pipeline.build_indices(dataset_path=args.dataset_path)
    result = pipeline.query(args.question)

    output = {
        "question": result["question"],
        "answer": result["answer"],
        "retrieved_chunk_ids": [
            item["chunk_id"] for item in result["retrieved_chunks"]
        ],
        "graph_result_count": result["graph_result_count"],
        "vector_result_count": result["vector_result_count"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
