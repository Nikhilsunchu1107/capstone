"""Run Graph RAG evaluation on the 15-document curated insurance policy set.

Loads (or builds) the policy graph + ChromaDB index for the 15-doc corpus,
runs all questions through the pipeline, and saves per-question results in
the same schema as graph_rag_eval.json so score_evaluation.py works without
modification.

Usage:
    python scripts/run_policy_evaluation.py
    python scripts/run_policy_evaluation.py --max-questions 20   # smoke test
    python scripts/run_policy_evaluation.py --resume             # after crash
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # capstone/
GRAPH_RAG_DIR = Path(__file__).resolve().parents[1]  # capstone/graph_rag/
for p in [str(PROJECT_ROOT), str(GRAPH_RAG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from graph_rag.config import RANDOM_SEED  # noqa: E402
from src.pipeline import GraphRAGPipeline  # noqa: E402

# Pause between NIM calls to stay within token-per-minute limits.
# 1.5 s ≈ 40 req/min sustained — comfortable margin for the free tier.
INTER_REQUEST_DELAY = 1.5


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the policy evaluation runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "policy_qa.json"),
        help="Path to the synthetic policy QA JSON file (from generate_policy_qa.py).",
    )
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "Policy_Documents_Curated_15"),
        help="Directory containing the 15 curated policy PDFs.",
    )
    parser.add_argument(
        "--graph-cache",
        type=str,
        default=str(GRAPH_RAG_DIR / "outputs" / "policy_graph.pkl"),
        help="Path to persist / load the pickled NetworkX graph.",
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
        help="ChromaDB collection name for policy chunks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=GRAPH_RAG_DIR / "outputs" / "evaluations" / "policy_eval.json",
        help="Where to write the raw evaluation results JSON.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing a smaller subset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If the output file already contains partial results, skip those "
            "questions and append new ones. Useful after a rate-limit crash."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=INTER_REQUEST_DELAY,
        help=f"Seconds to sleep between NIM calls (default: {INTER_REQUEST_DELAY}).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore the graph cache and rebuild from scratch.",
    )
    return parser.parse_args()


def _infer_question_type(question: str) -> str:
    """Assign a coarse question type for variance reporting."""
    q = question.lower().strip()
    if not q:
        return "single_hop"

    negative_patterns = (" not ", " except ", " false", " incorrect", " excluded", " never ")
    conflicting_patterns = ("compare", "difference", "versus", " vs ", "contradict", "conflict")
    multi_hop_patterns = (
        " and ", " both ", " first ", " second ", " former ", " latter ",
        "which of the following", " before ", " after ",
    )

    if any(p in q for p in negative_patterns):
        return "negative"
    if any(p in q for p in conflicting_patterns):
        return "conflicting"
    if any(p in q for p in multi_hop_patterns) or q.count("?") > 1:
        return "multi_hop"
    return "single_hop"


def _load_checkpoint(output_path: Path) -> list[dict[str, Any]]:
    """Load already-completed samples from a partial output file."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("samples", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _save_checkpoint(
    output_path: Path,
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    started_at: datetime,
    run_started: float,
) -> None:
    """Write partial results to disk immediately."""
    payload = {
        "strategy_name": "graph_rag_policy",
        "run_started_at": started_at.isoformat(),
        "runtime_seconds": round(time.perf_counter() - run_started, 3),
        "config": config,
        "sample_size": len(samples),
        "samples": samples,
        "metrics": {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def main() -> None:
    """Build/load policy indices and run evaluation on all QA pairs."""
    args = parse_args()
    load_dotenv()

    # --- Load QA pairs ---
    qa_path = Path(args.qa_file)
    if not qa_path.exists():
        print(f"Error: QA file not found: {qa_path}")
        print("Run scripts/generate_policy_qa.py first to generate it.")
        sys.exit(1)

    with open(qa_path, encoding="utf-8") as fh:
        qa_pairs: list[dict[str, Any]] = json.load(fh)

    if args.max_questions is not None:
        qa_pairs = qa_pairs[: args.max_questions]

    print(f"Loaded {len(qa_pairs)} QA pairs from: {qa_path}")

    output_path = args.output_path

    # --- Resume support ---
    already_done: set[str] = set()
    samples: list[dict[str, Any]] = []
    if args.resume:
        samples = _load_checkpoint(output_path)
        already_done = {s["question_id"] for s in samples}
        if already_done:
            print(f"Resuming: {len(already_done)} questions already completed, skipping them.")

    # --- Build / load pipeline ---
    pipeline = GraphRAGPipeline(policy_mode=True)
    pipeline.build_indices_from_policy_docs(
        policy_dir=args.policy_dir,
        graph_cache_path=args.graph_cache,
        chroma_persist_dir=args.chroma_dir,
        chroma_collection=args.chroma_collection,
        force_rebuild=args.force_rebuild,
    )

    # --- Run evaluation ---
    started_at = datetime.now(timezone.utc)
    run_started = time.perf_counter()

    run_config = {
        "qa_file": str(qa_path),
        "policy_dir": args.policy_dir,
        "graph_cache": args.graph_cache,
        "chroma_dir": args.chroma_dir,
        "chroma_collection": args.chroma_collection,
        "random_seed": RANDOM_SEED,
        "max_questions": args.max_questions,
        "inter_request_delay": args.delay,
    }

    remaining = [
        (idx, qa) for idx, qa in enumerate(qa_pairs)
        if str(qa.get("id", f"policy_qa_{idx:04d}")) not in already_done
    ]

    print(f"\nRunning evaluation on {len(remaining)} questions "
          f"(inter-request delay: {args.delay}s)...")
    print("-" * 60)

    errors = 0
    for batch_idx, (idx, qa) in enumerate(remaining):
        question = str(qa.get("question", "")).strip()
        if not question:
            continue

        # Inter-request delay — applied before every call except the first
        if batch_idx > 0:
            time.sleep(args.delay)

        query_started = time.perf_counter()
        try:
            result = pipeline.query(question)
            latency_ms = (time.perf_counter() - query_started) * 1000.0
            prediction = result["answer"]
            context = result["context"]
            retrieved_chunks = result["retrieved_chunks"]
            graph_result_count = result["graph_result_count"]
            vector_result_count = result["vector_result_count"]
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - query_started) * 1000.0
            errors += 1
            print(f"  [ERROR] Q{idx} failed: {exc}")
            prediction = ""
            context = ""
            retrieved_chunks = []
            graph_result_count = 0
            vector_result_count = 0

        samples.append(
            {
                "row_index": idx,
                "question_id": str(qa.get("id", f"policy_qa_{idx:04d}")),
                "dataset_name": "policy_documents",
                "question": question,
                "reference": str(qa.get("answer", "")),
                "question_type": _infer_question_type(question),
                "category": qa.get("category", ""),
                "source_title": qa.get("source_title", ""),
                "prediction": prediction,
                "context": context,
                "retrieved_chunks": retrieved_chunks,
                "graph_result_count": graph_result_count,
                "vector_result_count": vector_result_count,
                "latency_ms": latency_ms,
            }
        )

        # Checkpoint every 10 questions
        if len(samples) % 10 == 0:
            _save_checkpoint(output_path, samples, run_config, started_at, run_started)
            total_done = len(already_done) + batch_idx + 1
            avg_lat = sum(s["latency_ms"] for s in samples) / len(samples)
            print(
                f"  [{total_done}/{len(qa_pairs)}] "
                f"avg latency: {avg_lat:.0f}ms | errors so far: {errors} | checkpoint saved"
            )

    # --- Final save ---
    _save_checkpoint(output_path, samples, run_config, started_at, run_started)

    runtime = round(time.perf_counter() - run_started, 3)
    avg_latency = sum(s["latency_ms"] for s in samples) / len(samples) if samples else 0
    total_completed = len(already_done) + len(samples)

    print()
    print("=" * 60)
    print("Policy evaluation complete!")
    print(f"  Questions evaluated : {total_completed}/{len(qa_pairs)}")
    print(f"  Errors              : {errors}")
    print(f"  Total runtime       : {runtime:.1f}s")
    print(f"  Avg latency/query   : {avg_latency:.0f}ms")
    print(f"  Results saved to    : {output_path}")
    print("=" * 60)
    print("\nNext step: run score_evaluation.py to compute RAGAS + F1 metrics.")


if __name__ == "__main__":
    main()
