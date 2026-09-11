"""Score raw Graph RAG evaluation outputs using Groq API and build the final report."""

from __future__ import annotations

import argparse
import json
import os
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from datasets import Dataset
from dotenv import load_dotenv
from ragas import evaluate
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.llms import llm_factory
import sys
from openai import OpenAI
import instructor

# Monkeypatch instructor to use MD_JSON mode so RAGAS can parse JSON from
# reasoning models (like groq/compound) that output thinking/markdown.
_orig_from_openai = instructor.from_openai
def patched_from_openai(client, mode=None, **kwargs):
    return _orig_from_openai(client, mode=instructor.Mode.MD_JSON, **kwargs)
instructor.from_openai = patched_from_openai

# Ensure capstone/ (the project root) is on sys.path so that
# 'graph_rag' is importable as a package when this script is
# run directly from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # capstone/
GRAPH_RAG_DIR = Path(__file__).resolve().parents[1]  # capstone/graph_rag/
ENV_FILE = GRAPH_RAG_DIR / ".env"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the raw JSON file produced by run_evaluation.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/evaluations/policy_eval_summary.json"),
        help="Where to write the final summary JSON report.",
    )
    parser.add_argument(
        "--ragas-batch-size",
        type=int,
        default=10,
        help="Number of samples per RAGAS scoring batch (default: 10).",
    )
    parser.add_argument(
        "--ragas-batch-delay",
        type=float,
        default=30.0,
        help="Seconds to wait between RAGAS batches to avoid rate limits (default: 30).",
    )
    return parser.parse_args()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 when prediction matches ground truth after normalization."""
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_text(text: str) -> str:
    """Normalize text for metric calculation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.strip()


def _load_raw_records(input_path: Path) -> dict[str, Any]:
    """Load raw evaluation JSON from disk."""
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _ragas_dataset(samples: list[dict[str, Any]]) -> tuple[Dataset, int]:
    """Build a RAGAS-compatible dataset from raw samples."""
    rows = []
    skipped = 0
    for sample in samples:
        contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in sample.get("retrieved_chunks", [])
            if str(chunk.get("text", "")).strip()
        ]
        # Truncate context chunks to a max of 150 words to avoid 413 (Entity Too Large)
        # and stay within the 30k Tokens-Per-Minute limit.
        truncated_contexts = []
        for ctx in contexts:
            words = ctx.split()
            if len(words) > 150:
                truncated_contexts.append(" ".join(words[:150]) + "...")
            else:
                truncated_contexts.append(ctx)
        contexts = truncated_contexts

        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("prediction", "")).strip()
        reference = str(sample.get("reference", "")).strip()

        if not question or not answer or not reference or not contexts:
            skipped += 1
            continue

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference,
            }
        )
    return Dataset.from_list(rows), skipped


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return mean and population standard deviation for a list of scores."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def _safe_float(value: Any) -> float:
    """Convert a metric value to float, handling RAGAS scalar wrappers."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _metric_values(scores: Any, name: str) -> list[float]:
    """Extract a metric column as a list of floats from a RAGAS result object."""
    value = scores[name]
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [_safe_float(item) for item in value]
    try:
        return [_safe_float(item) for item in list(value)]
    except TypeError:
        return [_safe_float(value)]


def _run_ragas_scores(
    dataset: Dataset,
    batch_size: int = 10,
    batch_delay: float = 30.0,
) -> dict[str, list[float]]:
    """Compute RAGAS metrics in small batches to avoid Groq rate limits."""
    empty: dict[str, list[float]] = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    if len(dataset) == 0:
        return empty

    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing from environment")

        print("  [RAGAS] Using Groq API for evaluation scoring...")
        client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=60.0,
        )
        # Wrap the create method to add a 1-second delay before every API call
        # to guarantee we stay below the limits on Groq
        _orig_create = client.chat.completions.create
        def delayed_create(*args, **kwargs):
            import time  # noqa: PLC0415
            time.sleep(1.0)
            return _orig_create(*args, **kwargs)
        client.chat.completions.create = delayed_create

        eval_model = "groq/compound-mini"

        llm = llm_factory(
            eval_model,
            provider="openai",
            client=client,
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("RAGAS_EMBED_MODEL", "BAAI/bge-m3"),
            model_kwargs={"device": os.getenv("RAGAS_EMBED_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        logger.exception("RAGAS setup failed")
        return empty

    # Split dataset into batches and collect per-metric lists
    all_results: dict[str, list[float]] = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "answer_relevancy": [],
    }

    n = len(dataset)
    n_batches = (n + batch_size - 1) // batch_size

    from ragas.run_config import RunConfig
    run_config = RunConfig(
        timeout=60,
        max_retries=3,
        max_workers=1,  # Sequential execution to stay under TPM limits
    )

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch = dataset.select(range(start, end))

        print(
            f"  [RAGAS] Scoring batch {batch_idx + 1}/{n_batches} "
            f"(samples {start + 1}–{end})..."
        )

        try:
            scores = evaluate(
                batch,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config,
                raise_exceptions=False,
                show_progress=False,
            )
            for metric in all_results:
                all_results[metric].extend(_metric_values(scores, metric))
        except Exception:
            logger.exception(f"RAGAS batch {batch_idx + 1} failed — filling with zeros")
            batch_len = end - start
            for metric in all_results:
                all_results[metric].extend([0.0] * batch_len)

        # Sleep between batches (not after the last one)
        if batch_idx < n_batches - 1:
            import time  # noqa: PLC0415
            print(f"  [RAGAS] Waiting {batch_delay:.0f}s before next batch...")
            time.sleep(batch_delay)

    return all_results


def main() -> None:
    """Score raw outputs and generate the final report JSON."""
    args = parse_args()
    load_dotenv(ENV_FILE)
    raw = _load_raw_records(args.input_path)
    samples = raw.get("samples", [])

    dataset, skipped_for_ragas = _ragas_dataset(samples)
    metric_values = _run_ragas_scores(
        dataset,
        batch_size=args.ragas_batch_size,
        batch_delay=args.ragas_batch_delay,
    )

    per_sample = []
    metric_series = defaultdict(list)
    by_question_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "em": []}
    )
    by_config: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "em": [], "faithfulness": []}
    )

    for sample in samples:
        prediction = sample.get("prediction", "")
        reference = sample.get("reference", "")
        question_type = sample.get("question_type", "single_hop")
        config_name = str(sample.get("dataset_name", "unknown")).split("_")[0]

        em = exact_match(_normalize_text(prediction), _normalize_text(reference))
        f1 = token_f1(_normalize_text(prediction), _normalize_text(reference))
        latency = float(sample.get("latency_ms", 0.0))

        per_sample.append(
            {
                "question_id": sample.get("question_id", ""),
                "dataset_name": sample.get("dataset_name", "unknown"),
                "question_type": question_type,
                "exact_match": em,
                "f1_score": f1,
                "latency_ms": latency,
            }
        )

        by_question_type[question_type]["f1"].append(f1)
        by_question_type[question_type]["em"].append(em)
        by_config[config_name]["f1"].append(f1)
        by_config[config_name]["em"].append(em)

        metric_series["latency_ms"].append(latency)
        metric_series["exact_match"].append(em)
        metric_series["f1_score"].append(f1)

    summary_metrics = {
        "context_precision": _mean_std(metric_values["context_precision"]),
        "context_recall": _mean_std(metric_values["context_recall"]),
        "faithfulness": _mean_std(metric_values["faithfulness"]),
        "answer_relevancy": _mean_std(metric_values["answer_relevancy"]),
        "exact_match": _mean_std(metric_series["exact_match"]),
        "f1_score": _mean_std(metric_series["f1_score"]),
        "latency_ms": _mean_std(metric_series["latency_ms"]),
    }

    faithfulness_values = metric_values["faithfulness"]
    for sample, faithfulness_score in zip(samples, faithfulness_values, strict=False):
        config_name = str(sample.get("dataset_name", "unknown")).split("_")[0]
        by_config[config_name]["faithfulness"].append(faithfulness_score)

    type_means = [
        _mean_std(values["f1"]) for values in by_question_type.values()
    ]
    config_means = [_mean_std(values["f1"]) for values in by_config.values()]

    variance_across_types = float(np.var([entry["mean"] for entry in type_means])) if type_means else 0.0
    variance_across_configs = float(np.var([entry["mean"] for entry in config_means])) if config_means else 0.0

    report = {
        "strategy_name": raw.get("strategy_name", "graph_rag"),
        "run_started_at": raw.get("run_started_at"),
        "config": raw.get("config", {}),
        "dataset": {
            "primary_configs": raw.get("config", {}).get("configs", []),
            "sample_size": raw.get("sample_size", len(samples)),
            "ragas_sample_size": len(dataset),
            "ragas_skipped_samples": skipped_for_ragas,
        },
        "metrics": summary_metrics,
        "by_question_type": {
            key: {
                "f1": _mean_std(vals["f1"]),
                "em": _mean_std(vals["em"]),
            }
            for key, vals in by_question_type.items()
        },
        "by_config": {
            key: {
                "f1": _mean_std(vals["f1"]),
                "em": _mean_std(vals["em"]),
                "faithfulness": _mean_std(vals["faithfulness"]),
            }
            for key, vals in by_config.items()
        },
        "variance_across_types": variance_across_types,
        "variance_across_configs": variance_across_configs,
        "samples": per_sample,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Saved scored report to {args.output_path}")


if __name__ == "__main__":
    main()
