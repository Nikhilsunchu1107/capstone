"""Score raw Graph RAG evaluation outputs using a custom LLM-as-a-judge approach and build the final report."""

from __future__ import annotations

import argparse
import json
import os
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# Ensure capstone/ (the project root) is on sys.path so that
# 'graph_rag' is importable as a package when this script is
# run directly from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # capstone/
GRAPH_RAG_DIR = Path(__file__).resolve().parents[1]  # capstone/graph_rag/
ENV_FILE = GRAPH_RAG_DIR / ".env"

# --- Prompt Templates for LLM-as-a-Judge ---

UNIFIED_JUDGE_PROMPT = """
You are an expert evaluation judge. Evaluate the following RAG output on 4 standard metrics.

[Question]
{question}

[Retrieved Context]
{context}

[Generated Answer]
{answer}

[Ground Truth Reference]
{reference}

Evaluate the output carefully:
1. faithfulness (0.0 to 1.0): Are the claims in Generated Answer factually supported by the Retrieved Context?
2. answer_relevancy (0.0 to 1.0): Does the Generated Answer directly address the Question without irrelevant fluff?
3. context_precision (0.0 to 1.0): Is the Retrieved Context relevant to answering the Question?
4. context_recall (0.0 to 1.0): Does the Retrieved Context contain the key information from Ground Truth Reference?

Return your evaluation EXACTLY as a JSON object inside a ```json ``` block with these exact keys:
```json
{{
  "faithfulness": 0.90,
  "answer_relevancy": 0.95,
  "context_precision": 0.80,
  "context_recall": 0.85,
  "reasoning": "brief explanation..."
}}
```
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=GRAPH_RAG_DIR / "outputs/evaluations/policy_eval.json",
        help="Path to raw evaluation JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=GRAPH_RAG_DIR / "outputs/evaluations/policy_eval_summary.json",
        help="Path to save summary scoring report.",
    )
    parser.add_argument(
        "--query-delay",
        type=float,
        default=1.0,
        help="Seconds of delay between sequential LLM calls to stay under rate limits.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["groq", "nim"],
        default=None,
        help="API provider to use. If None, defaults to Groq if key exists, else NIM.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    """Normalize text for metric calculation."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match (0.0 or 1.0)."""
    return 1.0 if _normalize_text(prediction) == _normalize_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = _normalize_text(prediction).split()
    gt_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _load_raw_records(input_path: Path) -> dict[str, Any]:
    """Load raw evaluation JSON from disk."""
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_llm_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Initialize API client using available credentials."""
    groq_key = os.getenv("GROQ_API_KEY")
    nim_key = os.getenv("NVIDIA_NIM_API_KEY")

    # Determine provider (prefer NIM as it has no daily token limits and is verified working)
    if provider is None:
        provider = "nim" if nim_key else "groq"

    if provider == "nim":
        if not nim_key:
            raise RuntimeError("NVIDIA_NIM_API_KEY is missing from environment variables.")
        print("  [LLM Judge] API provider: NVIDIA NIM (Model: meta/llama-3.2-11b-vision-instruct)")
        return OpenAI(
            api_key=nim_key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=45.0,
        ), "meta/llama-3.2-11b-vision-instruct"

    if provider == "groq":
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is missing from environment variables.")
        print("  [LLM Judge] API provider: Groq (Model: groq/compound)")
        return OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=45.0,
        ), "groq/compound"

    raise RuntimeError(f"Unknown provider: {provider}")


def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 30,
    base_delay: float = 4.0,
) -> str:
    """Wrapper to make LLM calls with smart backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            # Parse recommended wait time if provided by API (e.g. "try again in 25.056s" or "try again in 9m14.688s")
            error_str = str(e)
            match = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", error_str, re.IGNORECASE)
            if match:
                minutes = float(match.group(1)) if match.group(1) else 0.0
                seconds = float(match.group(2))
                wait = (minutes * 60.0) + seconds + 3.0
            else:
                wait = min(base_delay * (2 ** min(attempt, 5)), 60.0)
            print(f"    [RateLimit 429] Waiting {wait:.1f} seconds before retrying (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    [API Error] {e}. Retrying in 3.0s...")
            time.sleep(3.0)
    return ""


def _parse_unified_response(content: str) -> dict[str, float]:
    """Robust JSON/Regex parser to extract all 4 metric scores from LLM response."""
    result = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }
    # Try parsing clean ```json ... ``` blocks
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    json_str = match.group(1) if match else content
    try:
        data = json.loads(json_str)
        for k in result:
            if k in data:
                result[k] = max(0.0, min(1.0, float(data[k])))
        return result
    except Exception:
        pass

    # Try checking for braces anywhere in the string
    braces_match = re.search(r"(\{.*\})", content, re.DOTALL)
    if braces_match:
        try:
            data = json.loads(braces_match.group(1))
            for k in result:
                if k in data:
                    result[k] = max(0.0, min(1.0, float(data[k])))
            return result
        except Exception:
            pass

    # Fallback regex for each metric individually
    for k in result:
        m = re.search(rf'"{k}"\s*:\s*([\d\.]+)', content, re.IGNORECASE)
        if m:
            result[k] = max(0.0, min(1.0, float(m.group(1))))

    return result


def _mean_std(values: list[float]) -> dict[str, float]:
    """Calculate mean and standard deviation."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def _build_and_save_report(
    raw: dict[str, Any],
    samples: list[dict[str, Any]],
    metric_values: dict[str, list[float]],
    by_question_type: dict[str, dict[str, list[float]]],
    by_config: dict[str, dict[str, list[float]]],
    per_sample: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Construct and save evaluation summary report."""
    summary_metrics = {
        "context_precision": _mean_std(metric_values["context_precision"]),
        "context_recall": _mean_std(metric_values["context_recall"]),
        "faithfulness": _mean_std(metric_values["faithfulness"]),
        "answer_relevancy": _mean_std(metric_values["answer_relevancy"]),
        "exact_match": _mean_std(metric_values["exact_match"]),
        "f1_score": _mean_std(metric_values["f1_score"]),
        "latency_ms": _mean_std(metric_values["latency_ms"]),
    }

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
            "ragas_sample_size": len(per_sample),
            "ragas_skipped_samples": 0,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Score raw outputs and generate the final report JSON."""
    args = parse_args()
    load_dotenv(ENV_FILE)
    
    raw = _load_raw_records(args.input_path)
    samples = raw.get("samples", [])
    
    if not samples:
        print("No samples found to evaluate.")
        return

    # Check for existing scored output to resume from
    existing_scored: dict[str, dict[str, Any]] = {}
    if args.output_path.exists():
        try:
            with args.output_path.open("r", encoding="utf-8") as f:
                prev_report = json.load(f)
                for s in prev_report.get("samples", []):
                    qid = s.get("question_id")
                    if qid and "faithfulness" in s:
                        existing_scored[qid] = s
            if existing_scored:
                print(f"  [Resume] Found {len(existing_scored)} existing scored samples in {args.output_path}. Resuming...")
        except Exception:
            pass

    # Setup client
    client, model = _get_llm_client(args.provider)

    per_sample = []
    metric_values = defaultdict(list)
    by_question_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "em": []}
    )
    by_config: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"f1": [], "em": [], "faithfulness": []}
    )

    n_samples = len(samples)
    print(f"Evaluating {n_samples} samples using LLM-as-a-Judge...")

    for i, sample in enumerate(samples):
        qid = sample.get("question_id", f"sample_{i}")
        print(f"\n  [Sample {i+1}/{n_samples}] Question ID: {qid}")
        
        prediction = str(sample.get("prediction", "")).strip()
        reference = str(sample.get("reference", "")).strip()
        question = str(sample.get("question", "")).strip()
        question_type = sample.get("question_type", "single_hop")
        config_name = str(sample.get("dataset_name", "unknown")).split("_")[0]
        
        # Local lexical scores
        em = exact_match(prediction, reference)
        f1 = token_f1(prediction, reference)
        latency = float(sample.get("latency_ms", 0.0))

        by_question_type[question_type]["f1"].append(f1)
        by_question_type[question_type]["em"].append(em)
        by_config[config_name]["f1"].append(f1)
        by_config[config_name]["em"].append(em)

        if qid in existing_scored:
            prev = existing_scored[qid]
            f_score = float(prev.get("faithfulness", 0.0))
            r_score = float(prev.get("answer_relevancy", 0.0))
            c_prec = float(prev.get("context_precision", 0.0))
            c_rec = float(prev.get("context_recall", 0.0))
            print(f"    [Cached] Reusing previous scores: Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f} | Precision: {c_prec:.2f} | Recall: {c_rec:.2f}")
        else:
            # Format context
            contexts = [
                str(chunk.get("text", "")).strip()
                for chunk in sample.get("retrieved_chunks", [])
                if str(chunk.get("text", "")).strip()
            ]
            
            # Truncate context chunks to 100 words to save tokens and stay within daily quota
            truncated_contexts = []
            for ctx in contexts:
                words = ctx.split()
                if len(words) > 100:
                    truncated_contexts.append(" ".join(words[:100]) + "...")
                else:
                    truncated_contexts.append(ctx)
            contexts = truncated_contexts

            full_context_str = "\n\n".join(contexts)

            # Check for invalid fields
            if not prediction or not reference or not question or not contexts:
                f_score, r_score, c_prec, c_rec = 0.0, 0.0, 0.0, 0.0
                print("    [Skipped LLM] Missing question/answer/contexts.")
            else:
                prompt_unified = UNIFIED_JUDGE_PROMPT.format(
                    question=question,
                    context=full_context_str,
                    answer=prediction,
                    reference=reference,
                )
                time.sleep(args.query_delay)
                res = _call_llm_with_retry(client, model, prompt_unified)
                scores = _parse_unified_response(res)
                f_score = scores["faithfulness"]
                r_score = scores["answer_relevancy"]
                c_prec = scores["context_precision"]
                c_rec = scores["context_recall"]

            print(f"    Scores -> Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f} | Precision: {c_prec:.2f} | Recall: {c_rec:.2f}")

        # Save scores
        metric_values["faithfulness"].append(f_score)
        metric_values["answer_relevancy"].append(r_score)
        metric_values["context_precision"].append(c_prec)
        metric_values["context_recall"].append(c_rec)
        metric_values["exact_match"].append(em)
        metric_values["f1_score"].append(f1)
        metric_values["latency_ms"].append(latency)
        
        by_config[config_name]["faithfulness"].append(f_score)

        per_sample.append(
            {
                "question_id": qid,
                "dataset_name": sample.get("dataset_name", "unknown"),
                "question_type": question_type,
                "exact_match": em,
                "f1_score": f1,
                "latency_ms": latency,
                "faithfulness": f_score,
                "answer_relevancy": r_score,
                "context_precision": c_prec,
                "context_recall": c_rec,
            }
        )

        # Checkpoint after every sample so no progress is ever lost
        _build_and_save_report(raw, samples, metric_values, by_question_type, by_config, per_sample, args.output_path)

    # Final save
    _build_and_save_report(raw, samples, metric_values, by_question_type, by_config, per_sample, args.output_path)
    print(f"\nSuccessfully saved custom scored report to: {args.output_path}")


if __name__ == "__main__":
    main()
