"""Score raw Graph RAG evaluation outputs using a custom LLM-as-a-judge approach and build the final report."""

from __future__ import annotations

import argparse
import json
import logging
import os
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

FAITHFULNESS_PROMPT = """
You are an evaluation judge. Your task is to evaluate the Faithfulness of an Answer given a Context.
Faithfulness measures whether the facts and claims in the Answer are supported by the Context.

Context:
{context}

Answer:
{answer}

Instructions:
1. Break down the Answer into individual factual claims.
2. For each claim, check if it is directly supported by the Context.
3. Calculate the faithfulness score as: (number of supported claims) / (total number of claims).
   If the Answer contains no claims or is empty, the score is 1.0.
4. Output your reasoning and the final score between 0.0 and 1.0.

Return the output EXACTLY in this JSON format inside a ```json ``` block:
```json
{{
  "reasoning": "Explain your analysis here...",
  "score": 0.85
}}
```
"""

RELEVANCY_PROMPT = """
You are an evaluation judge. Your task is to evaluate the Answer Relevancy.
Answer Relevancy measures how well the Answer directly addresses the Question.

Question:
{question}

Answer:
{answer}

Instructions:
1. Analyze if the Answer directly answers the Question, is on-topic, and does not contain irrelevant or redundant details.
2. Grade the relevancy on a continuous scale from 0.0 (completely irrelevant) to 1.0 (perfectly relevant and direct).
3. Output your reasoning and the final score between 0.0 and 1.0.

Return the output EXACTLY in this JSON format inside a ```json ``` block:
```json
{{
  "reasoning": "Explain your analysis here...",
  "score": 0.90
}}
```
"""

RECALL_PROMPT = """
You are an evaluation judge. Your task is to evaluate the Context Recall.
Context Recall measures if the retrieved Context contains all key points mentioned in the Ground Truth (Reference) answer.

Context:
{context}

Ground Truth Answer:
{reference}

Instructions:
1. Identify all key facts and information points in the Ground Truth Answer.
2. For each key point, check if it is present or mentioned in the Context.
3. Calculate the recall score as: (number of key facts present in Context) / (total number of key facts in Ground Truth).
4. Output your reasoning and the final score between 0.0 and 1.0.

Return the output EXACTLY in this JSON format inside a ```json ``` block:
```json
{{
  "reasoning": "Explain your analysis here...",
  "score": 0.75
}}
```
"""

PRECISION_PROMPT = """
You are an evaluation judge. Your task is to evaluate the Context Precision.
Context Precision measures how relevant the retrieved context chunks are to answering the Question.

Question:
{question}

Retrieved Chunks:
{chunks}

Instructions:
1. Analyze each chunk in the Retrieved Chunks list.
2. For each chunk, determine if it contains information relevant to answering the Question (Yes/No).
3. Calculate precision as: (number of relevant chunks) / (total number of chunks).
4. Output your reasoning and the final score between 0.0 and 1.0.

Return the output EXACTLY in this JSON format inside a ```json ``` block:
```json
{{
  "reasoning": "Explain your analysis here...",
  "score": 0.80
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

    # Determine provider
    if provider is None:
        provider = "groq" if groq_key else "nim"

    if provider == "groq":
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY is missing from environment variables.")
        print("  [LLM Judge] API provider: Groq (Model: groq/compound-mini)")
        return OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=30.0,
        ), "groq/compound-mini"

    if provider == "nim":
        if not nim_key:
            raise RuntimeError("NVIDIA_NIM_API_KEY is missing from environment variables.")
        print("  [LLM Judge] API provider: NVIDIA NIM (Model: meta/llama-3.1-70b-instruct)")
        return OpenAI(
            api_key=nim_key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=30.0,
        ), "meta/llama-3.1-70b-instruct"

    raise RuntimeError(f"Unknown provider: {provider}")


def _call_llm_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 5,
    base_delay: float = 4.0,
) -> str:
    """Wrapper to make LLM calls with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = base_delay * (2 ** attempt)
            print(f"    [RateLimit 429] Waiting {wait:.1f} seconds before retrying...")
            time.sleep(wait)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"    [API Error] {e}. Retrying in 2.0s...")
            time.sleep(2.0)
    return ""


def _parse_score_response(content: str) -> float:
    """Robust JSON/Regex parser to extract the score value from LLM responses."""
    # Try parsing clean ```json ... ``` blocks
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    json_str = match.group(1) if match else content
    try:
        data = json.loads(json_str)
        if "score" in data:
            return float(data["score"])
    except Exception:  
        # Try checking for braces anywhere in the string
        braces_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if braces_match:
            try:
                data = json.loads(braces_match.group(1))
                if "score" in data:
                    return float(data["score"])
            except Exception:
                pass

    # Regex fallback to parse raw "score": value pattern
    score_match = re.search(r'"score"\s*:\s*(0\.\d+|1\.0|0|1)', content)
    if score_match:
        return float(score_match.group(1))

    # General number backup
    num_match = re.search(r"\b(0\.\d+|1\.0|0|1)\b", content)
    if num_match:
        return float(num_match.group(1))

    logger.warning(f"Could not parse score from completion: {content[:200]}")
    return 0.0


def _mean_std(values: list[float]) -> dict[str, float]:
    """Calculate mean and standard deviation."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    """Score raw outputs and generate the final report JSON."""
    args = parse_args()
    load_dotenv(ENV_FILE)
    
    raw = _load_raw_records(args.input_path)
    samples = raw.get("samples", [])
    
    if not samples:
        print("No samples found to evaluate.")
        return

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
        print(f"\n  [Sample {i+1}/{n_samples}] Question ID: {sample.get('question_id', 'unknown')}")
        
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
        
        # Format context
        contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in sample.get("retrieved_chunks", [])
            if str(chunk.get("text", "")).strip()
        ]
        
        # Truncate context chunks to 150 words to avoid entity limits and save tokens
        truncated_contexts = []
        for ctx in contexts:
            words = ctx.split()
            if len(words) > 150:
                truncated_contexts.append(" ".join(words[:150]) + "...")
            else:
                truncated_contexts.append(ctx)
        contexts = truncated_contexts

        full_context_str = "\n\n".join(contexts)
        chunks_precision_str = ""
        for idx, chunk in enumerate(contexts):
            chunks_precision_str += f"Chunk {idx+1}:\n{chunk}\n\n"

        # Check for invalid fields
        if not prediction or not reference or not question or not contexts:
            # Skip LLM calls for invalid or empty samples
            f_score, r_score, c_prec, c_rec = 0.0, 0.0, 0.0, 0.0
            print("    [Skipped LLM] Missing question/answer/contexts.")
        else:
            # Evaluate Faithfulness
            prompt_f = FAITHFULNESS_PROMPT.format(context=full_context_str, answer=prediction)
            time.sleep(args.query_delay)
            res_f = _call_llm_with_retry(client, model, prompt_f)
            f_score = _parse_score_response(res_f)

            # Evaluate Relevancy
            prompt_r = RELEVANCY_PROMPT.format(question=question, answer=prediction)
            time.sleep(args.query_delay)
            res_r = _call_llm_with_retry(client, model, prompt_r)
            r_score = _parse_score_response(res_r)

            # Evaluate Context Precision
            prompt_p = PRECISION_PROMPT.format(question=question, chunks=chunks_precision_str)
            time.sleep(args.query_delay)
            res_p = _call_llm_with_retry(client, model, prompt_p)
            c_prec = _parse_score_response(res_p)

            # Evaluate Context Recall
            prompt_rec = RECALL_PROMPT.format(context=full_context_str, reference=reference)
            time.sleep(args.query_delay)
            res_rec = _call_llm_with_retry(client, model, prompt_rec)
            c_rec = _parse_score_response(res_rec)

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
                "question_id": sample.get("question_id", ""),
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

    # Compute summaries
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
            "ragas_sample_size": len(samples),
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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully saved custom scored report to: {args.output_path}")


if __name__ == "__main__":
    main()
