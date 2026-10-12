"""The one judge for all three RAG strategies.

Input row: {"question", "answer", "ground_truth", "contexts": [str, ...]}
Output: per-metric {"score": float | None, "reasoning": str}. A score is None when
the judge reply cannot be parsed twice; it is never replaced by a default number.
"""

import json
from concurrent.futures import ThreadPoolExecutor

from eval import config, llm

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness")

_FORMAT = (
    '\n\nRespond with one JSON object only, in this exact shape:\n'
    '{"reasoning": "<your step by step check>", "score": <number between 0.0 and 1.0>}\n'
    'Both keys are required. The "score" key must always be present, as the last key, after the reasoning.'
)

SYSTEM = {
    "faithfulness": """You are an expert evaluator assessing whether an AI answer is faithful to its source context.

TASK:
1. List every factual claim made in the Generated Answer.
2. For each claim, determine if it is directly supported by the Retrieved Context.
3. Score = (number of supported claims) / (total claims). If there are no claims, score 1.0.
If no context was retrieved, score is 0.0.""",
    "answer_relevancy": """You are an expert evaluator assessing whether an AI answer is relevant to the question asked.

TASK:
Score how directly and completely the answer addresses the question.
- 1.0 = answer directly addresses all parts of the question
- 0.7 = answer mostly relevant but misses a part or adds off-topic content
- 0.4 = answer is vaguely related but does not really answer the question
- 0.0 = answer is completely off-topic or refuses to answer""",
    "context_precision": """You are an expert evaluator assessing the quality of retrieved context for a RAG system.

TASK:
For each retrieved chunk, decide if it is relevant to answering the question (given what the ground truth says).
Score = (number of relevant chunks) / (total chunks).
If no chunks were retrieved, score is 0.0.""",
    "context_recall": """You are an expert evaluator assessing whether a RAG system retrieved all necessary information.

TASK:
1. List every key piece of information in the Ground Truth Answer.
2. For each key piece, check if it is present in the Retrieved Context.
3. Score = (pieces present in context) / (total key pieces).
If no context was retrieved, score is 0.0.""",
    "answer_correctness": """You are an expert evaluator assessing whether an AI answer is factually correct against a ground truth answer.

TASK:
1. List every key fact in the Ground Truth Answer.
2. For each key fact, check if the Generated Answer states it correctly. A fact that is missing or contradicted counts as not correct.
3. Score = (key facts stated correctly) / (total key facts). Wording differences do not matter. Extra correct detail does not lower the score. A contradiction of the ground truth caps the score at 0.5.""",
}


def _user_prompt(metric: str, row: dict) -> str:
    ctx = row.get("contexts") or []
    joined = "\n\n".join(ctx) if ctx else "[No context retrieved]"
    numbered = "\n\n".join(f"[Chunk {i + 1}]:\n{c}" for i, c in enumerate(ctx)) if ctx else "[No context retrieved]"
    q, a, gt = row["question"], row["answer"], row["ground_truth"]
    return {
        "faithfulness": f"QUESTION: {q}\n\nRETRIEVED CONTEXT:\n{joined}\n\nGENERATED ANSWER:\n{a}",
        "answer_relevancy": f"QUESTION: {q}\n\nGENERATED ANSWER:\n{a}",
        "context_precision": f"QUESTION: {q}\n\nGROUND TRUTH ANSWER:\n{gt}\n\nRETRIEVED CONTEXT CHUNKS:\n{numbered}",
        "context_recall": f"QUESTION: {q}\n\nGROUND TRUTH ANSWER:\n{gt}\n\nRETRIEVED CONTEXT:\n{joined}",
        "answer_correctness": f"QUESTION: {q}\n\nGROUND TRUTH ANSWER:\n{gt}\n\nGENERATED ANSWER:\n{a}",
    }[metric]


def _parse(raw: str) -> dict:
    obj = json.loads(raw)
    score = float(obj["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score out of range: {score}")
    return {"score": score, "reasoning": str(obj.get("reasoning", ""))}


def score_metric(metric: str, row: dict) -> dict:
    prompt = _user_prompt(metric, row)
    err = raw = ""
    for tag in ("", "retry1"):
        raw = llm.chat(
            prompt,
            system=SYSTEM[metric] + _FORMAT,
            model=config.JUDGE_MODEL,
            max_tokens=config.JUDGE_MAX_TOKENS,
            json_mode=True,
            tag=tag,
        )
        try:
            return _parse(raw)
        except (ValueError, KeyError, TypeError) as e:
            err = repr(e)
    return {"score": None, "reasoning": f"PARSE_FAIL {err} | raw: {raw[:200]}"}


def score_row(row: dict) -> dict:
    return {m: score_metric(m, row) for m in METRICS}


def score_rows(rows: list[dict]) -> list[dict]:
    """Score every row for every metric in parallel. Order is kept."""
    jobs = [(i, m) for i in range(len(rows)) for m in METRICS]
    with ThreadPoolExecutor(config.JUDGE_WORKERS) as pool:
        results = list(pool.map(lambda j: score_metric(j[1], rows[j[0]]), jobs))
    out = [{} for _ in rows]
    for (i, m), r in zip(jobs, results):
        out[i][m] = r
    return out
