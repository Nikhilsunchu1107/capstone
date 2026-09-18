import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from client import PROJECT_DIR, REGISTRY_PATH, pi_client
from ollama_client import ollama_call, ollama_call_json
from pageindex import utils
from router import route_query
from search import dedupe_text

GENERATOR_MODEL = "gemma2-8k"  # local Ollama, for test-set generation
JUDGE_MODEL = "granite4.2-8k"  # local Ollama, for evaluation scoring
# *-8k tags are local Ollama models created from Modelfiles with a baked-in
# `PARAMETER num_ctx` — Ollama's OpenAI-compatibility endpoint does not
# reliably honor a per-request num_ctx override via extra_body, so the
# context window has to be set on the model itself instead. granite4.2-8k is
# tagged at 65536; gemma2-8k is tagged at 8192, gemma2:2b's actual native
# max — raising it further wouldn't give it more real context.

# gemma2:2b cannot be given more context via num_ctx than it was trained on
# (8192, its real architectural limit) — Ollama's OpenAI-compat endpoint
# silently truncates an oversized prompt instead of erroring, which is worse
# than a loud failure: the model quietly answers from a content-blind
# fragment and looks like a normal success. GENERATOR_MODEL_CTX_TOKENS lets
# the answer-generation call size its own context to what gemma2-8k can
# actually use, truncating chunks explicitly (and visibly, in the resulting
# context) rather than leaving that to chance.
GENERATOR_MODEL_CTX_TOKENS = 8192
GENERATOR_MODEL_MAX_TOKENS = 384
GENERATOR_MODEL_SAFETY_MARGIN = 200  # buffer for tokenizer-estimate vs. real-model mismatch


def _answer_prompt(question: str, context: str) -> str:
    return f"""Answer the question based only on the context below.
Question: {question}

Context:
{context}

Instructions:
- Use plain simple language
- Start with a one-sentence summary
- End with "Bottom line:" telling the user what to know"""


def _fit_chunks_to_ctx(
    chunks: list[str],
    question: str,
    ctx_window: int = GENERATOR_MODEL_CTX_TOKENS,
    max_tokens: int = GENERATOR_MODEL_MAX_TOKENS,
    safety_margin: int = GENERATOR_MODEL_SAFETY_MARGIN,
) -> str:
    """Join `chunks` (already ranked most-relevant-first) into a context
    string that fits `ctx_window`'s real token budget for GENERATOR_MODEL,
    minus room for the prompt template/question, the model's own output, and
    a safety margin. Chunks are included whole where they fit; the first
    chunk that doesn't fit whole is truncated to the remaining budget
    (visibly marked) instead of silently overflowing the model's context."""
    overhead = utils.count_tokens(_answer_prompt(question, ""), model=None)
    budget = ctx_window - overhead - max_tokens - safety_margin
    if budget <= 0:
        return ""

    included = []
    used = 0
    for chunk in chunks:
        chunk_tokens = utils.count_tokens(chunk, model=None)
        if used + chunk_tokens <= budget:
            included.append(chunk)
            used += chunk_tokens
            continue

        remaining = budget - used
        if remaining > 0:
            lo, hi = 0, len(chunk)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if utils.count_tokens(chunk[:mid], model=None) <= remaining:
                    lo = mid
                else:
                    hi = mid - 1
            if lo > 0:
                included.append(chunk[:lo] + "\n...[truncated to fit generator model's context]")
        break

    return "\n\n---\n\n".join(included)
# ─────────────────────────────────────────────────────────────────────────────
# Bigger judge = more reliable scores. Generator model can stay small.
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache/")
CACHE_DIR.mkdir(exist_ok=True)

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "exact_match",
    "f1_score",
]


# exact_match/token_f1: same as graph_rag/scripts/score_evaluation_custom.py,
# kept identical so scores are comparable across RAG strategies.
def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", "", text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
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

TREE_CACHE: dict = {}
NODE_MAP_CACHE: dict = {}
_REGISTRY_CACHE: dict | None = None

# ── The 4 evaluation prompts ──────────────────────────────────────────────────
# Each returns JSON: {"score": float, "reasoning": str}

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI answer is faithful to its source context.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

TASK:
1. List every factual claim made in the Generated Answer.
2. For each claim, determine if it is directly supported by the Retrieved Context.
3. Score = (number of supported claims) / (total claims). If there are no claims, score 1.0.

Respond in this EXACT JSON format only (no markdown, no extra text):
{{"score": 0.0, "reasoning": "claim 1: supported/not supported because... claim 2: ..."}}

Score must be between 0.0 and 1.0."""


ANSWER_RELEVANCY_PROMPT = """You are an expert evaluator assessing whether an AI answer is relevant to the question asked.

QUESTION: {question}

GENERATED ANSWER:
{answer}

TASK:
Score how directly and completely the answer addresses the question.
- 1.0 = answer directly addresses all parts of the question
- 0.7 = answer mostly relevant but misses a part or adds off-topic content
- 0.4 = answer is vaguely related but does not really answer the question
- 0.0 = answer is completely off-topic or refuses to answer

Respond in this EXACT JSON format only:
{{"score": 0.0, "reasoning": "explanation of why this score was given"}}"""


CONTEXT_PRECISION_PROMPT = """You are an expert evaluator assessing the quality of retrieved context for a RAG system.

QUESTION: {question}

GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXT CHUNKS:
{context_numbered}

TASK:
For each retrieved chunk, decide if it is relevant to answering the question (given what the ground truth says).
Score = (number of relevant chunks) / (total chunks).
If no chunks were retrieved, score is 0.0.

Respond in this EXACT JSON format only:
{{"score": 0.0, "reasoning": "Chunk 1: relevant/not relevant because... Chunk 2: ..."}}"""


CONTEXT_RECALL_PROMPT = """You are an expert evaluator assessing whether a RAG system retrieved all necessary information.

QUESTION: {question}

GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED CONTEXT:
{context}

TASK:
1. List every key piece of information in the Ground Truth Answer.
2. For each key piece, check if it is present in the Retrieved Context.
3. Score = (pieces present in context) / (total key pieces).
If no context was retrieved, score is 0.0.

Respond in this EXACT JSON format only:
{{"score": 0.0, "reasoning": "Key point 1: found/not found in context... Key point 2: ..."}}"""


def load_test_set() -> list[dict]:
    # Shared across all RAG strategies (graph_rag/vector_rag/pageindex_rag) so
    # evaluation questions stay identical for a fair comparison. Generated by
    # graph_rag/scripts/generate_policy_qa.py; only that script writes this file.
    testset_file = PROJECT_DIR / "data" / "policy_qa.json"
    if not testset_file.exists():
        raise FileNotFoundError(
            f"Test set not found at {testset_file}. Generate it first with "
            "graph_rag/scripts/generate_policy_qa.py."
        )
    with open(testset_file) as f:
        raw_test_set = json.load(f)

    # policy_qa.json schema (RAGBench-style) uses "answer"/"filename";
    # the rest of this script expects "ground_truth"/"source_file".
    test_set = [
        {**item, "ground_truth": item["answer"], "source_file": item["filename"]}
        for item in raw_test_set
    ]
    print(f"✅ Loaded {len(test_set)} shared eval pairs from {testset_file}")
    return test_set


def _load_registry_cached() -> dict:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            _REGISTRY_CACHE = json.load(f)
    return _REGISTRY_CACHE


def doc_id_for_eval_item(item: dict, registry: dict) -> str | None:
    """Eval rows already know their source PDF, so avoid an LLM router call."""
    source_file = os.path.basename(item.get("source_file", ""))
    for doc_id, meta in registry.items():
        if os.path.basename(meta["filename"]) == source_file:
            return doc_id
    return None


def get_tree_and_node_map(doc_id: str):
    if doc_id not in TREE_CACHE:
        tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
        TREE_CACHE[doc_id] = tree
        NODE_MAP_CACHE[doc_id] = utils.create_node_mapping(tree)
    return TREE_CACHE[doc_id], NODE_MAP_CACHE[doc_id]


def _snippet_tree(node, max_len=500):
    """Copy of `node`/`nodes` with each 'text' field truncated to a short
    preview instead of removed outright, so node selection sees title +
    summary + a text snippet rather than title + summary alone."""
    if isinstance(node, dict):
        copy = dict(node)
        if "text" in copy and isinstance(copy["text"], str):
            text = copy["text"]
            copy["text"] = text[:max_len] + "..." if len(text) > max_len else text
        if "nodes" in copy:
            copy["nodes"] = _snippet_tree(copy["nodes"], max_len)
        return copy
    if isinstance(node, list):
        return [_snippet_tree(item, max_len) for item in node]
    return node


def search_nodes_cached(doc_id: str, query: str, max_chunks: int = 2) -> list[str]:
    """Same PageIndex tree selection as search_nodes(), but caches local tree work."""
    if not pi_client.is_retrieval_ready(doc_id):
        print(f"  Doc {doc_id} not ready - skipping.")
        return []

    registry = _load_registry_cached()
    filename = registry[doc_id]["filename"]
    tree, node_map = get_tree_and_node_map(doc_id)
    tree_with_snippets = _snippet_tree(tree.copy())

    prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, title, summary, and a short text preview.
Find up to {max_chunks} nodes likely to contain the answer to the question.

Question: {query}

Document tree:
{json.dumps(tree_with_snippets, indent=2)}

Reply ONLY with this JSON:
{{
    "thinking": "<your reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""

    # Local model may wrap the JSON object in prose/markdown, or occasionally
    # drop a malformed one — ollama_call_json retries with a "JSON only" nudge
    # instead of failing on a single bad generation. max_tokens is generous
    # because the requested "thinking" field alone can run past 1-2k tokens
    # on trees with dozens of nodes — too low a budget truncates before the
    # model ever reaches node_list, yielding an empty/incomplete response.
    result = ollama_call_json(prompt, model="granite4.2-8k", max_tokens=4096)
    node_list = result.get("node_list", [])

    # Doc has exactly one node (title+summary too coarse to judge relevance
    # against) — it's the only candidate either way, so use it regardless.
    if not node_list and len(node_map) == 1:
        node_list = list(node_map.keys())

    chunks = []
    for node_id in node_list[:max_chunks]:
        node = node_map.get(node_id)
        if not node:
            continue
        chunks.append(
            f"[Source: {filename}, Page {node['page_index']}]\n{dedupe_text(node['text'])}"
        )

    print(f"  {filename}: {len(chunks)} relevant node(s) found")
    return chunks


def run_pageindex_pipeline(
    items: list[dict],
    sleep_between_questions: float = 0.5,
    max_chunks: int = 2,
    on_result=None,
) -> list[dict]:
    """
    Fast eval path over exactly the given `items` (caller filters out
    already-done ones for resumability):
    - uses each test row's source_file instead of the LLM router
    - searches only that PDF, matching ask()'s specific-document path
    - caches PageIndex trees/node maps
    If `on_result` is given, it's called with each output row as soon as
    it's computed, so a caller can checkpoint to disk incrementally.
    """
    registry = _load_registry_cached()
    outputs = []

    print(f"Running PageIndex RAG on {len(items)} question(s)...\n")

    for offset, item in enumerate(items, 1):
        q = item["question"]
        print(f"[{offset}/{len(items)}] {q[:70]}...")

        try:
            doc_id = doc_id_for_eval_item(item, registry)
            q_type = "eval_source_file"

            if doc_id is None:
                routing = route_query(q)
                q_type = routing["type"]
                doc_ids_to_search = routing.get("doc_ids", [])[:1]
            else:
                doc_ids_to_search = [doc_id]

            all_chunks = []
            for doc_id_to_search in doc_ids_to_search:
                all_chunks.extend(
                    search_nodes_cached(doc_id_to_search, q, max_chunks=max_chunks)
                )

            context = _fit_chunks_to_ctx(all_chunks, q) if all_chunks else ""
            if context:
                answer = ollama_call(
                    _answer_prompt(q, context),
                    model=GENERATOR_MODEL,
                    max_tokens=GENERATOR_MODEL_MAX_TOKENS,
                )
            else:
                answer = "No relevant content found."

            outputs.append(
                {
                    "id": item.get("id", q),
                    "question": q,
                    "ground_truth": item["ground_truth"],
                    "contexts": all_chunks,
                    "answer": answer,
                    "q_type": q_type,
                    "source_file": item["source_file"],
                    "doc_ids_searched": doc_ids_to_search,
                }
            )
            if on_result is not None:
                on_result(outputs[-1])
            print(f"  OK {len(all_chunks)} chunks | Answer: {answer[:60]}...")

        except Exception as e:
            print(f"  ERROR: {e}")
            outputs.append(
                {
                    "id": item.get("id", q),
                    "question": q,
                    "ground_truth": item["ground_truth"],
                    "contexts": [],
                    "answer": f"ERROR: {e}",
                    "q_type": "ERROR",
                    "source_file": item["source_file"],
                    "doc_ids_searched": [],
                }
            )
            if on_result is not None:
                on_result(outputs[-1])

        time.sleep(sleep_between_questions)

    return outputs


def load_or_run_pageindex_pipeline(
    test_set: list[dict],
    max_chunks: int = 2,
) -> list[dict]:
    """Runs the full test_set, resuming by question id: anything already
    present in the cache file is skipped, only missing questions are run."""
    pageindex_outputs_file = CACHE_DIR / "pageindex_outputs.json"

    question_to_id = {item["question"]: item.get("id", item["question"]) for item in test_set}

    existing: dict[str, dict] = {}
    if pageindex_outputs_file.exists():
        with open(pageindex_outputs_file) as f:
            for row in json.load(f):
                key = row.get("id") or question_to_id.get(row["question"], row["question"])
                existing[key] = row
        print(f"Loaded {len(existing)} cached PageIndex output(s) from {pageindex_outputs_file}")

    missing = [item for item in test_set if item.get("id", item["question"]) not in existing]

    if missing:
        def checkpoint(row):
            existing[row["id"]] = row
            with open(pageindex_outputs_file, "w") as f:
                json.dump(list(existing.values()), f, indent=2)

        run_pageindex_pipeline(missing, max_chunks=max_chunks, on_result=checkpoint)
        print(f"Saved {len(existing)} total PageIndex outputs to {pageindex_outputs_file}")
    else:
        print("All questions already have cached PageIndex outputs — nothing to run.")

    # Return in test_set order
    order = {item.get("id", item["question"]): i for i, item in enumerate(test_set)}
    return sorted(existing.values(), key=lambda row: order.get(row.get("id", row["question"]), 1 << 30))


def _score_and_reasoning(parsed: dict) -> dict:
    score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
    return {"score": score, "reasoning": parsed.get("reasoning", "")}


def evaluate_single(item: dict) -> dict:
    """
    Evaluates one (question, contexts, answer, ground_truth) entry.
    Makes 4 judge calls sequentially. Returns scores + reasoning for all 4
    metrics. A judge call that still doesn't parse as JSON after
    ollama_call_json's retries raises — the caller (evaluate_pipeline) skips
    the whole row on exception, so an unparseable judge reply excludes the
    row from the aggregate instead of silently averaging it in as a 0.0.
    """
    q = item["question"]
    a = item["answer"]
    gt = item["ground_truth"]
    ctx = item.get("contexts", [])

    # Build context strings for prompts
    context_joined = "\n\n".join(ctx) if ctx else "[No context retrieved]"
    context_numbered = (
        "\n\n".join(f"[Chunk {i + 1}]:\n{c}" for i, c in enumerate(ctx))
        if ctx
        else "[No context retrieved]"
    )

    scores = {}

    # 1. Faithfulness
    parsed = ollama_call_json(
        FAITHFULNESS_PROMPT.format(question=q, context=context_joined, answer=a),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["faithfulness"] = _score_and_reasoning(parsed)

    # 2. Answer Relevancy
    parsed = ollama_call_json(
        ANSWER_RELEVANCY_PROMPT.format(question=q, answer=a),
        model=JUDGE_MODEL,
        max_tokens=300,
    )
    scores["answer_relevancy"] = _score_and_reasoning(parsed)

    # 3. Context Precision
    parsed = ollama_call_json(
        CONTEXT_PRECISION_PROMPT.format(
            question=q, ground_truth=gt, context_numbered=context_numbered
        ),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["context_precision"] = _score_and_reasoning(parsed)

    # 4. Context Recall
    parsed = ollama_call_json(
        CONTEXT_RECALL_PROMPT.format(question=q, ground_truth=gt, context=context_joined),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["context_recall"] = _score_and_reasoning(parsed)

    return scores


def evaluate_pipeline(pipeline_outputs: list[dict], strategy_name: str, on_result=None) -> list[dict]:
    """
    Evaluates the given pipeline outputs (caller filters out already-scored
    ones for resumability). Returns a list of per-question results with scores.
    If `on_result` is given, it's called with each result as soon as it's
    computed, so a caller can checkpoint to disk incrementally.
    """
    results = []
    total = len(pipeline_outputs)
    print(f"\n{'=' * 60}")
    print(
        f"Evaluating: {strategy_name} ({total} questions × 4 metrics = {total * 4} NIM calls)"
    )
    print(f"{'=' * 60}")

    for i, item in enumerate(pipeline_outputs, 1):
        print(f"\n[{i}/{total}] {item['question'][:65]}...")

        # Skip entries that errored during pipeline run
        if item["answer"].startswith("ERROR") or item["answer"] == "TODO":
            print("  ⏭ Skipped (pipeline error)")
            continue

        try:
            scores = evaluate_single(item)
        except Exception as e:
            print(f"  ⚠ Scoring failed, skipping this question: {e}")
            continue

        result = {
            "id": item.get("id", item["question"]),
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item["ground_truth"],
            "n_contexts": len(item.get("contexts", [])),
            "faithfulness": scores["faithfulness"]["score"],
            "answer_relevancy": scores["answer_relevancy"]["score"],
            "context_precision": scores["context_precision"]["score"],
            "context_recall": scores["context_recall"]["score"],
            "exact_match": exact_match(item["answer"], item["ground_truth"]),
            "f1_score": token_f1(item["answer"], item["ground_truth"]),
            "reasoning": scores,
        }
        results.append(result)
        if on_result is not None:
            on_result(result)

        # Print live scores
        print(
            f"  F={result['faithfulness']:.2f}  "
            f"AR={result['answer_relevancy']:.2f}  "
            f"CP={result['context_precision']:.2f}  "
            f"CR={result['context_recall']:.2f}  "
            f"EM={result['exact_match']:.2f}  "
            f"F1={result['f1_score']:.2f}"
        )

    print(f"\n✅ Done: {len(results)}/{total} questions evaluated")
    return results


def load_or_run_evaluation(pageindex_outputs: list[dict]) -> list[dict]:
    """Resumes by question id: only outputs not already scored are evaluated."""
    pageindex_eval_file = CACHE_DIR / "pageindex_eval_results.json"

    question_to_id = {item["question"]: item.get("id", item["question"]) for item in pageindex_outputs}

    existing: dict[str, dict] = {}
    if pageindex_eval_file.exists():
        with open(pageindex_eval_file) as f:
            for row in json.load(f):
                key = row.get("id") or question_to_id.get(row["question"], row["question"])
                existing[key] = row
        print(f"✅ Loaded {len(existing)} cached PageIndex eval result(s) from {pageindex_eval_file}")

    missing = [
        item for item in pageindex_outputs
        if item.get("id", item["question"]) not in existing
    ]

    if missing:
        def checkpoint(row):
            existing[row["id"]] = row
            with open(pageindex_eval_file, "w") as f:
                json.dump(list(existing.values()), f, indent=2)

        evaluate_pipeline(missing, "PageIndex RAG", on_result=checkpoint)
        print(f"✅ Saved {len(existing)} total eval results to {pageindex_eval_file}")
    else:
        print("All pipeline outputs already evaluated — nothing to score.")

    order = {item.get("id", item["question"]): i for i, item in enumerate(pageindex_outputs)}
    return sorted(existing.values(), key=lambda row: order.get(row.get("id", row["question"]), 1 << 30))


def compute_averages(results: list[dict], strategy_name: str) -> dict:
    if not results:
        return {"Strategy": strategy_name, **{m: None for m in METRICS}}
    df = pd.DataFrame(results)
    avgs = df[METRICS].mean().round(3)
    return {"Strategy": strategy_name, **avgs.to_dict()}


if __name__ == "__main__":
    test_set = load_test_set()
    pageindex_outputs = load_or_run_pageindex_pipeline(test_set)
    pageindex_eval_results = load_or_run_evaluation(pageindex_outputs)

    summary_df = pd.DataFrame(
        [compute_averages(pageindex_eval_results, "PageIndex RAG")]
    ).set_index("Strategy")
    print(summary_df.to_string())
