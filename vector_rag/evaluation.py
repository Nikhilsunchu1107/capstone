import json
import os
import pickle
import random
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from bootstrap import build_index
from config import CACHE_DIR
from pipeline import llm, rag_query
from retriever import RAGRetriever

load_dotenv()
EVAL_API_KEY = os.getenv("EVAL_API_KEY")

# ── Single NIM client — used for EVERYTHING in this module ───────────────────
nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=EVAL_API_KEY,
)

GENERATOR_MODEL = "nvidia/nemotron-nano-3-30b-a3b"  # for test-set generation
JUDGE_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # for evaluation scoring
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: If you have access to qwen/qwq-32b, set JUDGE_MODEL to that.
# A bigger judge = more reliable scores. Generator model can stay small.
# ─────────────────────────────────────────────────────────────────────────────

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

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


def nim_call(prompt: str, model: str = GENERATOR_MODEL, max_tokens: int = 512) -> str:
    """
    Single NIM API call with retry logic.
    Sleeps 1s between calls automatically to avoid 429s on free tier.
    """
    for attempt in range(3):
        try:
            time.sleep(1)  # Rate limit buffer — DO NOT REMOVE
            response = nim.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""  # Return empty string on total failure — handled downstream


def generate_test_set(
    chunks,
    questions_per_pdf: int = 2,
    batch_size: int = 5,
    batch_pause: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Generates question-answer pairs from chunks, 2 per PDF.

    Processes in batches of `batch_size` with a `batch_pause` second
    sleep between batches to avoid NVIDIA API rate limits.
    """
    random.seed(seed)

    # Group chunks by source PDF
    from collections import defaultdict

    pdf_chunks: dict[str, list] = defaultdict(list)
    for c in chunks:
        if len(c.page_content) > 200:
            src = c.metadata.get("source_file", "unknown")
            pdf_chunks[src].append(c)

    # Sample 2 chunks from each PDF
    selected = []
    for src, pool in pdf_chunks.items():
        pick = random.sample(pool, min(questions_per_pdf, len(pool)))
        selected.extend(pick)

    random.shuffle(selected)
    total = len(selected)
    print(
        f"Selected {total} chunks from {len(pdf_chunks)} PDFs ({questions_per_pdf} each)\n"
    )

    test_set = []
    batch_num = 0

    for i, chunk in enumerate(selected, 1):
        # Pause between batches
        if i > 1 and (i - 1) % batch_size == 0:
            batch_num += 1
            print(
                f"\n⏸  Batch complete. Sleeping {batch_pause}s to avoid rate limits...\n"
            )
            time.sleep(batch_pause)

        src_file = chunk.metadata.get("source_file", "?")
        page = chunk.metadata.get("page", "?")
        print(f"[{i}/{total}] Source: {src_file} p.{page}")

        prompt = f"""You are an insurance policy expert helping build a test set.

Read the policy text below carefully. Then:
1. Write ONE specific question that a real policyholder would ask about this text.
   - The question must be answerable from this text alone
   - Ask about a concrete detail: a number, a condition, a process, or a coverage rule
   - Do NOT ask vague questions like "what is insurance?"

2. Write the correct answer to that question.
   - Use only information from the text below
   - Be specific and complete (1-3 sentences)
   - Do not add information not present in the text

Policy text:
{chunk.page_content}

Respond in this EXACT JSON format (no extra text, no markdown):
{{"question": "...", "answer": "..."}}"""

        raw = nim_call(prompt, model=GENERATOR_MODEL, max_tokens=300)

        try:
            raw_clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw_clean)
            q = parsed.get("question", "").strip()
            a = parsed.get("answer", "").strip()

            if len(q) > 15 and len(a) > 20:
                test_set.append(
                    {
                        "question": q,
                        "ground_truth": a,
                        "source_chunk": chunk.page_content,
                        "source_file": src_file,
                        "page": page,
                    }
                )
                print(f"  ✅ Q: {q[:80]}...")
            else:
                print(f"  ⚠ Bad parse, skipping")

        except json.JSONDecodeError:
            print(f"  ⚠ JSON decode failed. Raw: {raw[:80]}")

    print(f"\n✅ Generated {len(test_set)} test pairs")
    return test_set


def load_or_generate_test_set(chunks: list) -> list[dict]:
    testset_file = CACHE_DIR / "eval_testset.json"

    if testset_file.exists():
        with open(testset_file) as f:
            test_set = json.load(f)
        print(f"✅ Loaded {len(test_set)} cached test pairs — skipping generation")
        return test_set

    test_set = generate_test_set(chunks, questions_per_pdf=2, batch_size=5, batch_pause=10)
    with open(testset_file, "w") as f:
        json.dump(test_set, f, indent=2)
    print(f"\n✅ Saved to {testset_file}")
    return test_set


def run_vector_rag_pipeline(test_set: list[dict], rag_retriever: RAGRetriever) -> list[dict]:
    """
    Runs Vector RAG on every question and collects outputs.
    """
    outputs = []
    print(f"Running Vector RAG on {len(test_set)} questions...\n")

    for i, item in enumerate(test_set, 1):
        q = item["question"]
        print(f"[{i}/{len(test_set)}] {q[:70]}...")

        try:
            # Direct retrieval for contexts
            retrieved = rag_retriever.retrieve(q, top_k=5, score_threshold=0.2)
            contexts = [doc["content"] for doc in retrieved]

            # Full pipeline for the answer
            result = rag_query(query=q, retriever=rag_retriever, llm=llm, top_k=5)
            answer = result.get("answer", "")
            category = result.get("category", "RELEVANT")

            outputs.append(
                {
                    "question": q,
                    "ground_truth": item["ground_truth"],
                    "contexts": contexts,
                    "answer": answer,
                    "category": category,
                    "source_file": item["source_file"],
                }
            )
            print(f"  ✅ Answer: {answer[:80]}...")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            outputs.append(
                {
                    "question": q,
                    "ground_truth": item["ground_truth"],
                    "contexts": [],
                    "answer": f"ERROR: {e}",
                    "category": "ERROR",
                    "source_file": item["source_file"],
                }
            )

        time.sleep(1)  # Be gentle with NIM free tier

    return outputs


def load_or_run_vector_rag_pipeline(test_set: list[dict], rag_retriever: RAGRetriever) -> list[dict]:
    vector_rag_outputs_file = CACHE_DIR / "vector_rag_outputs.json"

    if vector_rag_outputs_file.exists():
        with open(vector_rag_outputs_file) as f:
            vector_rag_outputs = json.load(f)
        print(f"✅ Loaded cached Vector RAG outputs ({len(vector_rag_outputs)} entries)")
        return vector_rag_outputs

    vector_rag_outputs = run_vector_rag_pipeline(test_set, rag_retriever)
    with open(vector_rag_outputs_file, "w") as f:
        json.dump(vector_rag_outputs, f, indent=2)
    print(f"✅ Saved Vector RAG outputs to {vector_rag_outputs_file}")
    return vector_rag_outputs


def safe_parse_score(raw: str) -> dict:
    """
    Parses NIM's JSON response robustly.
    Handles cases where the model adds markdown fences or extra text.
    Returns {"score": float, "reasoning": str} or a fallback.
    """
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        # Find the JSON object even if there's trailing text
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found")
        parsed = json.loads(clean[start:end])
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        return {"score": score, "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        return {"score": 0.0, "reasoning": f"Parse error: {e} | Raw: {raw[:100]}"}


def evaluate_single(item: dict) -> dict:
    """
    Evaluates one (question, contexts, answer, ground_truth) entry.
    Makes 4 NIM calls sequentially with sleep between each.
    Returns scores + reasoning for all 4 metrics.
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
    raw = nim_call(
        FAITHFULNESS_PROMPT.format(question=q, context=context_joined, answer=a),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["faithfulness"] = safe_parse_score(raw)

    # 2. Answer Relevancy
    raw = nim_call(
        ANSWER_RELEVANCY_PROMPT.format(question=q, answer=a),
        model=JUDGE_MODEL,
        max_tokens=300,
    )
    scores["answer_relevancy"] = safe_parse_score(raw)

    # 3. Context Precision
    raw = nim_call(
        CONTEXT_PRECISION_PROMPT.format(
            question=q, ground_truth=gt, context_numbered=context_numbered
        ),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["context_precision"] = safe_parse_score(raw)

    # 4. Context Recall
    raw = nim_call(
        CONTEXT_RECALL_PROMPT.format(question=q, ground_truth=gt, context=context_joined),
        model=JUDGE_MODEL,
        max_tokens=400,
    )
    scores["context_recall"] = safe_parse_score(raw)

    return scores


def evaluate_pipeline(pipeline_outputs: list[dict], strategy_name: str) -> list[dict]:
    """
    Evaluates all outputs of one RAG strategy.
    Returns a list of per-question results with scores.
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

        scores = evaluate_single(item)

        result = {
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item["ground_truth"],
            "n_contexts": len(item.get("contexts", [])),
            "faithfulness": scores["faithfulness"]["score"],
            "answer_relevancy": scores["answer_relevancy"]["score"],
            "context_precision": scores["context_precision"]["score"],
            "context_recall": scores["context_recall"]["score"],
            "reasoning": scores,
        }
        results.append(result)

        # Print live scores
        print(
            f"  F={result['faithfulness']:.2f}  "
            f"AR={result['answer_relevancy']:.2f}  "
            f"CP={result['context_precision']:.2f}  "
            f"CR={result['context_recall']:.2f}"
        )

    print(f"\n✅ Done: {len(results)}/{total} questions evaluated")
    return results


def load_or_run_evaluation(vector_rag_outputs: list[dict]) -> list[dict]:
    vector_eval_file = CACHE_DIR / "vector_rag_eval_results.json"

    if vector_eval_file.exists():
        with open(vector_eval_file) as f:
            vector_eval_results = json.load(f)
        print(f"✅ Loaded cached Vector RAG eval results ({len(vector_eval_results)} entries)")
        return vector_eval_results

    vector_eval_results = evaluate_pipeline(vector_rag_outputs, "Vector RAG (ChromaDB)")
    with open(vector_eval_file, "w") as f:
        json.dump(vector_eval_results, f, indent=2)
    print(f"✅ Saved to {vector_eval_file}")
    return vector_eval_results


def compute_averages(results: list[dict], strategy_name: str) -> dict:
    if not results:
        return {"Strategy": strategy_name, **{m: None for m in METRICS}}
    df = pd.DataFrame(results)
    avgs = df[METRICS].mean().round(3)
    return {"Strategy": strategy_name, **avgs.to_dict()}


if __name__ == "__main__":
    from retriever import RAGRetriever

    chunks, embedding_manager, vector_store = build_index()
    rag_retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)

    test_set = load_or_generate_test_set(chunks)
    vector_rag_outputs = load_or_run_vector_rag_pipeline(test_set, rag_retriever)
    vector_eval_results = load_or_run_evaluation(vector_rag_outputs)

    summary_rows = [compute_averages(vector_eval_results, "Vector RAG (ChromaDB)")]
    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")

    print("\n" + "=" * 65)
    print("  COMPARATIVE RAG EVALUATION RESULTS")
    print("=" * 65)
    print(summary_df.to_string())
    print("=" * 65)
    print("  All scores 0.0 – 1.0   |   Higher is better")
    print(
        "  F=Faithfulness | AR=Answer Relevancy | CP=Context Precision | CR=Context Recall"
    )
