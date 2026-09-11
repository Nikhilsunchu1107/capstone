"""Generate synthetic insurance QA pairs from the 15-document curated policy set.

Produces ~90 QA pairs (6 per document × 15 documents) saved as
data/policy_qa.json in the same schema as RAGBench, so score_evaluation.py
works without modification.

Usage:
    python scripts/generate_policy_qa.py
    python scripts/generate_policy_qa.py --questions-per-doc 6
    python scripts/generate_policy_qa.py --resume   # continue after a crash
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIStatusError, APITimeoutError

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # capstone/
GRAPH_RAG_DIR = Path(__file__).resolve().parents[1]  # capstone/graph_rag/
for p in [str(PROJECT_ROOT), str(GRAPH_RAG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from graph_rag.config import LLM_MODEL, RANDOM_SEED  # noqa: E402
from src.policy_loader import load_policy_documents  # noqa: E402


# ---------------------------------------------------------------------------
# Question categories — one full rotation per document
# ---------------------------------------------------------------------------
_QA_CATEGORIES = [
    "coverage lookup",
    "exclusion lookup",
    "premium or benefit amount",
    "claim procedure",
    "general / eligibility",
    "waiting period or tenure",
]

_CATEGORY_PROMPTS: dict[str, str] = {
    "coverage lookup": (
        "Generate a question asking what is covered or included under this insurance policy. "
        "The question should be specific and answerable from the text provided."
    ),
    "exclusion lookup": (
        "Generate a question asking what is NOT covered, excluded, or what conditions void "
        "a claim under this insurance policy."
    ),
    "premium or benefit amount": (
        "Generate a question asking about a specific financial amount, sum assured, premium, "
        "payout, or benefit stated in this policy text."
    ),
    "claim procedure": (
        "Generate a question asking how to file a claim, what documents are required, "
        "or what steps must be followed under this insurance policy."
    ),
    "general / eligibility": (
        "Generate a general question about the policy such as who is eligible, "
        "what the policy covers broadly, or what type of insurance this is."
    ),
    "waiting period or tenure": (
        "Generate a question about waiting periods, policy tenure, renewal conditions, "
        "or any time-based restrictions stated in this policy."
    ),
}

_GENERATION_PROMPT_TEMPLATE = """\
You are an insurance domain expert. Read the following insurance policy text carefully.

{category_instruction}

Generate exactly ONE question and ONE concise answer based solely on the text below.
The answer must be directly supported by the text. Do not add information not present.

Format your response EXACTLY as:
Question: <your question here>
Answer: <your answer here>

Policy text:
{chunk_text}
"""

_QA_PARSE_RE = re.compile(
    r"Question:\s*(.+?)\s*Answer:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)

# Similarity threshold for deduplication (per document)
_DEDUP_THRESHOLD = 0.80


def _parse_qa(response_text: str) -> tuple[str, str] | None:
    """Parse 'Question: ... Answer: ...' from LLM output."""
    m = _QA_PARSE_RE.search(response_text)
    if not m:
        return None
    question = m.group(1).strip().splitlines()[0].strip()
    answer = m.group(2).strip()
    if not question or not answer:
        return None
    return question, answer


def _is_duplicate(candidate: str, existing: list[str], threshold: float = _DEDUP_THRESHOLD) -> bool:
    """Return True if *candidate* is too similar to any question in *existing*."""
    for q in existing:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), q.lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def _call_llm_with_retry(
    client: Any,
    prompt: str,
    model: str,
    max_retries: int = 5,
    base_delay: float = 30.0,
) -> str | None:
    """Call the NIM LLM with exponential backoff on rate-limit and transient errors.

    Returns the response text on success, or None if all retries are exhausted.

    Retry strategy:
    - 403 / 429 (rate limit / quota): wait base_delay × 2^attempt seconds.
    - 502 / 503 / 504 (transient server error): wait a shorter fixed delay.
    - Timeout: wait a moderate delay and retry.
    - Other errors: skip immediately (no retry).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                timeout=60,
            )
            return response.choices[0].message.content or ""

        except APIStatusError as exc:
            status = exc.status_code
            if status in (403, 429):  # rate limit / quota
                wait = base_delay * (2 ** attempt)  # 30 s, 60 s, 120 s …
                print(
                    f"  [Rate limit {status}] attempt {attempt + 1}/{max_retries} — "
                    f"waiting {wait:.0f}s before retry..."
                )
                time.sleep(wait)
            elif status in (502, 503, 504):  # transient server error
                wait = 10.0 * (attempt + 1)
                print(
                    f"  [Server error {status}] attempt {attempt + 1}/{max_retries} — "
                    f"waiting {wait:.0f}s before retry..."
                )
                time.sleep(wait)
            else:
                print(f"  [WARN] Unrecoverable API error {status}: {exc}. Skipping chunk.")
                return None

        except APITimeoutError:
            wait = 15.0 * (attempt + 1)
            print(
                f"  [Timeout] attempt {attempt + 1}/{max_retries} — "
                f"waiting {wait:.0f}s before retry..."
            )
            time.sleep(wait)

        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Unexpected error: {exc}. Skipping chunk.")
            return None

    print(f"  [WARN] Exhausted {max_retries} retries. Skipping chunk.")
    return None


def generate_qa_pairs(
    policy_dir: str | Path,
    questions_per_doc: int,
    nim_api_key: str,
    output_path: Path,
    existing_pairs: list[dict[str, Any]] | None = None,
    random_seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Generate *questions_per_doc* QA pairs for each PDF in *policy_dir*.

    Categories are cycled through so each document gets one question of each
    type (up to *questions_per_doc*). Results are checkpointed to *output_path*
    after every document so a crash loses at most one doc's worth of work.

    Args:
        policy_dir: Directory containing the curated policy PDFs.
        questions_per_doc: Number of QA pairs to generate per document.
        nim_api_key: NVIDIA NIM API key for LLM inference.
        output_path: Where to checkpoint / write final results.
        existing_pairs: Pre-loaded pairs from a previous partial run (resume mode).
        random_seed: Random seed for reproducible chunk sampling.

    Returns:
        List of dicts with keys: id, question, answer, source_title,
        filename, category, chunk_id, documents.
    """
    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(
        timeout=60,
        api_key=nim_api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    rng = random.Random(random_seed)  # noqa: S311

    # Load and group chunks by source document
    print(f"Loading policy documents from: {policy_dir}")
    all_chunks = load_policy_documents(policy_dir, verbose=True)

    chunks_by_doc: dict[str, list] = defaultdict(list)
    for chunk in all_chunks:
        chunks_by_doc[chunk.source_id].append(chunk)

    # Resume: identify which source_ids already have pairs
    qa_pairs: list[dict[str, Any]] = list(existing_pairs) if existing_pairs else []
    done_source_ids: set[str] = {
        p["chunk_id"].rsplit("::", 1)[0] for p in qa_pairs
    }
    if done_source_ids:
        print(f"\nResuming: {len(done_source_ids)} documents already done, skipping them.")

    remaining_docs = [
        (sid, chunks)
        for sid, chunks in sorted(chunks_by_doc.items())
        if sid not in done_source_ids
    ]

    total_docs = len(chunks_by_doc)
    done_count = total_docs - len(remaining_docs)
    print(f"\nDocuments loaded: {total_docs} | Remaining: {len(remaining_docs)}")
    print(f"Target: {questions_per_doc} QA pairs × {len(remaining_docs)} remaining docs")
    print("-" * 60)

    def _checkpoint() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(qa_pairs, fh, indent=2, ensure_ascii=False)
        print(f"  💾 Checkpoint: {len(qa_pairs)} pairs saved.")

    for rel_idx, (source_id, doc_chunks) in enumerate(remaining_docs):
        doc_title = doc_chunks[0].source_title
        abs_doc_num = done_count + rel_idx + 1
        print(f"\n[{abs_doc_num}/{total_docs}] {doc_title} ({len(doc_chunks)} chunks)")

        # Shuffle chunks for this document so different runs sample different parts
        shuffled = list(doc_chunks)
        rng.shuffle(shuffled)

        doc_questions: list[str] = []
        cat_idx = 0
        chunk_ptr = 0
        generated = 0

        while generated < questions_per_doc and chunk_ptr < len(shuffled):
            chunk = shuffled[chunk_ptr]
            chunk_ptr += 1

            category = _QA_CATEGORIES[cat_idx % len(_QA_CATEGORIES)]
            category_instruction = _CATEGORY_PROMPTS[category]

            prompt = _GENERATION_PROMPT_TEMPLATE.format(
                category_instruction=category_instruction,
                chunk_text=chunk.text[:1500],
            )

            raw = _call_llm_with_retry(client, prompt, LLM_MODEL)
            time.sleep(2.0)  # gentle pacing between NIM calls
            if raw is None:
                continue  # retries exhausted — move to next chunk

            parsed = _parse_qa(raw)
            if not parsed:
                continue

            question, answer = parsed

            if _is_duplicate(question, doc_questions):
                continue

            doc_questions.append(question)
            qa_pairs.append(
                {
                    "id": f"policy_qa_{len(qa_pairs):04d}",
                    "question": question,
                    "answer": answer,
                    "documents": [chunk.text],
                    "source_title": doc_title,
                    "filename": chunk.metadata.get("filename", ""),
                    "chunk_id": chunk.chunk_id,
                    "category": category,
                }
            )
            generated += 1
            cat_idx += 1
            print(f"  [{generated}/{questions_per_doc}] [{category}] ✓")

        if generated < questions_per_doc:
            print(f"  ⚠ Only generated {generated}/{questions_per_doc} — chunk pool exhausted.")

        # Checkpoint after every completed document
        _checkpoint()

    print(f"\n{'=' * 60}")
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    return qa_pairs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "Policy_Documents_Curated_15"),
        help="Directory containing the 15 curated policy PDF files.",
    )
    parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=6,
        help="QA pairs to generate per document (default: 6 → 90 total for 15 docs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "policy_qa.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for chunk sampling.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If --output already contains partial results, skip completed documents "
            "and append new pairs. Safe to use after any crash or interruption."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: generate QA pairs and save to disk."""
    args = parse_args()
    load_dotenv()

    import os  # noqa: PLC0415

    nim_api_key = os.getenv("NVIDIA_NIM_API_KEY", "")
    if not nim_api_key or nim_api_key == "your_nim_key_here":
        print("Error: NVIDIA_NIM_API_KEY not set in .env")
        sys.exit(1)

    output_path = Path(args.output)

    # Load existing pairs if resuming
    existing_pairs: list[dict[str, Any]] = []
    if args.resume and output_path.exists():
        with open(output_path, encoding="utf-8") as fh:
            existing_pairs = json.load(fh)
        print(f"Resuming from {output_path}: {len(existing_pairs)} pairs already done.")
    elif args.resume:
        print("--resume specified but no existing output found — starting fresh.")

    qa_pairs = generate_qa_pairs(
        policy_dir=args.policy_dir,
        questions_per_doc=args.questions_per_doc,
        nim_api_key=nim_api_key,
        output_path=output_path,
        existing_pairs=existing_pairs,
        random_seed=args.seed,
    )

    # Final definitive save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(qa_pairs, fh, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(qa_pairs)} QA pairs to: {output_path}")

    from collections import Counter  # noqa: PLC0415

    cat_counts = Counter(p["category"] for p in qa_pairs)
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<35s}: {count}")


if __name__ == "__main__":
    main()
