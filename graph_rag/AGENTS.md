# AGENTS.md

## Scope + Ground Truth
- **Project root** (`../`) has `pyproject.toml` + `uv.lock` that define dependencies.
- `graph_rag/` contains the package code, `config.py`, and `main.py`.
- `../common_rules.md` is the team comparability contract; runtime deviations are in `differences.md`.
- `README.md` is at root level; rely on code + scripts as source of truth.

## Corpus

The active evaluation corpus is the **15-document curated insurance policy set**:
- Location: `../data/Policy_Documents_Curated_15/`
- See `../data/Policy_Documents_Curated_15/SELECTION.md` for document list and selection rationale.
- The larger 72-document set is **not used** for evaluation (rate-limit issue at scoring stage).

## End-to-End Evaluation Workflow

Run all commands from `graph_rag/`:

```bash
# Step 1 — Generate QA pairs (one-time, ~10 min, requires NVIDIA_NIM_API_KEY)
../.venv/bin/python scripts/generate_policy_qa.py
# Output: ../data/policy_qa.json  (~90 pairs, 6 per doc × 15 docs)

# Step 2 — Run Graph-RAG retrieval + generation (~20–30 min)
../.venv/bin/python scripts/run_policy_evaluation.py
# Output: outputs/evaluations/policy_eval.json
# Options: --max-questions 10 (smoke test), --resume (after crash)

# Step 3 — Score with LLM-as-a-judge / RAGAS
../.venv/bin/python scripts/score_evaluation_custom.py \
    --input-path  outputs/evaluations/policy_eval.json \
    --output-path outputs/evaluations/policy_eval_summary.json
```

## Setup That Actually Works
1. From project root:
   ```bash
   mise install python 3.12
   .venv/bin/uv sync
   .venv/bin/python -m spacy download en_core_web_sm
   ```
2. Copy `.env.example` → `graph_rag/.env` and fill in:
   - `NVIDIA_NIM_API_KEY` — required for both generation and scoring
   - `VOYAGE_API_KEY` — required for embeddings (VoyageAI)

## Env Vars
| Variable | Required for | Notes |
|---|---|---|
| `NVIDIA_NIM_API_KEY` | Generation + scoring | NIM free tier: ~1000 RPM |
| `VOYAGE_API_KEY` | Embeddings | Used by `src/vector_store.py` |
| `RAGAS_EMBED_MODEL` | Scoring | Default: `BAAI/bge-m3` (local HuggingFace) |
| `RAGAS_EMBED_DEVICE` | Scoring | Default: `cpu` |

## Code Map
- `main.py` — CLI wrapper for a quick single-query smoke test on policy documents.
- `scripts/generate_policy_qa.py` — generate 90 QA pairs from the 15 curated PDFs.
- `scripts/run_policy_evaluation.py` — run full retrieval + generation over all QA pairs.
- `scripts/score_evaluation_custom.py` — custom LLM-as-a-judge scorer (Faithfulness, Relevancy, Precision, Recall).
- `scripts/score_evaluation_groq.py` — Groq-accelerated scoring variant.
- `scripts/ingest_policy_docs.py` — standalone ingestion CLI (useful for rebuilding the graph cache only).
- `src/pipeline.py` — orchestration (load/chunk → graph+vector retrieval → merge → generate).
- `src/policy_loader.py` — PDF loading and chunking.
- `src/graph.py` — in-memory NetworkX graph index with Document -> Chunk -> Entity hierarchy.
- `src/vector_store.py` — ChromaDB index + VoyageAI / local BGE embeddings backend.
- `src/llm_client.py` — NVIDIA NIM prompt + completion with retry backoff.
- `config.py` — shared constants (model names, chunk size, TOP_K, etc.).

## Output Files (fresh run)
All outputs are **gitignored** and rebuilt on each run:
- `outputs/policy_graph.pkl` — pickled NetworkX graph (15-doc corpus)
- `outputs/policy_chromadb/` — ChromaDB persistence directory
- `outputs/evaluations/policy_eval.json` — raw per-question results
- `outputs/evaluations/policy_eval_summary.json` — final scored report

## Rate Limit Budget
| Stage | Count | Notes |
|---|---|---|
| QA generation | ~90 NIM calls | 1 per QA pair, 1 s pacing |
| Answer generation | ~90 NIM calls | 1.5 s pacing |
| Scoring | ~9 batches × 10 samples | 60 s between batches |

## Current Status
- [x] 15-doc curated corpus selected (`data/Policy_Documents_Curated_15/`)
- [x] RAGBench completely decommissioned
- [x] Scripts updated for 15-doc defaults
- [x] QA pairs generated (`data/policy_qa.json`)
- [x] Evaluation run completed (`outputs/evaluations/policy_eval.json`)
- [x] RAGAS & Custom LLM-as-a-judge scores obtained (0.96 Faithfulness, 0.94 Relevancy)

