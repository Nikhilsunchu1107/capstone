# Strategy Differences Log

This file records differences and evolutions from the initial project planning:

## Phase 4 Update — Insurance Policy Domain (Active)

- **Corpus Transition:** RAGBench has been completely deprecated and removed from the codebase.
- **Active Dataset:** The active evaluation corpus is the **15-document curated insurance policy set** located in `data/Policy_Documents_Curated_15/` accompanied by `SELECTION.md`.
- **Benchmark Test Set:** 90 domain QA pairs generated across the 15 policies stored in `data/policy_qa.json`.
- **Hybrid Domain NER:** Fully integrated GLiNER (`urchade/gliner_medium-v2.1`) combined with spaCy in `src/policy_ner.py` extracting 10 insurance entities.
- **LLM Provider:** NVIDIA NIM (`openai/gpt-oss-120b`) with exponential backoff handling for 429 rate limits in `src/llm_client.py`.


## 1) Embedding model/provider mismatch

- **Expected (common rules):**
  - `EMBED_MODEL = voyage-3-lite`
  - `EMBED_PROVIDER = voyageai`
- **Current MVP:**
  - `graph_rag/config.py:2` sets `EMBED_MODEL = "jina-embeddings-v4"`.
  - `graph_rag/src/vector_store.py` uses local SentenceTransformer embeddings by default (`USE_LOCAL_EMBEDDINGS=1`) and otherwise calls Jina API (`JINA_API_KEY`), not Voyage.
  - Voyage calls are present only as commented reference lines in `graph_rag/src/vector_store.py`.
- **Why this happened:** to avoid external embedding API dependency/cost/rate-limit issues during MVP build and keep local iteration fast.

## 2) Python runtime baseline is stricter than plan

- **Expected (implementation/common rules):** Python 3.10+.
- **Current MVP:** `pyproject.toml:6` requires `>=3.12` (at project root).
- **Why this happened:** local environment standardization during MVP setup.

## 3) MVP sample size differs from Phase 1 recommendation

- **Expected (common rules Phase 1):** hotpotqa sample size 100.
- **Current MVP:**
  - `graph_rag/main.py` defaults to 50 samples.
  - CLI default dataset path is also `../data/ragbench_50`.
- **Why this happened:** faster indexing/query turnaround for MVP debugging.

## 4) Dataset download flow is MVP-centric, not team-shared flow yet

- **Expected (common rules Rule 2.5):**
  - one-time download of required configs to `./data/ragbench/<config>`
  - load with `load_from_disk()`
  - share one team-wide `shared_eval_indices.json`
- **Current MVP:**
  - script pulls one config sample and saves one local sampled dataset folder.
  - indices file is local MVP-specific (`mvp_eval_indices_50.json`), not team-shared mixed-config indices.
- **Why this happened:** MVP focused on proving single-config end-to-end pipeline first.

## 5) NER stack is partial vs implementation guide

- **Expected (implementation plan):** spaCy + GLiNER (zero-shot NER) in ingestion/NER.
- **Current MVP:** `graph_rag/src/ner.py` uses spaCy (`en_core_web_sm`) only; GLiNER is not integrated.
- **Why this happened:** reduce complexity and dependency load for MVP.

## 6) Graph storage layer is still NetworkX-only

- **Expected (overall architecture):** transition to Neo4j in core implementation.
- **Current MVP:** graph retrieval is implemented in-memory with NetworkX (`graph_rag/src/graph.py`); no active Neo4j storage/query path is used in runtime pipeline.
- **Why this happened:** aligns with MVP-first sequencing (NetworkX before Neo4j), but differs from later-phase target architecture.

## 7) Evaluation/experiment tracking not implemented yet

- **Expected (common rules):**
  - full RAGAS + custom metrics
  - standardized JSON results format
  - MLflow logging for every run
- **Current MVP:**
  - current CLI (`graph_rag/main.py`) runs one query and prints basic output fields only.
  - no RAGAS evaluation pipeline, no standardized metrics JSON output, and no MLflow instrumentation in active code.
- **Why this happened:** MVP milestone prioritized retrieval + generation functionality over full benchmarking harness.

## 8) Additional dependencies for local embedding workaround

- **Expected team alignment intent:** Voyage-based embedding path.
- **Current MVP:** root `pyproject.toml` includes `sentence-transformers` and `torch`, supporting `src/embeddings_local.py`.
- **Why this happened:** required for local embedding fallback path used during MVP.

## 9) Note on `.env` handling

- **Rule expectation:** keep secrets in `.env` and never commit it.
- **Current state:** `.env` is already ignored in `graph_rag/.gitignore:13`.
- **Status:** this part is compliant.

## Current recommendation

Keep these differences documented for MVP transparency, then progressively align to team-wide settings before Phase 2/Phase 3 comparative evaluation runs (especially embedding provider/model, dataset sampling protocol, metrics/MLflow, and shared indices).

## Dataset Configuration (Phase 2 Aligned)

The MVP now uses the full RAGBench dataset configs instead of the MVP sample:

- **Available configs:** hotpotqa, msmarco, hagrid (primary), delucionqa, cuad, emanual (secondary)
- **Full dataset:** 2,957 examples (all 6 configs, test split)
- **Sampled (eval-mode):** 700 examples (500 primary + 200 secondary via shared indices)
- **CLI defaults:**
  - Non-eval mode: PRIMARY_CONFIGS (hotpotqa + msmarco + hagrid) = 2,131 examples
  - Eval mode: all 6 configs via `shared_eval_indices.json` = 700 examples
