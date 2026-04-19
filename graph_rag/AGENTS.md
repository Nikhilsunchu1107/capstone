# AGENTS.md

## Repo Reality
- This is a single-package Python repo managed with `uv` (`pyproject.toml` + `uv.lock`).
- `main.py` is the working CLI entrypoint; core pipeline lives in `src/`.
- No CI, lint, typecheck, or real test suite is configured (`tests/` is empty).

## Non-Negotiable Team Alignment
- `../common_rules.md` is authoritative for shared experiment settings and comparability.
- Do not change shared config values unilaterally (LLM, embedding model/provider, chunking, `TOP_K`, temperature, seed).
- Current code contains temporary embedding paths (`src/vector_store.py`) that use local BGE or Jina; treat those as implementation workarounds, not new team standards.

## Setup That Actually Works
- From repo root:
  - `uv venv .venv --python 3.12`
  - `source .venv/bin/activate`
  - `uv sync`
  - `uv run python -m spacy download en_core_web_sm`
- Run everything from repo root so default relative paths (`data/...`, `outputs/...`) resolve.

## High-Value Commands
- Build deterministic sample dataset:
  - `uv run python scripts/download_ragbench_sample.py --config hotpotqa --sample-size 50`
- Run one end-to-end query:
  - `uv run python main.py --question "..." --dataset-path data/ragbench_50`

## Data + Reproducibility Gotchas
- Sampling script defaults are part of current MVP contract: `data/ragbench_50` + `data/mvp_eval_indices_50.json`.
- Determinism comes from `RANDOM_SEED` in `config.py`.
- Chroma is persistent at `outputs/chromadb`; if you change dataset shape or indexing assumptions, clear this directory before re-indexing to avoid stale mixed collections.

## Env Vars You Will Need
- Required for generation: `GROQ_API_KEY`.
- Embeddings path in current code:
  - Default is local embeddings (`USE_LOCAL_EMBEDDINGS=1`, model from `LOCAL_EMBED_MODEL`, default `BAAI/bge-m3`).
  - If `USE_LOCAL_EMBEDDINGS=0`, `JINA_API_KEY` is required.
- Keep `.env` untracked; this repo currently does not ignore `.env` by default.

## Code Map (for fast navigation)
- `main.py`: CLI wrapper that builds indices then runs one query.
- `src/pipeline.py`: orchestration (`build_indices`, hybrid retrieve, prompt generation call).
- `src/data_loader.py`: local dataset load + chunk creation.
- `src/graph.py`: NetworkX graph construction + traversal retrieval.
- `src/vector_store.py`: Chroma index + embedding backend selection.
- `src/llm_client.py`: Groq completion client and prompt template.
