# AGENTS.md

## Scope + Ground Truth
- **Project root** (`../`) has `pyproject.toml` + `uv.lock` that define dependencies.
- `graph_rag/` contains the package code, `config.py`, and `main.py`.
- `../common_rules.md` is the team comparability contract; runtime deviations are in `differences.md`.
- `README.md` is at root level; rely on code + scripts as source of truth.

## Setup That Actually Works
1. From project root:
   ```bash
   # Install mise if not present, then:
   mise install python 3.12
   .venv/bin/uv sync
   .venv/bin/python -m spacy download en_core_web_sm
   ```
2. Run from `graph_rag/`:
   ```bash
   cd graph_rag
   ../.venv/bin/python main.py --question "..." --dataset-path ../data/ragbench_50
   ```

## Fast Run Commands
- Build evaluation indices (from project root):
  - `../.venv/bin/python scripts/generate_eval_samples.py`
- Run one end-to-end query:
  - `cd graph_rag && ../.venv/bin/python main.py --question "..." --dataset-path ../data/ragbench_50`

## Current Runtime Behavior (Easy To Miss)
- `main.py` defaults to `../data/ragbench_50`; pipeline also defaults to this path.
- Chroma persistence is `outputs/chromadb`; changing dataset/index assumptions without clearing this can mix stale vectors with new runs.

## Env Vars + Embedding Path
- `GROQ_API_KEY` is required even for basic CLI runs (`GraphRAGPipeline.__init__` instantiates `GroqGenerator` immediately).
- Default embedding path is local SentenceTransformer (`USE_LOCAL_EMBEDDINGS=1`, default model `BAAI/bge-m3`).
- If `USE_LOCAL_EMBEDDINGS=0`, `JINA_API_KEY` is required (`JINA_EMBED_MODEL` optional, default `jina-embeddings-v4`).
- `.env` is already ignored by `graph_rag/.gitignore`.

## Team-Alignment Caveat
- Shared rules expect Voyage embeddings (`voyage-3-lite`), but active code paths in `src/vector_store.py` use local BGE or Jina; treat this as MVP workaround, not final evaluation baseline.

## Code Map
- `main.py`: CLI wrapper (build indices, run one query, print JSON result summary).
- `src/pipeline.py`: orchestration (load/chunk -> graph+vector retrieval -> merge -> generate).
- `src/data_loader.py`: `load_from_disk` dataset loader + whitespace chunking into `ChunkRecord`.
- `src/graph.py`: in-memory NetworkX graph index and traversal retrieval.
- `src/vector_store.py`: ChromaDB index + local/Jina embeddings backend switch.
- `src/llm_client.py`: Groq prompt + completion call.

## Verification Reality
- No CI, lint config, typecheck config, or tests are currently set up (`tests/` is empty).
- Practical verification is a smoke run via `main.py` with a real `GROQ_API_KEY`.
