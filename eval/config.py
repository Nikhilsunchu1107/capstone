"""Single source of settings for all three RAG strategies and the judge."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Any OpenAI-compatible chat endpoint. Settings come from .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
# The older names DEEPSEEK_API_KEY and DEEPSEEK_MODEL are still read as a fallback.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
# Exact ID comes from `python -m eval.llm` (models.list). Flash tier only.
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "")
FORBIDDEN_MODEL_MARKERS = ("pro", "reasoner")

GENERATOR_MODEL = LLM_MODEL
JUDGE_MODEL = LLM_MODEL
TEMPERATURE = 0.0
THINKING = "disabled"
ANSWER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 800
JUDGE_WORKERS = 8

# The chat endpoint has no embeddings API, so embeddings use a second OpenAI-compatible endpoint.
# Settings: EMBED_BASE_URL, EMBED_API_KEY (fallback: NVIDIA_API_KEY), EMBED_MODEL.
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY") or os.getenv("NVIDIA_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nemotron-3-embed-1b")
EMBED_DIM = 2048
EMBED_BATCH = 32
EMBED_CACHE_DIR = ROOT / "cache" / "embeddings"
TOP_K = 15  # chunks retrieved per query (Vector and Graph RAG)
PAGEINDEX_MAX_NODES = 5  # PageIndex returns at most this many tree nodes
# Generation has NO context cap: each strategy passes its natural retrieval unit.
# The cap below is used only by the retrieval-check 'budget' metric (reported as an extra row).
RETRIEVAL_BUDGET_CHARS = 8000
RANDOM_SEED = 42

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "Policy_Documents_Curated_15"
# 80 questions (q41-q120). Frozen: eval/run_strategy.py stops if the file does not match its hash.
EVAL_SET = DATA_DIR / "eval_set_v2.json"
EVAL_SET_SHA256 = "982a01e2962683386e868c614c76a4dde73240196d58b752207e4b28f5564c14"  # frozen 2026-10-13
LLM_CACHE_DIR = ROOT / "cache" / "llm"
RESULTS_DIR = ROOT / "eval" / "results"
