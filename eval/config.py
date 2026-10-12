"""Single source of settings for all three RAG strategies and the judge."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
# Exact ID comes from `python -m eval.llm` (models.list). Flash tier only.
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "")
FORBIDDEN_MODEL_MARKERS = ("pro", "reasoner")

GENERATOR_MODEL = DEEPSEEK_MODEL
JUDGE_MODEL = DEEPSEEK_MODEL
TEMPERATURE = 0.0
THINKING = "disabled"
ANSWER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 800
JUDGE_WORKERS = 8

# DeepSeek has no embeddings endpoint, so embeddings use NVIDIA's OpenAI-compatible endpoint.
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
EMBED_DIM = 2048
EMBED_BATCH = 32
EMBED_CACHE_DIR = ROOT / "cache" / "nvidia" / "embed"
TOP_K = 5
MAX_CONTEXT_CHARS = 8000
RANDOM_SEED = 42

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "Policy_Documents_Curated_15"
EVAL_SET = DATA_DIR / "eval_set.json"
EVAL_SET_SHA256 = "ce3683e0c5f9813abdf125302a2010afc6e359cacaaf4ebae055e7e70eb4b61a"
LLM_CACHE_DIR = ROOT / "cache" / "deepseek" / "llm"
RESULTS_DIR = ROOT / "eval" / "results"
