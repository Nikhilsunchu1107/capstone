import os
from pathlib import Path

API_KEY = os.getenv("NVIDIA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
EVAL_API_KEY = os.getenv("EVAL_API_KEY")

CACHE_DIR = Path("../cache")
CACHE_DIR.mkdir(exist_ok=True)
