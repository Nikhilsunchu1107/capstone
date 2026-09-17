import os
from pathlib import Path

from dotenv import load_dotenv

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
load_dotenv(MODULE_DIR.parent / ".env")
NVIDIA_API_KEY = os.environ["NV_API_KEY"]
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ.setdefault("NVIDIA_NIM_API_KEY", NVIDIA_API_KEY)  # litellm's nvidia_nim provider reads this name

import pageindex.utils as _u

# 1) Don't hammer NVIDIA's free tier → no 429s
_u.SUMMARY_CONCURRENCY = 2

# 2) Cap oversized summary prompts (PORT_PACKAGE has a 200k-char node) → no 504s
_orig_acomp = _u.llm_acompletion


async def _capped(model, prompt):
    if len(prompt) > 6000:
        prompt = prompt[:6000] + "\n...[truncated]"
    return await _orig_acomp(model, prompt)


_u.llm_acompletion = _capped
from pageindex import PageIndexLocalClient

pi_client = PageIndexLocalClient(
    model="nvidia_nim/poolside/laguna-xs-2.1",
    summary_model="nvidia_nim/poolside/laguna-xs-2.1",
    retrieve_model="nvidia_nim/poolside/laguna-xs-2.1",
    storage_path=str(MODULE_DIR / ".pageindex"),
)

PDF_SOURCE_DIR = str(PROJECT_DIR / "data" / "Policy_Documents_Curated_15")
REGISTRY_PATH = str(MODULE_DIR / "pdf_registry.json")

if __name__ == "__main__":
    print(pi_client.__class__.__name__)
    from openai import OpenAI

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY
    )
    for model in client.models.list():
        print(model.id)
