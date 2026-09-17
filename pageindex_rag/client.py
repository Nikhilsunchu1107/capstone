from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

import pageindex.utils as _u

# Local Ollama has no per-account rate limit, but keep concurrency modest —
# this machine's CPU/GPU still has finite capacity.
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
    model="ollama_chat/granite4.2:8b",
    summary_model="ollama_chat/granite4.2:8b",
    retrieve_model="ollama_chat/granite4.2:8b",
    storage_path=str(MODULE_DIR / ".pageindex"),
)

PDF_SOURCE_DIR = str(PROJECT_DIR / "data" / "Policy_Documents_Curated_15")
REGISTRY_PATH = str(MODULE_DIR / "pdf_registry.json")

if __name__ == "__main__":
    print(pi_client.__class__.__name__)
