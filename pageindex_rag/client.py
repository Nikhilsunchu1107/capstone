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
    if len(prompt) > 20000:
        prompt = prompt[:20000] + "\n...[truncated]"
    return await _orig_acomp(model, prompt)


_u.llm_acompletion = _capped
from pageindex import PageIndexLocalClient

# granite4.2-8k / gemma2-8k are local Ollama tags (see pageindex_rag/README or
# `ollama show granite4.2-8k`) created via a Modelfile with `PARAMETER num_ctx
# 8192` baked in — Ollama's OpenAI-compatibility endpoint does not reliably
# honor a per-request `num_ctx` override passed via extra_body, so the
# context window has to be set on the model itself instead.
pi_client = PageIndexLocalClient(
    model="ollama_chat/granite4.2-8k",
    summary_model="ollama_chat/granite4.2-8k",
    retrieve_model="ollama_chat/granite4.2-8k",
    storage_path=str(MODULE_DIR / ".pageindex"),
)

PDF_SOURCE_DIR = str(PROJECT_DIR / "data" / "Policy_Documents_Curated_15")
REGISTRY_PATH = str(MODULE_DIR / "pdf_registry.json")

if __name__ == "__main__":
    print(pi_client.__class__.__name__)
