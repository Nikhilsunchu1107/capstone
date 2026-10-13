"""Route every LLM call inside the `pageindex` package to DeepSeek through eval.llm.

Call install() before the first PageIndexLocalClient call. All calls then use the
cached, flash-only client. No prompt truncation is applied.
"""

import asyncio
import sys

from eval import config, llm

MAX_TOKENS = 4096


def _completion(model=None, prompt="", chat_history=None, return_finish_reason=False):
    if chat_history:
        history = "\n\n".join(f"{m['role']}: {m['content']}" for m in chat_history)
        prompt = f"{history}\n\nuser: {prompt}"
    text, finish = llm.chat_full(prompt, max_tokens=MAX_TOKENS)
    if return_finish_reason:
        return text, "max_output_reached" if finish == "length" else "finished"
    return text


async def _acompletion(model=None, prompt=""):
    return await asyncio.to_thread(_completion, model, prompt)


def install(summary_concurrency: int = 8) -> None:
    import pageindex  # noqa: F401
    import pageindex.utils as u

    u.SUMMARY_CONCURRENCY = summary_concurrency
    u.llm_completion, u.llm_acompletion = _completion, _acompletion
    # modules that did `from .utils import *` hold their own references
    for name, mod in list(sys.modules.items()):
        if name.startswith("pageindex") and mod is not None and mod is not u:
            if hasattr(mod, "llm_completion"):
                mod.llm_completion = _completion
            if hasattr(mod, "llm_acompletion"):
                mod.llm_acompletion = _acompletion
