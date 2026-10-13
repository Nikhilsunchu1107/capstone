"""Shared answer generation. Every strategy uses the same prompt and format.

The context is the strategy's natural retrieval unit with no character cap.
"""

from openai import APIStatusError

from eval import config, llm

SYSTEM = """You answer questions about insurance policy documents.
Use only the context given. Do not use outside knowledge.
Answer every part of the question. Give exact numbers, periods and names from the context.
If the context does not contain the answer, say that it does not."""


def format_context(items: list[dict]) -> str:
    if not items:
        return "[No context retrieved]"
    return "\n\n---\n\n".join(
        f"[Source: {i['file']}" + (f", page {i['page']}" if i.get("page") else "") + f"]\n{i['text']}" for i in items
    )


def answer(question: str, items: list[dict]) -> str:
    prompt = f"Context:\n{format_context(items)}\n\nQuestion: {question}\n\nAnswer:"
    try:
        return llm.chat(prompt, system=SYSTEM, model=config.GENERATOR_MODEL, max_tokens=config.ANSWER_MAX_TOKENS).strip()
    except APIStatusError as e:
        return f"ERROR {e.status_code}: {str(e)[:200]}"
