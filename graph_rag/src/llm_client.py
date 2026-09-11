"""LLM client wrapper for answer generation in Graph RAG.

Uses the NVIDIA NIM OpenAI-compatible endpoint.
Supports two prompt modes:
- Default (RAGBench): generic helpful-assistant prompt.
- Policy mode (Phase 4): stricter insurance-domain prompt that prevents
  hallucinating coverage terms, amounts, or exclusions.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI, RateLimitError

from graph_rag.config import LLM_MODEL, MAX_TOKENS, NIM_BASE_URL, TEMPERATURE

# --- Prompt templates ---

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context.\n"
    "If the answer is not in context, reply exactly: I don't know.\n"
    "Always cite the source chunk IDs in brackets."
)

_POLICY_SYSTEM_PROMPT = (
    "You are an expert insurance policy assistant. "
    "Answer questions about insurance policies using ONLY the provided context. "
    "If the policy document does not mention the requested information, say exactly: "
    "'Not specified in the provided policy documents.' "
    "Always cite the policy document name and chunk ID in your answer. "
    "Use clear, precise language. "
    "Do NOT invent coverage terms, exclusions, waiting periods, or benefit amounts "
    "that are not explicitly stated in the context."
)


class LLMGenerator:
    """Generate grounded answers from retrieved context using NVIDIA NIM."""

    def __init__(self, policy_mode: bool = False) -> None:
        """Initialize NVIDIA NIM client from environment.

        Args:
            policy_mode: If True, use the stricter insurance-domain system prompt.
        """
        api_key = os.getenv("NVIDIA_NIM_API_KEY")
        if not api_key or api_key == "your_nim_key_here":
            msg = "NVIDIA_NIM_API_KEY is missing. Set it in .env before generation."
            raise ValueError(msg)
        self.client = OpenAI(
            api_key=api_key,
            base_url=NIM_BASE_URL,
        )
        self.system_prompt = _POLICY_SYSTEM_PROMPT if policy_mode else _DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        question: str,
        context: str,
        max_retries: int = 6,
        base_delay: float = 10.0,
    ) -> str:
        """Generate an answer constrained to provided context.

        Retries up to *max_retries* times on rate-limit (429) errors,
        with exponential backoff starting at *base_delay* seconds.
        """
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                return response.choices[0].message.content or ""

            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = base_delay * (2 ** attempt)  # 10s, 20s, 40s, 80s …
                print(
                    f"  [RateLimit] NIM 429 on attempt {attempt + 1}/{max_retries}. "
                    f"Waiting {wait:.0f}s before retry..."
                )
                time.sleep(wait)

        return ""  # unreachable — satisfies type checker


# ---------------------------------------------------------------------------
# Backward-compatibility alias — remove once all call-sites are updated
# ---------------------------------------------------------------------------
GroqGenerator = LLMGenerator
