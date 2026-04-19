"""Groq client wrapper for answer generation in Graph RAG MVP."""

from __future__ import annotations

import os

from groq import Groq

from config import LLM_MODEL, MAX_TOKENS, TEMPERATURE


class GroqGenerator:
    """Generate grounded answers from retrieved context using Groq."""

    def __init__(self) -> None:
        """Initialize Groq API client from environment."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_key_here":
            msg = "GROQ_API_KEY is missing. Set it in .env before generation."
            raise ValueError(msg)
        self.client = Groq(api_key=api_key)

    def generate(self, question: str, context: str) -> str:
        """Generate an answer constrained to provided context."""
        prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context.\n"
            "If the answer is not in context, reply exactly: I don't know.\n"
            "Always cite the source chunk IDs in brackets.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
