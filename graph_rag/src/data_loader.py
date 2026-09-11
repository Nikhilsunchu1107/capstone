"""Text splitting and chunk serialization utilities for Graph RAG."""

from __future__ import annotations

import re
from dataclasses import asdict

from src.types import ChunkRecord



def _sentence_split(text: str) -> list[str]:
    """Split *text* into sentences using punctuation and paragraph boundaries.

    Uses a regex that handles common sentence terminators (.  !  ?) and
    treats double-newlines (paragraph breaks) as hard sentence boundaries.
    Returns a list of non-empty sentence strings.
    """
    import re  # noqa: PLC0415

    # Treat paragraph breaks as sentence separators first
    text = re.sub(r"\n{2,}", "  .  ", text)
    # Split on sentence-ending punctuation followed by whitespace + capital/quote
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            sentences.append(part)
    return sentences if sentences else [text.strip()]


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks that always end on a sentence boundary.

    Sentences are detected by :func:`_sentence_split` and then greedily packed
    into chunks up to *chunk_size* whitespace-tokens.  When a single sentence
    is longer than *chunk_size* tokens it is placed in its own chunk (no
    mid-sentence cut).  The last *chunk_overlap* tokens of each chunk are
    prepended to the next chunk to preserve cross-boundary context.
    """
    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)

    text = text.strip()
    if not text:
        return []

    sentences = _sentence_split(text)
    chunks: list[str] = []
    current_tokens: list[str] = []

    for sentence in sentences:
        sentence_tokens = sentence.split()
        if not sentence_tokens:
            continue

        # If adding this sentence would overflow, flush current chunk first
        if current_tokens and len(current_tokens) + len(sentence_tokens) > chunk_size:
            chunks.append(" ".join(current_tokens))
            # Carry over the last chunk_overlap tokens into the next chunk
            current_tokens = current_tokens[-chunk_overlap:] if chunk_overlap else []

        # If a single sentence is longer than chunk_size, split it on token
        # boundaries as a fallback (avoids infinite loop)
        if len(sentence_tokens) > chunk_size:
            # Flush whatever we have first
            if current_tokens:
                chunks.append(" ".join(current_tokens))
                current_tokens = []
            step = chunk_size - chunk_overlap
            for start in range(0, len(sentence_tokens), step):
                window = sentence_tokens[start : start + chunk_size]
                if window:
                    chunks.append(" ".join(window))
                if start + chunk_size >= len(sentence_tokens):
                    break
        else:
            current_tokens.extend(sentence_tokens)

    # Flush the last partial chunk
    if current_tokens:
        chunks.append(" ".join(current_tokens))

    return chunks


def chunk_records_to_dicts(chunk_records: list[ChunkRecord]) -> list[dict]:
    """Serialize chunk records to plain dictionaries for storage/output."""
    return [asdict(chunk) for chunk in chunk_records]
