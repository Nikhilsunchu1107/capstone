"""Shared typed structures for the Graph RAG MVP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkRecord:
    """Represent a text chunk and its source metadata."""

    chunk_id: str
    text: str
    source_id: str
    source_title: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityMention:
    """Represent one entity mention extracted from text."""

    name: str
    label: str
    norm_name: str


@dataclass(slots=True)
class RetrievedChunk:
    """Represent a retrieved chunk with score and provenance."""

    chunk_id: str
    text: str
    source_id: str
    source_title: str
    score: float
    retrieval_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
