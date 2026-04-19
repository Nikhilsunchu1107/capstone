"""Entity extraction utilities for Graph RAG MVP."""

from __future__ import annotations

import re

import spacy

from src.types import EntityMention

_WHITESPACE_RE = re.compile(r"\s+")


class NERExtractor:
    """Extract named entities using spaCy for chunks and queries."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize spaCy model for NER extraction."""
        self.nlp = spacy.load(model_name)

    @staticmethod
    def _normalize_entity_text(text: str) -> str:
        """Normalize entity text for deduplication in graph indexing."""
        return _WHITESPACE_RE.sub(" ", text.strip().lower())

    def extract(self, text: str) -> list[EntityMention]:
        """Extract and normalize unique entity mentions from input text."""
        if not text.strip():
            return []

        doc = self.nlp(text)
        entities: list[EntityMention] = []
        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            name = ent.text.strip()
            if not name:
                continue
            norm_name = self._normalize_entity_text(name)
            key = (norm_name, ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append(EntityMention(name=name, label=ent.label_, norm_name=norm_name))

        return entities
