"""Insurance-domain NER extractor combining spaCy (general) + GLiNER (domain-specific)."""

from __future__ import annotations

import re
from functools import lru_cache

import spacy

from src.types import EntityMention

_WHITESPACE_RE = re.compile(r"\s+")

# Insurance-domain label names passed to GLiNER at inference time.
# These map to the canonical label strings stored on EntityMention.
GLINER_INSURANCE_LABELS: list[str] = [
    "policy type",
    "coverage",
    "exclusion",
    "benefit",
    "premium",
    "claimant",
    "insurer",
    "waiting period",
    "policy tenure",
    "regulatory body",
]

# Map GLiNER's free-text label → canonical uppercase label for graph nodes
_GLINER_LABEL_MAP: dict[str, str] = {
    "policy type": "POLICY_TYPE",
    "coverage": "COVERAGE",
    "exclusion": "EXCLUSION",
    "benefit": "BENEFIT",
    "premium": "PREMIUM",
    "claimant": "CLAIMANT",
    "insurer": "INSURER",
    "waiting period": "WAITING_PERIOD",
    "policy tenure": "TENURE",
    "regulatory body": "REGULATORY_BODY",
}

# GLiNER confidence threshold — entities below this score are discarded.
GLINER_THRESHOLD = 0.5

# spaCy entity labels to keep (skip numeric/misc noise labels)
_SPACY_KEEP_LABELS = {"ORG", "PERSON", "GPE", "DATE", "MONEY", "PERCENT", "LOC", "FAC"}


@lru_cache(maxsize=1)
def _load_gliner_model():
    """Lazily load GLiNER model (cached after first call — ~300 MB download once)."""
    from gliner import GLiNER  # noqa: PLC0415

    print("[PolicyNER] Loading GLiNER model 'urchade/gliner_medium-v2.1' (first run may download)...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    print("[PolicyNER] GLiNER model loaded.")
    return model


@lru_cache(maxsize=1)
def _load_spacy_model(model_name: str = "en_core_web_sm"):
    """Lazily load spaCy model (cached)."""
    return spacy.load(model_name)


class PolicyNERExtractor:
    """Extract entities from insurance policy text using spaCy + GLiNER.

    spaCy handles generic entity types (ORG, MONEY, DATE, PERCENT …).
    GLiNER handles insurance-domain types (COVERAGE, EXCLUSION, BENEFIT …).
    Results are merged and deduplicated; domain-specific labels take precedence
    over generic ones when spans overlap.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        gliner_threshold: float = GLINER_THRESHOLD,
        use_gliner: bool = True,
    ) -> None:
        """Initialise both NER backends.

        Args:
            spacy_model: spaCy model name for generic NER.
            gliner_threshold: Minimum GLiNER confidence score to keep an entity.
            use_gliner: Set False to fall back to spaCy-only (useful for query NER
                where GLiNER's batch inference overhead is unnecessary).
        """
        self.nlp = _load_spacy_model(spacy_model)
        self.gliner_threshold = gliner_threshold
        self.use_gliner = use_gliner

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase + collapse whitespace for deduplication."""
        return _WHITESPACE_RE.sub(" ", text.strip().lower())

    def _run_spacy(self, text: str) -> list[EntityMention]:
        """Run spaCy NER and return filtered entity mentions."""
        doc = self.nlp(text)
        results: list[EntityMention] = []
        seen: set[tuple[str, str]] = set()
        for ent in doc.ents:
            if ent.label_ not in _SPACY_KEEP_LABELS:
                continue
            name = ent.text.strip()
            if not name:
                continue
            norm = self._normalize(name)
            key = (norm, ent.label_)
            if key in seen:
                continue
            seen.add(key)
            results.append(EntityMention(name=name, label=ent.label_, norm_name=norm))
        return results

    def _run_gliner(self, text: str) -> list[EntityMention]:
        """Run GLiNER zero-shot NER for insurance-specific entity types.

        GLiNER has a hard 384 subword-token limit. Insurance PDF text is
        extremely dense (tables, amounts, abbreviations) and can tokenize at
        3-4x the whitespace-token count, so we split into small sub-windows
        of at most GLINER_MAX_WORDS (80) whitespace tokens before inference.
        80 words × worst-case 4x ratio = 320 subword tokens < 384.
        """
        # Conservative limit: 80 whitespace words × 4× worst-case subword
        # ratio = 320 subword tokens — safely under GLiNER's 384 hard cap.
        GLINER_MAX_WORDS = 80   # noqa: N806
        GLINER_OVERLAP = 15     # noqa: N806

        import warnings  # noqa: PLC0415

        model = _load_gliner_model()
        words = text.split()

        # Build sub-windows; every chunk goes through windowing regardless
        # of length so a single dense chunk never exceeds the limit.
        if len(words) <= GLINER_MAX_WORDS:
            windows = [text]
        else:
            step = GLINER_MAX_WORDS - GLINER_OVERLAP
            windows = []
            for start in range(0, len(words), step):
                window_words = words[start : start + GLINER_MAX_WORDS]
                if window_words:
                    windows.append(" ".join(window_words))
                if start + GLINER_MAX_WORDS >= len(words):
                    break

        results: list[EntityMention] = []
        seen: set[tuple[str, str]] = set()

        for window in windows:
            # Suppress the third-party truncation UserWarning — with our
            # windowing it should never fire, but filter it as a safety net.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*has been truncated.*",
                    category=UserWarning,
                )
                raw = model.predict_entities(
                    window,
                    GLINER_INSURANCE_LABELS,
                    threshold=self.gliner_threshold,
                )
            for ent in raw:
                canonical_label = _GLINER_LABEL_MAP.get(ent["label"], ent["label"].upper())
                name = ent["text"].strip()
                if not name:
                    continue
                norm = self._normalize(name)
                key = (norm, canonical_label)
                if key in seen:
                    continue
                seen.add(key)
                results.append(EntityMention(name=name, label=canonical_label, norm_name=norm))

        return results

    def extract(self, text: str) -> list[EntityMention]:
        """Extract and merge entity mentions from *text*.

        GLiNER domain-specific entities take precedence over spaCy generic
        entities when they share the same normalised surface form.
        """
        if not text.strip():
            return []

        spacy_ents = self._run_spacy(text)
        gliner_ents = self._run_gliner(text) if self.use_gliner else []

        # Build merged dict keyed by norm_name; GLiNER wins on conflict
        merged: dict[str, EntityMention] = {}
        for ent in spacy_ents:
            merged[ent.norm_name] = ent
        for ent in gliner_ents:
            merged[ent.norm_name] = ent  # overwrite spaCy if same surface form

        return list(merged.values())

    def extract_query(self, text: str) -> list[EntityMention]:
        """Lightweight extraction for query-time NER (spaCy only, no GLiNER).

        At query time we cannot afford GLiNER's transformer inference cost for
        a single short sentence. spaCy is sufficient to seed the graph traversal.
        """
        return self._run_spacy(text)
