"""Entity extraction with the shared LLM client (replaces spaCy + GLiNER).

Same interface as the old extractors: extract(text) -> list[EntityMention].
Calls are cached on disk by eval.llm, so a rebuild costs nothing.
"""

from __future__ import annotations

import json
import re

from eval import llm
from src.types import EntityMention

MAX_ENTITIES = 12
_PROMPT = """Extract the key entities from the text below for a knowledge graph about an insurance policy.
Entities are specific provisions, coverages, exclusions, benefits, conditions, amounts, time periods,
named parties, documents or clauses. Use the wording of the text.
Do NOT extract generic words such as: policy, insured, insurer, company, claim, premium, cover, loss.
Return at most {n} entities.

Text:
{text}

Reply with one JSON object only: {{"entities": [{{"name": "<entity>", "type": "<one word type>"}}]}}"""


def normalize(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    n = re.sub(r"\s+", " ", n).strip()
    return re.sub(r"^(the|a|an) ", "", n)


class LLMNERExtractor:
    def __init__(self) -> None:
        self.stop: set[str] = set()  # normalized names too common to be useful, set after indexing

    def raw(self, text: str) -> list[EntityMention]:
        prompt = _PROMPT.format(n=MAX_ENTITIES, text=text[:6000])
        out = []
        for tag in ("", "retry"):
            try:
                data = json.loads(llm.chat(prompt, max_tokens=500, json_mode=True, tag=tag))
                for e in data.get("entities", []):
                    name = str(e.get("name", "")).strip()
                    norm = normalize(name)
                    if len(norm) >= 3:
                        out.append(EntityMention(name=name, label=str(e.get("type", "")).upper(), norm_name=norm))
                break
            except (ValueError, AttributeError, TypeError):
                continue
        seen, uniq = set(), []
        for e in out[:MAX_ENTITIES]:
            if e.norm_name not in seen:
                seen.add(e.norm_name)
                uniq.append(e)
        return uniq

    def extract(self, text: str) -> list[EntityMention]:
        return [e for e in self.raw(text) if e.norm_name not in self.stop]
