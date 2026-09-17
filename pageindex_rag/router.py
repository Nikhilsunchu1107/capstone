import json

from nim_client import ollama_call
from registry import load_registry


def route_query(query: str) -> dict:
    registry = load_registry()

    catalog = [
        {
            "doc_id": v["doc_id"],
            "filename": v["filename"],
            "description": v["description"],
        }
        for v in registry.values()
    ]

    prompt = f"""You are a document router for an insurance Q&A assistant.
A user has asked a question. Decide which documents to search and classify the question type.

User question: {query}

Available documents:
{json.dumps(catalog, indent=2)}

Classify the question into ONE of these types:
- "general"    : conceptual, definitional, or comparative questions ("what is X?", "explain Y", "how does Z work", "difference between A and B"). Answerable from general knowledge — EVEN IF one of the documents happens to cover that topic.
- "specific"   : the user asks about a concrete provision, condition, claim, or coverage detail of a policy — the kind of thing you'd only find in the actual wording (grace periods, exclusions, claim process, deductibles, coverage limits).
- "ambiguous"  : too vague to route (e.g. "tell me about my policy").

Reply ONLY with this JSON, nothing else:
{{
    "thinking": "<your reasoning>",
    "type": "specific",
    "doc_ids": ["doc_id_1"],
    "clarification": ""
}}

Rules:
- type "general"   → doc_ids must be [], clarification must be ""
- type "ambiguous" → doc_ids must be [], clarification must be a follow-up question
- type "specific"  → clarification must be ""
- Rule of thumb: if the answer wouldn't change between documents, it's "general". Only search documents when the user needs THEIR policy's actual terms.
- For "specific": list at most the 3 most relevant doc_ids, never more.
"""

    result = ollama_call(prompt, model="granite4.2:8b")
    if result is None:
        raise ValueError("ollama_call returned no response")

    # Reasoning model: `result` may be chain-of-thought text followed by the
    # JSON answer, not pure JSON -- extract the {...} block.
    start = result.find("{")
    end = result.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {result[:200]!r}")
    parsed = json.loads(result[start:end])

    print(f"🔍 Type     : {parsed['type']}")
    print(f"📄 Doc IDs  : {parsed['doc_ids']}")
    print(f"💭 Reasoning: {parsed['thinking']}")

    return parsed


if __name__ == "__main__":
    # Test all three types
    route_query("What happens if I miss a premium payment?")  # should be specific
    route_query("What is term insurance?")  # should be general
    route_query("Tell me about my policy")  # should be ambiguous
