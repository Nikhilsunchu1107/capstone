import json

from client import REGISTRY_PATH, pi_client
from ollama_client import ollama_call, ollama_call_json
from pageindex import utils
from router import route_query


def dedupe_text(text: str) -> str:
    """Remove duplicate paragraphs from node text while preserving order
    and keeping short lines (headers, numbers, list items) untouched."""
    seen = set()
    out = []
    for para in text.split('\n'):
        key = para.strip()
        if len(key) < 40:
            out.append(para)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(para)
    return '\n'.join(out)


def _snippet_tree(node, max_len=500):
    """Copy of `node`/`nodes` with each 'text' field truncated to a short
    preview instead of removed outright, so node selection sees title +
    summary + a text snippet rather than title + summary alone. Text is
    deduped before truncating so the preview isn't just the same repeated
    paragraph over and over on nodes with duplicated boilerplate."""
    if isinstance(node, dict):
        copy = dict(node)
        if "text" in copy and isinstance(copy["text"], str):
            text = dedupe_text(copy["text"])
            copy["text"] = text[:max_len] + "..." if len(text) > max_len else text
        if "nodes" in copy:
            copy["nodes"] = _snippet_tree(copy["nodes"], max_len)
        return copy
    if isinstance(node, list):
        return [_snippet_tree(item, max_len) for item in node]
    return node


def search_nodes(doc_id: str, query: str) -> list[str]:
    """
    Searches the node tree of a single PDF for content relevant to the query.
    Returns a list of text chunks tagged with source filename and page number.
    """
    if not pi_client.is_retrieval_ready(doc_id):
        print(f"⚠️  Doc {doc_id} not ready — skipping.")
        return []

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    filename = registry[doc_id]["filename"]

    # fetch tree for this specific doc
    tree = pi_client.get_tree(doc_id, node_summary=True)["result"]
    tree_with_snippets = _snippet_tree(tree.copy())

    prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, title, summary, and a short text preview.
Find all nodes likely to contain the answer to the question.

Question: {query}

Document tree:
{json.dumps(tree_with_snippets, indent=2)}

Reply ONLY with this JSON:
{{
    "thinking": "<your reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""

    # Local model may wrap the JSON object in prose/markdown, or occasionally
    # drop a malformed one — ollama_call_json retries with a "JSON only" nudge
    # instead of failing on a single bad generation. max_tokens is generous
    # because the requested "thinking" field alone can run past 1-2k tokens
    # on trees with dozens of nodes — too low a budget truncates before the
    # model ever reaches node_list, yielding an empty/incomplete response.
    result = ollama_call_json(prompt, model="granite4.2-8k", max_tokens=4096)

    node_map = utils.create_node_mapping(tree)
    node_list = result.get("node_list", [])

    # Doc has exactly one node (title+summary too coarse to judge relevance
    # against) — it's the only candidate either way, so use it regardless.
    if not node_list and len(node_map) == 1:
        node_list = list(node_map.keys())

    chunks = []
    for node_id in node_list:
        if node_id not in node_map:
            continue
        node = node_map[node_id]
        chunks.append(
            f"[Source: {filename}, Page {node['page_index']}]\n{node['text']}"
        )

    print(f"  📑 {filename}: {len(chunks)} relevant node(s) found")
    return chunks


def ask(query: str):
    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    # Step 1: route — classify query and get relevant doc_ids
    routing = route_query(query)
    q_type = routing["type"]

    # Step 2: branch based on question type
    if q_type == "ambiguous":
        print(f"\nCould you clarify: {routing['clarification']}")
        return

    elif q_type == "general":
        print("\nGeneral question — answering from LLM knowledge.\n")
        prompt = f"""Answer this insurance question in simple plain language.
        Start with a one-sentence summary. Avoid jargon.
        Question: {query}"""
        answer = ollama_call(
            prompt,
            model="granite4.2-8k",
        )
        utils.print_wrapped(answer)

    elif q_type == "specific":
        print(f"\nSearching {len(routing['doc_ids'])} document(s)...")

        # Step 3: search nodes in each routed doc and collect chunks
        all_chunks = []
        for doc_id in routing["doc_ids"]:
            chunks = search_nodes(doc_id, query)
            all_chunks.extend(chunks)

        if not all_chunks:
            print("No relevant content found.")
            return

        context = "\n\n---\n\n".join(all_chunks)

        # Step 4: generate answer from retrieved context
        prompt = f"""Answer the question based only on the context below.
If context comes from multiple documents, mention which document each point is from.

Question: {query}

Context:
{context}

Instructions:
- Use plain simple language, avoid legal jargon
- Use "you" and "your" instead of "the policyholder"
- Start with a one-sentence summary
- End with "Bottom line:" telling the user what to actually do or know
"""
        answer = ollama_call(prompt, model="granite4.2-8k")
        print("\n📝 Answer:\n")
        utils.print_wrapped(answer)


if __name__ == "__main__":
    ask("What happens if I miss a premium payment?")
    ask("What is the SECTION V Of CYBER VAULTEDGE policy wording?")
