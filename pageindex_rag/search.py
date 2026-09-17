import json

from client import REGISTRY_PATH, pi_client
from nim_client import call_nim, nim_call
from pageindex import utils
from router import route_query


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
    tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])

    prompt = f"""You are given a question and a tree structure of a document.
Each node contains a node id, title, and summary.
Find all nodes likely to contain the answer to the question.

Question: {query}

Document tree:
{json.dumps(tree_without_text, indent=2)}

Reply ONLY with this JSON:
{{
    "thinking": "<your reasoning>",
    "node_list": ["node_id_1", "node_id_2"]
}}
"""

    response_text = nim_call(prompt, model="poolside/laguna-xs-2.1")
    if not response_text:
        raise ValueError("nim_call returned no response after 3 retries")
    result = json.loads(response_text)

    node_map = utils.create_node_mapping(tree)

    chunks = []
    for node_id in result["node_list"]:
        if node_id not in node_map:
            continue
        node = node_map[node_id]
        chunks.append(
            f"[Source: {filename}, Page {node['page_index']}]\n{node['text'][:4000]}"
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
        answer = call_nim(
            prompt,
            model="poolside/laguna-xs-2.1",
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
        answer = call_nim(prompt, model="poolside/laguna-xs-2.1")
        print("\n📝 Answer:\n")
        utils.print_wrapped(answer)


if __name__ == "__main__":
    ask("What happens if I miss a premium payment?")
    ask("What is the SECTION V Of CYBER VAULTEDGE policy wording?")
