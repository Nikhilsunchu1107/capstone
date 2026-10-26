"""PageIndex retrieval: LLM picks documents from the catalog, then picks tree nodes in each.

The router never sees the gold source file. Items returned are nodes (file, page, text).
"""

import json

from eval import config, llm
from pageindex_rag import llm_patch

STORE = config.ROOT / "pageindex_rag" / ".pageindex"
MAX_DOCS = 3
NODES_PER_DOC = 3

_state = {}


def _load():
    if _state:
        return
    llm_patch.install()
    from pageindex import PageIndexLocalClient, utils

    client = PageIndexLocalClient(model=config.LLM_MODEL, summary_model=config.LLM_MODEL,
                                  retrieve_model=config.LLM_MODEL, storage_path=str(STORE))
    docs = client.list_documents()["documents"]
    _state.update(client=client, utils=utils, docs={d["id"]: d for d in docs}, trees={})


def _json(prompt: str, tag: str = "") -> dict | None:
    for t in (tag, tag + "retry"):
        raw = llm.chat(prompt, max_tokens=512, json_mode=True, tag=t)
        try:
            return json.loads(raw)
        except ValueError:
            continue
    return None


def route(question: str) -> list[str]:
    catalog = [{"doc_id": i, "filename": d["name"], "description": d["description"]} for i, d in _state["docs"].items()]
    prompt = f"""You are a document router for an insurance Q&A assistant.
Choose the documents that must be searched to answer the question. Choose at most {MAX_DOCS}.
Choose a document only if the question refers to that policy or to its type of insurance.

Question: {question}

Available documents:
{json.dumps(catalog, separators=(",", ":"))}

Reply with one JSON object only: {{"doc_ids": ["<doc_id>", ...]}}"""
    out = _json(prompt) or {}
    return [d for d in out.get("doc_ids", []) if d in _state["docs"]][:MAX_DOCS]


def _tree(doc_id: str):
    if doc_id not in _state["trees"]:
        u = _state["utils"]
        tree = _state["client"].get_tree(doc_id, node_summary=True)["result"]
        _state["trees"][doc_id] = (u.remove_fields(json.loads(json.dumps(tree)), fields=["text"]), u.create_node_mapping(tree))
    return _state["trees"][doc_id]


def search(doc_id: str, question: str) -> list[dict]:
    tree, node_map = _tree(doc_id)
    prompt = f"""You are given the tree structure of a document, then a question.
Each node has a node id, title, and summary.
Choose up to {NODES_PER_DOC} nodes that are most likely to contain the answer. Put the best node first.

Document tree:
{json.dumps(tree, separators=(",", ":"))}

Question: {question}

Reply with one JSON object only: {{"node_list": ["<node_id>", ...]}}"""
    out = _json(prompt) or {}
    name = _state["docs"][doc_id]["name"]
    items = []
    for nid in out.get("node_list", [])[:NODES_PER_DOC]:
        node = node_map.get(nid)
        if node:
            items.append({"file": name, "page": int(node["page_index"]), "text": node["text"], "node_id": nid})
    return items


def retrieve(question: str, top_k: int = config.PAGEINDEX_MAX_NODES) -> list[dict]:
    """Nodes from each routed document, interleaved by node rank, cut to top_k."""
    _load()
    per_doc = [search(d, question) for d in route(question)]
    merged = []
    for rank in range(NODES_PER_DOC):
        for items in per_doc:
            if rank < len(items):
                merged.append(items[rank])
    return [{**m, "rank": r + 1, "score": 0.0} for r, m in enumerate(merged[:top_k])]
