"""Graph RAG retrieval: embedding-linked entity search plus vector search, merged with rank fusion.

Chunks and vectors are shared with Vector RAG (vector_rag/index.py).
Entities come from the shared LLM client (graph_rag/src/llm_ner.py).
Build the index first: python -m graph_rag.build_index
"""

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent)):  # `src.*` and `graph_rag.*` imports used by the old code
    if p not in sys.path:
        sys.path.insert(0, p)

from eval import config, embed  # noqa: E402

TOP_K = config.TOP_K

CACHE = HERE / "cache"
GRAPH = CACHE / "graph.pkl"

_state = {}


MAX_DF = 0.05  # drop entities found in more than 5% of chunks (too generic to link anything)
WORKERS = 8


def build():
    """Build or load (graph, chunk ids, vector matrix). Chunks and vectors are shared with Vector RAG."""
    from concurrent.futures import ThreadPoolExecutor

    from src.graph import GraphIndex
    from src.llm_ner import LLMNERExtractor
    from src.types import ChunkRecord
    from vector_rag import index

    ner = LLMNERExtractor()
    gi = GraphIndex(ner_extractor=ner, policy_mode=True)
    if GRAPH.exists():
        with open(GRAPH, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        gi.graph, gi.chunk_lookup = payload["graph"], payload["chunk_lookup"]
        gi.entity_norm_to_node, gi.document_lookup = payload["entity_norm_to_node"], payload["document_lookup"]
        ner.stop = payload["stop"]
        return gi, payload["order"], payload["vecs"]
    chunks, vecs = index.build()
    # one cached LLM call per chunk, in parallel
    with ThreadPoolExecutor(WORKERS) as pool:
        raw = list(pool.map(lambda c: ner.raw(c["text"]), chunks))
    df: dict[str, int] = {}
    for ents in raw:
        for n in {e.norm_name for e in ents}:
            df[n] = df.get(n, 0) + 1
    ner.stop = {n for n, c in df.items() if c > MAX_DF * len(chunks)}
    records = [
        ChunkRecord(chunk_id=f"{c['file']}::{c['id']}", text=c["text"], source_id=f"policy::{c['file']}",
                    source_title=c["file"], metadata={"filename": c["file"], "page": c["page"]})
        for c in chunks
    ]
    gi.build(records)
    order = [r.chunk_id for r in records]
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(GRAPH, "wb") as fh:
        pickle.dump({"graph": gi.graph, "chunk_lookup": gi.chunk_lookup, "entity_norm_to_node": gi.entity_norm_to_node,
                     "document_lookup": gi.document_lookup, "order": order, "stop": ner.stop, "vecs": vecs}, fh)
    print(f"entities dropped as too common (> {MAX_DF:.0%} of chunks): {len(ner.stop)}")
    return gi, order, vecs


def _load():
    if not _state:
        _state["gi"], _state["order"], _state["vecs"] = build()


def _item(chunk_id: str, score: float, source: str) -> dict:
    c = _state["gi"].chunk_lookup[chunk_id]
    return {"id": chunk_id, "file": c.metadata["filename"], "page": c.metadata.get("page"), "text": c.text, "score": score, "source": source}


def vector_results(question: str, top_k: int = TOP_K) -> list[dict]:
    _load()
    q = embed.embed([question], "query")[0]
    sims = _state["vecs"] @ q
    return [_item(_state["order"][i], float(sims[i]), "vector") for i in np.argsort(-sims)[:top_k]]


LINK_MIN_SIM = 0.6   # query entity -> graph entity link threshold (cosine)
LINK_PER_ENTITY = 3  # at most this many graph entities per query entity
RRF_K = 60


def _entity_index():
    """Embeddings of every graph entity name (cached), plus document frequency per entity."""
    if "ents" not in _state:
        gi = _state["gi"]
        names = sorted(gi.entity_norm_to_node)
        _state["ents"] = names
        _state["ent_vecs"] = embed.embed(names, "passage")
        g = gi.graph
        _state["ent_chunks"] = {
            n: [c for c in g.neighbors(gi.entity_norm_to_node[n]) if g.nodes[c].get("node_type") == "chunk"] for n in names
        }
    return _state["ents"], _state["ent_vecs"], _state["ent_chunks"]


def graph_results(question: str, top_k: int = TOP_K) -> list[dict]:
    """Graph retrieval with embedding-based entity linking and idf-weighted chunk scores.

    Each query entity links to its most similar graph entities. A chunk scores the sum of
    (similarity x idf) over the distinct linked entities it mentions. One hop, no co-occurrence hop.
    """
    _load()
    names, vecs, ent_chunks = _entity_index()
    qents = [e.name for e in _state["gi"].ner.extract(question)]
    if not qents:
        return []
    n_chunks = len(_state["order"])
    weight: dict[str, float] = {}
    for sims in embed.embed(qents, "query") @ vecs.T:
        for j in np.argsort(-sims)[:LINK_PER_ENTITY]:
            if sims[j] >= LINK_MIN_SIM:
                idf = np.log(n_chunks / max(1, len(ent_chunks[names[j]])))
                weight[names[j]] = max(weight.get(names[j], 0.0), float(sims[j]) * idf)
    score: dict[str, float] = {}
    for n, w in weight.items():
        for c in ent_chunks[n]:
            score[c] = score.get(c, 0.0) + w
    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [_item(c, sc, "graph") for c, sc in ranked]


def _rrf(lists: list[list[dict]], top_k: int) -> list[dict]:
    fused: dict[str, float] = {}
    first: dict[str, dict] = {}
    for items in lists:
        for rank, it in enumerate(items, 1):
            fused[it["id"]] = fused.get(it["id"], 0.0) + 1.0 / (RRF_K + rank)
            first.setdefault(it["id"], it)
    top = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [{**first[i], "score": fused[i], "rank": r + 1} for r, i in enumerate(top)]


def retrieve(question: str, top_k: int = TOP_K, mode: str = "hybrid") -> list[dict]:
    """Modes: graph (linked entities only), vector, hybrid (rank fusion of both)."""
    if mode == "graph":
        return graph_results(question, top_k)
    if mode == "vector":
        return vector_results(question, top_k)
    return _rrf([graph_results(question, top_k), vector_results(question, top_k)], top_k)
