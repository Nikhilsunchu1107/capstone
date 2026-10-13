"""Exact cosine retrieval over the chunk matrix (vectors are L2-normalised)."""

import numpy as np

from eval import config, embed
from vector_rag import index

_state = {}


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    if not _state:
        _state["chunks"], _state["vecs"] = index.build()
    q = embed.embed([query], "query")[0]
    sims = _state["vecs"] @ q
    top = np.argsort(-sims)[:top_k]
    return [{**_state["chunks"][i], "score": float(sims[i]), "rank": r + 1} for r, i in enumerate(top)]
