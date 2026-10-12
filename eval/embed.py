"""Embeddings through NVIDIA's OpenAI-compatible endpoint, cached on disk per text.

Passages and queries use different `input_type` values, as the model requires.
"""

import hashlib
import random
import time

import numpy as np
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from eval import config

_client = None


def _cli() -> OpenAI:
    global _client
    if _client is None:
        if not config.NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set in .env")
        _client = OpenAI(base_url=config.NVIDIA_BASE_URL, api_key=config.NVIDIA_API_KEY)
    return _client


def _path(text: str, input_type: str):
    h = hashlib.sha256(f"{config.EMBED_MODEL}|{input_type}|{text}".encode()).hexdigest()
    return config.EMBED_CACHE_DIR / h[:2] / f"{h}.npy"


def _call(batch: list[str], input_type: str, max_retries: int = 6) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            r = _cli().embeddings.create(
                model=config.EMBED_MODEL,
                input=batch,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        except (RateLimitError, APIConnectionError) as e:
            err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            err = e
        if attempt == max_retries - 1:
            raise err
        time.sleep(min(60, 2**attempt) + random.random())


def embed(texts: list[str], input_type: str) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (len(texts), EMBED_DIM)."""
    assert input_type in ("passage", "query")
    out = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        p = _path(t, input_type)
        if p.exists():
            out[i] = np.load(p)
        else:
            todo.append(i)
    for s in range(0, len(todo), config.EMBED_BATCH):
        idx = todo[s : s + config.EMBED_BATCH]
        vecs = _call([texts[i] for i in idx], input_type)
        for i, v in zip(idx, vecs):
            v = np.asarray(v, dtype=np.float32)
            v /= np.linalg.norm(v)
            p = _path(texts[i], input_type)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, v)
            out[i] = v
    return np.vstack(out) if out else np.zeros((0, config.EMBED_DIM), np.float32)


if __name__ == "__main__":
    docs = ["The free look period is 30 days.", "Grace period is 15 days for monthly mode."]
    d = embed(docs, "passage")
    q = embed(["How long is the free look period?"], "query")
    print("shape", d.shape, "scores", (d @ q.T).ravel().round(3))
    assert d.shape[1] == config.EMBED_DIM
    n = sum(1 for _ in config.EMBED_CACHE_DIR.rglob("*.npy"))
    embed(docs, "passage")
    assert n == sum(1 for _ in config.EMBED_CACHE_DIR.rglob("*.npy"))
    print("cache ok")
