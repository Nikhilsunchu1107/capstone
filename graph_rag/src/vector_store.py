"""ChromaDB + Jina embeddings vector indexing and retrieval for MVP."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import chromadb

# Kept for reference per project request (do not remove):
# import voyageai

# Kept for reference per project request (do not remove):
# from config import EMBED_MODEL
from src.embeddings_local import BGEEmbedder
from src.types import ChunkRecord, RetrievedChunk


class VectorIndex:
    """Manage chunk embedding storage and semantic retrieval."""

    def __init__(self, persist_dir: str = "outputs/chromadb", collection_name: str = "graph_rag_chunks") -> None:
        """Initialize Chroma client and selected embedding backend."""
        self.use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self.local_model_name = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")

        api_key = os.getenv("JINA_API_KEY")
        if not self.use_local_embeddings and (not api_key or api_key == "your_jina_key_here"):
            msg = "JINA_API_KEY is missing. Set it in .env before vector operations."
            raise ValueError(msg)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.jina_api_key = api_key or ""
        self.embed_model = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v4")
        self.local_embedder = BGEEmbedder(self.local_model_name) if self.use_local_embeddings else None

        # Kept for reference per project request (do not remove):
        # self.voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        self.chunk_lookup: dict[str, ChunkRecord] = {}

    def _embed_with_jina(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings using Jina Embeddings API."""
        if not texts:
            return []
        if not self.jina_api_key:
            msg = "JINA_API_KEY is missing while local embeddings are disabled."
            raise RuntimeError(msg)

        payload = {"model": self.embed_model, "input": texts}
        request = urllib.request.Request(
            url="https://api.jina.ai/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.jina_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="ignore")
            if error.code == 403 and "1010" in error_body:
                msg = (
                    "Jina request blocked (error 1010). This is usually WAF/IP protection. "
                    "Try a different network/VPN-off state, or switch to local embeddings for now."
                )
                raise RuntimeError(msg) from error
            msg = f"Jina embedding request failed ({error.code}): {error_body}"
            raise RuntimeError(msg) from error

        data = body.get("data", [])
        data_sorted = sorted(data, key=lambda row: row.get("index", 0))
        return [row["embedding"] for row in data_sorted]

    def index_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Embed and upsert chunk records into ChromaDB."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source_id": chunk.source_id,
                "source_title": chunk.source_title,
                "question_id": str(chunk.metadata.get("question_id", "")),
                "dataset_name": str(chunk.metadata.get("dataset_name", "")),
            }
            for chunk in chunks
        ]

        if self.use_local_embeddings and self.local_embedder is not None:
            embeddings = self.local_embedder.embed(documents)
        else:
            embeddings = self._embed_with_jina(documents)

        # Kept for reference per project request (do not remove):
        # embeddings = self.voyage_client.embed(documents, model=EMBED_MODEL).embeddings
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

        for chunk in chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk

    def query(self, query_text: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-k chunks by semantic similarity search."""
        if self.use_local_embeddings and self.local_embedder is not None:
            query_embedding = self.local_embedder.embed([query_text])[0]
        else:
            query_embedding = self._embed_with_jina([query_text])[0]

        # Kept for reference per project request (do not remove):
        # query_embedding = self.voyage_client.embed([query_text], model=EMBED_MODEL).embeddings[0]
        result = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(ids)

        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            chunk = self.chunk_lookup.get(chunk_id)
            source_id = chunk.source_id if chunk else str(metadata.get("source_id", ""))
            source_title = chunk.source_title if chunk else str(metadata.get("source_title", ""))
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 0.0
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_id=source_id,
                    source_title=source_title,
                    score=score,
                    retrieval_source="vector",
                    metadata=chunk.metadata if chunk else dict(metadata),
                )
            )
        return retrieved
