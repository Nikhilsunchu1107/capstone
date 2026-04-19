"""Local embedding client wrappers for offline vectorization."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """Generate embeddings locally with a SentenceTransformer model."""

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        """Initialize local embedding model."""
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed input texts and return vectors as Python lists."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
