import time

import numpy as np
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import API_KEY


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "nvidia/nv-embed-v1",
        api_key: str | None = API_KEY,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._embedding_dimension: int | None = None

        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = NVIDIAEmbeddings(
                model=self.model_name,
                nvidia_api_key=self.api_key,
                base_url=self.base_url,
                max_batch_size=20,
                truncate="END",
            )
            self.model._client.timeout = 120
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {self.model_name}: {e}")
            raise

    def generate_embeddings(
        self, texts: list[str], batch_size: int = 20, max_retries: int = 3
    ) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")

        all_embeddings = []
        total = len(texts)
        print(f"Generating embeddings for {total} texts in batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"  Batch {batch_num}/{total_batches} (attempt {attempt})...")
                    result = self.model.embed_documents(batch)
                    all_embeddings.extend(result)
                    print(
                        f"  Batch {batch_num}/{total_batches} done ({len(all_embeddings)}/{total} total)"
                    )
                    break
                except Exception as e:
                    print(f"  Batch {batch_num} attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        wait = 2**attempt
                        print(f"  Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise

        print(f"Generated embeddings count: {len(all_embeddings)}")
        return np.asarray(all_embeddings)

    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise ValueError("Model not loaded")

        if self._embedding_dimension is None:
            sample = self.model.embed_query("probe")
            self._embedding_dimension = len(sample)

        return self._embedding_dimension

    def generate_query_embedding(self, query: str) -> np.ndarray:
        embedding = self.model.embed_query(query)
        return np.asarray(embedding)
