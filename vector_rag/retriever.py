from typing import Any

from embeddings import EmbeddingManager
from vector_store import VectorStore


class RAGRetriever:
    def __init__(
        self, vector_store: VectorStore, embedding_manager: EmbeddingManager
    ) -> None:
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        print(f"Retrieving documents for query: {query}")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        try:
            if self.vector_store is None or self.vector_store.collection is None:
                raise ValueError("Vector store collection is not initialized.")

            query_embedding = self.embedding_manager.generate_query_embedding(query)

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents: list[str] = results["documents"][0]
                metadatas: list[dict] = results["metadatas"][0]  # type: ignore
                distances: list[float] = results["distances"][0]  # type: ignore
                ids: list[str] = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = (
                        1 - distance
                    )  # ← ChromaDB cosine distance = 1 - similarity

                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "rank": i + 1,
                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:  # noqa: BLE001
            print(f"Error retrieving documents: {e}")
            return []
