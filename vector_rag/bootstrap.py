import pickle

import numpy as np

from chunking import split_documents
from config import CACHE_DIR
from embeddings import EmbeddingManager
from ingestion import process_all_pdf
from vector_store import VectorStore


def load_or_build_chunks(pdf_directory: str = "../Policy Documents Curated 15/") -> list:
    chunks_file = CACHE_DIR / "chunks.pkl"

    if chunks_file.exists():
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        print(f"✓ Loaded {len(chunks)} chunks from cache")
        return chunks

    all_pdf_documents = process_all_pdf(pdf_directory)
    chunks = split_documents(all_pdf_documents)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✓ Saved {len(chunks)} chunks to cache")
    return chunks


def load_or_build_embeddings(
    chunks: list, embedding_manager: EmbeddingManager
) -> np.ndarray:
    embeddings_file = CACHE_DIR / "embeddings.npy"

    if embeddings_file.exists():
        embeddings = np.load(embeddings_file)
        print(f"✓ Loaded embeddings from cache: {embeddings.shape}")
        return embeddings

    embeddings = embedding_manager.generate_embeddings(
        [doc.page_content for doc in chunks]
    )
    np.save(embeddings_file, embeddings)
    print(f"✓ Saved embeddings to cache: {embeddings.shape}")
    return embeddings


def build_vector_store(
    chunks: list, embeddings: np.ndarray, persist_directory: str = "../vectorDB"
) -> VectorStore:
    vector_store = VectorStore(
        collection_name="policy_documents",
        persist_directory=persist_directory,
    )
    if vector_store.collection.count() == 0:
        vector_store.add_documents(embeddings=embeddings, documents=chunks)
        print("✓ Vector DB populated and persisted")
    else:
        print(
            f"✓ Vector DB already has {vector_store.collection.count()} docs — skipping ingestion"
        )
    return vector_store


def build_index() -> tuple[list, EmbeddingManager, VectorStore]:
    """Loads/builds chunks, embeddings, and the ChromaDB vector store, using on-disk caches."""
    chunks = load_or_build_chunks()
    embedding_manager = EmbeddingManager()
    embeddings = load_or_build_embeddings(chunks, embedding_manager)
    vector_store = build_vector_store(chunks, embeddings)
    return chunks, embedding_manager, vector_store


if __name__ == "__main__":
    build_index()
