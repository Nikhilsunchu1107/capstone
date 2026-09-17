import os
import uuid
from typing import Any

import chromadb
import numpy as np


class VectorStore:
    def __init__(
        self, collection_name: str, persist_directory: str = "../vector_store/"
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF Document embeddings for RAG",
                    "hnsw:space": "cosine",
                },
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Exception intializing vector store: {e}")
            raise

    def add_documents(self, documents: list[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vectore store")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            assert self.collection is not None
            batch_size = 1000
            for start in range(0, len(documents), batch_size):
                end = start + batch_size
                self.collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end],
                )
            print(f"Successfully added {len(documents)} documents to vector DB")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
