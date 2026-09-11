"""End-to-end Graph RAG pipeline orchestration for insurance policy documents."""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

from graph_rag.config import TOP_K
from src.graph import GraphIndex
from src.llm_client import LLMGenerator
from src.ner import NERExtractor
from src.policy_loader import load_policy_documents
from src.policy_ner import PolicyNERExtractor
from src.types import RetrievedChunk
from src.vector_store import VectorIndex


class GraphRAGPipeline:
    """Orchestrate data loading, indexing, retrieval, and answer generation."""

    def __init__(self, policy_mode: bool = True) -> None:
        """Initialize Graph RAG components.

        Args:
            policy_mode: If True, use PolicyNERExtractor (spaCy + GLiNER) and
                enable Document nodes in the graph. Defaults to True.
        """
        self.policy_mode = policy_mode
        if policy_mode:
            self.ner = PolicyNERExtractor()
        else:
            self.ner = NERExtractor()
        self.graph_index = GraphIndex(ner_extractor=self.ner, policy_mode=policy_mode)
        self.vector_index = VectorIndex()
        self.generator = LLMGenerator(policy_mode=policy_mode)
        self._is_indexed = False

    def build_indices(
        self,
        policy_dir: str | Path = "data/Policy_Documents_Curated_15",
        graph_cache_path: str | Path | None = "outputs/policy_graph.pkl",
        chroma_persist_dir: str = "outputs/policy_chromadb",
        chroma_collection: str = "policy_chunks",
        force_rebuild: bool = False,
    ) -> None:
        """Build graph and vector indices from insurance policy PDFs.

        Args:
            policy_dir: Directory containing policy PDFs.
            graph_cache_path: Path to persist/load the pickled NetworkX graph.
            chroma_persist_dir: ChromaDB persistence directory for policy chunks.
            chroma_collection: ChromaDB collection name for policy chunks.
            force_rebuild: If True, ignore graph cache and rebuild from scratch.
        """
        self.build_indices_from_policy_docs(
            policy_dir=policy_dir,
            graph_cache_path=graph_cache_path,
            chroma_persist_dir=chroma_persist_dir,
            chroma_collection=chroma_collection,
            force_rebuild=force_rebuild,
        )

    # ------------------------------------------------------------------
    # Policy-mode ingestion
    # ------------------------------------------------------------------

    def build_indices_from_policy_docs(
        self,
        policy_dir: str | Path = "data/Policy_Documents_Curated_15",
        graph_cache_path: str | Path | None = "outputs/policy_graph.pkl",
        chroma_persist_dir: str = "outputs/policy_chromadb",
        chroma_collection: str = "policy_chunks",
        force_rebuild: bool = False,
    ) -> None:
        """Build graph and vector indices from insurance policy PDFs.

        On first run this loads all PDFs, runs NER, and builds the graph.
        The resulting NetworkX graph is pickled to *graph_cache_path* so subsequent
        runs load instantly from cache unless *force_rebuild* is True.

        Args:
            policy_dir: Directory containing the policy PDFs.
            graph_cache_path: Where to persist/load the pickled NetworkX graph.
                Set to None to disable caching.
            chroma_persist_dir: ChromaDB persistence directory for policy chunks.
            chroma_collection: ChromaDB collection name for policy chunks.
            force_rebuild: If True, ignore the graph cache and rebuild from scratch.
        """
        if not self.policy_mode:
            msg = (
                "build_indices_from_policy_docs() requires policy_mode=True. "
                "Reinitialise with GraphRAGPipeline(policy_mode=True)."
            )
            raise RuntimeError(msg)

        # Re-configure vector store to use the policy-specific collection
        self.vector_index = VectorIndex(
            persist_dir=chroma_persist_dir,
            collection_name=chroma_collection,
        )

        graph_cache = Path(graph_cache_path) if graph_cache_path else None

        if graph_cache and graph_cache.exists() and not force_rebuild:
            print(f"[Pipeline] Loading graph from cache: {graph_cache}")
            self.load_graph(graph_cache)
            # Still need to index into ChromaDB if collection is empty
            if self.vector_index.collection.count() == 0:
                print("[Pipeline] ChromaDB collection empty — re-indexing chunks...")
                chunks = list(self.graph_index.chunk_lookup.values())
                self.vector_index.index_chunks(chunks)
        else:
            print(f"[Pipeline] Loading policy documents from: {policy_dir}")
            chunks = load_policy_documents(policy_dir)
            print(f"[Pipeline] Building graph from {len(chunks)} chunks...")
            self.graph_index.build(chunks)
            self.vector_index.index_chunks(chunks)
            if graph_cache:
                self.save_graph(graph_cache)

        self._is_indexed = True

    def save_graph(self, path: str | Path) -> None:
        """Persist the NetworkX graph and chunk lookup to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": self.graph_index.graph,
            "chunk_lookup": self.graph_index.chunk_lookup,
            "entity_norm_to_node": self.graph_index.entity_norm_to_node,
            "document_lookup": self.graph_index.document_lookup,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[Pipeline] Graph saved to {path} ({path.stat().st_size // 1024} KB)")

    def load_graph(self, path: str | Path) -> None:
        """Restore a previously pickled graph and chunk lookup."""
        path = Path(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)  # noqa: S301
        self.graph_index.graph = payload["graph"]
        self.graph_index.chunk_lookup = payload["chunk_lookup"]
        self.graph_index.entity_norm_to_node = payload["entity_norm_to_node"]
        self.graph_index.document_lookup = payload.get("document_lookup", {})
        print(f"[Pipeline] Graph loaded from {path}")

    # ------------------------------------------------------------------
    # Retrieval & generation (shared by both modes)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_results(
        graph_results: list[RetrievedChunk],
        vector_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Merge graph/vector retrieval results by chunk ID and score."""
        merged: dict[str, RetrievedChunk] = {}

        for result in graph_results + vector_results:
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = result
                continue

            if result.score > existing.score:
                merged[result.chunk_id] = result
            else:
                existing.retrieval_source = "hybrid"

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _build_context(retrieved_chunks: list[RetrievedChunk]) -> str:
        """Create context text block with chunk IDs for grounded prompting."""
        lines: list[str] = []
        for chunk in retrieved_chunks:
            lines.append(f"[{chunk.chunk_id}] {chunk.text}")
        return "\n\n".join(lines)

    def query(self, question: str) -> dict:
        """Run full hybrid retrieval + generation for a user question."""
        if not self._is_indexed:
            msg = "Indices are not built. Call build_indices() first."
            raise RuntimeError(msg)

        graph_results = self.graph_index.retrieve(question, top_k=TOP_K)
        vector_results = self.vector_index.query(question, top_k=TOP_K)
        retrieved_chunks = self._merge_results(
            graph_results, vector_results, top_k=TOP_K
        )
        context = self._build_context(retrieved_chunks)
        answer = self.generator.generate(question=question, context=context)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_chunks": [asdict(chunk) for chunk in retrieved_chunks],
            "graph_result_count": len(graph_results),
            "vector_result_count": len(vector_results),
        }
