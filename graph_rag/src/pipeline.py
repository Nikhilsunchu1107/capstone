"""End-to-end Graph RAG MVP pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict

from config import TOP_K
from src.data_loader import create_chunk_records, load_local_ragbench
from src.graph import GraphIndex
from src.llm_client import GroqGenerator
from src.ner import NERExtractor
from src.types import RetrievedChunk
from src.vector_store import VectorIndex


class GraphRAGPipeline:
    """Orchestrate data loading, indexing, retrieval, and answer generation."""

    def __init__(self) -> None:
        """Initialize Graph RAG components for the MVP stack."""
        self.ner = NERExtractor()
        self.graph_index = GraphIndex(ner_extractor=self.ner)
        self.vector_index = VectorIndex()
        self.generator = GroqGenerator()
        self._is_indexed = False

    def build_indices(self, dataset_path: str = "data/ragbench_50") -> None:
        """Build graph and vector indices from local RAGBench sample."""
        dataset = load_local_ragbench(dataset_path)
        chunks = create_chunk_records(dataset)
        self.graph_index.build(chunks)
        self.vector_index.index_chunks(chunks)
        self._is_indexed = True

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
        retrieved_chunks = self._merge_results(graph_results, vector_results, top_k=TOP_K)
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
