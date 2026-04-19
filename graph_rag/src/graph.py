"""NetworkX knowledge graph construction and traversal for MVP."""

from __future__ import annotations

from collections import deque

import networkx as nx

from src.ner import NERExtractor
from src.types import ChunkRecord, RetrievedChunk


class GraphIndex:
    """Build and query a chunk-entity graph for Graph RAG retrieval."""

    def __init__(self, ner_extractor: NERExtractor) -> None:
        """Initialize graph index with a configured NER extractor."""
        self.ner = ner_extractor
        self.graph = nx.DiGraph()
        self.chunk_lookup: dict[str, ChunkRecord] = {}
        self.entity_norm_to_node: dict[str, str] = {}

    def build(self, chunks: list[ChunkRecord]) -> None:
        """Populate graph nodes and edges from chunk records."""
        self.graph.clear()
        self.chunk_lookup.clear()
        self.entity_norm_to_node.clear()

        for chunk in chunks:
            self.chunk_lookup[chunk.chunk_id] = chunk
            self.graph.add_node(
                chunk.chunk_id,
                node_type="chunk",
                text=chunk.text,
                source_id=chunk.source_id,
                source_title=chunk.source_title,
                metadata=chunk.metadata,
            )

            entities = self.ner.extract(chunk.text)
            chunk.metadata["entities"] = [(entity.name, entity.label) for entity in entities]

            entity_nodes: list[str] = []
            for entity in entities:
                entity_node = self._ensure_entity_node(entity.name, entity.label, entity.norm_name)
                entity_nodes.append(entity_node)
                self.graph.add_edge(chunk.chunk_id, entity_node, relation="MENTIONS")
                self.graph.add_edge(entity_node, chunk.chunk_id, relation="MENTIONED_IN")

            for i, src_entity in enumerate(entity_nodes):
                for dst_entity in entity_nodes[i + 1 :]:
                    self.graph.add_edge(src_entity, dst_entity, relation="RELATED_TO")
                    self.graph.add_edge(dst_entity, src_entity, relation="RELATED_TO")

    def _ensure_entity_node(self, name: str, label: str, norm_name: str) -> str:
        """Create or reuse an entity node keyed by normalized entity text."""
        existing = self.entity_norm_to_node.get(norm_name)
        if existing:
            return existing

        node_id = f"entity::{norm_name}"
        self.graph.add_node(node_id, node_type="entity", name=name, label=label, norm_name=norm_name)
        self.entity_norm_to_node[norm_name] = node_id
        return node_id

    def retrieve(self, query: str, top_k: int = 5, max_hops: int = 2) -> list[RetrievedChunk]:
        """Retrieve chunk candidates via entity-seeded graph traversal."""
        query_entities = self.ner.extract(query)
        if not query_entities:
            return []

        seed_entities = [
            self.entity_norm_to_node[entity.norm_name]
            for entity in query_entities
            if entity.norm_name in self.entity_norm_to_node
        ]
        if not seed_entities:
            return []

        chunk_scores: dict[str, float] = {}
        visited_depth: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((entity_node, 0) for entity_node in seed_entities)

        while queue:
            node_id, depth = queue.popleft()
            prev_depth = visited_depth.get(node_id)
            if prev_depth is not None and prev_depth <= depth:
                continue
            visited_depth[node_id] = depth

            if depth > max_hops:
                continue

            node_type = self.graph.nodes[node_id].get("node_type")
            if node_type == "chunk":
                score = 1.0 / (depth + 1)
                current = chunk_scores.get(node_id, 0.0)
                chunk_scores[node_id] = max(current, score)

            for neighbor in self.graph.neighbors(node_id):
                if depth + 1 <= max_hops:
                    queue.append((neighbor, depth + 1))

        ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: list[RetrievedChunk] = []
        for chunk_id, score in ranked:
            chunk = self.chunk_lookup[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_id=chunk.source_id,
                    source_title=chunk.source_title,
                    score=score,
                    retrieval_source="graph",
                    metadata=chunk.metadata,
                )
            )
        return results
