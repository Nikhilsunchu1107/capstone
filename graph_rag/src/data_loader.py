"""Dataset loading and chunk preparation for Graph RAG MVP."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from datasets import Dataset, load_from_disk

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.types import ChunkRecord


def load_local_ragbench(dataset_path: str | Path = "data/ragbench_50") -> Dataset:
    """Load the local RAGBench sample dataset from disk."""
    return load_from_disk(str(dataset_path))


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping token-like chunks using whitespace tokens."""
    tokens = text.split()
    if not tokens:
        return []

    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            continue
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def create_chunk_records(dataset: Dataset) -> list[ChunkRecord]:
    """Convert RAGBench rows into chunk records for indexing and retrieval."""
    chunk_records: list[ChunkRecord] = []

    for row in dataset:
        question_id = str(row["id"])
        question = row["question"].strip()
        documents = row.get("documents", []) or []

        for doc_idx, doc_text in enumerate(documents):
            if not doc_text or not str(doc_text).strip():
                continue

            source_id = f"{question_id}_doc_{doc_idx}"
            source_title = f"ragbench::{row.get('dataset_name', 'unknown')}::doc_{doc_idx}"

            doc_chunks = _split_text(
                text=str(doc_text),
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk_idx, chunk_text in enumerate(doc_chunks):
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=f"{source_id}_chunk_{chunk_idx}",
                        text=chunk_text,
                        source_id=source_id,
                        source_title=source_title,
                        metadata={
                            "question_id": question_id,
                            "question": question,
                            "dataset_name": row.get("dataset_name", "unknown"),
                            "doc_index": doc_idx,
                            "chunk_index": chunk_idx,
                        },
                    )
                )

    return chunk_records


def chunk_records_to_dicts(chunk_records: list[ChunkRecord]) -> list[dict]:
    """Serialize chunk records to plain dictionaries for storage/output."""
    return [asdict(chunk) for chunk in chunk_records]
