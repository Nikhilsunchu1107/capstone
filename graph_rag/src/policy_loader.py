"""PDF loading and chunking for insurance policy documents (Phase 4)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import fitz  # PyMuPDF

from graph_rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.data_loader import _split_text
from src.types import ChunkRecord

# PDFs yielding fewer than this many whitespace-tokens are skipped as unreadable.
MIN_TOKENS_THRESHOLD = 50

# Regex patterns used to clean extracted PDF text.
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_FORM_FEED_RE = re.compile(r"\f")
_HEADER_FOOTER_RE = re.compile(
    r"(?m)^\s*(page\s+\d+(\s+of\s+\d+)?|©.*|www\..+|\d{1,3})\s*$",
    re.IGNORECASE,
)


def _clean_pdf_text(raw: str) -> str:
    """Normalize raw PDF text: remove headers/footers, collapse whitespace."""
    text = _FORM_FEED_RE.sub("\n", raw)
    text = _HEADER_FOOTER_RE.sub("", text)
    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def _derive_policy_name(pdf_path: Path) -> str:
    """Derive a human-readable policy name from a PDF filename.

    Strips trailing hash suffixes (e.g. ``_78baa23b5a``) and underscores.
    """
    stem = pdf_path.stem
    # Remove trailing 8+ hex character hash segments added by some portals
    stem = re.sub(r"_[0-9a-f]{8,}$", "", stem)
    # Replace underscores/hyphens with spaces and title-case
    return re.sub(r"[_\-]+", " ", stem).strip()


def load_policy_documents(
    policy_dir: str | Path = "data/Policy_Documents",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_tokens: int = MIN_TOKENS_THRESHOLD,
    verbose: bool = True,
) -> list[ChunkRecord]:
    """Load all PDF policy documents from *policy_dir* and return chunk records.

    Args:
        policy_dir: Directory containing ``*.pdf`` files.
        chunk_size: Token window size for chunking (matches shared config).
        chunk_overlap: Token overlap between consecutive chunks.
        min_tokens: PDFs with fewer extracted tokens are skipped as unreadable.
        verbose: If True, print a per-document summary after loading.

    Returns:
        List of :class:`~src.types.ChunkRecord` objects ready for indexing.
    """
    policy_dir = Path(policy_dir)
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy document directory not found: {policy_dir}")

    pdf_paths = sorted(policy_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {policy_dir}")

    all_chunks: list[ChunkRecord] = []
    skipped: list[str] = []

    for pdf_path in pdf_paths:
        policy_name = _derive_policy_name(pdf_path)
        source_id = f"policy::{pdf_path.stem}"

        # --- Extract text with PyMuPDF ---
        try:
            with fitz.open(str(pdf_path)) as doc:
                raw_pages: list[str] = [page.get_text() for page in doc]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[PolicyLoader] Could not open '{pdf_path.name}': {exc}. Skipping.",
                stacklevel=2,
            )
            skipped.append(pdf_path.name)
            continue

        full_text = _clean_pdf_text("\n".join(raw_pages))
        token_count = len(full_text.split())

        if token_count < min_tokens:
            warnings.warn(
                f"[PolicyLoader] '{pdf_path.name}' yielded only {token_count} tokens "
                f"(threshold: {min_tokens}). Likely a scanned/image PDF. Skipping.",
                stacklevel=2,
            )
            skipped.append(pdf_path.name)
            continue

        # --- Chunk ---
        raw_chunks = _split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            all_chunks.append(
                ChunkRecord(
                    chunk_id=f"{source_id}::chunk_{chunk_idx}",
                    text=chunk_text,
                    source_id=source_id,
                    source_title=policy_name,
                    metadata={
                        "policy_name": policy_name,
                        "filename": pdf_path.name,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(raw_chunks),
                        "doc_tokens": token_count,
                    },
                )
            )

        if verbose:
            print(
                f"  [OK] {pdf_path.name:<70s} "
                f"{token_count:>6} tokens → {len(raw_chunks)} chunks"
            )

    if verbose:
        print(
            f"\nLoaded {len(all_chunks)} chunks from "
            f"{len(pdf_paths) - len(skipped)}/{len(pdf_paths)} PDFs."
        )
        if skipped:
            print(f"Skipped ({len(skipped)}): {', '.join(skipped)}")

    return all_chunks
