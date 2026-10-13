"""Vector RAG index: PDF pages -> chunks -> embeddings -> one matrix on disk.

Run: python -m vector_rag.index
Writes vector_rag/cache/chunks.json and embeddings.npy (same row order).
"""

import json

import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval import config, embed, textclean

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK = 80
STRIP_BOILERPLATE = True  # drop lines repeated on >=40% of a PDF's pages (see eval/textclean.py)
CACHE = config.ROOT / "vector_rag" / "cache"
OUT_DIR = CACHE if STRIP_BOILERPLATE else CACHE / "ablation_raw_pages"
CHUNKS = OUT_DIR / "chunks.json"
EMBEDDINGS = OUT_DIR / "embeddings.npy"


def build_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    chunks = []
    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        pages = PyMuPDFLoader(str(pdf)).load()
        drop = textclean.repeated_lines([p.page_content for p in pages]) if STRIP_BOILERPLATE else set()
        for page in pages:
            body = "\n".join(l for l in page.page_content.splitlines() if l.strip() not in drop)
            for piece in splitter.split_text(body):
                text = piece.strip()
                if len(text) < MIN_CHUNK or sum(c.isalnum() for c in text) / len(text) < 0.3:
                    continue
                chunks.append({"id": len(chunks), "file": pdf.name, "page": page.metadata["page"] + 1, "text": text})
    return chunks


def build() -> tuple[list[dict], np.ndarray]:
    if CHUNKS.exists() and EMBEDDINGS.exists():
        return json.loads(CHUNKS.read_text()), np.load(EMBEDDINGS)
    chunks = build_chunks()
    vecs = embed.embed([c["text"] for c in chunks], "passage")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False))
    np.save(EMBEDDINGS, vecs)
    return chunks, vecs


if __name__ == "__main__":
    chunks, vecs = build()
    print(f"{len(chunks)} chunks, matrix {vecs.shape}")
