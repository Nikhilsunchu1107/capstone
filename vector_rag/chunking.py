from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    min_chunk_size: int = 80,
) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n\n",  # Section breaks
            "\n\n",  # Paragraph breaks
            "\n",  # Numbered lists / clause lines
            ". ",  # Sentence boundaries
            "? ",
            "! ",
            "; ",  # Legal list semicolons
            ", ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )

    split_docs = text_splitter.split_documents(documents)
    original_count = len(split_docs)

    cleaned_docs = []
    for doc in split_docs:
        stripped_text = doc.page_content.strip()

        # Filter 1: Drop chunks below min character threshold
        if len(stripped_text) < min_chunk_size:
            continue

        # Filter 2: Drop chunks that are mostly non-alphanumeric junk
        alnum_chars = sum(c.isalnum() for c in stripped_text)
        if (alnum_chars / max(len(stripped_text), 1)) < 0.3:
            continue

        # Filter 3: Drop exact single-line header/footer noise
        lines = [line.strip() for line in stripped_text.split("\n") if line.strip()]
        if len(lines) == 1 and any(
            lines[0].lower().startswith(p) for p in ["page ", "www.", "copyright"]
        ):
            continue

        # Use cleaned text, not raw
        doc.page_content = stripped_text
        cleaned_docs.append(doc)

    # Enrich metadata
    for i, doc in enumerate(cleaned_docs):
        doc.metadata["chunk_index"] = i
        doc.metadata["chunk_size"] = len(doc.page_content)
        doc.metadata["chunk_word_count"] = len(doc.page_content.split())

    print(f"Split {len(documents)} documents into {original_count} chunks")
    print(f"After filtering: {len(cleaned_docs)} chunks")
    print(f"Filtered out: {original_count - len(cleaned_docs)} noisy/tiny chunks")

    if cleaned_docs:
        sizes = [len(d.page_content) for d in cleaned_docs]
        print("\nChunk size stats:")
        print(f"  Min: {min(sizes)} chars")
        print(f"  Max: {max(sizes)} chars")
        print(f"  Avg: {sum(sizes) // len(sizes)} chars")
        print("\nExample chunk preview:")
        print(f"  Content: {cleaned_docs[0].page_content[:200]}...")
        print(f"  Metadata: {cleaned_docs[0].metadata}")

    return cleaned_docs
