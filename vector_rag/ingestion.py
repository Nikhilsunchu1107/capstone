from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def process_all_pdf(pdf_directory):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF Files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:  # noqa: BLE001
            print(f"Error: {e}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    all_pdf_documents = process_all_pdf("../Policy Documents Curated 15/")

    directory = Path("../Policy Documents Curated 15")
    pdf_count = sum(1 for file in directory.glob("*.pdf"))
    print(f"Total PDFs: {pdf_count}")
