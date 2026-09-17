import json
import os

from pageindex import utils

if __package__:
    from .client import PDF_SOURCE_DIR, pi_client
else:
    from client import PDF_SOURCE_DIR, pi_client


def submit_pdfs(pdf_source_dir: str = PDF_SOURCE_DIR) -> dict[str, str]:
    """Submits every PDF in `pdf_source_dir` for indexing, skipping already-indexed files."""
    pdf_files = sorted(
        os.path.join(pdf_source_dir, f)
        for f in os.listdir(pdf_source_dir)
        if f.lower().endswith(".pdf")
    )
    print(f"Found {len(pdf_files)} PDFs")

    existing = {d["name"]: d["id"] for d in pi_client.list_documents()["documents"]}
    doc_ids = {}
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        if filename in existing:
            doc_ids[filename] = existing[filename]
            print(f"⏭️  Already indexed: {filename}")
            continue
        doc_ids[filename] = pi_client.submit_document(pdf_path)["doc_id"]  # synchronous
        print(f"✅ Submitted: {filename} → {doc_ids[filename]}")

    print("\nAll doc_ids:", doc_ids)
    return doc_ids


if __name__ == "__main__":
    doc_ids = submit_pdfs()
    print(json.dumps(doc_ids, indent=4))

    documents = pi_client.list_documents()["documents"]
    if not documents:
        print("No indexed documents available for tree inspection.")
        raise SystemExit(0)

    d = documents[0]["id"]
    t = pi_client.get_tree(d, node_summary=True)["result"]
    print(len(t), "top-level nodes")
    print(utils.create_node_mapping(t).keys().__len__(), "total nodes")
