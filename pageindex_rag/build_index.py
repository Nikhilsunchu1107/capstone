"""Build the PageIndex store for all PDFs. Resumable: finished PDFs are skipped.

Run: python -m pageindex_rag.build_index
"""

import time

from eval import config, llm
from pageindex_rag import llm_patch

STORE = config.ROOT / "pageindex_rag" / ".pageindex"


def count(nodes):
    return sum(1 + count(n.get("nodes") or []) for n in nodes)


if __name__ == "__main__":
    llm_patch.install()
    from pageindex import PageIndexLocalClient

    client = PageIndexLocalClient(model=config.LLM_MODEL, summary_model=config.LLM_MODEL,
                                  retrieve_model=config.LLM_MODEL, storage_path=str(STORE))
    done = {d["name"] for d in client.list_documents()["documents"]}
    failed = []
    for pdf in sorted(config.PDF_DIR.glob("*.pdf")):
        if pdf.name in done:
            print(f"skip   {pdf.name[:50]}", flush=True)
            continue
        before, t0 = llm.usage_totals(), time.time()
        try:
            doc_id = client.submit_document(str(pdf))["doc_id"]
        except Exception as e:
            failed.append(pdf.name)
            print(f"FAILED {pdf.name[:50]}: {str(e)[:200]}", flush=True)
            continue
        after = llm.usage_totals()
        n = count(client.get_tree(doc_id)["result"])
        print(f"built  {pdf.name[:50]:50s} nodes={n:3d} calls={after['calls'] - before['calls']:3d} "
              f"prompt={after['prompt_tokens'] - before['prompt_tokens']:7d} "
              f"completion={after['completion_tokens'] - before['completion_tokens']:6d} {time.time() - t0:4.0f}s", flush=True)
    print("totals:", llm.usage_totals(), "| failed:", failed, flush=True)
