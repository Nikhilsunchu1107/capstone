"""Index ONE PDF with DeepSeek flash and report token use. Usage:
python -m pageindex_rag.dry_run [pdf name fragment]
"""

import sys
import time

from eval import config, llm
from pageindex_rag import llm_patch

STORE = config.ROOT / "pageindex_rag" / ".pageindex"

if __name__ == "__main__":
    frag = sys.argv[1] if len(sys.argv) > 1 else "SBI_General_Livestock"
    pdf = next(p for p in sorted(config.PDF_DIR.glob("*.pdf")) if frag in p.name)
    llm_patch.install()
    from pageindex import PageIndexLocalClient

    client = PageIndexLocalClient(model=config.LLM_MODEL, summary_model=config.LLM_MODEL,
                                  retrieve_model=config.LLM_MODEL, storage_path=str(STORE))
    before = llm.usage_totals()
    t0 = time.time()
    res = client.submit_document(str(pdf))
    after = llm.usage_totals()
    tree = client.get_tree(res["doc_id"], node_summary=True)["result"]

    def count(nodes):
        return sum(1 + count(n.get("nodes") or []) for n in nodes)

    print(f"{pdf.name}: {time.time() - t0:.0f}s, top nodes={len(tree)}, total nodes={count(tree)}")
    d = {k: after[k] - before[k] for k in after}
    print("this PDF:", d)
