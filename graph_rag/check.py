"""Phase 3a for Graph RAG: python -m graph_rag.check  (graph-only, vector-only, hybrid)"""

import json

from eval import config, retrieval_check
from graph_rag import retrieve

if __name__ == "__main__":
    out = {}
    for mode in ("graph", "vector", "hybrid"):
        res = retrieval_check.evaluate(lambda q, k, m=mode: retrieve.retrieve(q, k, mode=m), top_k=config.TOP_K)
        retrieval_check.print_summary(f"Graph RAG [{mode}]", res)
        out[mode] = res
    p = config.RESULTS_DIR / "retrieval_graph_rag_v2.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
