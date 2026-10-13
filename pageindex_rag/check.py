"""Phase 3a for PageIndex: python -m pageindex_rag.check"""

import json

from eval import config, retrieval_check
from pageindex_rag import retrieve

if __name__ == "__main__":
    res = retrieval_check.evaluate(retrieve.retrieve, top_k=5)
    retrieval_check.print_summary("PageIndex RAG (LLM router + tree search)", res)
    out = config.RESULTS_DIR / "retrieval_pageindex_rag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
