"""Phase 3a for Vector RAG: python -m vector_rag.check"""

import json

from eval import config, retrieval_check
from vector_rag import retrieve

if __name__ == "__main__":
    res = retrieval_check.evaluate(retrieve.retrieve)
    retrieval_check.print_summary("Vector RAG (nemotron-3-embed-1b, chunk 1000/150)", res)
    out = config.RESULTS_DIR / "retrieval_vector_rag_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
