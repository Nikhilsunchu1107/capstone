"""Build the Graph RAG index (chunks, NER graph, embeddings). python -m graph_rag.build_index"""

import time

from graph_rag import retrieve

if __name__ == "__main__":
    t0 = time.time()
    gi, order, vecs = retrieve.build()
    g = gi.graph
    kinds = {}
    for _, d in g.nodes(data=True):
        kinds[d.get("node_type")] = kinds.get(d.get("node_type"), 0) + 1
    print(f"chunks={len(order)} vectors={vecs.shape} nodes={kinds} edges={g.number_of_edges()} {time.time() - t0:.0f}s")
