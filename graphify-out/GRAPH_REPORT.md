# Graph Report - capstone  (2026-08-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 277 nodes · 432 edges · 17 communities (15 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e265437d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GraphRAGPipeline
- pipeline.py
- score_evaluation.py
- PolicyNERExtractor
- run_evaluation.py
- generate_eval_samples.py
- run_policy_evaluation.py
- download_datasets.py
- download_ragbench_sample.py
- create_chunk_records
- BGEEmbedder
- ingest_policy_docs.py
- main
- graphify.js
- graph-rag

## God Nodes (most connected - your core abstractions)
1. `GraphRAGPipeline` - 20 edges
2. `GraphIndex` - 12 edges
3. `VectorIndex` - 12 edges
4. `PolicyNERExtractor` - 11 edges
5. `main()` - 11 edges
6. `NERExtractor` - 10 edges
7. `main()` - 10 edges
8. `ChunkRecord` - 9 edges
9. `RetrievedChunk` - 9 edges
10. `EntityMention` - 9 edges

## Surprising Connections (you probably didn't know these)
- `NERExtractor` --uses--> `EntityMention`  [INFERRED]
  graph_rag/src/ner.py → graph_rag/src/types.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/main.py → graph_rag/src/pipeline.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/run_evaluation.py → graph_rag/src/pipeline.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/scripts/ingest_policy_docs.py → graph_rag/src/pipeline.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/scripts/run_policy_evaluation.py → graph_rag/src/pipeline.py

## Import Cycles
- None detected.

## Communities (17 total, 2 thin omitted)

### Community 0 - "GraphRAGPipeline"
Cohesion: 0.06
Nodes (34): GraphIndex, Retrieve chunk candidates via entity-seeded graph traversal., Build and query a chunk-entity graph for Graph RAG retrieval., Initialize graph index with a configured NER extractor. Args: ner_extractor:…, Create or reuse a Document node keyed by source_id., Populate graph nodes and edges from chunk records., Create or reuse an entity node keyed by normalized entity text., LLMGenerator (+26 more)

### Community 1 - "pipeline.py"
Cohesion: 0.06
Nodes (35): CLI entrypoint for running a quick Graph RAG MVP query., generate_qa_pairs(), _is_duplicate(), main(), parse_args(), _parse_qa(), Any, Namespace (+27 more)

### Community 2 - "score_evaluation.py"
Cohesion: 0.11
Nodes (27): exact_match(), _load_raw_records(), main(), _mean_std(), _metric_values(), _normalize_text(), parse_args(), Any (+19 more)

### Community 3 - "PolicyNERExtractor"
Cohesion: 0.12
Nodes (16): Normalize entity text for deduplication in graph indexing., Extract and normalize unique entity mentions from input text., _load_gliner_model(), _load_spacy_model(), PolicyNERExtractor, Run spaCy NER and return filtered entity mentions., Run GLiNER zero-shot NER for insurance-specific entity types., Extract and merge entity mentions from *text*. GLiNER domain-specific entities… (+8 more)

### Community 4 - "run_evaluation.py"
Cohesion: 0.15
Nodes (20): _get_ground_truth(), _infer_question_type(), _limit_dataset(), _load_eval_dataset(), main(), parse_args(), _parse_configs(), Any (+12 more)

### Community 5 - "generate_eval_samples.py"
Cohesion: 0.16
Nodes (18): _allocation(), _build_group_payload(), _load_local_split(), main(), _normalize_configs(), parse_args(), Dataset, Namespace (+10 more)

### Community 6 - "run_policy_evaluation.py"
Cohesion: 0.19
Nodes (15): datetime, _infer_question_type(), _load_checkpoint(), main(), parse_args(), Any, Namespace, Path (+7 more)

### Community 7 - "download_datasets.py"
Cohesion: 0.20
Nodes (15): download_config(), _is_complete_saved_dataset(), main(), _normalize_configs(), parse_args(), Namespace, Path, Download and persist team-standard RAGBench configs for local reuse. (+7 more)

### Community 8 - "download_ragbench_sample.py"
Cohesion: 0.21
Nodes (13): download_sample(), main(), parse_args(), Namespace, Path, Download a deterministic RAGBench sample for MVP runs., Parse command-line arguments for sample download., Load RAGBench and return a deterministic sampled dataset and indices. (+5 more)

### Community 9 - "create_chunk_records"
Cohesion: 0.24
Nodes (10): create_chunk_records(), load_local_ragbench(), load_ragbench_configs(), Dataset, Path, Load the local RAGBench sample dataset from disk., Convert RAGBench rows into chunk records for indexing and retrieval., Load one or more RAGBench configs from disk. Args: configs: List of config… (+2 more)

### Community 10 - "BGEEmbedder"
Cohesion: 0.20
Nodes (6): BGEEmbedder, Local embedding client wrappers for offline vectorization., Initialize local embedding model., Embed input texts and return vectors as Python lists., Generate embeddings locally with a SentenceTransformer model., Initialize Chroma client and selected embedding backend.

### Community 11 - "ingest_policy_docs.py"
Cohesion: 0.33
Nodes (6): main(), parse_args(), Namespace, One-shot CLI script to ingest all insurance policy PDFs into Graph RAG indices.…, Parse CLI arguments for the policy ingestion script., Build graph and vector indices from all policy PDFs.

### Community 12 - "main"
Cohesion: 0.40
Nodes (5): main(), parse_args(), Namespace, Parse command-line arguments for the MVP query run., Build MVP indices and execute one Graph RAG query.

## Knowledge Gaps
- **1 isolated node(s):** `graph-rag`
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraphRAGPipeline` connect `GraphRAGPipeline` to `pipeline.py`, `PolicyNERExtractor`, `run_evaluation.py`, `run_policy_evaluation.py`, `ingest_policy_docs.py`, `main`?**
  _High betweenness centrality (0.238) - this node is a cross-community bridge._
- **Why does `VectorIndex` connect `GraphRAGPipeline` to `pipeline.py`, `BGEEmbedder`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Why does `PolicyNERExtractor` connect `PolicyNERExtractor` to `GraphRAGPipeline`, `pipeline.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `GraphRAGPipeline` (e.g. with `main()` and `main()`) actually correct?**
  _`GraphRAGPipeline` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `GraphIndex` (e.g. with `NERExtractor` and `ChunkRecord`) actually correct?**
  _`GraphIndex` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `VectorIndex` (e.g. with `GraphRAGPipeline` and `.build_indices_from_policy_docs()`) actually correct?**
  _`VectorIndex` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `PolicyNERExtractor` (e.g. with `GraphRAGPipeline` and `.__init__()`) actually correct?**
  _`PolicyNERExtractor` has 3 INFERRED edges - model-reasoned connections that need verification._