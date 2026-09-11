# Graph Report - capstone  (2026-09-09)

## Corpus Check
- 52 files · ~86,793 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 472 nodes · 561 edges · 45 communities (33 shown, 12 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `505a13ad`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GraphRAGPipeline
- pipeline.py
- Graph RAG — Project Implementation Plan
- PolicyNERExtractor
- Graph RAG Monorepo
- score_evaluation_groq.py
- run_policy_evaluation.py
- What You Must Do When Invoked
- score_evaluation_custom.py
- What You Must Do When Invoked
- BGEEmbedder
- ingest_policy_docs.py
- main
- graphify.js
- graph-rag
- generate_policy_qa.py
- Team Common Rules — RAG Evaluation Project
- Strategy Differences Log
- graph_rag/AGENTS.md
- graphify reference: extra exports and benchmark
- policy_loader.py
- graphify reference: extra exports and benchmark
- data_loader.py
- graphify reference: query, path, explain
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- opencode.json
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- Curated Policy Document Set
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- AGENTS.md
- rules/graphify.md
- .agents/skills/graphify/references/extraction-spec.md
- workflows/graphify.md
- .opencode/skills/graphify/references/extraction-spec.md

## God Nodes (most connected - your core abstractions)
1. `GraphRAGPipeline` - 19 edges
2. `Strategy Differences Log` - 13 edges
3. `VectorIndex` - 12 edges
4. `What You Must Do When Invoked` - 12 edges
5. `What You Must Do When Invoked` - 12 edges
6. `GraphIndex` - 11 edges
7. `PolicyNERExtractor` - 11 edges
8. `Team Common Rules — RAG Evaluation Project` - 11 edges
9. `generate_qa_pairs()` - 10 edges
10. `main()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/main.py → graph_rag/src/pipeline.py
- `generate_qa_pairs()` --calls--> `load_policy_documents()`  [INFERRED]
  graph_rag/scripts/generate_policy_qa.py → graph_rag/src/policy_loader.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/scripts/ingest_policy_docs.py → graph_rag/src/pipeline.py
- `main()` --uses--> `GraphRAGPipeline`  [INFERRED]
  graph_rag/scripts/run_policy_evaluation.py → graph_rag/src/pipeline.py
- `load_policy_documents()` --calls--> `_split_text()`  [INFERRED]
  graph_rag/src/policy_loader.py → graph_rag/src/data_loader.py

## Import Cycles
- None detected.

## Communities (45 total, 12 thin omitted)

### Community 0 - "GraphRAGPipeline"
Cohesion: 0.05
Nodes (36): GraphIndex, Any, Retrieve chunk candidates via entity-seeded graph traversal., Build and query a chunk-entity graph for Graph RAG retrieval., Initialize graph index with a configured NER extractor. Args: ner_extractor:…, Create or reuse a Document node keyed by source_id., Populate graph nodes and edges from chunk records., Create or reuse an entity node keyed by normalized entity text. (+28 more)

### Community 1 - "pipeline.py"
Cohesion: 0.15
Nodes (9): CLI entrypoint for running a quick Graph RAG MVP query., NetworkX knowledge graph construction and traversal for Insurance Policy Graph…, Graph RAG MVP package exports., LLM client wrapper for answer generation in Graph RAG. Uses the NVIDIA NIM…, Entity extraction utilities for Graph RAG MVP., End-to-end Graph RAG pipeline orchestration for insurance policy documents., Insurance-domain NER extractor combining spaCy (general) + GLiNER (domain-…, Shared typed structures for the Graph RAG MVP pipeline. (+1 more)

### Community 2 - "Graph RAG — Project Implementation Plan"
Cohesion: 0.06
Nodes (35): 1.1 Research Objective, 1.2 Team Structure, 1.3 Key Constraints, 1. Project Overview, 2.1 Architecture Overview, 2.2 Layer 1 — Data Ingestion, 2.3 Layer 2 — Graph Construction, 2.4 Layer 3 — Storage (+27 more)

### Community 3 - "PolicyNERExtractor"
Cohesion: 0.15
Nodes (14): _load_gliner_model(), _load_spacy_model(), PolicyNERExtractor, Run spaCy NER and return filtered entity mentions., Run GLiNER zero-shot NER for insurance-specific entity types. GLiNER has a hard…, Extract and merge entity mentions from *text*. GLiNER domain-specific entities…, Lightweight extraction for query-time NER (spaCy only, no GLiNER). At query…, Lazily load GLiNER model (cached after first call — ~300 MB download once). (+6 more)

### Community 4 - "Graph RAG Monorepo"
Cohesion: 0.07
Nodes (29): API Keys Setup, "command not found: mise", Creating a New Strategy, Environment Setup, From a Git Repository, From a Local Directory, Git LFS files not downloading, Graph RAG Monorepo (+21 more)

### Community 5 - "score_evaluation_groq.py"
Cohesion: 0.11
Nodes (27): Dataset, exact_match(), _load_raw_records(), main(), _mean_std(), _metric_values(), _normalize_text(), parse_args() (+19 more)

### Community 6 - "run_policy_evaluation.py"
Cohesion: 0.19
Nodes (15): datetime, _infer_question_type(), _load_checkpoint(), main(), parse_args(), Any, Namespace, Path (+7 more)

### Community 7 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 8 - "score_evaluation_custom.py"
Cohesion: 0.11
Nodes (24): _call_llm_with_retry(), exact_match(), _get_llm_client(), _load_raw_records(), main(), _mean_std(), _normalize_text(), parse_args() (+16 more)

### Community 9 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 10 - "BGEEmbedder"
Cohesion: 0.20
Nodes (6): BGEEmbedder, Local embedding client wrappers for offline vectorization., Initialize local embedding model., Embed input texts and return vectors as Python lists., Generate embeddings locally with a SentenceTransformer model., Initialize Chroma client and selected embedding backend.

### Community 11 - "ingest_policy_docs.py"
Cohesion: 0.33
Nodes (6): main(), parse_args(), Namespace, One-shot CLI script to ingest all insurance policy PDFs into Graph RAG indices.…, Parse CLI arguments for the policy ingestion script., Build graph and vector indices from all policy PDFs.

### Community 12 - "main"
Cohesion: 0.40
Nodes (5): main(), parse_args(), Namespace, Parse command-line arguments for the Insurance Policy Graph RAG query., Build or load policy indices and execute one Graph RAG query.

### Community 17 - "generate_policy_qa.py"
Cohesion: 0.15
Nodes (17): _call_llm_with_retry(), generate_qa_pairs(), _is_duplicate(), main(), parse_args(), _parse_qa(), Any, Namespace (+9 more)

### Community 18 - "Team Common Rules — RAG Evaluation Project"
Cohesion: 0.11
Nodes (17): 2.1 Why a Multi-Config Mixture, 2.2 Complete Config Reference, 2.3 Recommended Config Combinations by Phase, 2.4 Dataset Size Reference, 2.5 Local Download Instructions, 3.1 Custom Metric Implementations, Rule 1 — Shared Configuration (Non-Negotiable), Rule 2 — Dataset Selection and Sampling (+9 more)

### Community 19 - "Strategy Differences Log"
Cohesion: 0.14
Nodes (13): 1) Embedding model/provider mismatch, 2) Python runtime baseline is stricter than plan, 3) MVP sample size differs from Phase 1 recommendation, 4) Dataset download flow is MVP-centric, not team-shared flow yet, 5) NER stack is partial vs implementation guide, 6) Graph storage layer is still NetworkX-only, 7) Evaluation/experiment tracking not implemented yet, 8) Additional dependencies for local embedding workaround (+5 more)

### Community 20 - "graph_rag/AGENTS.md"
Cohesion: 0.18
Nodes (9): Code Map, Corpus, Current Status, End-to-End Evaluation Workflow, Env Vars, Output Files (fresh run), Rate Limit Budget, Scope + Ground Truth (+1 more)

### Community 21 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 22 - "policy_loader.py"
Cohesion: 0.31
Nodes (8): _clean_pdf_text(), _derive_policy_name(), load_policy_documents(), Path, PDF loading and chunking for insurance policy documents (Phase 4)., Normalize raw PDF text: remove headers/footers, collapse whitespace., Derive a human-readable policy name from a PDF filename. Strips trailing hash…, Load all PDF policy documents from *policy_dir* and return chunk records. Args:…

### Community 23 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 24 - "data_loader.py"
Cohesion: 0.29
Nodes (7): chunk_records_to_dicts(), Text splitting and chunk serialization utilities for Graph RAG., Split *text* into sentences using punctuation and paragraph boundaries. Uses a…, Split text into overlapping chunks that always end on a sentence boundary.…, Serialize chunk records to plain dictionaries for storage/output., _sentence_split(), _split_text()

### Community 25 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 26 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 27 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 28 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 29 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 30 - "opencode.json"
Cohesion: 0.50
Nodes (3): plugin, $schema, .opencode/plugins/graphify.js

### Community 31 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 32 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 33 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

## Knowledge Gaps
- **175 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `graph-rag`, `graphify`, `Usage` (+170 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraphRAGPipeline` connect `GraphRAGPipeline` to `pipeline.py`, `PolicyNERExtractor`, `run_policy_evaluation.py`, `ingest_policy_docs.py`, `main`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `LLMGenerator` connect `GraphRAGPipeline` to `pipeline.py`, `generate_policy_qa.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `load_policy_documents()` connect `policy_loader.py` to `GraphRAGPipeline`, `generate_policy_qa.py`, `data_loader.py`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `GraphRAGPipeline` (e.g. with `main()` and `main()`) actually correct?**
  _`GraphRAGPipeline` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `VectorIndex` (e.g. with `GraphRAGPipeline` and `.build_indices_from_policy_docs()`) actually correct?**
  _`VectorIndex` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `.opencode/plugins/graphify.js`, `graph-rag` to the rest of the system?**
  _175 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `GraphRAGPipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.053246753246753244 - nodes in this community are weakly interconnected._