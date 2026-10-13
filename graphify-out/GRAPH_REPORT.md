# Graph Report - capstone-deepseek  (2026-10-13)

## Corpus Check
- 153 files · ~2,285,191 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 885 nodes · 1069 edges · 80 communities (65 shown, 15 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ae561a28`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GraphRAGPipeline
- pipeline.py
- Graph RAG — Project Implementation Plan
- EntityMention
- Graph RAG Monorepo
- score_evaluation_groq.py
- run_policy_evaluation.py
- What You Must Do When Invoked
- score_evaluation_custom.py
- What You Must Do When Invoked
- BGEEmbedder
- ingest_policy_docs.py
- Eval set v2 — 40 new questions (DRAFT, for review)
- graphify.js
- graph-rag
- policy_loader.py
- Team Common Rules — RAG Evaluation Project
- Strategy Differences Log
- graph_rag/AGENTS.md
- graphify reference: extra exports and benchmark
- docs
- graphify reference: extra exports and benchmark
- final_plan.md — DeepSeek migration + RAG eval fix
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
- GraphIndex
- eval/config.py
- VectorIndex
- graph_rag/retrieve.py
- Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n
- retrieval_check.py
- Phase 3 results explained
- pi-0a89c4406925404d96ba6bf858050dc9
- pi-1cc88166fb454c2192f6157b6065f58a
- pi-1e5cffbc4ade4224b1a0c3f449edddaf
- pi-24ad0cb16a524340af8a4cfdc8a217dd
- pi-42f758c2276f46039437ccf5df58f9ce
- pi-45cec3af828a4b5583aa637da53a3eed
- pi-5dea76e20a44438084501a07d3417b80
- .__init__
- NERExtractor
- RAG Strategy Comparison — PageIndex vs Vector vs Graph
- Instructions for the construction
- Curated Policy Document Set
- pi-7414c3afd2c24f5cb46ab7fff5c3c910
- pi-81aa6514f3cc41a69ccf156aeb9706ad
- pi-81b4c4966c3247c1940d49e5d8e5e169
- pi-8394957513f7468aaf4a5b525ce40de3
- pi-9f1c1cff5eaa49d3aba25cc09783b477
- pi-b9867eaab5a840e9b457f9920996c390
- pi-d83401c000ac4faca3a0dc71982559b5
- pi-f567e57ef3b046ef88d14e64b9fbc8de
- Work Summary: DeepSeek Migration
- Results summary (n=40 questions, judge = deepseek-flash, temperature 0)
- main
- _load_spacy_model

## God Nodes (most connected - your core abstractions)
1. `Eval set v2 — 40 new questions (DRAFT, for review)` - 41 edges
2. `GraphRAGPipeline` - 19 edges
3. `docs` - 16 edges
4. `Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n` - 16 edges
5. `final_plan.md — DeepSeek migration + RAG eval fix` - 13 edges
6. `Strategy Differences Log` - 13 edges
7. `GraphIndex` - 12 edges
8. `EntityMention` - 12 edges
9. `VectorIndex` - 12 edges
10. `What You Must Do When Invoked` - 12 edges

## Surprising Connections (you probably didn't know these)
- `_run_ragas_scores()` --calls--> `evaluate()`  [INFERRED]
  graph_rag/scripts/score_evaluation_groq.py → eval/retrieval_check.py
- `_entity_index()` --calls--> `embed()`  [EXTRACTED]
  graph_rag/retrieve.py → eval/embed.py
- `linked_graph_results()` --calls--> `embed()`  [EXTRACTED]
  graph_rag/retrieve.py → eval/embed.py
- `vector_results()` --calls--> `embed()`  [EXTRACTED]
  graph_rag/retrieve.py → eval/embed.py
- `build()` --calls--> `embed()`  [EXTRACTED]
  vector_rag/index.py → eval/embed.py

## Import Cycles
- None detected.

## Communities (80 total, 15 thin omitted)

### Community 0 - "GraphRAGPipeline"
Cohesion: 0.16
Nodes (12): GraphRAGPipeline, Path, Persist the NetworkX graph and chunk lookup to a pickle file., Restore a previously pickled graph and chunk lookup., Merge graph/vector retrieval results by chunk ID and score., Create context text block with chunk IDs for grounded prompting., Run full hybrid retrieval + generation for a user question., Orchestrate data loading, indexing, retrieval, and answer generation. (+4 more)

### Community 1 - "pipeline.py"
Cohesion: 0.13
Nodes (11): CLI entrypoint for running a quick Graph RAG MVP query., NetworkX knowledge graph construction and traversal for Insurance Policy Graph…, Graph RAG MVP package exports., LLM client wrapper for answer generation in Graph RAG. Uses the NVIDIA NIM…, Entity extraction utilities for Graph RAG MVP., End-to-end Graph RAG pipeline orchestration for insurance policy documents., _load_gliner_model(), Insurance-domain NER extractor combining spaCy (general) + GLiNER (domain-… (+3 more)

### Community 2 - "Graph RAG — Project Implementation Plan"
Cohesion: 0.06
Nodes (35): 1.1 Research Objective, 1.2 Team Structure, 1.3 Key Constraints, 1. Project Overview, 2.1 Architecture Overview, 2.2 Layer 1 — Data Ingestion, 2.3 Layer 2 — Graph Construction, 2.4 Layer 3 — Storage (+27 more)

### Community 3 - "EntityMention"
Cohesion: 0.16
Nodes (12): LLMNERExtractor, normalize(), Entity extraction with the shared LLM client (replaces spaCy + GLiNER). Same…, PolicyNERExtractor, Run spaCy NER and return filtered entity mentions., Run GLiNER zero-shot NER for insurance-specific entity types. GLiNER has a hard…, Extract and merge entity mentions from *text*. GLiNER domain-specific entities…, Lightweight extraction for query-time NER (spaCy only, no GLiNER). At query… (+4 more)

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
Nodes (27): _build_and_save_report(), _call_llm_with_retry(), exact_match(), _get_llm_client(), _load_raw_records(), main(), _mean_std(), _normalize_text() (+19 more)

### Community 9 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 10 - "BGEEmbedder"
Cohesion: 0.20
Nodes (6): BGEEmbedder, Local embedding client wrappers for offline vectorization., Initialize local embedding model., Embed input texts and return vectors as Python lists., Generate embeddings locally with a SentenceTransformer model., Initialize Chroma client and selected embedding backend.

### Community 11 - "ingest_policy_docs.py"
Cohesion: 0.33
Nodes (6): main(), parse_args(), Namespace, One-shot CLI script to ingest all insurance policy PDFs into Graph RAG indices.…, Parse CLI arguments for the policy ingestion script., Build graph and vector indices from all policy PDFs.

### Community 12 - "Eval set v2 — 40 new questions (DRAFT, for review)"
Cohesion: 0.05
Nodes (41): Eval set v2 — 40 new questions (DRAFT, for review), q01 · A_single_fact, q02 · A_single_fact, q03 · A_single_fact, q04 · A_single_fact, q05 · A_single_fact, q06 · A_single_fact, q07 · A_single_fact (+33 more)

### Community 17 - "policy_loader.py"
Cohesion: 0.07
Nodes (35): Page header and footer removal, shared by every strategy that reads PDFs., Lines that appear on at least SHARE of a PDF's pages (headers, footers)., repeated_lines(), strip_pages(), _call_llm_with_retry(), generate_qa_pairs(), _is_duplicate(), main() (+27 more)

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

### Community 23 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 24 - "final_plan.md — DeepSeek migration + RAG eval fix"
Cohesion: 0.09
Nodes (21): 0. Goal, 10. Next action, 11. Status, 1. Hard rules, 2. Findings (from reading code), 3. Decision: git worktree (no nuke, no archive), 4. Target layout (inside the worktree), 5. Decisions (D1-D5: user took recommendations, 2026-10-12) (+13 more)

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

### Community 45 - "GraphIndex"
Cohesion: 0.18
Nodes (8): GraphIndex, Any, Retrieve chunk candidates via entity-seeded graph traversal., Build and query a chunk-entity graph for Graph RAG retrieval., Initialize graph index with a configured NER extractor. Args: ner_extractor:…, Create or reuse a Document node keyed by source_id., Populate graph nodes and edges from chunk records., Create or reuse an entity node keyed by normalized entity text.

### Community 46 - "eval/config.py"
Cohesion: 0.06
Nodes (49): Single source of settings for all three RAG strategies and the judge., answer(), format_context(), Shared answer generation. Every strategy uses the same prompt and format. The…, _parse(), The one judge for all three RAG strategies. Input row: {"question", "answer",…, Score every row for every metric in parallel. Order is kept., Score one metric. The reply must hold a numeric "score" key. On a bad reply,… (+41 more)

### Community 47 - "VectorIndex"
Cohesion: 0.27
Nodes (7): ChunkRecord, Represent a text chunk and its source metadata., Retrieve top-k chunks by semantic similarity search., Manage chunk embedding storage and semantic retrieval., Create embeddings using Jina Embeddings API., Embed and upsert chunk records into ChromaDB., VectorIndex

### Community 48 - "graph_rag/retrieve.py"
Cohesion: 0.11
Nodes (28): _call(), _cli(), embed(), _path(), ndarray, OpenAI, Embeddings through an OpenAI-compatible endpoint, cached on disk per text.…, Return L2-normalised float32 vectors, shape (len(texts), EMBED_DIM). (+20 more)

### Community 49 - "Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n"
Cohesion: 0.12
Nodes (16): 10. PageIndex RAG / q38 (D_cross_document), 11. Graph RAG (linked) / q28 (B_multi_part), 12. Graph RAG (linked) / q03 (A_single_fact), 13. Graph RAG (linked) / q02 (A_single_fact), 14. Graph RAG (linked) / q06 (A_single_fact), 15. Graph RAG (linked) / q14 (A_single_fact), 1. Vector RAG / q08 (A_single_fact), 2. Vector RAG / q02 (A_single_fact) (+8 more)

### Community 50 - "retrieval_check.py"
Cohesion: 0.17
Nodes (13): _agg(), _anchors(), budget_rank(), evaluate(), first_hit_rank(), norm(), Phase 3a: retrieval-only metrics against the frozen eval set. No LLM call. A…, Return (strict_rank, page_rank), 1-based, or None. (+5 more)

### Community 51 - "Phase 3 results explained"
Cohesion: 0.13
Nodes (14): 1. Words used in this file, 2. What the check does, 3.1 The letter k, 3.2 any@5, 3.3 all@5, 3.4 MRR, 3.5 Worked example, 3. The numbers (+6 more)

### Community 52 - "pi-0a89c4406925404d96ba6bf858050dc9"
Cohesion: 0.20
Nodes (10): pi-0a89c4406925404d96ba6bf858050dc9, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 53 - "pi-1cc88166fb454c2192f6157b6065f58a"
Cohesion: 0.20
Nodes (10): pi-1cc88166fb454c2192f6157b6065f58a, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 54 - "pi-1e5cffbc4ade4224b1a0c3f449edddaf"
Cohesion: 0.20
Nodes (10): pi-1e5cffbc4ade4224b1a0c3f449edddaf, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 55 - "pi-24ad0cb16a524340af8a4cfdc8a217dd"
Cohesion: 0.20
Nodes (10): pi-24ad0cb16a524340af8a4cfdc8a217dd, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 56 - "pi-42f758c2276f46039437ccf5df58f9ce"
Cohesion: 0.20
Nodes (10): pi-42f758c2276f46039437ccf5df58f9ce, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 57 - "pi-45cec3af828a4b5583aa637da53a3eed"
Cohesion: 0.20
Nodes (10): pi-45cec3af828a4b5583aa637da53a3eed, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 58 - "pi-5dea76e20a44438084501a07d3417b80"
Cohesion: 0.20
Nodes (10): pi-5dea76e20a44438084501a07d3417b80, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 59 - ".__init__"
Cohesion: 0.25
Nodes (5): LLMGenerator, Generate grounded answers from retrieved context using NVIDIA NIM., Initialize NVIDIA NIM client from environment. Args: policy_mode: If True, use…, Generate an answer constrained to provided context. Retries up to *max_retries*…, Initialize Graph RAG components. Args: policy_mode: If True, use…

### Community 60 - "NERExtractor"
Cohesion: 0.29
Nodes (5): NERExtractor, Extract named entities using spaCy for chunks and queries., Initialize spaCy model for NER extraction., Normalize entity text for deduplication in graph indexing., Extract and normalize unique entity mentions from input text.

### Community 61 - "RAG Strategy Comparison — PageIndex vs Vector vs Graph"
Cohesion: 0.33
Nodes (5): ⚠️ Data-quality caveat: Graph RAG's scores are not trustworthy as-is, RAG Strategy Comparison — PageIndex vs Vector vs Graph, Recommendation, Results, What the trustworthy numbers say

### Community 66 - "pi-7414c3afd2c24f5cb46ab7fff5c3c910"
Cohesion: 0.20
Nodes (10): pi-7414c3afd2c24f5cb46ab7fff5c3c910, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 67 - "pi-81aa6514f3cc41a69ccf156aeb9706ad"
Cohesion: 0.20
Nodes (10): pi-81aa6514f3cc41a69ccf156aeb9706ad, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 68 - "pi-81b4c4966c3247c1940d49e5d8e5e169"
Cohesion: 0.20
Nodes (10): pi-81b4c4966c3247c1940d49e5d8e5e169, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 69 - "pi-8394957513f7468aaf4a5b525ce40de3"
Cohesion: 0.20
Nodes (10): pi-8394957513f7468aaf4a5b525ce40de3, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 70 - "pi-9f1c1cff5eaa49d3aba25cc09783b477"
Cohesion: 0.20
Nodes (10): pi-9f1c1cff5eaa49d3aba25cc09783b477, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 71 - "pi-b9867eaab5a840e9b457f9920996c390"
Cohesion: 0.20
Nodes (10): pi-b9867eaab5a840e9b457f9920996c390, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 72 - "pi-d83401c000ac4faca3a0dc71982559b5"
Cohesion: 0.20
Nodes (10): pi-d83401c000ac4faca3a0dc71982559b5, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 73 - "pi-f567e57ef3b046ef88d14e64b9fbc8de"
Cohesion: 0.20
Nodes (10): pi-f567e57ef3b046ef88d14e64b9fbc8de, createdAt, description, folderId, id, metadata, mode, name (+2 more)

### Community 74 - "Work Summary: DeepSeek Migration"
Cohesion: 0.12
Nodes (15): 10. Next steps, 1. Goal, 2. Status, 3.1 Null scores in detail, 3. Judge, 4. Findings per strategy, 5. Decision you made, 6. Full-run results (+7 more)

### Community 75 - "Results summary (n=40 questions, judge = deepseek-flash, temperature 0)"
Cohesion: 0.40
Nodes (4): Answer correctness by tier (mean), Mean score with 95% bootstrap CI, Paired differences vs Vector RAG (same questions), Results summary (n=40 questions, judge = deepseek-flash, temperature 0)

### Community 76 - "main"
Cohesion: 0.40
Nodes (5): main(), parse_args(), Namespace, Parse command-line arguments for the Insurance Policy Graph RAG query., Build or load policy indices and execute one Graph RAG query.

### Community 77 - "_load_spacy_model"
Cohesion: 0.50
Nodes (3): _load_spacy_model(), Lazily load spaCy model (cached)., Initialise both NER backends. Args: spacy_model: spaCy model name for generic…

## Knowledge Gaps
- **417 isolated node(s):** `$schema`, `.opencode/plugins/graphify.js`, `id`, `name`, `description` (+412 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `docs` connect `docs` to `pi-7414c3afd2c24f5cb46ab7fff5c3c910`, `pi-81aa6514f3cc41a69ccf156aeb9706ad`, `pi-81b4c4966c3247c1940d49e5d8e5e169`, `pi-8394957513f7468aaf4a5b525ce40de3`, `pi-9f1c1cff5eaa49d3aba25cc09783b477`, `pi-b9867eaab5a840e9b457f9920996c390`, `pi-d83401c000ac4faca3a0dc71982559b5`, `pi-f567e57ef3b046ef88d14e64b9fbc8de`, `pi-0a89c4406925404d96ba6bf858050dc9`, `pi-1cc88166fb454c2192f6157b6065f58a`, `pi-1e5cffbc4ade4224b1a0c3f449edddaf`, `pi-24ad0cb16a524340af8a4cfdc8a217dd`, `pi-42f758c2276f46039437ccf5df58f9ce`, `pi-45cec3af828a4b5583aa637da53a3eed`, `pi-5dea76e20a44438084501a07d3417b80`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Why does `evaluate()` connect `retrieval_check.py` to `score_evaluation_groq.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Why does `_run_ragas_scores()` connect `score_evaluation_groq.py` to `retrieval_check.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `GraphRAGPipeline` (e.g. with `main()` and `main()`) actually correct?**
  _`GraphRAGPipeline` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `.opencode/plugins/graphify.js`, `id` to the rest of the system?**
  _417 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pipeline.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12857142857142856 - nodes in this community are weakly interconnected._
- **Should `Graph RAG — Project Implementation Plan` be split into smaller, more focused modules?**
  _Cohesion score 0.05555555555555555 - nodes in this community are weakly interconnected._