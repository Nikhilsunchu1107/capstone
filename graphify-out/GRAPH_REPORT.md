# Graph Report - capstone-deepseek  (2026-10-26)

## Corpus Check
- 154 files · ~3,055,826 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1006 nodes · 1311 edges · 94 communities (78 shown, 16 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 37 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f566dab7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GraphRAGPipeline
- types.py
- Graph RAG — Project Implementation Plan
- PolicyNERExtractor
- Graph RAG Monorepo
- score_evaluation_groq.py
- Tier G · unanswerable / trap
- report.py
- score_evaluation_custom.py
- Tier E · multi-hop, one policy
- BGEEmbedder
- main.py
- eval/config.py
- Tier F · cross-document
- What You Must Do When Invoked
- graph-rag
- generate_policy_qa.py
- Team Common Rules — RAG Evaluation Project
- Strategy Differences Log
- graph_rag/AGENTS.md
- Tier H · paraphrased
- docs
- What You Must Do When Invoked
- final_plan.md — DeepSeek migration + RAG eval fix
- Comparison of the Retrieval Strategies
- pipeline.py
- policy_loader.py
- Eval set v2: 80 new, harder questions (DRAFT, for review)
- graph_rag/retrieve.py
- opencode.json
- judge.py
- llm_patch.py
- graphify reference: extra exports and benchmark
- EntityMention
- graphify reference: extra exports and benchmark
- Curated Policy Document Set
- pageindex_rag/retrieve.py
- graphify reference: query, path, explain
- AGENTS.md
- textclean.py
- graphify.js
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- GraphIndex
- llm.py
- VectorIndex
- graphify reference: commit hook and native CLAUDE.md integration
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
- graphify reference: incremental update and cluster-only
- pi-7414c3afd2c24f5cb46ab7fff5c3c910
- pi-81aa6514f3cc41a69ccf156aeb9706ad
- pi-81b4c4966c3247c1940d49e5d8e5e169
- pi-8394957513f7468aaf4a5b525ce40de3
- pi-9f1c1cff5eaa49d3aba25cc09783b477
- pi-b9867eaab5a840e9b457f9920996c390
- pi-d83401c000ac4faca3a0dc71982559b5
- pi-f567e57ef3b046ef88d14e64b9fbc8de
- Work Summary: DeepSeek Migration
- Results summary (n=80 questions, judge = deepseek-flash, temperature 0)
- _load_spacy_model
- policy_ner.py
- run_policy_evaluation.py
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- rules/graphify.md
- .agents/skills/graphify/references/extraction-spec.md
- workflows/graphify.md
- .opencode/skills/graphify/references/extraction-spec.md

## God Nodes (most connected - your core abstractions)
1. `Tier E · multi-hop, one policy` - 25 edges
2. `Tier F · cross-document` - 25 edges
3. `Tier H · paraphrased` - 21 edges
4. `GraphRAGPipeline` - 19 edges
5. `docs` - 16 edges
6. `Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n` - 16 edges
7. `Tier G · unanswerable / trap` - 13 edges
8. `final_plan.md — DeepSeek migration + RAG eval fix` - 13 edges
9. `Strategy Differences Log` - 13 edges
10. `GraphIndex` - 12 edges

## Surprising Connections (you probably didn't know these)
- `_run_ragas_scores()` --calls--> `evaluate()`  [INFERRED]
  graph_rag/scripts/score_evaluation_groq.py → eval/retrieval_check.py
- `_json()` --calls--> `chat()`  [EXTRACTED]
  pageindex_rag/retrieve.py → eval/llm.py
- `_completion()` --calls--> `chat_full()`  [EXTRACTED]
  pageindex_rag/llm_patch.py → eval/llm.py
- `load_policy_documents()` --calls--> `strip_pages()`  [EXTRACTED]
  graph_rag/src/policy_loader.py → eval/textclean.py
- `_entity_index()` --calls--> `embed()`  [EXTRACTED]
  graph_rag/retrieve.py → eval/embed.py

## Import Cycles
- None detected.

## Communities (94 total, 16 thin omitted)

### Community 0 - "GraphRAGPipeline"
Cohesion: 0.16
Nodes (12): GraphRAGPipeline, Path, Persist the NetworkX graph and chunk lookup to a pickle file., Restore a previously pickled graph and chunk lookup., Merge graph/vector retrieval results by chunk ID and score., Create context text block with chunk IDs for grounded prompting., Run full hybrid retrieval + generation for a user question., Orchestrate data loading, indexing, retrieval, and answer generation. (+4 more)

### Community 1 - "types.py"
Cohesion: 0.16
Nodes (12): dataclasses, chunk_records_to_dicts(), Text splitting and chunk serialization utilities for Graph RAG., Split *text* into sentences using punctuation and paragraph boundaries. Uses a…, Split text into overlapping chunks that always end on a sentence boundary.…, Serialize chunk records to plain dictionaries for storage/output., _sentence_split(), _split_text() (+4 more)

### Community 2 - "Graph RAG — Project Implementation Plan"
Cohesion: 0.06
Nodes (35): 1.1 Research Objective, 1.2 Team Structure, 1.3 Key Constraints, 1. Project Overview, 2.1 Architecture Overview, 2.2 Layer 1 — Data Ingestion, 2.3 Layer 2 — Graph Construction, 2.4 Layer 3 — Storage (+27 more)

### Community 3 - "PolicyNERExtractor"
Cohesion: 0.20
Nodes (9): _load_gliner_model(), PolicyNERExtractor, Run spaCy NER and return filtered entity mentions., Run GLiNER zero-shot NER for insurance-specific entity types. GLiNER has a hard…, Extract and merge entity mentions from *text*. GLiNER domain-specific entities…, Lightweight extraction for query-time NER (spaCy only, no GLiNER). At query…, Lazily load GLiNER model (cached after first call — ~300 MB download once)., Extract entities from insurance policy text using spaCy + GLiNER. spaCy handles… (+1 more)

### Community 4 - "Graph RAG Monorepo"
Cohesion: 0.07
Nodes (29): API Keys Setup, "command not found: mise", Creating a New Strategy, Environment Setup, From a Git Repository, From a Local Directory, Git LFS files not downloading, Graph RAG Monorepo (+21 more)

### Community 5 - "score_evaluation_groq.py"
Cohesion: 0.07
Nodes (36): Dataset, datasets, exact_match(), _load_raw_records(), main(), _mean_std(), _metric_values(), _normalize_text() (+28 more)

### Community 6 - "Tier G · unanswerable / trap"
Cohesion: 0.15
Nodes (13): q100 · G_unanswerable, q89 · G_unanswerable, q90 · G_unanswerable, q91 · G_unanswerable, q92 · G_unanswerable, q93 · G_unanswerable, q94 · G_unanswerable, q95 · G_unanswerable (+5 more)

### Community 7 - "report.py"
Cohesion: 0.21
Nodes (17): csv, ci(), common_ids(), complete(), handcheck(), load(), main(), paired() (+9 more)

### Community 8 - "score_evaluation_custom.py"
Cohesion: 0.11
Nodes (28): _build_and_save_report(), _call_llm_with_retry(), exact_match(), _get_llm_client(), _load_raw_records(), main(), _mean_std(), _normalize_text() (+20 more)

### Community 9 - "Tier E · multi-hop, one policy"
Cohesion: 0.08
Nodes (25): q41 · E_multi_hop, q42 · E_multi_hop, q43 · E_multi_hop, q44 · E_multi_hop, q45 · E_multi_hop, q46 · E_multi_hop, q47 · E_multi_hop, q48 · E_multi_hop (+17 more)

### Community 10 - "BGEEmbedder"
Cohesion: 0.18
Nodes (7): BGEEmbedder, Local embedding client wrappers for offline vectorization., Initialize local embedding model., Embed input texts and return vectors as Python lists., Generate embeddings locally with a SentenceTransformer model., Initialize Chroma client and selected embedding backend., sentence_transformers

### Community 11 - "main.py"
Cohesion: 0.15
Nodes (15): argparse, dotenv, main(), parse_args(), Namespace, CLI entrypoint for running a quick Graph RAG MVP query., Parse command-line arguments for the Insurance Policy Graph RAG query., Build or load policy indices and execute one Graph RAG query. (+7 more)

### Community 12 - "eval/config.py"
Cohesion: 0.19
Nodes (9): Single source of settings for all three RAG strategies and the judge., Phase 3a for Graph RAG: python -m graph_rag.check (graph-only, vector-only,…, json, langchain_text_splitters, numpy, Phase 3a for PageIndex: python -m pageindex_rag.check, Phase 3a for Vector RAG: python -m vector_rag.check, Vector RAG index: PDF pages -> chunks -> embeddings -> one matrix on disk. Run:… (+1 more)

### Community 13 - "Tier F · cross-document"
Cohesion: 0.08
Nodes (25): q65 · F_cross_document, q66 · F_cross_document, q67 · F_cross_document, q68 · F_cross_document, q69 · F_cross_document, q70 · F_cross_document, q71 · F_cross_document, q72 · F_cross_document (+17 more)

### Community 14 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 17 - "generate_policy_qa.py"
Cohesion: 0.17
Nodes (16): difflib, _call_llm_with_retry(), generate_qa_pairs(), _is_duplicate(), main(), parse_args(), _parse_qa(), Any (+8 more)

### Community 18 - "Team Common Rules — RAG Evaluation Project"
Cohesion: 0.11
Nodes (17): 2.1 Why a Multi-Config Mixture, 2.2 Complete Config Reference, 2.3 Recommended Config Combinations by Phase, 2.4 Dataset Size Reference, 2.5 Local Download Instructions, 3.1 Custom Metric Implementations, Rule 1 — Shared Configuration (Non-Negotiable), Rule 2 — Dataset Selection and Sampling (+9 more)

### Community 19 - "Strategy Differences Log"
Cohesion: 0.14
Nodes (13): 1) Embedding model/provider mismatch, 2) Python runtime baseline is stricter than plan, 3) MVP sample size differs from Phase 1 recommendation, 4) Dataset download flow is MVP-centric, not team-shared flow yet, 5) NER stack is partial vs implementation guide, 6) Graph storage layer is still NetworkX-only, 7) Evaluation/experiment tracking not implemented yet, 8) Additional dependencies for local embedding workaround (+5 more)

### Community 20 - "graph_rag/AGENTS.md"
Cohesion: 0.18
Nodes (9): Code Map, Corpus, Current Status, End-to-End Evaluation Workflow, Env Vars, Output Files (fresh run), Rate Limit Budget, Scope + Ground Truth (+1 more)

### Community 21 - "Tier H · paraphrased"
Cohesion: 0.10
Nodes (21): q101 · H_paraphrased, q102 · H_paraphrased, q103 · H_paraphrased, q104 · H_paraphrased, q105 · H_paraphrased, q106 · H_paraphrased, q107 · H_paraphrased, q108 · H_paraphrased (+13 more)

### Community 23 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 24 - "final_plan.md — DeepSeek migration + RAG eval fix"
Cohesion: 0.09
Nodes (21): 0. Goal, 10. Next action, 11. Status, 1. Hard rules, 2. Findings (from reading code), 3. Decision: git worktree (no nuke, no archive), 4. Target layout (inside the worktree), 5. Decisions (D1-D5: user took recommendations, 2026-10-12) (+13 more)

### Community 25 - "Comparison of the Retrieval Strategies"
Cohesion: 0.15
Nodes (12): 10. Limits, 1. Strategies, 2. The eval set, 3. Answer quality, 4. Statistical test, 5. Retrieval only, 6. Cost, 7. Does PageIndex win because of context size? (+4 more)

### Community 26 - "pipeline.py"
Cohesion: 0.16
Nodes (10): chromadb, Graph RAG MVP package exports., LLM client wrapper for answer generation in Graph RAG. Uses the NVIDIA NIM…, End-to-end Graph RAG pipeline orchestration for insurance policy documents., ChromaDB + Jina embeddings vector indexing and retrieval for MVP., openai, os, pickle (+2 more)

### Community 27 - "policy_loader.py"
Cohesion: 0.24
Nodes (10): fitz, _clean_pdf_text(), _derive_policy_name(), load_policy_documents(), Path, PDF loading and chunking for insurance policy documents (Phase 4)., Normalize raw PDF text: remove headers/footers, collapse whitespace., Derive a human-readable policy name from a PDF filename. Strips trailing hash… (+2 more)

### Community 29 - "graph_rag/retrieve.py"
Cohesion: 0.13
Nodes (25): _call(), _cli(), embed(), _path(), ndarray, OpenAI, Embeddings through an OpenAI-compatible endpoint, cached on disk per text.…, Return L2-normalised float32 vectors, shape (len(texts), EMBED_DIM). (+17 more)

### Community 31 - "judge.py"
Cohesion: 0.16
Nodes (16): concurrent_futures, applies(), is_null(), _parse(), The one judge for all three RAG strategies. Input row: {"question", "answer",…, A judge failure. A not-applicable cell is not a failure., Score every row for every metric in parallel. Order is kept., Score one metric. The reply must hold a numeric "score" key. On a bad reply,… (+8 more)

### Community 32 - "llm_patch.py"
Cohesion: 0.16
Nodes (11): asyncio, Build the Graph RAG index (chunks, NER graph, embeddings). python -m…, pageindex, Build the PageIndex store for all PDFs. Resumable: finished PDFs are skipped.…, Index ONE PDF with DeepSeek flash and report token use. Usage: python -m…, _acompletion(), _completion(), install() (+3 more)

### Community 33 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 34 - "EntityMention"
Cohesion: 0.33
Nodes (5): LLMNERExtractor, normalize(), Entity extraction with the shared LLM client (replaces spaCy + GLiNER). Same…, EntityMention, Represent one entity mention extracted from text.

### Community 35 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 37 - "pageindex_rag/retrieve.py"
Cohesion: 0.39
Nodes (8): _json(), _load(), PageIndex retrieval: LLM picks documents from the catalog, then picks tree…, Nodes from each routed document, interleaved by node rank, cut to top_k., retrieve(), route(), search(), _tree()

### Community 38 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 40 - "textclean.py"
Cohesion: 0.40
Nodes (5): Page header and footer removal, shared by every strategy that reads PDFs., Lines that appear on at least SHARE of a PDF's pages (headers, footers)., repeated_lines(), strip_pages(), build_chunks()

### Community 41 - "graphify.js"
Cohesion: 0.47
Nodes (5): hasGraph(), location, IMPORTANT: keep the reminder string free of backticks and $(...) constructs., server(), setup()

### Community 42 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 43 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 45 - "GraphIndex"
Cohesion: 0.18
Nodes (8): GraphIndex, Any, Retrieve chunk candidates via entity-seeded graph traversal., Build and query a chunk-entity graph for Graph RAG retrieval., Initialize graph index with a configured NER extractor. Args: ner_extractor:…, Create or reuse a Document node keyed by source_id., Populate graph nodes and edges from chunk records., Create or reuse an entity node keyed by normalized entity text.

### Community 46 - "llm.py"
Cohesion: 0.15
Nodes (18): answer(), format_context(), Shared answer generation. Every strategy uses the same prompt and format. The…, chat(), chat_full(), _check_model(), client(), _key() (+10 more)

### Community 47 - "VectorIndex"
Cohesion: 0.27
Nodes (7): ChunkRecord, Represent a text chunk and its source metadata., Retrieve top-k chunks by semantic similarity search., Manage chunk embedding storage and semantic retrieval., Create embeddings using Jina Embeddings API., Embed and upsert chunk records into ChromaDB., VectorIndex

### Community 48 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 49 - "Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n"
Cohesion: 0.12
Nodes (16): 10. PageIndex RAG / q95 (G_unanswerable), 11. Graph RAG / q74 (F_cross_document), 12. Graph RAG / q104 (H_paraphrased), 13. Graph RAG / q103 (H_paraphrased), 14. Graph RAG / q111 (H_paraphrased), 15. Graph RAG / q47 (E_multi_hop), 1. Vector RAG / q114 (H_paraphrased), 2. Vector RAG / q103 (H_paraphrased) (+8 more)

### Community 50 - "retrieval_check.py"
Cohesion: 0.16
Nodes (18): collections, _agg(), _anchors(), budget_rank(), evaluate(), first_hit_rank(), norm(), Phase 3a: retrieval-only metrics against the frozen eval set. No LLM call. A… (+10 more)

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

### Community 64 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

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
Nodes (16): 10. The eval set, 11. Next steps, 1. Goal, 2. Status, 3.1 Null scores in detail, 3. Judge, 4. Findings per strategy, 5. Decision you made (+8 more)

### Community 75 - "Results summary (n=80 questions, judge = deepseek-flash, temperature 0)"
Cohesion: 0.40
Nodes (4): Answer correctness by tier (mean), Mean score with 95% bootstrap CI, Paired differences vs Vector RAG (same questions), Results summary (n=80 questions, judge = deepseek-flash, temperature 0)

### Community 76 - "_load_spacy_model"
Cohesion: 0.50
Nodes (3): _load_spacy_model(), Lazily load spaCy model (cached)., Initialise both NER backends. Args: spacy_model: spaCy model name for generic…

### Community 77 - "policy_ner.py"
Cohesion: 0.33
Nodes (5): functools, Entity extraction utilities for Graph RAG MVP., Insurance-domain NER extractor combining spaCy (general) + GLiNER (domain-…, re, spacy

### Community 78 - "run_policy_evaluation.py"
Cohesion: 0.19
Nodes (15): datetime, _infer_question_type(), _load_checkpoint(), main(), parse_args(), Any, Namespace, Path (+7 more)

### Community 79 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 80 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 81 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

## Knowledge Gaps
- **468 isolated node(s):** `$schema`, `location`, `id`, `name`, `description` (+463 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GraphRAGPipeline` connect `GraphRAGPipeline` to `PolicyNERExtractor`, `main.py`, `GraphIndex`, `run_policy_evaluation.py`, `VectorIndex`, `pipeline.py`, `.__init__`, `NERExtractor`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `docs` connect `docs` to `pi-7414c3afd2c24f5cb46ab7fff5c3c910`, `pi-81aa6514f3cc41a69ccf156aeb9706ad`, `pi-81b4c4966c3247c1940d49e5d8e5e169`, `pi-8394957513f7468aaf4a5b525ce40de3`, `pi-9f1c1cff5eaa49d3aba25cc09783b477`, `pi-b9867eaab5a840e9b457f9920996c390`, `pi-d83401c000ac4faca3a0dc71982559b5`, `pi-f567e57ef3b046ef88d14e64b9fbc8de`, `pi-0a89c4406925404d96ba6bf858050dc9`, `pi-1cc88166fb454c2192f6157b6065f58a`, `pi-1e5cffbc4ade4224b1a0c3f449edddaf`, `pi-24ad0cb16a524340af8a4cfdc8a217dd`, `pi-42f758c2276f46039437ccf5df58f9ce`, `pi-45cec3af828a4b5583aa637da53a3eed`, `pi-5dea76e20a44438084501a07d3417b80`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Why does `VectorIndex` connect `VectorIndex` to `GraphRAGPipeline`, `BGEEmbedder`, `pipeline.py`, `.__init__`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `GraphRAGPipeline` (e.g. with `main()` and `main()`) actually correct?**
  _`GraphRAGPipeline` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `location`, `id` to the rest of the system?**
  _468 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Graph RAG — Project Implementation Plan` be split into smaller, more focused modules?**
  _Cohesion score 0.05555555555555555 - nodes in this community are weakly interconnected._
- **Should `Graph RAG Monorepo` be split into smaller, more focused modules?**
  _Cohesion score 0.06666666666666667 - nodes in this community are weakly interconnected._