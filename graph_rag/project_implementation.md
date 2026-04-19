# Graph RAG — Project Implementation Plan

**Knowledge Graph-Based Retrieval Augmented Generation**
Version 1.0 | April 2026 | Final Year Research Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Recommended Tech Stack](#3-recommended-tech-stack)
4. [Implementation Plan](#4-implementation-plan)
5. [Sample Code](#5-sample-code)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Risks and Mitigations](#7-risks-and-mitigations)
8. [What Not to Overengineer](#8-what-not-to-overengineer)

---

## 1. Project Overview

This document is the complete implementation guide for the **Graph RAG** strategy — one of six RAG strategies being evaluated in a team-based final-year research project. The objective is to build, evaluate, and compare multiple Retrieval-Augmented Generation (RAG) architectures to determine the most optimal strategy or combination, which will then be implemented in the **Insurance domain**.

### 1.1 Research Objective

Evaluate Graph RAG against five other RAG strategies (Naive RAG, Hybrid RAG, HyDE, Agentic RAG, Page Indexing) using a common benchmark dataset and standardized metrics, then implement the optimal strategy on Insurance domain data.

### 1.2 Team Structure

| Team Member | Assigned Strategy | Status |
|---|---|---|
| You (this doc) | Graph RAG | Active |
| Teammate 2 | Naive RAG + HyDE | Parallel |
| Teammate 3 | Hybrid RAG | Parallel |
| Teammate 4 | Agentic RAG + Page Indexing | Parallel |

### 1.3 Key Constraints

- **Programming Language:** Python 3.10+
- **LLM:** Llama 3.1 8B via Groq API (laptop) or Ollama (college GPU server)
- **Embeddings:** Voyage AI API — `voyage-3-lite` (50M free tokens/month)
- **Budget:** Minimal — all tools must be free or open-source
- **Timeline:** MVP by June 2026, full implementation by September 2026
- **Team Constraint:** All members use identical LLM, embeddings, dataset, and metrics

---

## 2. System Architecture

The Graph RAG system is composed of **six layers**. Each layer is independently buildable and testable, which is critical for a prototype built by a beginner team.

### 2.1 Architecture Overview

| Layer | Name | Responsibility |
|---|---|---|
| 1 | Data Ingestion | Load, clean, chunk, and embed documents |
| 2 | Graph Construction | Extract entities and relationships, build the knowledge graph |
| 3 | Storage | Persist graph (Neo4j) and embeddings (ChromaDB) |
| 4 | Retrieval | Multi-hop graph traversal + vector search + context assembly |
| 5 | LLM Generation | Build prompt, call LLM, return answer + supporting context |
| 6 | Evaluation | Measure performance using RAGAS + latency + F1 score |

---

### 2.2 Layer 1 — Data Ingestion

Responsible for loading raw documents, preprocessing them into clean chunks, extracting named entities and relationships, and generating vector embeddings.

**Steps:**

1. Load documents using LlamaIndex `SimpleDirectoryReader` (supports PDF, TXT, HTML)
2. Split into chunks of 512 tokens with 64-token overlap using `SentenceSplitter`
3. Clean text — remove special characters, normalize whitespace
4. Run spaCy NER pipeline to extract entities (persons, organizations, locations, concepts)
5. Generate embeddings for each chunk using Voyage AI API (`voyage-3-lite`)
6. Store chunk text + metadata (source, page, chunk_id) for later retrieval

**Important Configuration:**

| Parameter | Value | Note |
|---|---|---|
| Chunk size | 512 tokens | Must be identical across all teammates |
| Chunk overlap | 64 tokens | Must be identical across all teammates |
| Embedding model | `voyage-3-lite` | Must be identical across all teammates |
| NER model | spaCy `en_core_web_sm` + GLiNER | GLiNER for zero-shot entity detection |

---

### 2.3 Layer 2 — Graph Construction

This is the **core differentiator** of Graph RAG. Entities become nodes and semantic relationships between them become edges. The graph captures relational context that flat vector search cannot.

**Graph Schema:**

| Element | Type | Examples |
|---|---|---|
| Node | Document | A source document or PDF file |
| Node | Chunk | A text chunk from a document |
| Node | Entity | Person, organization, location, concept, product |
| Edge | CONTAINS | Document → Chunk |
| Edge | MENTIONS | Chunk → Entity |
| Edge | RELATED_TO | Entity ↔ Entity (co-occurrence) |
| Edge | IS_A | Entity → Category (taxonomy) |
| Edge | PART_OF | Entity → Entity (hierarchy) |

**Construction Strategy:**

1. For each chunk, run NER to extract entity mentions
2. Create a node for each unique entity (deduplicated by name + type)
3. Create `MENTIONS` edges between chunks and their entities
4. Create `RELATED_TO` edges between entities that co-occur in the same chunk
5. Create `CONTAINS` edges from documents to their chunks
6. Assign embedding vectors as node properties for hybrid retrieval

---

### 2.4 Layer 3 — Storage

Graph RAG requires **two storage systems** running in parallel — a graph database for structural/relational queries and a vector store for semantic similarity search.

| Store | Tool | What it holds | Why |
|---|---|---|---|
| Graph DB | Neo4j Community (free) | Nodes, edges, entity properties, relationships | Industry standard, Cypher query language, free tier |
| Vector Store | ChromaDB (local) | Chunk embeddings + metadata | Zero setup, runs locally, Python-native |
| Document Store | JSON files / SQLite | Raw chunk text + source metadata | Lightweight, no extra dependencies |

---

### 2.5 Layer 4 — Retrieval Strategy

This is the **most important layer** for Graph RAG and the key differentiator from other strategies. The retrieval process runs in four stages sequentially.

**Stage 1 — Entity Extraction from Query:**
- Run NER on the user's query to identify mentioned entities
- These become the starting nodes for graph traversal

**Stage 2 — Graph Traversal (Multi-hop):**
- Start from seed entities identified in the query
- Traverse up to 2–3 hops across `RELATED_TO` and `PART_OF` edges
- Collect all chunks connected to traversed entities via `MENTIONS` edges
- Use Cypher queries in Neo4j for efficient traversal

**Stage 3 — Vector Search (Semantic Fallback):**
- Embed the query using Voyage AI
- Search ChromaDB for top-k=5 most similar chunks
- Merge with graph traversal results (deduplicate by chunk_id)

**Stage 4 — Context Assembly:**
- Rank merged results by relevance score (graph proximity + vector similarity)
- Select top-k=5 final chunks to form the context window
- Preserve source metadata (document name, page, chunk_id) for citation

---

### 2.6 Layer 5 — LLM Generation

The assembled context and user query are packaged into a structured prompt and sent to the LLM.

**Prompt Template:**

```
System: You are a helpful assistant. Answer the question using ONLY the provided context.
        If the answer is not in the context, say "I don't know".
        Always cite which document/chunk your answer comes from.

Context: {assembled_chunks_with_metadata}

Question: {user_query}

Answer: [Direct answer]
Supporting context: [Relevant excerpt + source citation]
```

**LLM Configuration:**

| Parameter | Value | Note |
|---|---|---|
| Model | `llama-3.1-8b-instant` | Via Groq API |
| Temperature | `0.0` | Mandatory — for reproducible evaluation |
| Max tokens | `1024` | Consistent across all teammates |

---

### 2.7 Layer 6 — Evaluation

All evaluation must be run with the same configuration as teammates. See `common_rules.md` for full details.

| Metric | Tool | What it measures |
|---|---|---|
| Context Precision | RAGAS | Are the retrieved chunks relevant to the query? |
| Context Recall | RAGAS | Were all necessary chunks retrieved? |
| Faithfulness | RAGAS | Does the answer stay faithful to the context (no hallucination)? |
| Answer Relevancy | RAGAS | Is the answer relevant to the original question? |
| Exact Match (EM) | Custom | Does the answer exactly match the ground truth? |
| F1 Score | Custom | Token-level overlap between predicted and ground truth answer |
| Latency (ms) | Custom | End-to-end response time per query |
| Variance (σ²) | Custom | Performance consistency across question types |

---

## 3. Recommended Tech Stack

| Layer | Tool / Library | Version | Install Command |
|---|---|---|---|
| Document loading | LlamaIndex | latest | `pip install llama-index` |
| NER / NLP | spaCy | 3.7+ | `pip install spacy && python -m spacy download en_core_web_sm` |
| Zero-shot NER | GLiNER | latest | `pip install gliner` |
| Graph (MVP) | NetworkX | 3.x | `pip install networkx pyvis` |
| Graph DB | Neo4j Community | 5.x | Download from neo4j.com (free) |
| Neo4j Python driver | neo4j | 5.x | `pip install neo4j` |
| Vector store | ChromaDB | latest | `pip install chromadb` |
| Embeddings | Voyage AI API | latest | `pip install voyageai` |
| LLM API | Groq | latest | `pip install groq` |
| LLM (local) | Ollama + Llama 3.1 | latest | ollama.com (free download) |
| RAG pipeline | LlamaIndex Graph | latest | `pip install llama-index-graph-stores-neo4j` |
| Evaluation | RAGAS | latest | `pip install ragas` |
| Dataset | HuggingFace datasets | latest | `pip install datasets` |
| Experiment tracking | MLflow | latest | `pip install mlflow` |
| Development | Jupyter + VS Code | latest | `pip install jupyter` |

**Full install command (copy and run once):**

```bash
pip install llama-index llama-index-graph-stores-neo4j spacy gliner chromadb
pip install voyageai groq neo4j ragas datasets mlflow networkx pyvis jupyter pandas
python -m spacy download en_core_web_sm
```

**Why LlamaIndex over LangChain?**
LlamaIndex has a built-in `KnowledgeGraphIndex` specifically designed for Graph RAG, saving weeks of custom wiring. LangChain is more general purpose but requires significantly more manual work for graph-based retrieval.

**Why NetworkX first, then Neo4j?**
Start with NetworkX for your MVP — it is pure Python with zero database setup. Once your graph logic is verified and working, migrate to Neo4j in Phase 2. This is the most practical approach for beginners.

---

## 4. Implementation Plan

The project is divided into four phases. **Complete each phase before starting the next.** Do not skip Phase 1 — the MVP is the foundation everything else builds on.

### Phase 1 — MVP (April 2026 – May 2026)

> **Goal:** Get a working end-to-end Graph RAG pipeline on a small sample of data. No Neo4j yet. Use NetworkX for the graph.

| Milestone | Task | Expected Output | Deadline |
|---|---|---|---|
| M1.1 | Environment setup | All libraries installed, API keys working | Week 1 |
| M1.2 | Data loading | Load RAGBench sample (100 QA pairs) locally | Week 1 |
| M1.3 | Basic NER pipeline | Extract entities from 10 sample chunks | Week 2 |
| M1.4 | NetworkX graph | Build small in-memory graph, visualize with pyvis | Week 2 |
| M1.5 | ChromaDB setup | Store embeddings, run first similarity search | Week 3 |
| M1.6 | LLM connection | Send first prompt to Groq API, get response | Week 3 |
| M1.7 | End-to-end pipeline | Answer 5 questions using Graph RAG (no evaluation yet) | Week 4 |

---

### Phase 2 — Core Implementation (May 2026 – July 2026)

> **Goal:** Replace NetworkX with Neo4j, scale to full RAGBench dataset, run first evaluation.

| Milestone | Task | Expected Output | Deadline |
|---|---|---|---|
| M2.1 | Neo4j setup | Neo4j running locally, connected via Python driver | Week 5 |
| M2.2 | Graph migration | Export NetworkX graph to Neo4j, verify with Cypher | Week 6 |
| M2.3 | Cypher retrieval | Multi-hop traversal queries working correctly | Week 7 |
| M2.4 | Full data pipeline | All RAGBench data ingested and indexed | Week 8 |
| M2.5 | Hybrid retrieval | Graph + vector results merged correctly | Week 9 |
| M2.6 | First evaluation run | RAGAS scores on 100 sample questions | Week 10 |
| M2.7 | Baseline comparison | Share results with team for cross-strategy comparison | Week 11 |

---

### Phase 3 — Optimization (July 2026 – August 2026)

> **Goal:** Improve Graph RAG performance based on Phase 2 evaluation results, run full evaluation.

| Milestone | Task | Expected Output | Deadline |
|---|---|---|---|
| M3.1 | Error analysis | Identify failure patterns from Phase 2 results | Week 12 |
| M3.2 | Graph schema tuning | Refine entity types and relationship types | Week 13 |
| M3.3 | Retrieval tuning | Tune hop depth, top-k values, re-ranking strategy | Week 14 |
| M3.4 | Prompt optimization | Test 3 prompt variants, pick best performing | Week 15 |
| M3.5 | Full evaluation | Run on complete RAGBench primary + secondary datasets | Week 16 |
| M3.6 | Results documentation | Document scores, variances, failure cases | Week 17 |

---

### Phase 4 — Insurance Domain Implementation (August 2026 – September 2026)

> **Goal:** Apply the optimized Graph RAG system to Insurance domain data. This phase only begins after Phase 3 is complete.

| Milestone | Task | Expected Output | Deadline |
|---|---|---|---|
| M4.1 | Insurance data prep | Collect and clean policy, claims, regulatory, FAQ docs | Week 18 |
| M4.2 | Domain schema design | Design Insurance-specific graph schema (Policy, Claim, etc.) | Week 18 |
| M4.3 | Domain ingestion | Ingest all Insurance documents into the pipeline | Week 19 |
| M4.4 | Synthetic QA generation | Generate 200 Insurance-specific QA pairs using LLM | Week 19 |
| M4.5 | Domain evaluation | Run RAGAS on Insurance QA pairs | Week 20 |
| M4.6 | Final results | Complete comparison: benchmark vs domain performance | Week 21 |

---

## 5. Sample Code

### 5.1 Environment Setup

```python
import os
from groq import Groq
import voyageai
import chromadb
from neo4j import GraphDatabase
import spacy
import networkx as nx
from dotenv import load_dotenv

# Load API keys from .env file — never hardcode
load_dotenv()
GROQ_API_KEY   = os.getenv('GROQ_API_KEY')
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')

# Shared config — identical for ALL teammates
LLM_MODEL     = 'llama-3.1-8b-instant'
EMBED_MODEL   = 'voyage-3-lite'
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5
TEMPERATURE   = 0.0
RANDOM_SEED   = 42
```

---

### 5.2 Document Ingestion and NER

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Load documents from local directory
documents = SimpleDirectoryReader('data/documents/').load_data()

# Chunk documents
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.get_nodes_from_documents(documents)

# Extract entities using spaCy
nlp = spacy.load('en_core_web_sm')

def extract_entities(text):
    """Extract named entities from text using spaCy NER."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

for chunk in chunks:
    chunk.metadata['entities'] = extract_entities(chunk.text)
```

---

### 5.3 Graph Construction (NetworkX — MVP Phase Only)

```python
import networkx as nx
from pyvis.network import Network

G = nx.DiGraph()

for chunk in chunks:
    chunk_id = chunk.node_id
    G.add_node(chunk_id, type='chunk', text=chunk.text[:100])

    for entity, label in chunk.metadata['entities']:
        if entity not in G:
            G.add_node(entity, type='entity', label=label)
        G.add_edge(chunk_id, entity, relation='MENTIONS')

        # Add RELATED_TO edges between co-occurring entities
        for other_entity, _ in chunk.metadata['entities']:
            if other_entity != entity:
                G.add_edge(entity, other_entity, relation='RELATED_TO')

# Visualize the graph in browser
net = Network(height='600px', directed=True)
net.from_nx(G)
net.show('graph.html')
```

---

### 5.4 Neo4j Graph Storage (Phase 2 onwards)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    'bolt://localhost:7687',
    auth=('neo4j', os.getenv('NEO4J_PASSWORD'))
)

def store_chunk_and_entities(tx, chunk_id, text, entities):
    """Store a chunk and its entities as nodes in Neo4j."""
    tx.run('MERGE (c:Chunk {id: $id}) SET c.text = $text',
           id=chunk_id, text=text)
    for entity, label in entities:
        tx.run('''
            MERGE (e:Entity {name: $name, type: $type})
            WITH e
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
        ''', name=entity, type=label, chunk_id=chunk_id)

with driver.session() as session:
    for chunk in chunks:
        session.execute_write(
            store_chunk_and_entities,
            chunk.node_id,
            chunk.text,
            chunk.metadata['entities']
        )
```

---

### 5.5 Graph Traversal Retrieval

```python
def graph_retrieve(query, driver, top_k=TOP_K):
    """Retrieve relevant chunks via multi-hop graph traversal."""
    query_entities = [e for e, _ in extract_entities(query)]

    cypher = '''
    MATCH (e:Entity)
    WHERE e.name IN $entities
    MATCH (e)<-[:MENTIONS]-(c:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e2:Entity)<-[:MENTIONS]-(c2:Chunk)
    RETURN DISTINCT c.id AS chunk_id, c.text AS text
    LIMIT $top_k
    '''

    with driver.session() as session:
        results = session.run(cypher, entities=query_entities, top_k=top_k)
        return [{'chunk_id': r['chunk_id'], 'text': r['text']} for r in results]
```

---

### 5.6 Full RAG Pipeline

```python
import voyageai
from groq import Groq

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
groq_client   = Groq(api_key=GROQ_API_KEY)

def graph_rag_query(query, driver, chroma_collection):
    """Run a full Graph RAG query — graph traversal + vector search + LLM generation."""

    # Step 1: Graph traversal
    graph_chunks = graph_retrieve(query, driver, top_k=TOP_K)

    # Step 2: Vector search (semantic fallback)
    query_embedding = voyage_client.embed([query], model=EMBED_MODEL).embeddings[0]
    vector_results  = chroma_collection.query(
        query_embeddings=[query_embedding], n_results=TOP_K
    )
    vector_chunks = [{'text': t} for t in vector_results['documents'][0]]

    # Step 3: Merge and deduplicate
    all_chunks, seen, unique_chunks = graph_chunks + vector_chunks, set(), []
    for c in all_chunks:
        if c['text'] not in seen:
            seen.add(c['text'])
            unique_chunks.append(c)

    context = '\n\n'.join([c['text'] for c in unique_chunks[:TOP_K]])

    # Step 4: LLM generation
    prompt = f"""You are a helpful assistant.
Answer using ONLY the context below. Always cite your sources.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
"""
    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=TEMPERATURE,
        max_tokens=1024
    )
    return response.choices[0].message.content
```

---

### 5.7 RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import numpy as np

def run_evaluation(questions, ground_truths, driver, chroma_col):
    """Run full RAGAS evaluation on a set of questions."""
    results = []
    for q, gt in zip(questions, ground_truths):
        answer       = graph_rag_query(q, driver, chroma_col)
        graph_chunks = graph_retrieve(q, driver)
        results.append({
            'question':     q,
            'answer':       answer,
            'contexts':     [c['text'] for c in graph_chunks],
            'ground_truth': gt
        })

    dataset = Dataset.from_list(results)
    scores  = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    return scores

def compute_variance(scores_by_type):
    """Compute variance across question types — key optimality metric."""
    values = list(scores_by_type.values())
    return float(np.var(values))
```

---

## 6. Evaluation Methodology

### 6.1 Datasets

> See `common_rules.md` Rule 2 for full dataset selection rationale, config details, and local download instructions.

| Dataset | Purpose | Configs used | Size |
|---|---|---|---|
| RAGBench (primary) | Main benchmark | `hotpotqa` + `msmarco` + `hagrid` | 500 QA pairs |
| RAGBench (secondary) | Robustness validation | `delucionqa` + `cuad` + `emanual` | 200 QA pairs |
| Synthetic Twin | Domain-neutral control | LLM-generated from neutral corpus | 200 QA pairs |
| Insurance QA | Phase 4 domain evaluation | Synthetic from Insurance documents | 200 QA pairs |

### 6.2 Evaluation Protocol

1. Fix all variables: same LLM, same embeddings, same temperature, same top-k
2. Run all strategies on **identical** question sets using shared indices file
3. Record scores for every metric per question — not just averages
4. Stratify results by question type (single-hop vs multi-hop vs negative vs conflicting)
5. Report mean, standard deviation, and variance for each metric
6. Report scores broken down by RAGBench config (hotpotqa vs msmarco vs hagrid)
7. Calculate performance gap between question types to measure consistency

### 6.3 Optimality Criterion

> The optimal strategy is **NOT** the one with the highest average score. It is the one with the **lowest variance** across question types.

A strategy scoring 85% on simple and 80% on multi-hop is **better** for a general-purpose Insurance system than one scoring 95% on simple and 50% on multi-hop.

### 6.4 Baselines

| Baseline | Description |
|---|---|
| BM25 (keyword search) | Classical retrieval — no LLM, no embeddings. Establishes the minimum bar. |
| Naive RAG (teammate) | Simple chunk + embed + retrieve. The most common RAG baseline. |
| Random retrieval | Randomly selected chunks + LLM. Tests whether retrieval matters at all. |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Neo4j too complex for beginners | High | High | Start with NetworkX MVP — migrate to Neo4j in Phase 2 only after pipeline is working |
| Groq API rate limits during evaluation | Medium | Medium | Batch evaluation runs, add retry logic with exponential backoff |
| Poor NER quality affecting graph | Medium | High | Manually validate NER output on 20 samples before scaling |
| Voyage AI free tier exhausted | Low | Medium | Cache all embeddings locally after first generation — never re-embed the same text |
| Inconsistent results vs teammates | Medium | High | Strictly follow `common_rules.md` — especially LLM config and dataset sampling |
| Timeline slippage | Medium | Medium | Never skip Phase 1 MVP. A working simple system beats a broken complex one. |

---

## 8. What Not to Overengineer

This is a **research prototype**, not a production system.

**Do NOT build or implement:**
- A frontend or UI — command line and Jupyter notebooks are sufficient
- A perfect NER system — spaCy `en_core_web_sm` is good enough for MVP
- Multiple LLMs — stick to one model for the entire project
- Custom graph algorithms — Neo4j built-in Cypher traversal is sufficient
- Speed optimizations in Phase 1 — correctness first, performance later
- Complex re-ranking systems in Phase 1 — simple merge by relevance score is fine
- Mid-project embedding model changes — all data must use the same embedding space

**Do prioritize:**
- A working end-to-end pipeline (even if slow or simple)
- Consistent evaluation methodology shared with all teammates
- Clean, readable, well-commented code
- Reproducible results (fixed seeds, fixed temperature, logged configs)
- Good experiment tracking from Day 1 (MLflow)

---

*End of Document — Graph RAG Implementation Plan v1.0*
*Confidential — Research Project*
