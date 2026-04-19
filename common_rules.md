# Team Common Rules — RAG Evaluation Project

**Mandatory rules for ALL team members across ALL RAG strategies**
Version 2.0 | April 2026

---

## Table of Contents

1. [Why These Rules Exist](#why-these-rules-exist)
2. [Rule 1 — Shared Configuration](#rule-1--shared-configuration-non-negotiable)
3. [Rule 2 — Dataset Selection and Sampling](#rule-2--dataset-selection-and-sampling)
4. [Rule 3 — Evaluation Metrics](#rule-3--evaluation-metrics)
5. [Rule 4 — Results Reporting Format](#rule-4--results-reporting-format)
6. [Rule 5 — Experiment Tracking](#rule-5--experiment-tracking)
7. [Rule 6 — Code Standards](#rule-6--code-standards)
8. [Rule 7 — Team Coordination](#rule-7--team-coordination)
9. [Rule 8 — Defining Optimality](#rule-8--defining-optimality)

---

## Why These Rules Exist

This project compares six different RAG strategies to find the most optimal one. For the comparison to be scientifically valid, every team member must use **identical settings** for everything that is not their strategy. Deviating from these rules — even slightly — will make your results incomparable and **invalidate the entire team comparison**.

> **CRITICAL:** If you change ANY of the shared configuration settings below, you must inform the entire team immediately and all experiments must be re-run from scratch.

---

## Rule 1 — Shared Configuration (Non-Negotiable)

Every team member must use these exact settings. Copy them into your config file.

```python
# ============================================
# SHARED CONFIG — DO NOT CHANGE WITHOUT TEAM AGREEMENT
# ============================================

LLM_MODEL        = 'llama-3.1-8b-instant'   # Groq API model name
LLM_PROVIDER     = 'groq'                    # or 'ollama' on GPU server
TEMPERATURE      = 0.0                       # MUST be 0.0 for reproducibility
MAX_TOKENS       = 1024                      # LLM output limit

EMBED_MODEL      = 'voyage-3-lite'           # Voyage AI embedding model
EMBED_PROVIDER   = 'voyageai'
EMBED_DIMENSION  = 512                       # voyage-3-lite output dimension

CHUNK_SIZE       = 512                       # tokens per chunk
CHUNK_OVERLAP    = 64                        # token overlap between chunks
TOP_K            = 5                         # chunks retrieved per query

RANDOM_SEED      = 42                        # for any random sampling
EVAL_SAMPLE_SIZE = 500                       # questions per evaluation run
```

---

## Rule 2 — Dataset Selection and Sampling

### 2.1 Why a Multi-Config Mixture

RAGBench is a collection of 12 distinct dataset configs, each with different question types and domains. Using a single config would create a structural bias favouring whichever strategy aligns with that config's question style. We use a carefully chosen **mixture** that stress-tests all strategies equally — so no single strategy has a home-field advantage.

### 2.2 Complete Config Reference

The table below lists all 12 RAGBench configs with their properties and whether your team should use them.

| Config | Domain | Question Type | Use? | Reason |
|---|---|---|---|---|
| `hotpotqa` | General knowledge | Multi-hop reasoning | **Primary** | Tests graph traversal and relational reasoning — core strength of Graph RAG |
| `msmarco` | General web search | Simple factual retrieval | **Primary** | Tests basic retrieval — good baseline for Naive RAG comparison |
| `hagrid` | General / Wikipedia | Mixed types + hallucination | **Primary** | Balanced question types — good for variance testing across all strategies |
| `delucionqa` | General | Hallucination / conflict | **Optional** | Tests faithfulness — critical for Insurance domain later |
| `cuad` | Legal contracts | Domain-specific extraction | **Optional** | Closest to Insurance policy documents — good secondary validation |
| `emanual` | Technical manuals | Procedural / factual | **Optional** | Similar structure to Insurance FAQs and procedures |
| `expertqa` | Expert knowledge | Complex, open-ended | No | Too complex and open-ended for fair cross-strategy comparison in MVP |
| `finqa` | Financial tables | Numerical + table reasoning | No | Requires table parsing — creates unfair advantage for some strategies |
| `tatqa` | Financial tables | Numerical + text hybrid | No | Same issue as `finqa` — numerical bias skews results |
| `pubmedqa` | Medical research | Yes / No / Maybe | No | Too narrow question format — not representative of Insurance use case |
| `covidqa` | Medical / COVID | Domain-specific factual | No | Domain too specific and unrelated to the project |
| `techqa` | Technical support | Domain-specific Q&A | No | Domain not relevant to project goals |

### 2.3 Recommended Config Combinations by Phase

| Phase | Configs to Use | Total Questions | Purpose |
|---|---|---|---|
| Phase 1 — MVP testing | `hotpotqa` only | 100 (sample) | Quick sanity check — does your pipeline produce answers? |
| Phase 2 — First evaluation | `hotpotqa` + `msmarco` + `hagrid` | 500 (mixed) | Primary benchmark — balanced cross-strategy comparison |
| Phase 3 — Full evaluation | `hotpotqa` + `msmarco` + `hagrid` | 500 (full) | Final benchmark scores for the research paper |
| Phase 3 — Secondary validation | `delucionqa` + `cuad` + `emanual` | 200 (mixed) | Robustness check and domain proximity validation |
| Phase 4 — Insurance domain | Synthetic QA from Insurance documents | 200 (generated) | Domain-specific evaluation after Insurance implementation |

### 2.4 Dataset Size Reference

| Config | Train size | Validation size | Test size | Split to use |
|---|---|---|---|---|
| `hotpotqa` | ~7,400 | ~1,300 | ~1,300 | `test` |
| `msmarco` | ~10,000 | ~1,000 | ~1,000 | `test` |
| `hagrid` | ~4,600 | ~700 | ~700 | `test` |
| `delucionqa` | ~2,800 | ~500 | ~500 | `test` |
| `cuad` | ~8,600 | ~1,400 | ~1,400 | `test` |
| `emanual` | ~2,200 | ~300 | ~300 | `test` |

### 2.5 Local Download Instructions

> **Do NOT fetch datasets from HuggingFace on every run.** Download once, save locally, and share the saved files with all teammates via Git or shared drive.

**Why local download matters:**
- Fetching from HuggingFace on every run is slow and network-dependent
- HuggingFace API can be unavailable — this will break your evaluation runs at the worst time
- The college GPU server may have restricted internet access
- HuggingFace may update a dataset — a local copy ensures everyone uses identical data
- Re-downloading wastes your Voyage AI and Groq API quota during development

**Step 1 — Run this download script ONCE (one team member only):**

```python
from datasets import load_dataset
import os

SAVE_DIR = './data/ragbench'
os.makedirs(SAVE_DIR, exist_ok=True)

# Primary configs — required for all teammates
primary_configs  = ['hotpotqa', 'msmarco', 'hagrid']

# Optional configs — for secondary validation in Phase 3
optional_configs = ['delucionqa', 'cuad', 'emanual']

all_configs = primary_configs + optional_configs

for config in all_configs:
    print(f'Downloading {config}...')
    dataset = load_dataset('rungalileo/ragbench', config)
    dataset.save_to_disk(f'{SAVE_DIR}/{config}')
    print(f'Saved {config} to {SAVE_DIR}/{config}')

print('All configs downloaded successfully.')
```

**Step 2 — Share the downloaded folder with all teammates:**
- Commit the `./data/ragbench/` folder to your team Git repository, **OR**
- Upload to a shared Google Drive / OneDrive folder
- Every teammate must use the **exact same saved files** — not independently downloaded copies

**Step 3 — Load from disk in all future code (use this instead of `load_dataset()`):**

```python
from datasets import load_from_disk, concatenate_datasets
import random, json

random.seed(42)
SAVE_DIR = './data/ragbench'

# Load primary configs from local disk
hotpotqa = load_from_disk(f'{SAVE_DIR}/hotpotqa')['test']
msmarco  = load_from_disk(f'{SAVE_DIR}/msmarco')['test']
hagrid   = load_from_disk(f'{SAVE_DIR}/hagrid')['test']

# Sample equally from each config (~167 per config = 500 total)
n_per_config = 167

combined = concatenate_datasets([
    hotpotqa.select(random.sample(range(len(hotpotqa)), min(n_per_config, len(hotpotqa)))),
    msmarco.select(random.sample(range(len(msmarco)),   min(n_per_config, len(msmarco)))),
    hagrid.select(random.sample(range(len(hagrid)),     min(n_per_config, len(hagrid)))),
])

# Save and share the exact sample indices with all teammates
sample_indices = {
    'hotpotqa': list(range(n_per_config)),
    'msmarco':  list(range(n_per_config)),
    'hagrid':   list(range(n_per_config)),
}
with open('shared_eval_indices.json', 'w') as f:
    json.dump(sample_indices, f)

print(f'Evaluation set ready: {len(combined)} questions')
```

**Step 4 — For optional secondary validation configs (Phase 3 only):**

```python
# Load optional configs from local disk
delucionqa = load_from_disk(f'{SAVE_DIR}/delucionqa')['test']
cuad       = load_from_disk(f'{SAVE_DIR}/cuad')['test']
emanual    = load_from_disk(f'{SAVE_DIR}/emanual')['test']

# Sample ~67 from each = 200 total for secondary validation
n_secondary = 67

secondary = concatenate_datasets([
    delucionqa.select(random.sample(range(len(delucionqa)), min(n_secondary, len(delucionqa)))),
    cuad.select(random.sample(range(len(cuad)),             min(n_secondary, len(cuad)))),
    emanual.select(random.sample(range(len(emanual)),       min(n_secondary, len(emanual)))),
])
```

> **Action item:** Share `shared_eval_indices.json` with all teammates via Git before Phase 2 begins. Every teammate must evaluate on the **exact same questions**.

---

## Rule 3 — Evaluation Metrics

All team members must report **ALL** of the following metrics. Do not selectively report only the metrics where your strategy performs well.

| Metric | Tool | How to compute | Report as |
|---|---|---|---|
| Context Precision | RAGAS | `ragas.metrics.context_precision` | 0.0 to 1.0 (higher = better) |
| Context Recall | RAGAS | `ragas.metrics.context_recall` | 0.0 to 1.0 (higher = better) |
| Faithfulness | RAGAS | `ragas.metrics.faithfulness` | 0.0 to 1.0 (higher = better) |
| Answer Relevancy | RAGAS | `ragas.metrics.answer_relevancy` | 0.0 to 1.0 (higher = better) |
| Exact Match (EM) | Custom | `exact_match(prediction, ground_truth)` | 0.0 to 1.0 (higher = better) |
| F1 Score | Custom | `token_f1(prediction, ground_truth)` | 0.0 to 1.0 (higher = better) |
| Latency (ms) | Custom | `time.time()` around full pipeline | milliseconds (lower = better) |
| Variance (σ²) | Custom | `np.var(scores_by_question_type)` | lower = more consistent |

### 3.1 Custom Metric Implementations

```python
import numpy as np
from collections import Counter
import time

def exact_match(prediction, ground_truth):
    """Exact string match between prediction and ground truth."""
    return float(prediction.strip().lower() == ground_truth.strip().lower())

def token_f1(prediction, ground_truth):
    """Token-level F1 score — gives partial credit for overlapping tokens."""
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()
    common      = Counter(pred_tokens) & Counter(gt_tokens)
    num_same    = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def measure_latency(query_fn, query, *args):
    """Measure end-to-end pipeline latency in milliseconds."""
    start      = time.time()
    result     = query_fn(query, *args)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms

def compute_variance(scores_by_type: dict) -> float:
    """Compute variance across question types — the key optimality metric."""
    return float(np.var(list(scores_by_type.values())))
```

---

## Rule 4 — Results Reporting Format

All team members must submit results in the same JSON format so the team can aggregate and compare automatically.

```json
{
  "strategy_name": "graph_rag",
  "date_run": "2026-07-15",
  "config": {
    "llm_model": "llama-3.1-8b-instant",
    "embed_model": "voyage-3-lite",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "top_k": 5,
    "temperature": 0.0
  },
  "dataset": {
    "primary_configs": ["hotpotqa", "msmarco", "hagrid"],
    "secondary_configs": ["delucionqa", "cuad", "emanual"],
    "sample_size_primary": 500,
    "sample_size_secondary": 200,
    "indices_file": "shared_eval_indices.json"
  },
  "metrics": {
    "context_precision": { "mean": 0.0, "std": 0.0 },
    "context_recall":    { "mean": 0.0, "std": 0.0 },
    "faithfulness":      { "mean": 0.0, "std": 0.0 },
    "answer_relevancy":  { "mean": 0.0, "std": 0.0 },
    "exact_match":       { "mean": 0.0, "std": 0.0 },
    "f1_score":          { "mean": 0.0, "std": 0.0 },
    "latency_ms":        { "mean": 0.0, "std": 0.0 }
  },
  "by_question_type": {
    "single_hop":  { "f1": 0.0, "em": 0.0 },
    "multi_hop":   { "f1": 0.0, "em": 0.0 },
    "negative":    { "f1": 0.0, "em": 0.0 },
    "conflicting": { "f1": 0.0, "em": 0.0 }
  },
  "by_config": {
    "hotpotqa": { "f1": 0.0, "em": 0.0, "faithfulness": 0.0 },
    "msmarco":  { "f1": 0.0, "em": 0.0, "faithfulness": 0.0 },
    "hagrid":   { "f1": 0.0, "em": 0.0, "faithfulness": 0.0 }
  },
  "variance_across_types":   0.0,
  "variance_across_configs": 0.0
}
```

---

## Rule 5 — Experiment Tracking

- All team members must use **MLflow** for experiment tracking
- Log every evaluation run — do not delete runs even if results are poor
- Use the team shared MLflow tracking URI, or run locally and share your export
- Every run must log: all config parameters, all metrics, dataset configs used, sample indices file

```python
import mlflow

mlflow.set_experiment('rag_strategy_comparison')

with mlflow.start_run(run_name='graph_rag_phase2_run1'):

    # Log shared config
    mlflow.log_params({
        'strategy':        'graph_rag',
        'llm_model':       LLM_MODEL,
        'embed_model':     EMBED_MODEL,
        'chunk_size':      CHUNK_SIZE,
        'top_k':           TOP_K,
        'temperature':     TEMPERATURE,
        'dataset_configs': 'hotpotqa,msmarco,hagrid',
        'indices_file':    'shared_eval_indices.json'
    })

    # Log all metrics
    mlflow.log_metrics({
        'faithfulness':      scores['faithfulness'],
        'context_precision': scores['context_precision'],
        'context_recall':    scores['context_recall'],
        'answer_relevancy':  scores['answer_relevancy'],
        'f1_score':          f1_mean,
        'exact_match':       em_mean,
        'latency_ms':        latency_mean,
        'variance':          variance_score
    })
```

---

## Rule 6 — Code Standards

- Use **Python 3.10** or higher
- Store all API keys in a `.env` file — **NEVER hardcode keys** in source code
- Use `python-dotenv` to load environment variables
- Add `.env` to your `.gitignore` — never commit API keys to Git
- Comment every function with a one-line docstring
- Use the shared config variables — never use magic numbers in code
- Cache all generated embeddings to disk — never re-embed the same text twice
- Always use `load_from_disk()` for datasets — never call `load_dataset()` from HuggingFace during evaluation runs

```bash
# .env file structure — create this file locally, never commit it
GROQ_API_KEY=your_groq_key_here
VOYAGE_API_KEY=your_voyage_key_here
NEO4J_PASSWORD=your_neo4j_password
```

```python
# Load API keys in Python
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY   = os.getenv('GROQ_API_KEY')
VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
```

---

## Rule 7 — Team Coordination

| What to share | When | How |
|---|---|---|
| Downloaded RAGBench data folder (`./data/ragbench/`) | Before Phase 1 begins | Git repo or shared drive — ONE person downloads, shares to all |
| Shared eval indices (`shared_eval_indices.json`) | Before Phase 2 begins | Git repo — generate once, everyone uses same file |
| Intermediate RAGAS scores (after Phase 2) | End of Week 11 | Slack / team meeting + JSON results file |
| Final results JSON | End of Phase 3 (Week 17) | Git repo |
| Any config change proposal | Immediately when identified | Team meeting — unanimous agreement required |
| Blockers or bugs affecting shared tools | Same day | Team group chat |

---

## Rule 8 — Defining Optimality

When the team compares final results, the optimal strategy is determined by this **weighted criterion:**

| Criterion | Weight | Why |
|---|---|---|
| Lowest variance across question types | 40% | Consistency matters more than peak performance for a general-purpose system |
| Highest average F1 score | 25% | Core answer quality metric |
| Highest faithfulness (RAGAS) | 20% | Hallucination resistance is critical for the Insurance domain |
| Lowest latency | 15% | Practical usability in a real Insurance system |

If no single strategy wins across all criteria, the team will evaluate **combinations** (e.g., Naive RAG for simple queries + Graph RAG for multi-hop queries) as the final recommendation.

---

*End of Document — Team Common Rules v2.0 — Applies to ALL members*
