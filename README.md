# Graph RAG Monorepo

A reproducible research monorepo for Graph RAG strategies. This repository serves as the single source of truth for all team strategies, with shared dependencies, configuration, and evaluation data.

## Quick Start

```bash
# Clone and setup (5 commands)
git clone <repo-url>
cd capstone
git lfs install          # Enable Git LFS (one time)
mise install           # Install Python 3.12 + uv
uv sync               # Create .venv and install dependencies
.venv/bin/python -m spacy download en_core_web_sm

# Run a query
cd graph_rag
../.venv/bin/python main.py --question "Your question here"
```

---

## Project Structure

```
capstone/                    # Project root
├── .mise.toml               # mise tool definitions (Python 3.12, uv)
├── pyproject.toml           # Python dependencies (shared across all strategies)
├── uv.lock                 # Locked dependency versions
├── shared_config.py        # Team-wide configuration (copy to your strategy as config.py)
├── common_rules.md         # Team alignment contract
├── .env.example           # Template for API keys
├── .venv/                 # Virtual environment (auto-created by uv sync)
├── scripts/               # Shared scripts
│   └── generate_eval_samples.py
├── data/                  # Shared evaluation data (Git LFS)
│   └── ragbench/
├── graph_rag/            # Strategy 1: Graph RAG implementation
│   ├── config.py        # Strategy-specific config (derived from shared_config.py)
│   ├── main.py        # CLI entry point
│   ├── src/          # Source code
│   └── outputs/      # Pipeline outputs (chromadb, logs)
└── <new-strategy>/   # Future strategies follow same pattern
```

---

## Environment Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Git | Any recent version | |
| mise | See [mise docs](https://mise.run/) | For Python + uv management |

### Step 1: Install mise

Choose your OS:

**macOS / Linux:**
```bash
# Using curl
curl https://mise.run | sh

# Or using Homebrew
brew install mise
```

**Windows:**
```powershell
# Using winget
winget install mise-cli.Mise

# Or using Chocolatey
choco install mise
```

### Step 2: Clone Repository

```bash
git clone <your-repo-url>
cd capstone
```

### Step 3: Install Tools & Dependencies

The `.mise.toml` file defines Python 3.12 and uv. Just run:

```bash
mise install      # Installs Python 3.12 + uv (defined in .mise.toml)
uv sync          # Creates .venv and installs all dependencies
```

### Step 4: Download spaCy Model

```bash
.venv/bin/python -m spacy download en_core_web_sm
```

---

## Git LFS Setup

This repository uses Git LFS for large evaluation data files in `data/`.

### Install Git LFS

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
# Use your distro's package manager (apt, dnf, pacman, etc.)
# Example for Debian/Ubuntu:
sudo apt install git-lfs
```

**Windows:** Already included in Git for Windows — no install needed.

### Enable LFS

After cloning, run once to configure LFS:

```bash
git lfs install
```

Then download LFS files (or just run `git pull`, which also downloads LFS):

```bash
git pull
# or explicitly:
git lfs pull
```

---

## API Keys Setup

Required and optional API keys:

| Variable | Required | Purpose |
|----------|----------|---------|
| `GROQ_API_KEY` | Yes | LLM generation (Groq) |
| `JINA_API_KEY` | No | Embeddings (if `USE_LOCAL_EMBEDDINGS=0`) |
| `VOYAGE_API_KEY` | No | Alternative embeddings |
| `USE_LOCAL_EMBEDDINGS` | No | Set to `1` for local embeddings |

### Setup Steps

1. Copy the example env file:
   ```bash
   cp .env.example graph_rag/.env   # For graph_rag strategy
   ```

2. Add your API keys:
   ```bash
   # Edit the .env file and replace your_key_here with actual keys
   vim graph_rag/.env
   ```

3. **Never commit `.env` files** — they're in `.gitignore`.

---

## Creating a New Strategy

To create a new strategy (e.g., `my_strategy/`):

### Step 1: Create Directory Structure

```bash
mkdir -p my_strategy/src my_strategy/outputs
```

### Step 2: Copy Config Template

```bash
cp shared_config.py my_strategy/config.py
# Edit config.py with strategy-specific values
```

### Step 3: Create main.py

Create `my_strategy/main.py`:

```python
"""CLI entrypoint for my_strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRAPH_STRATEGY_DIR = Path(__file__).resolve().parent
for p in [str(PROJECT_ROOT), str(GRAPH_STRATEGY_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from my_strategy.config import RANDOM_SEED
from src.pipeline import MyStrategyPipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", type=str, default="Your default question")
    parser.add_argument("--dataset-path", type=Path, default=Path("../data/ragbench_50"))
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = MyStrategyPipeline()
    result = pipeline.run(args.question, dataset_path=args.dataset_path)
    print(result)


if __name__ == "__main__":
    main()
```

### Step 4: Create src/ Modules

Follow the pattern in `graph_rag/src/`:
- `__init__.py` — exports
- `pipeline.py` — main orchestration
- `data_loader.py` — data loading
- `graph.py` — graph index
- `vector_store.py` — vector index
- `llm_client.py` — LLM client
- `types.py` — data types

---

## Migrating Existing Strategy

### From a Git Repository

If you have an existing strategy in a separate git repo:

```bash
# Step 1: Clone your existing strategy
git clone <your-strategy-url> temp_strategy

# Step 2: Move to this monorepo
mv temp_strategy/* ./temp_strategy/
rm -rf temp_strategy

# Step 3: Update imports (see Creating a New Strategy above)
# - Add sys.path.insert pattern to main.py
# - Update any hardcoded paths

# Step 4: Copy API keys
cp .env.example temp_strategy/.env
# Add your keys
```

### From a Local Directory

If you have a strategy in a local directory (not git):

```bash
# Step 1: Copy to monorepo
cp -r /path/to/your/strategy ./temp_strategy/

# Step 2: Update imports
# - Add sys.path.insert pattern to main.py
# - Update any hardcoded paths

# Step 3: Copy data (if any)
cp -r /path/to/your/data ./data/

# Step 4: Copy API keys
cp .env.example temp_strategy/.env
# Add your keys
```

---

## Running the Pipeline

### graph_rag Strategy

```bash
cd graph_rag
../.venv/bin/python main.py --question "Your question" --dataset-path ../data/ragbench_50
```

### Other Strategies

```bash
cd <strategy-name>
../.venv/bin/python main.py --question "Your question"
```

---

## Troubleshooting

### "command not found: mise"

Install mise first. See [mise.run](https://mise.run/).

### "No module named 'graph_rag'"

Ensure you're running from the strategy directory:
```bash
cd graph_rag
../.venv/bin/python main.py
```

Or use absolute paths:
```bash
./.venv/bin/python graph_rag/main.py
```

### Import errors after migration

Did you add the `sys.path.insert` pattern to your `main.py`? See the template in "Creating a New Strategy".

### spaCy model not found

```bash
.venv/bin/python -m spacy download en_core_web_sm
```

### Git LFS files not downloading

```bash
git lfs install
git lfs pull
```

---

## Next Steps

- **Team Rules**: See `common_rules.md`
- **Strategy Details**: See `graph_rag/AGENTS.md`
- **Configuration**: See `shared_config.py`
- **Differences**: See `graph_rag/differences.md`