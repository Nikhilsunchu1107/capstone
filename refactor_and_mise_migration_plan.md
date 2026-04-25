---
Phase 1: Environment Migration (Mise Setup)
Objective: Establish a new, reproducible mise-managed environment at the project root before changing any file paths.
1.  Install mise:
    *   Ensure mise is installed on your system. If not, run the installer:
                curl https://mise.run | sh
            *   Follow the instructions to add mise to your shell's PATH.
2.  Define Project Runtimes:
    *   Create a .mise.toml file in the project root (/home/darkhat/Projects/capstone/.mise.toml) to specify the exact Python version. This replaces .python-version.
                [tools]
        python = "3.12"
        
3.  Update Root .gitignore:
    *   Create or update the .gitignore file at the project root (/home/darkhat/Projects/capstone/.gitignore).
    *   Add entries to ignore mise artifacts, the new root virtual environment, and the now-obsolete graph_rag virtual environment.
                # Mise
        .mise/
        # Python Virtual Environments
        .venv/
        graph_rag/.venv/
        # Python-generated files
        __pycache__/
        *.py[oc]
        build/
        dist/
        wheels/
        *.egg-info
        # Environment variables
        .env
        .env.local
        
4.  Activate mise and Create New Virtual Environment:
    *   Run mise trust in the root directory to approve the new configuration.
    *   Run mise install to ensure the specified Python version (3.12) is installed.
    *   Create the new, clean virtual environment at the project root using mise's Python:
                python -m venv .venv
            *   Important: Deactivate any active conda environment first. After this, your shell, when in this directory, will automatically use the Python from .venv/bin/python.
5.  Sync Dependencies into New Environment:
    *   Install uv into the new environment.
                .venv/bin/pip install uv
            *   Use uv to sync the project dependencies from graph_rag/pyproject.toml into the new root .venv.
                uv sync --python .venv/bin/python --venv .venv -p graph_rag/pyproject.toml
        
6.  Verify New Environment:
    *   Run a command to confirm the dependencies are installed in the new location.
                .venv/bin/python -m spacy --version
        
---
Phase 2: Repository Refactoring
Objective: Move all shared files and configurations from graph_rag/ to the project root, leaving only strategy-specific code inside graph_rag/.
1.  Move Core Configuration Files:
    *   Move pyproject.toml, uv.lock, .python-version, and .gitattributes from graph_rag/ to the project root.
                mv graph_rag/pyproject.toml .
        mv graph_rag/uv.lock .
        mv graph_rag/.python-version .
        mv graph_rag/.gitattributes .
        
2.  Move Shared Data and Scripts:
    *   Move the data and scripts directories to the project root.
                mv graph_rag/data .
        mv graph_rag/scripts .
        
3.  Update Python Import Paths:
    *   The scripts in scripts/ previously relied on being inside the graph_rag package. They now need to be updated to correctly import modules from graph_rag/src.
    *   Example Change in scripts/download_datasets.py (and others):
        *   The sys.path manipulation at the top of the files will need to be adjusted or made more robust to reflect the new structure.
4.  Clean Up Obsolete Files and Directories:
    *   Delete the now-empty and obsolete virtual environment from its old location.
                rm -rf graph_rag/.venv
        
5.  Verify Project Integrity:
    *   Run one of the scripts to ensure the path changes work correctly and it can still find the graph_rag module.
                python scripts/generate_eval_samples.py --help
---
