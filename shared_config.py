# ============================================
# SHARED CONFIG — DO NOT CHANGE WITHOUT TEAM AGREEMENT
# Copy this file to your strategy directory as 'config.py'
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
