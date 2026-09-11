# LLM provider: NVIDIA NIM (OpenAI-compatible endpoint)
# Verify the exact model ID at https://build.nvidia.com/explore/reasoning
LLM_MODEL = "openai/gpt-oss-120b"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_PROVIDER = "nim"

# Separate model for RAGAS evaluation scoring.
# Must be a standard instruction model (not a reasoning model) so RAGAS
# prompts fit within the context window without chain-of-thought overhead.
RAGAS_EVAL_MODEL = "meta/llama-3.1-70b-instruct"

EMBED_MODEL = "jina-embeddings-v4"
# GLiNER (urchade/gliner_medium-v2.1) has a hard 384 subword-token limit.
# At ~1.5 subword tokens per whitespace token, 200 whitespace tokens ≈ 300
# subword tokens — safely below the limit with headroom for longer words.
CHUNK_SIZE = 200
TOP_K = 5
TEMPERATURE = 0.0
MAX_TOKENS = 1024
EMBED_DIMENSION = 512
EMBED_PROVIDER = "voyageai"
CHUNK_OVERLAP = 30
RANDOM_SEED = 42
