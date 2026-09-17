# NVIDIA NIM — Available Models (account\-verified, Sept 2026) {#nvidia-nim-available-models-account-verified-sept-2026}

Verified against `https://integrate.api.nvidia.com/v1` with your API key by testing every catalog entry from `client.models.list()` with a live chat completion call. Catalog listing ≠ actual access — many models 404 despite being listed.

## Working — general\-purpose (safe for pi\_client) {#working-general-purpose-safe-for-pi_client}

| Model ID | Notes |
| --- | --- |
| `poolside/laguna-xs-2.1` | Smallest by naming, clean response — first choice for lightweight |
| `meta/llama-3.2-11b-vision-instruct` | 11B, reliable, good fallback quality |
| `mistralai/mistral-nemotron` | Clean response, mid\-size |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | MoE, \~3B active — known\-good, already proven end\-to\-end |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | MoE reasoning variant |
| `openai/gpt-oss-20b` | 20B open\-weight |
| `nvidia/nemotron-3-super-120b-a12b` | Large MoE |
| `nvidia/nemotron-3-ultra-550b-a55b` | Largest MoE tested |

## Working — but not general chat (skip for summarization) {#working-but-not-general-chat-skip-for-summarization}

| Model ID | Actual purpose |
| --- | --- |
| `nvidia/riva-translate-4b-instruct-v1.1` | Translation only |
| `nvidia/riva-translate-4b-instruct-v2` | Translation only |
| `nvidia/llama-3.1-nemoguard-8b-content-safety` | Safety classifier |
| `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | Safety classifier |
| `nvidia/nemotron-3.5-content-safety` | Safety classifier |
| `nvidia/nemotron-parse-2.0` | Document/layout parsing (OCR\-style) |
| `nvidia/ising-calibration-1.5-31b` | Specialized/unclear general use |
