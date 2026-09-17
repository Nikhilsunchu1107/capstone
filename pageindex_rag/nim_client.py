import os
import time

from openai import OpenAI

from client import NVIDIA_API_KEY  # also loads repo .env as an import side effect

EVAL_API_KEY = os.getenv("EVAL_API_KEY")

# ── Single NIM client — used for eval + tree search ───────────────────────────
nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=EVAL_API_KEY,
)

# ── Local Ollama client — no API key needed, runs on your own hardware ────────
ollama = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # unused by Ollama, the SDK just requires a non-empty value
)


def ollama_call(
    prompt: str, model: str, max_tokens: int = 512, retries: int = 3, num_ctx: int | None = None
) -> str:
    """Local Ollama call with retry logic, same shape as call_nim/nim_call.
    `num_ctx` overrides the context window for this call (Ollama-specific,
    passed through extra_body) — leave None to use the model's default."""
    extra_body = {"options": {"num_ctx": num_ctx}} if num_ctx else {}
    for attempt in range(retries):
        try:
            response = ollama.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 3 * (attempt + 1)
            print(f"  ⚠ ollama_call attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def nim_call(prompt: str, model: str, max_tokens: int = 512) -> str:
    """
    NIM API call with retry logic, used by tree search and evaluation.
    Sleeps 1s between calls automatically to avoid 429s on free tier.
    """
    for attempt in range(3):
        try:
            time.sleep(1)  # Rate limit buffer — DO NOT REMOVE
            response = nim.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    return ""  # Return empty string on total failure — handled downstream


def call_nim(
    prompt,
    model: str,
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
    temperature=0,
    max_tokens=1024,
    retries=3,
):
    client = OpenAI(base_url=base_url, api_key=api_key)
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            print(f"  ⚠ call_nim attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


if __name__ == "__main__":
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    for model in client.models.list():
        print(model.id)

    response = call_nim(
        prompt="What is the capital of India?", model="nvidia/nemotron-3.5-lightning-30b-a3b"
    )
    print(response)
