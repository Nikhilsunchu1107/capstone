import time

from openai import OpenAI

# ── Local Ollama client — no API key needed, runs on your own hardware ────────
ollama = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # unused by Ollama, the SDK just requires a non-empty value
)


def ollama_call(
    prompt: str, model: str, max_tokens: int = 512, retries: int = 3, num_ctx: int | None = None
) -> str:
    """Local Ollama call with retry logic.
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


if __name__ == "__main__":
    response = ollama_call(prompt="What is the capital of India?", model="granite4.2:8b")
    print(response)
