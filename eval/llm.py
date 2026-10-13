"""DeepSeek chat client (OpenAI-compatible) with retry, disk cache and token log.

Run `python -m eval.llm` to verify the endpoint and list model IDs.
"""

import hashlib
import json
import random
import time

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from eval import config

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.LLM_API_KEY:
            raise RuntimeError("LLM_API_KEY is not set in .env")
        _client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    return _client


def _check_model(model: str) -> None:
    if not model:
        raise RuntimeError("LLM_MODEL is not set in .env (run `python -m eval.llm`)")
    if any(m in model.lower() for m in config.FORBIDDEN_MODEL_MARKERS):
        raise RuntimeError(f"Model '{model}' is not allowed: flash tier only")


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def chat(prompt: str, **kw) -> str:
    """Return the completion text. See chat_full."""
    return chat_full(prompt, **kw)[0]


def chat_full(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = config.ANSWER_MAX_TOKENS,
    json_mode: bool = False,
    tag: str = "",
    max_retries: int = 6,
) -> tuple[str, str]:
    """Return (completion text, finish_reason). Identical calls are served from the disk cache.

    `tag` is part of the cache key only (not sent); change it to force a fresh call.
    """
    model = model or config.LLM_MODEL
    _check_model(model)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": config.TEMPERATURE,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "thinking": config.THINKING,
        "tag": tag,
    }
    path = config.LLM_CACHE_DIR / f"{_key(payload)}.json"
    if path.exists():
        rec = json.loads(path.read_text())
        return rec["content"], rec.get("finish_reason", "stop")

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=config.TEMPERATURE,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": config.THINKING}},
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            resp = client().chat.completions.create(**kwargs)
            break
        except (RateLimitError, APIConnectionError) as e:
            err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            err = e
        if attempt == max_retries - 1:
            raise err
        time.sleep(min(60, 2**attempt) + random.random())

    content = resp.choices[0].message.content or ""
    usage = resp.usage
    record = {
        "content": content,
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "finish_reason": resp.choices[0].finish_reason,
        "request": payload,
    }
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False))
    tmp.replace(path)
    return content, record["finish_reason"]


def usage_totals() -> dict:
    """Token totals over every cached call (each paid call is cached exactly once)."""
    calls = prompt = completion = 0
    for f in config.LLM_CACHE_DIR.glob("*.json"):
        r = json.loads(f.read_text())
        calls += 1
        prompt += r["prompt_tokens"]
        completion += r["completion_tokens"]
    return {"calls": calls, "prompt_tokens": prompt, "completion_tokens": completion}


if __name__ == "__main__":
    print("Models on this endpoint:")
    for m in client().models.list():
        print("  ", m.id)
    if not config.LLM_MODEL:
        raise SystemExit("\nSet LLM_MODEL in .env to the flash model ID above, then run again.")
    before = usage_totals()
    print(f"\nModel: {config.LLM_MODEL}")
    print("Reply 1:", chat("Reply with the single word: ready", max_tokens=16))
    mid = usage_totals()
    print("Reply 2:", chat("Reply with the single word: ready", max_tokens=16))
    after = usage_totals()
    print(f"New paid calls: first={mid['calls'] - before['calls']}, second={after['calls'] - mid['calls']} (second must be 0)")
    print("Totals:", after)
