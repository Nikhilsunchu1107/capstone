# RAG.sh — Known Issues & Fix Plan

Context: PageIndex-based RAG pipeline for insurance policy Q&A, using local
Ollama (`granite4.2:8b`) for summarization, routing, node selection, and
answer generation. Eval scores (faithfulness / answer_relevancy /
context_precision / context_recall) are near-zero across most of the eval
set (`pageindex_eval_results.json`). Root cause is a chain of context-
truncation bugs plus a harness bug in the judge, not primarily model
capability. Fix in the order below — each fix is cheap and should be
verified before assuming the model itself needs upgrading.

## Bug A — Summary truncation poisons the whole pipeline (`client.py`)

```python
async def _capped(model, prompt):
    if len(prompt) > 6000:
        prompt = prompt[:6000] + "\n...[truncated]"
    return await _orig_acomp(model, prompt)

_u.llm_acompletion = _capped
```

This patches `llm_acompletion` globally, so the 6000-char cap applies to
**summary generation calls too**, not just the retrieval/answer calls it
was presumably added for (the comment says "PORT_PACKAGE has a 200k-char
node node → no 504s"). Result: node summaries for large nodes are built
from the first ~3% of the node's text. Every downstream step
(node selection, routing) relies on these summaries being representative.

**Fix:** Don't apply one blanket cap to all call sites.

- Either raise the cap substantially for summary calls (e.g. 20000+), or
- Split long nodes into chunks, summarize each, and merge/re-summarize,
  instead of hard-truncating.
- At minimum, tag/distinguish summary-generation calls from other calls so
  the cap (if any) is call-site-specific, not global.

## Bug B — Node text truncated before reaching the answering model (`search.py`)

```python
chunks.append(
    f"[Source: {filename}, Page {node['page_index']}]\n{node['text'][:4000]}"
)
```

Even when the *correct* node is selected, only the first 4000 characters
of its text are passed into the final answer-generation context. The fact
that actually answers the question (a specific clause, number, or list)
frequently sits past that cutoff in longer nodes.

**Fix:** Remove or substantially raise this cap. Check what `num_ctx` the
final `ollama_call` in `ask()` (the answer-generation call in `search.py`)
uses — it is **not explicitly set**, unlike the routing call which sets
`num_ctx=8192`. Confirm the default doesn't silently truncate again even
if the `[:4000]` slice is removed.

## Bug C — Node selection is blind to node text (`search.py`)

```python
tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])
```

`search_nodes()` sends the LLM only `node_id` + `title` + `summary` to
decide which nodes are relevant — text is stripped out entirely. Combined
with Bug A (poisoned summaries), the 8B model is picking nodes almost by
title alone for many queries, which explains the wrong-chunk-retrieved
failures in the eval set (e.g. excluded-countries question returning a
policy-termination clause instead of the exclusions section).

**Fix:** Once Bug A is fixed, summaries improve "for free" and this step
should get better. Additionally consider passing a short text snippet
(e.g. first ~500 chars) per node alongside title+summary in the selection
prompt, especially for documents with few large nodes where summary
granularity alone is too coarse to disambiguate.

## Bug D — No retry/repair on judge JSON parse failure (harness, eval.py)

A large fraction of `pageindex_eval_results.json` rows show
`"Parse error: No JSON object found"` for every sub-metric, which zeroes
the entire row regardless of actual answer quality. This is a harness bug,
not a model-quality signal — it needs to be excluded from any "is the
model good enough" judgment until fixed.

`router.py` and `search.py` show the codebase's existing parse pattern:

```python
start = result.find("{")
end = result.rfind("}") + 1
if start == -1 or end == 0:
    raise ValueError(f"No JSON object found in response: {result[:200]!r}")
parsed = json.loads(result[start:end])
```

This has no retry — a single malformed generation is fatal (raises) here,
and in the judge, apparently resolves to a silent zero-score instead of a
raised exception (needs confirming once `evaluation.py` is available —
not yet uploaded).

**Fix:**

- Add a retry loop (2-3 attempts) with re-prompting on parse failure to
  `router.py`, `search.py`, and whatever the judge's LLM-call function is
  in `evaluation.py`.
- The judge specifically should **not** silently default to a zero score
  on parse failure — either retry, or mark the row as `invalid`/`excluded`
  so it doesn't get averaged into the aggregate score as if it were a
  genuine failing answer.
- `evaluation.py` was not available in this session — locate it and apply
  the same fix pattern once found.

## Fix order

1. Bug A (summary cap) — unblocks everything downstream, zero cost.
2. Bug C (node selection uses text) — depends on A being fixed first to
   see real improvement; can be done in the same pass.
3. Bug B (chunk truncation) — independent, do anytime.
4. Bug D (judge/router/search retry-on-parse-failure) — independent,
   needed to get a trustworthy score signal at all. Locate `evaluation.py`
   first.
5. **Re-run eval after 1–3.** Only after that, if faithfulness/precision
   are still bad on cases where the correct full-text context demonstrably
   reached the model, consider upgrading the answer-generation model
   (e.g. to a 14B+ local model). Don't upgrade model size before this —
   there's no way to tell if `granite4.2:8b` is actually the bottleneck
   while it's being fed truncated/wrong context.

## Notes for whoever picks this up

- Routing/node-selection (classification-like tasks) can likely stay on
  the smaller model even after everything above is fixed — it's the
  answer-generation step (final `ollama_call` in `search.py`'s `ask()`)
  that most needs either full context or a stronger model if quality is
  still lacking after the bug fixes.
- The answer-generation prompt in `search.py` forces a "one-sentence
  summary" + "Bottom line:" structure. This style nudges the model toward
  paraphrase over verbatim figures/clauses, which likely contributes to
  low faithfulness scores even on correctly-retrieved context. Worth a
  prompt tweak (e.g. "quote exact figures/clause language where relevant")
  once the retrieval-side bugs are fixed and it's possible to isolate this
  as a remaining issue.`
