# final_plan.md — DeepSeek migration + RAG eval fix

Style: STE100 + caveman. Short. One idea per line. Plan = for user AND for executor (me).
Status: **PLAN ONLY. No code changed. Wait for user "go".**

## 0. Goal

- All 3 RAG strategies (PageIndex, Vector, Graph) use ONE OpenAI-compatible LLM endpoint: DeepSeek.
- Same curated question set (`data/eval_set.json`, 40 QA, user-approved), same judge, same judge prompts, same metrics.
- Result: numbers valid for research paper.

## 1. Hard rules

1. Model = DeepSeek **flash** only. Never use pro. Budget is finite.
2. Model ID is UNVERIFIED. "deepseek-flash" may not be exact ID. First step: `client.models.list()`. Use exact ID found. Store in `.env` as `DEEPSEEK_MODEL`.
3. Key in `.env` only (`DEEPSEEK_API_KEY`). Never in code, never in git. `.env` in `.gitignore`.
4. Every LLM call cached to disk (json). Re-run = zero cost.
5. Smoke test on 5 questions BEFORE the full set (N=40). Print token count + cost estimate first.

## 2. Findings (from reading code)

| # | Finding | Impact |
|---|---|---|
| F1 | **DeepSeek has no embeddings endpoint.** | Cannot move embeddings to DeepSeek. |
| F2 | Old notebooks use NVIDIA NIM (OpenAI-compatible). Only `base_url`, `api_key`, `model` change. | Migration small for LLM calls. |
| F3 | Old `RAG_v1/run_pageindex_v4.ipynb`: pageindex lib (`pi_client`, `pageindex.utils`), router, `search_nodes`, `ask`, 4 judge prompts, `safe_parse_score`, `evaluate_pipeline`. | Reuse logic. |
| F4 | Old `VectorRAG/notebook/RAG_pipeline.ipynb` (user typed `noteboook`; real dir = `notebook`). Chunk 1000/150, Chroma, NV-Embed. | Reuse logic. |
| F5 | Current notebooks (Ollama) already hold caching, parallel judge, schema-JSON judge. Old notebooks do not. | Do NOT discard these 3 features. |
| F6 | Graph RAG: all 4 RAGAS scores have `std = 0.0` over 83 rows (old set). Batch-score broadcast bug. | Graph numbers invalid now. |
| F7 | Graph RAG still uses NIM + jina/voyage embeddings + GLiNER + own scripts (`score_evaluation_*.py`: 3 scorers). | 3 scorers = judge inconsistency risk. |
| F8 | Judges differ: PageIndex = local Ollama, Vector = other, Graph = llama-70b NIM. | Not comparable. |
| F9 | Context precision/recall ~0.3-0.4 for Vector and PageIndex. | Cause unknown: retrieval or judge prompt. Must debug. |
| F10 | Vector scored 82/83 rows. 1 row lost. | Must find why. |
| F11 | Old v4 notebook truncates prompts at 6000 chars (hack for 504). | Hidden quality bug. Do not copy. |

## 3. Decision: git worktree (no nuke, no archive)

**DECIDED (user):** Do the migration in a git worktree. Do not delete directories. Do not create `_archive/`. Do not create a `rag_common/` package.

- Worktree path: `~/Projects/capstone-deepseek`
- Branch: `deepseek-migration` (from `yash` at 902b121)
- Main checkout `~/Projects/capstone` (branch `yash`) stays untouched. Old Ollama code and results stay safe there.
- Work: edit each strategy dir IN PLACE inside the worktree. Port logic from the old NIM notebooks into the existing scripts.
- Merge to `yash` only after the full run passes all gates. Discard the branch if the migration fails.
- `.env` is not in git. Copy it by hand into the worktree. Add `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL`.
- Plan files (`final_plan.md`, `RAG PLAN.md`, `RAG_STRATEGY_COMPARISON.md`) are untracked. Copies exist in the worktree. Commit them there.

**Trade-off to accept:** with no shared package, the same code can end up in 3 places. This caused the 3-judge problem (F8). Rule to prevent it: the judge lives in ONE file, `eval/judge.py`. All 3 strategies call it. Each strategy keeps its own LLM and cache code in its own dir.

## 4. Target layout (inside the worktree)

```
capstone-deepseek/
  .env                      # DEEPSEEK_API_KEY, DEEPSEEK_MODEL (copy by hand, not in git)
  data/
    policy_qa.json          # old 83 Q, frozen, kept for reference
    eval_set.json           # NEW curated set (see Phase 0)
  vector_rag/  pageindex_rag/  graph_rag/   # edited in place; LLM calls point to DeepSeek
  eval/
    judge.py                # ONE judge: 4 prompts, JSON parse, parallel, cache
    config.py               # model IDs, seeds, TOP_K, chunk sizes: ONE place
    run_all.py              # eval_set -> 3 strategies -> same judge
    report.py               # reads results/*.json -> tables (md) + charts (png). No LLM call.
    results/                # per-question rows (json) + summary
  final_plan.md
```

## 5. Decisions (D1-D5: user took recommendations, 2026-10-12)

| ID | Question | Recommendation |
|---|---|---|
| D1 | Embeddings (F1) | **Local Ollama `nomic-embed-text`.** Free, already used by current Vector RAG. Same embedder for Vector AND Graph (fair). Alternative: NVIDIA NIM embeddings (old key, may be dead). |
| D2 | Judge model | **Same `deepseek` flash for all 3.** One judge = comparable. Note in paper: judge = generator family (self-preference bias). Mitigation: spot-check 15 rows by hand. |
| D3 | Generator model | Same flash model for all 3. |
| D4 | Thinking mode | Off for judge and generator. Cheaper. Deterministic. `temperature=0`. |
| D5 | Graph RAG scope | Keep in scope. Fix scorer. Graph RAG NER (GLiNER) stays local. Only LLM calls move. |
| D6 | **DECIDED:** Git worktree. No nuke, no archive, no `rag_common/` (§3). | — |
| D7 | **DECIDED:** No `.ipynb` files. Python scripts only. Run with `python`. Old notebooks stay in place as reading reference only. | — |
| D8 | **DECIDED:** No Jev / TypeSafe AI. One judge only: DeepSeek flash. Data stays on our machine and DeepSeek. | — |

## 6. Phases

Each phase ends with a **gate**. Gate fail = stop, fix, do not continue.

### Phase 0 — New eval set (40 questions, written fresh)

**DECIDED (user):** Discard the old 83. Use 40 NEW questions, written from the PDF text.

Why: the old set is flawed. About 22 gold answers say "text does not contain". Most questions say "this policy" with no name. Some gold answers are truncated or wrong.

Design (draft done: `data/eval_set_v2_candidates.json`, review file `data/eval_set_v2_review.md`):
- Each question names its policy. Each gold answer has evidence: filename, page, exact quote.
- 4 tiers, so results show WHERE strategies differ:
  - A single fact, one passage (16)
  - B multi-part answer, list or table (12)
  - C apply a rule to a case (6)
  - D cross-document, needs 2 PDFs (6)
- All 15 PDFs covered.
- Tiers are chosen by question type, not by which strategy is expected to win. Report scores per tier AND overall.

Steps left:
1. User reviews all 40. Marks reject or change.
2. Freeze as `data/eval_set.json` with content hash. No edit after freeze.

Gate: 40 questions, user-approved, hash recorded.

Statistics note: n=40 overall, but tiers C and D have n=6 each. Per-tier numbers are descriptive only. Do significance tests on the overall set.

### Phase 1 — Foundation (DeepSeek client in each strategy + `eval/config.py`)
- Verify endpoint: `models.list()`, 1 hello call. Record exact model ID.
- Build `llm.py`: OpenAI SDK, `base_url=https://api.deepseek.com`, retries on 429/5xx, exponential backoff, per-call disk cache, log prompt/completion tokens.
- Build `cache.py`, `config.py`.
- **Gate:** same call twice → 2nd call = cache hit, 0 tokens. Cost per 1k calls estimated and shown to user.

### Phase 2 — One judge (`judge.py`) — fixes F6, F7, F8
- Port the 4 prompts (Faithfulness, Answer Relevancy, Context Precision, Context Recall) from old v4 notebook. Keep text identical across strategies.
- Force JSON output (`response_format={"type":"json_object"}`). Remove fuzzy parsing where possible. On parse fail: retry once, then record `score=null` + reason. **Never silently default to a number.**
- Per-question call. Parallel (thread pool, 8-16 workers; tune to rate limit).
- Delete the 3 old Graph scorers (`score_evaluation_*.py`) after port.
- **Gate (judge sanity tests, no RAG needed):**
  - Perfect answer + gold context → high F/AR/CR.
  - Answer from different question → low AR.
  - Random context → low CP/CR.
  - Score `std > 0` across 20 mixed rows.
  - Null-rate < 2%.

### Phase 3 — Debug each strategy separately (user's `<mark>` item)
Rule: test retrieval ALONE before generation, generation ALONE before judge.

**3a. Retrieval-only metrics (no LLM judge, free):**
- For each of N Q: gold passage is known (`documents` field in `policy_qa.json`). Measure **hit@k** (gold chunk/page in top-k) and **MRR**.
- This answers F9: low CP/CR = retrieval problem or judge problem? If hit@5 is high but CP is low → judge/prompt problem. If hit@5 is low → retrieval problem.
- Check QA quality: 10 random Q read by hand. Is each Q answerable from one passage? Does question name a doc ("Item 12.2") that exists in many PDFs? Ambiguous Q = bad ground truth.

**3b. Per-strategy bug checklist**

| Strategy | Check |
|---|---|
| Vector | Why 82/83 (F10)? (find root cause on old set, then confirm fixed) Chunk size 1000/150 vs gold passage size. Chroma distance metric (cosine vs L2) and score conversion. Dedupe of overlap chunks. Is `top_k` same as other strategies? Metadata `source_file`/page kept? Does retrieval filter by doc? |
| PageIndex | Tree build uses LLM → rebuild trees with DeepSeek flash (cost: 15 PDFs; check `PORT_PACKAGE` 200k-char node). Do NOT truncate to 6000 chars (F11); instead split node. Is router used in eval? (Old code skips router when `source_file` known = **answer leak**. Eval must route by LLM, like real use. Else PageIndex is advantaged.) Page numbers off-by-one? |
| Graph | Fix F6 scorer. Check NER recall on policy text. Check graph edge/entity linking. Latency 41.8s avg, max 620s: find the slow step (likely LLM or graph traversal). Chunk size 200 tokens vs 1000 chars in Vector: unequal context size. |

**3c. Fairness controls (write into `config.py`, one place):**
- Same `TOP_K`, same max context chars passed to generator, same generator prompt template, same temperature.
- Same curated questions (N), same order, same seed.
- No strategy sees `source_file` of the gold answer unless ALL do (decide: none).

- **Gate:** per strategy: hit@k table printed; all N rows produce an answer; failures listed with reason.

### Phase 4 — Rebuild indexes with new stack
- Vector: re-chunk (cache), embed with D1 model (cache `.npy`), Chroma (persist).
- PageIndex: rebuild trees via DeepSeek flash. Cache trees + registry.
- Graph: re-ingest with D1 embedder + DeepSeek flash.
- **Gate:** index counts printed and match source PDFs (15).

### Phase 5 — Smoke run (5 Q × 3 strategies)
- Run end to end. Print: answer, retrieved context, 4 scores, tokens, cost.
- User reads 5 rows by eye.
- **Gate:** scores vary, make sense, cost/Q × N × 3 within budget user states.

### Phase 6 — Full run (N × 3)
- `eval/run_all.py`, checkpoint every question. Resume on crash.
- Output: `eval/results/{strategy}_rows.json`, `summary.json`, `summary.md`.
- Summary shows: mean, **std, 95% bootstrap CI, null count**, latency p50/p95, tokens.
- **Gate:** no strategy has `std = 0`. Null-rate < 2%. Rows = N each.

### Phase 7 — Paper-grade validation
- Hand-check 15 random rows (5 per strategy): does judge score match human view? Report agreement.
- Paired comparison: same questions → paired bootstrap or Wilcoxon between strategies. Report p / CI. Not only means.
- Ablation of noise: re-run judge once with different seed on 20 rows → score stability.
- Write method section facts: models, temps, chunk sizes, top-k, embedder, date, API version.
- Update `RAG_STRATEGY_COMPARISON.md` with new numbers. Remove old caveat.

## 7. Cost control

- Flash only. Thinking off. `max_tokens` capped (answer 512, judge 256).
- Context sent to judge: top-k chunks only, no truncation hack, but top-k fixed.
- Rough call count (N=40): generation 40×3=120; judge 40×3×4=480; router 40; PageIndex tree build ≈ 15 PDFs × many node summaries (largest cost — measure in Phase 1 via 1-PDF dry run). Total ≈ 640 + tree build. About 2x cheaper than N=83.
- Show token estimate from cached dry run before each phase. User approves.
- DeepSeek automatic prefix cache: put fixed instructions FIRST in prompt, variable text LAST → cheaper repeat calls.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Model ID wrong | `models.list()` first. |
| Rate limit 429 | Backoff + low parallelism first. |
| Judge = generator family bias | State in paper. Hand-check 15 rows. |
| Embedder differs from older results | Re-run everything. Old numbers discarded. |
| Account balance runs out mid-run | Checkpoint + cache. Resume = no repeat spend. |
| Graph RAG too slow | Find slow step in Phase 3 before full run. |
| API key leaks | `.env`, `.gitignore`, never print key. |

## 9. After code changes

Run `graphify update .` (AGENTS.md rule). Commit per phase. Commit message ends with attribution line.

## 10. Next action

## 11. Status

- Phase 0 DONE: `data/eval_set.json` frozen (sha256 ce3683e0…b61a, 40 Q).
- Phase 1 DONE: endpoint verified. Models: `deepseek-flash`, `deepseek-v4-pro` (never use pro). Cache gate passed (2nd call = 0 paid calls). Thinking-off param accepted.
- Phase 2 DONE: `eval/judge.py` + `eval/judge_sanity.py`. 9/9 checks pass, null rate 0/150. Judge now has 5 metrics: the 4 plan metrics + `answer_correctness` (gold answer vs generated answer). Reason: the 4 RAGAS metrics never compare the answer with the gold answer.
- Judge reply format: JSON, reasoning first, score last. First version lost the score key in 3/114 replies. Fixed by stating both keys are required.
- NEXT: Phase 3 (retrieval-only debug, hit@k, per strategy).
