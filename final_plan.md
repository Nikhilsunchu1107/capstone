# final_plan.md — DeepSeek migration + RAG eval fix

Style: STE100 + caveman. Short. One idea per line. Plan = for user AND for executor (me).
Status: **PLAN ONLY. No code changed. Wait for user "go".**

## 0. Goal

- All 3 RAG strategies (PageIndex, Vector, Graph) use ONE OpenAI-compatible LLM endpoint: DeepSeek.
- Same curated question set (`data/eval_set_v2.json`, 80 QA), same judge, same judge prompts, same metrics.
- Result: numbers valid for research paper.

## 1. Hard rules

1. Model = DeepSeek **flash** only. Never use pro. Budget is finite.
2. Model ID is UNVERIFIED. "deepseek-flash" may not be exact ID. First step: `client.models.list()`. Use exact ID found. Store in `.env` as `DEEPSEEK_MODEL`.
3. Key in `.env` only (`LLM_API_KEY`; old name `DEEPSEEK_API_KEY` still works). Never in code, never in git. `.env` in `.gitignore`.
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
- `.env` is not in git. Copy it by hand into the worktree. Add `LLM_API_KEY` and `LLM_MODEL` (old names `DEEPSEEK_*` still work).
- Plan files (`final_plan.md`, `RAG PLAN.md`, `RAG_STRATEGY_COMPARISON.md`) are untracked. Copies exist in the worktree. Commit them there.

**Trade-off to accept:** with no shared package, the same code can end up in 3 places. This caused the 3-judge problem (F8). Rule to prevent it: the judge lives in ONE file, `eval/judge.py`. All 3 strategies call it. Each strategy keeps its own LLM and cache code in its own dir.

## 4. Target layout (inside the worktree)

```
capstone-deepseek/
  .env                      # LLM_*, EMBED_* settings (copy by hand, not in git)
  data/
    policy_qa.json          # old 83 Q, frozen, kept for reference
    eval_set_v2.json        # curated set (see Phase 0)
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
| D1 | Embeddings (F1) | **DECIDED (user): `nvidia/nemotron-3-embed-1b` on NVIDIA's OpenAI-compatible endpoint.** 2048 dims. No Ollama anywhere. Same embedder for Vector AND Graph. Passages use `input_type=passage`, queries use `input_type=query`. Cached per text in `cache/nvidia/embed/`. |
| D2 | Judge model | **Same `deepseek` flash for all 3.** One judge = comparable. Note in paper: judge = generator family (self-preference bias). Mitigation: spot-check 15 rows by hand. |
| D3 | Generator model | Same flash model for all 3. |
| D4 | Thinking mode | Off for judge and generator. Cheaper. Deterministic. `temperature=0`. |
| D5 | Graph RAG scope | Keep in scope. Fix scorer. Graph RAG NER (GLiNER) stays local. Only LLM calls move. |
| D6 | **DECIDED:** Git worktree. No nuke, no archive, no `rag_common/` (§3). | — |
| D7 | **DECIDED:** No `.ipynb` files. Python scripts only. Run with `python`. Old notebooks stay in place as reading reference only. | — |
| D8 | **DECIDED:** No Jev / TypeSafe AI. One judge only: DeepSeek flash. Data stays on our machine and DeepSeek. | — |

## 6. Phases

Each phase ends with a **gate**. Gate fail = stop, fix, do not continue.

### Phase 0 — Eval set (80 questions, written from the PDF text)

**DECIDED (user):** Discard the old 83 questions. Use a new set written from the PDF text.

Why: the old set is flawed. About 22 gold answers say "text does not contain". Most questions say "this policy" with no name. Some gold answers are truncated or wrong.

Design (`data/eval_set_v2.json`, review sheet `data/eval_set_v2_review_q41-q120.md`):
- Each gold answer has evidence: filename, page, exact quote. Every quote is checked against the PDFs (`python -m eval.verify_quotes`).
- 4 tiers, so results show WHERE strategies differ:
  - E multi-hop in one policy (24)
  - F cross-document, needs 2 or 3 PDFs (24)
  - G unanswerable trap (12)
  - H paraphrased, 10 of them without the policy name (20)
- All 15 PDFs covered.
- Tiers are chosen by question type, not by which strategy is expected to win. Report scores per tier AND overall.
- Freeze with a content hash. No edit after freeze.

Gate: 80 questions, hash recorded.

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

## 9. After code changes (user commits; never commit for them, give message as text)

Run `graphify update .` (AGENTS.md rule). Commit per phase. Commit message ends with attribution line.

## 10. Next action

## 11. Status

- Eval set DONE: `data/eval_set_v2.json`, 80 questions (q41 to q120), FROZEN (sha256 982a01e2…c14, set in `eval/config.py`). `eval/run_strategy.py` stops if the file does not match its hash.
  - Tiers: E multi-hop in one policy (24), F cross-document with 2 or 3 PDFs (24), G unanswerable trap (12), H paraphrased (20). 10 of the H questions do not name the policy.
  - Every quote passes `python -m eval.verify_quotes` (0 problems). All 15 PDFs are used. Review sheet: `data/eval_set_v2_review_q41-q120.md`.
  - Limitation for the paper: an LLM wrote the set. Not all gold answers are reviewed by a person. q85, q104, q59, q64 and q119 are ambiguous or have 3 sub-questions (not changed, the file is frozen).
  - Limitation for the paper: design choices (header removal, Graph RAG entity linking) were made while looking at an earlier question set that is no longer part of this project.
- Phase 1 DONE: endpoint verified. Models: `deepseek-flash`, `deepseek-v4-pro` (never use pro). Cache gate passed (2nd call = 0 paid calls). Thinking-off param accepted.
- Phase 2 DONE: `eval/judge.py` + `eval/judge_sanity.py`. Judge has 5 metrics: the 4 plan metrics + `answer_correctness` (gold answer vs generated answer), because the 4 RAGAS metrics never compare the answer with the gold answer. Sanity test on this set: 10 of 11 checks pass, null rate 0/180. The failing check is "perfect: all metrics >= 0.85": context recall is 0.74, because several multi-hop gold answers contain derived statements that the short gold quotes do not spell out. Threshold not changed.
- Judge reply format: JSON, reasoning first, score last. The first version lost the score key in 3/114 replies. Fixed by stating both keys are required. A later version lost it in 8/200 Vector replies. Retry now shows the model its own reply and asks again (model supplies the number; no default). Use `--rescore-null`.
- Judge change for tier G (unanswerable): only answer correctness is scored. The other four metrics are marked not applicable (`na`), not null. Reason: a correct refusal scored 0.33 on faithfulness. `eval/judge_sanity.py` has 3 abstention checks (all pass).
- Embeddings DONE: `eval/embed.py`, NVIDIA `nemotron-3-embed-1b` (2048 dims, cache test passed). Needs `NVIDIA_API_KEY` in `.env`. Only 2 external services: DeepSeek (LLM) and NVIDIA (embeddings).
- Vector RAG DONE (`vector_rag/index.py`, `retrieve.py`, `check.py`). Page headers and footers (lines on >=40% of a PDF's pages) are stripped, because header chunks ("<Policy name> ... F&U Dated") won the top ranks (every question names its policy). The same stripping applies to Graph RAG and PageIndex input. Strict all@5 = 0.59, all@10 = 0.78, cross-document (tier F) all@5 = 0.29.
- Renamed to vendor-neutral names (user decision): `cache/llm/`, `cache/embeddings/`, `pageindex_rag/.pageindex/` (old Ollama store replaced), `pageindex_rag/llm_patch.py`, `vector_rag/cache/{chunks.json,embeddings.npy}`. Config: `LLM_BASE_URL/LLM_API_KEY/LLM_MODEL`, `EMBED_BASE_URL/EMBED_API_KEY/EMBED_MODEL`. `.env` was NOT edited: config falls back to `DEEPSEEK_*` and `NVIDIA_API_KEY`.
- PageIndex DONE (`pageindex_rag/`): index of 15 PDFs, 0 failures, about 520 calls, 0.96M prompt + 0.25M completion tokens. Node counts vs old Ollama trees: 7 of 14 identical, others +1 to +33, so tree structure is NOT model independent. LLM router (never sees the gold file) picks up to 3 documents, tree search picks up to 3 nodes in each, at most 5 nodes go to the generator (`PAGEINDEX_MAX_NODES`; this cut was in the code for every run). Strict all@5 = 0.90, but mean context is 63.6k chars vs 13k for Vector. Budget metric (gold inside first 8000 chars): Vector 0.59, PageIndex 0.40. The page metric is unfair to PageIndex (node has start page only).
- Graph RAG DONE. NER switched from spaCy+GLiNER to DeepSeek flash (`graph_rag/src/llm_ner.py`), chunks and vectors shared with Vector RAG. Graph: 1878 chunks, 11,619 entities, 222k edges. Build cost ~0.65M prompt / 0.37M completion tokens. A first design (exact-string entity matching, raw-score merge) retrieved poorly. One improvement pass: embedding entity linking + idf + RRF. Only the improved design is kept. Retrieval all@5: vector-only 0.59, graph-only 0.06, hybrid 0.49 (worse than vector alone).
- DECIDED (user): generation has NO context cap. Each strategy passes its natural retrieval unit. Paper reports accuracy AND cost (context chars, tokens) together, plus the 8000-char retrieval-budget row. Context precision favours big units, so compare it with care.
- Settings: Vector and Graph retrieve 15 chunks (`TOP_K` in `eval/config.py`).
- Full runs DONE (80 Q each, 0 null scores): `eval/results/{vector,pageindex,graph}_v2_rows.json`. Correctness: Vector 0.80 [0.73,0.86], PageIndex 0.82 [0.75,0.89], Graph 0.78 [0.71,0.85]. No paired difference in correctness is significant (PageIndex +0.021, p=0.56; Graph -0.017, p=0.49). PageIndex has higher recall (0.91 vs 0.84, p=0.11) and the best tier F score (0.89 vs 0.78) at 4.8x the context. PageIndex cost about 4.9M prompt tokens, Graph about 1.1M.
- Reports: `python -m eval.report` writes `eval/results/summary.md`, `handcheck.md` and `final_comparison.csv` (paper table: 4 metrics, one column per strategy). `COMPARISON.md` and `WORK_SUMMARY.md` are current.
- NEXT:
  1. User hand-checks 15 rows in `eval/results/handcheck.md`.
  2. Check the source of the PageIndex "99%" claim before the paper cites it.
  3. Decide: keep the 5-node cut for PageIndex, or remove it and re-run the affected questions (24 of 80 have exactly 5 nodes).
  4. Cleanup of old Ollama files (needs user OK), `graphify update .`, user commits and merges.
