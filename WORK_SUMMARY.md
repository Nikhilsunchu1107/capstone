# Work Summary: DeepSeek Migration

Branch: `deepseek-migration`. Source of facts: `final_plan.md`, section 11.

## 1. Goal

Compare three retrieval strategies on 40 questions. The strategies are Vector RAG, PageIndex RAG, and Graph RAG. The old setup used Ollama. The new setup uses two external services only:

- DeepSeek for the LLM (model `deepseek-flash`).
- NVIDIA for embeddings (model `nemotron-3-embed-1b`).

Never use the model `deepseek-v4-pro`.

## 2. Status

Phases 0 to 6 are complete. Phase 7 (hand check) is not complete. It needs you.

| Phase | Result |
|---|---|
| 0. Eval set | `data/eval_set.json` is frozen. It has 40 questions. |
| 1. Foundation | The DeepSeek endpoint works. The cache works. The second call costs 0 paid calls. |
| 2. Judge | `eval/judge.py` passes 9 of 9 checks. The null rate is 0 of 150. |
| 3. Debug each strategy | Done for Vector, PageIndex, and Graph. See section 4. |
| 4. Rebuild indexes | Done for all three strategies. |
| 5. Smoke run | Done on the first 5 questions. 0 null scores. |
| 6. Full run | Done. 40 questions per strategy. 0 null scores. |
| 7. Hand check | Waiting for you. |

## 3. Judge

The judge scores five metrics. The plan had four. A fifth metric, `answer_correctness`, compares the generated answer with the gold answer. The four original metrics never make this comparison.

The judge reply is JSON. The reasoning comes first. The score comes last.

I found and fixed two judge faults:

- The first version lost the score key in 3 of 114 replies. The prompt now says that both keys are required.
- A later version lost the score key in 8 of 200 Vector replies. The retry now shows the model its own reply and asks again. The model gives the number. The code sets no default. Use `--rescore-null` to run it.

### 3.1 Null scores in detail

**What a null score is.** The judge must return one number from 0.0 to 1.0 for each metric. A null score means the judge gave no valid number. The row keeps `score: null` and a reason text. The code never fills the gap with a default value (for example 0 or 0.5). A default would hide the fault and bias the mean.

**What the fault was.** The judge sometimes returned JSON without the `score` key. The `reasoning` key was present. The `score` key was missing. I did not find the root cause inside the model. I only measured the symptom and fixed it.

**When it happened.**

| Version | Fault rate | Fix |
|---|---|---|
| Judge v1 (sanity test) | 3 of 114 replies | The prompt now says: "Both keys are required." The score key goes last, after the reasoning. |
| Judge v2 (first Vector full run) | 8 of 200 replies | The retry now shows the model its own reply. It asks for the number again. |

**How the retry works** (`score_metric` in `eval/judge.py`):

1. The judge makes the first request.
2. The code parses the reply. A reply is valid when it has a numeric `score` from 0.0 to 1.0.
3. If the reply is not valid, the code makes one more request. This request includes the bad reply and a short instruction to answer with one JSON object.
4. If the second reply is also not valid, the code records `score: null` and the reason `PARSE_FAIL`.

The model always supplies the number. The code never invents it.

**A second cause of null.** If the API returns an error, the code records `score: null` and the reason `API_ERROR`. An example is a prompt that is longer than the model context. This case did not occur in the full runs. The longest context (318k characters) worked in the smoke run.

**How to find and fix null scores.**

- The summary table shows the null count for each metric. Mean scores skip null rows.
- Run `python -m eval.run_strategy <strategy> --rescore-null` to judge the null rows again.

**Final state.** All four full runs have 0 null scores. This is the reason you can trust that the means in section 6 use all 40 questions for all five metrics.

## 4. Findings per strategy

### Vector RAG

Old context precision was low. The cause was page headers and footers. Header chunks (for example "Policy name ... F&U Dated") won the top ranks, because every question names its policy.

The fix removes lines that appear on 40% or more of the pages of one PDF. The result for strict all@5 (every gold passage is in the top 5):

| Input | all@5 |
|---|---|
| Raw pages, chunk 1000/150 | 0.70 |
| After header and footer removal | **0.85** |

Tier D (cross-document) questions score 0.50. The top 5 cannot hold two documents. Report both rows in the paper.

The same removal applies to Graph RAG and PageIndex input. This keeps the comparison fair.

### PageIndex RAG

- The index covers 15 PDFs with 0 failures.
- The build used about 520 calls. It used 0.96M prompt tokens and 0.25M completion tokens. This is 2 times the first estimate.
- Tree structure depends on the model. 7 of 14 trees match the old Ollama trees. The other trees have 1 to 33 more nodes.
- An LLM router picks the document. The router never sees the gold file.
- Strict all@5 is 1.00. But the retrieved context is large: median 27k characters, mean 63k, maximum 388k. Vector returns 4.5k.
- Budget metric (gold inside the first 8000 characters): Vector 0.85, PageIndex 0.62.

### Graph RAG

- Entity extraction (NER) changed from spaCy and GLiNER to DeepSeek flash. The code is in `graph_rag/src/llm_ner.py`.
- The graph has 1878 chunks, 11,619 entities, and 222k edges. The build used about 0.65M prompt tokens and 0.37M completion tokens.
- The original design scored poorly. Retrieval all@5: vector only 0.85, graph only 0.17, original hybrid 0.75.
- I found two causes. The code matched entities by exact string. The code merged raw scores from different scales.
- I made one improvement pass: embedding-based entity linking, idf weighting, and RRF merge. No tuning used the frozen set.
- Result: linked hybrid 0.82.
- The paper reports two variants: "Graph RAG (original)" and "Graph RAG (linked)".

## 5. Decision you made

Generation has no context cap. Each strategy passes its natural retrieval unit. The paper reports accuracy and cost together. Cost means context size and tokens. The paper also reports the 8000-character retrieval-budget row.

Warning for the paper: large units raise context precision. Compare this metric with care.

## 6. Full-run results

40 questions per strategy. 0 null scores.

| Metric | Vector | PageIndex | Graph (original) | Graph (linked) |
|---|---|---|---|---|
| Answer correctness | 0.83 | 0.91 | 0.75 | 0.78 |
| 95% interval | 0.71 to 0.93 | 0.82 to 0.99 | 0.62 to 0.87 | 0.66 to 0.90 |

Other results for Vector and PageIndex:

| Metric | Vector | PageIndex |
|---|---|---|
| Faithfulness | 0.91 | 0.94 |
| Answer relevancy | 0.88 | 0.94 |
| Context precision | 0.27 | 0.75 |
| Context recall | 0.95 | 0.99 |
| Context size (characters) | 4.5k | 63k |

Correctness by tier, Vector and PageIndex:

| Tier | Vector | PageIndex |
|---|---|---|
| A | 0.88 | 1.00 |
| B | 1.00 | 0.96 |
| C | 0.71 | 0.67 |
| D | 0.47 | 0.83 |

**Main conclusion.** No paired difference in correctness is significant at n=40. The best case is PageIndex over Vector: +0.085, interval -0.02 to +0.19, p=0.20.

Only these differences are significant:

- PageIndex context precision is higher. The cause is unit size.
- Graph (original) has lower recall and lower precision.

## 7. Cost

| Stage | Tokens |
|---|---|
| Total so far | 5.2M prompt, 0.78M completion |
| Vector, per question | about 6k prompt |
| PageIndex, per question | about 105k prompt (smoke run) |
| PageIndex judge alone | 1.5M prompt |

## 8. Naming change

You chose vendor-neutral names:

- Caches: `cache/llm/`, `cache/embeddings/`.
- PageIndex store: `pageindex_rag/.pageindex/`.
- Patch file: `pageindex_rag/llm_patch.py`.
- Vector cache: `vector_rag/cache/{chunks.json,embeddings.npy}`.
- Raw-page baseline: `vector_rag/cache/ablation_raw_pages/`.
- Config variables: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL`.

The `.env` file is not edited. The config falls back to `DEEPSEEK_*` and `NVIDIA_API_KEY`. After the rename, the Vector numbers were identical. The judge cache needed 0 new calls.

## 9. Main files

- `eval/judge.py`, `eval/embed.py`, `eval/llm.py`, `eval/config.py`: shared client, judge, embeddings, config.
- `eval/generate.py`: shared answer prompt.
- `eval/run_strategy.py`: retrieve, answer, judge. It saves checkpoints.
- `eval/retrieval_check.py`: retrieval check for all strategies.
- `eval/report.py`: run `python -m eval.report` to write `eval/results/summary.md` and `eval/results/handcheck.md`.
- `vector_rag/`, `pageindex_rag/`, `graph_rag/`: each has `build_index.py` or `index.py`, `retrieve.py`, and `check.py`.
- `eval/results/*_rows.json`: per-question results.

## 10. Next steps

1. **You:** hand check the 15 rows in `eval/results/handcheck.md` (Phase 7).
2. **Needs your approval:** remove the old Ollama files:
   - `pageindex_rag/cache/ollama/*`
   - `vector_rag/cache/vector_rag_{eval_results,outputs}.json`
   - the old notebooks
3. **Claude:** run `graphify update .`.
4. **You:** commit and merge. Claude gives the commit message as text only.
