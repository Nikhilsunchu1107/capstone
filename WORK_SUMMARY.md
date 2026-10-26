# Work Summary: DeepSeek Migration

Branch: `deepseek-migration`. Source of facts: `final_plan.md`, section 11.

## 1. Goal

Compare three retrieval strategies on 80 questions. The strategies are Vector RAG, PageIndex RAG, and Graph RAG. The old setup used Ollama. The new setup uses two external services only:

- DeepSeek for the LLM (model `deepseek-flash`).
- NVIDIA for embeddings (model `nemotron-3-embed-1b`).

Never use the model `deepseek-v4-pro`.

## 2. Status

Phases 0 to 6 are complete. Phase 7 (hand check) is not complete. It needs you.

| Phase | Result |
|---|---|
| 0. Eval set | `data/eval_set_v2.json` is frozen. It has 80 questions (q41 to q120). |
| 1. Foundation | The DeepSeek endpoint works. The cache works. The second call costs 0 paid calls. |
| 2. Judge | `eval/judge.py` passes 10 of 11 sanity checks on this set. The null rate is 0 of 180. The failing check is explained in section 3. |
| 3. Debug each strategy | Done for Vector, PageIndex, and Graph. See section 4. |
| 4. Rebuild indexes | Done for all three strategies. |
| 5. Smoke run | Done on 8 questions (2 per tier). 0 null scores. |
| 6. Full run | Done. 80 questions per strategy. 0 null scores. |
| 7. Hand check | Waiting for you. |

## 3. Judge

The judge scores five metrics. The plan had four. A fifth metric, `answer_correctness`, compares the generated answer with the gold answer. The four original metrics never make this comparison.

The judge reply is JSON. The reasoning comes first. The score comes last.

I found and fixed two judge faults:

- The first version lost the score key in 3 of 114 replies. The prompt now says that both keys are required.
- A later version lost the score key in 8 of 200 Vector replies. The retry now shows the model its own reply and asks again. The model gives the number. The code sets no default. Use `--rescore-null` to run it.

**Sanity test.** `eval/judge_sanity.py` scores perfect, wrong-answer, wrong-context and refusal cases built from the first 10 questions. 10 of 11 checks pass. The failing check is "perfect: all metrics >= 0.85". Context recall is 0.74 for the perfect case. Several multi-hop gold answers contain derived statements (for example a computed figure) that the short gold quotes do not spell out, so the judge marks them missing. The judge is not at fault. I did not change the threshold. Treat context recall on multi-hop questions with care.

### 3.1 Null scores in detail

**What a null score is.** The judge must return one number from 0.0 to 1.0 for each metric. A null score means the judge gave no valid number. The row keeps `score: null` and a reason text. The code never fills the gap with a default value (for example 0 or 0.5). A default would hide the fault and bias the mean.

**What the fault was.** The judge sometimes returned JSON without the `score` key. The `reasoning` key was present. The `score` key was missing. I did not find the root cause inside the model. I only measured the symptom and fixed it.

**When it happened.**

| Version | Fault rate | Fix |
|---|---|---|
| First judge version (sanity test) | 3 of 114 replies | The prompt now says: "Both keys are required." The score key goes last, after the reasoning. |
| Second judge version (first Vector full run) | 8 of 200 replies | The retry now shows the model its own reply. It asks for the number again. |

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

**Final state.** All three full runs have 0 null scores. The means in section 6 use all 80 questions (tier G is scored on correctness only).

## 4. Findings per strategy

### Vector RAG

Old context precision was low. The cause was page headers and footers. Header chunks (for example "Policy name ... F&U Dated") won the top ranks, because every question names its policy.

The fix removes lines that appear on 40% or more of the pages of one PDF. The same removal applies to Graph RAG and PageIndex input. This keeps the comparison fair.

Retrieval on the 80 questions: strict all@5 is 0.59 (every gold passage is in the first 5), all@10 is 0.78. Cross-document questions (tier F) score 0.29. The top of a short list cannot hold two documents.

### PageIndex RAG

- The index covers 15 PDFs with 0 failures.
- The build used about 520 calls. It used 0.96M prompt tokens and 0.25M completion tokens. This is 2 times the first estimate.
- Tree structure depends on the model. 7 of 14 trees match the old Ollama trees. The other trees have 1 to 33 more nodes.
- An LLM router picks the document. The router never sees the gold file.
- Strict all@5 is 0.90, but the retrieved context is large: mean 63.6k characters. Vector returns 13k.
- Budget metric (gold inside the first 8000 characters): Vector 0.59, PageIndex 0.40.
- The generator gets at most 5 nodes. This cut was in the code for every run.

### Graph RAG

- Entity extraction (NER) changed from spaCy and GLiNER to DeepSeek flash. The code is in `graph_rag/src/llm_ner.py`.
- The graph has 1878 chunks, 11,619 entities, and 222k edges. The build used about 0.65M prompt tokens and 0.37M completion tokens.
- A first design matched entities by exact string and merged raw scores from different scales. It retrieved poorly, so I made one improvement pass: embedding-based entity linking, idf weighting, and RRF merge. No tuning used the frozen set. Only the improved design is kept.
- Retrieval all@5 on the 80 questions: vector only 0.59, graph only 0.06, hybrid (graph plus vector, RRF) 0.49. The graph signal ranks gold passages worse than vector alone. Answer quality still matches Vector, because the generator sees 15 chunks.
- The chunks and vectors are shared with Vector RAG.

## 5. Decision you made

Generation has no context cap. Each strategy passes its natural retrieval unit. The paper reports accuracy and cost together. Cost means context size and tokens. The paper also reports the 8000-character retrieval-budget row.

Warning for the paper: large units raise context precision. Compare this metric with care.

## 6. Results (80 questions)

0 null scores. Vector and Graph retrieve 15 chunks (`TOP_K` in `eval/config.py`). PageIndex returns at most 5 nodes.

| Metric | Vector | PageIndex | Graph |
|---|---|---|---|
| Answer correctness | 0.80 [0.73, 0.86] | 0.82 [0.75, 0.89] | 0.78 [0.71, 0.85] |
| Faithfulness | 0.92 | 0.92 | 0.93 |
| Answer relevancy | 0.90 | 0.88 | 0.88 |
| Context recall | 0.84 | 0.91 | 0.83 |
| Context precision | 0.21 | 0.77 | 0.21 |
| Context size (characters) | 13.3k | 63.6k | 13.3k |

Correctness by tier (E multi-hop, F cross-document, G unanswerable, H paraphrased): Vector 0.78, 0.78, 0.92, 0.78. PageIndex 0.72, 0.89, 1.00, 0.75. Graph 0.76, 0.76, 1.00, 0.71.

**Main conclusion.** No paired difference in correctness is significant. PageIndex against Vector: +0.021, interval [-0.044, +0.082], p=0.56. Graph against Vector: -0.017, p=0.49. PageIndex has higher context recall (+0.062, p=0.11) and the best tier F score (0.89), at about 4.8 times the context. Only the PageIndex context precision gain is significant, and the cause is unit size.

The paper table is `eval/results/final_comparison.csv`: four metrics (context precision left out), one column per strategy.

## 7. Cost

| Stage | Tokens |
|---|---|
| v2 run, PageIndex | about 4.9M prompt (about 61k per question) |
| v2 run, Graph | about 1.1M prompt |
| PageIndex judge | 3.12M prompt |

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
- `eval/report.py`: run `python -m eval.report` to write `eval/results/summary.md`, `handcheck.md` and `final_comparison.csv`.
- `vector_rag/`, `pageindex_rag/`, `graph_rag/`: each has `build_index.py` or `index.py`, `retrieve.py`, and `check.py`.
- `eval/results/*_rows.json`: per-question results.

## 10. The eval set

80 questions, frozen (sha256 982a01e2…c14). Tiers: E multi-hop (24), F cross-document (24), G unanswerable (12), H paraphrased (20). Every gold quote is verified (`python -m eval.verify_quotes`, 0 problems). In tier G only answer correctness is scored. The other four metrics are marked not applicable. Reason: a correct refusal scored 0.33 on faithfulness.

**Limits for the paper.**
- An LLM wrote the questions. The quotes are verified, the gold answers are not all reviewed by a person.
- Five questions (q59, q64, q85, q104, q119) are ambiguous or have 3 sub-questions. The file is frozen.
- Tier G has 12 questions. Treat its numbers as indications.
- The design choices (header removal, Graph RAG entity linking) were made while looking at an earlier question set that is no longer part of this project. State this in the paper.

**Report.** `python -m eval.report` writes `summary.md`, `handcheck.md` and `final_comparison.csv`.

Full tables are in `COMPARISON.md`.

## 11. Next steps

1. **You:** hand check the 15 rows in `eval/results/handcheck.md` (Phase 7).
2. **Needs your approval:** remove the old Ollama files:
   - `pageindex_rag/cache/ollama/*`
   - `vector_rag/cache/vector_rag_{eval_results,outputs}.json`
   - the old notebooks
3. Done: the "99%" claim is a vendor-reported 98.7% on FinanceBench for Mafin 2.5. See `COMPARISON.md` section 10.
4. **Claude:** run `graphify update .`.
5. **You:** commit and merge. Claude gives the commit message as text only.
