# Comparison of the Retrieval Strategies

Data: 80 questions (q41 to q120). Judge: `deepseek-flash`, temperature 0. All runs have 0 null scores.
Source files: `eval/results/summary.md`, `eval/results/final_comparison.csv`, `eval/results/retrieval_*_rag_v2.json`.

## 1. Strategies

| Strategy | Retrieval unit | How it finds the unit |
|---|---|---|
| Vector RAG | Chunk (1000 characters, 150 overlap) | Embedding similarity. Top 15. |
| PageIndex RAG | Tree node (a section of a document) | An LLM router picks up to 3 documents. An LLM picks up to 3 nodes in each. At most 5 nodes go to the generator. |
| Graph RAG | Same chunks as Vector | Query entities link to graph entities by embedding similarity, weighted by idf. Graph and vector rankings merge with RRF. Top 15. |

All strategies use the same cleaned text. The cleaning removes lines that appear on 40% or more of the pages of one PDF.

## 2. The eval set

Tiers: E multi-hop in one policy (24), F cross-document (24), G unanswerable (12), H paraphrased (20). The set is frozen (sha256 982a01e2…c14). Every gold quote passes `eval/verify_quotes.py` (0 problems, all 15 PDFs used). For tier G only answer correctness is scored. The other four metrics are marked not applicable, because a correct refusal scores low on faithfulness.

## 3. Answer quality

Mean score and 95% bootstrap interval. The best value in each row is in bold.

| Metric | Vector | PageIndex | Graph |
|---|---|---|---|
| **Answer correctness** | 0.80 [0.73, 0.86] | **0.82** [0.75, 0.89] | 0.78 [0.71, 0.85] |
| Faithfulness | 0.92 [0.86, 0.97] | 0.92 [0.86, 0.98] | **0.93** [0.88, 0.98] |
| Answer relevancy | **0.90** [0.83, 0.95] | 0.88 [0.80, 0.95] | 0.88 [0.81, 0.94] |
| Context recall | 0.84 [0.79, 0.90] | **0.91** [0.86, 0.95] | 0.83 [0.77, 0.89] |
| Context precision | 0.21 | 0.77 | 0.21 |
| Context chars | 13.3k | 63.6k | 13.3k |

Do not use context precision to compare strategies. It rewards large retrieval units. PageIndex scores high because its units are large.

The paper table is `eval/results/final_comparison.csv` (four metrics, without context precision).

### Correctness by tier

| Tier | Questions | Vector | PageIndex | Graph |
|---|---|---|---|---|
| E: multi-hop in one policy | 24 | **0.78** | 0.72 | 0.76 |
| F: cross-document | 24 | 0.78 | **0.89** | 0.76 |
| G: unanswerable | 12 | 0.92 | **1.00** | **1.00** |
| H: paraphrased | 20 | **0.78** | 0.75 | 0.71 |

Tier G has 12 questions. Treat its numbers as indications only. No tier-level test was run.

## 4. Statistical test

Paired difference against Vector (same 80 questions). A positive value means the strategy scores higher than Vector.

| Strategy | Metric | Difference | 95% interval | p |
|---|---|---|---|---|
| PageIndex | answer correctness | +0.021 | [-0.044, +0.082] | 0.555 |
| PageIndex | context recall | +0.062 | [+0.001, +0.122] | 0.111 |
| PageIndex | context precision | +0.563 | [+0.495, +0.628] | 0.000 |
| Graph | answer correctness | -0.017 | [-0.065, +0.032] | 0.485 |
| Graph | context recall | -0.014 | [-0.064, +0.039] | 0.549 |

**No difference in correctness is significant.** The PageIndex recall interval excludes zero, but the Wilcoxon test gives p=0.11, so treat it as a suggestion. The PageIndex precision gain is significant and is a unit-size effect.

## 5. Retrieval only

Strict all@5: every gold passage is in the first 5 retrieved items. 68 questions have gold passages (tier G has none). The Vector and Graph lists come from the top-15 retriever, and the metric reads the first 5.

| Measure | Vector | PageIndex | Graph (hybrid) | Graph (graph only) |
|---|---|---|---|---|
| Strict all@5 | 0.59 | 0.90 | 0.49 | 0.06 |
| Strict all@10 | 0.78 | 0.90 | 0.68 | 0.18 |
| Any@5 | 0.90 | 0.99 | 0.87 | 0.46 |
| Gold inside first 8000 characters | 0.59 | 0.40 | 0.49 | 0.06 |
| Cross-document (tier F), all@5 | 0.29 | 0.88 | 0.25 | 0.04 |

- Graph RAG ranks gold passages worse than Vector alone (0.49 against 0.59). The graph signal adds noise on this corpus. The answer scores still match Vector, because the generator sees 15 chunks.
- PageIndex finds every gold passage in 90% of questions because it returns large units. Under the 8000-character budget it scores 0.40 and Vector scores 0.59.
- PageIndex all@10 equals all@5 because it returns at most 5 nodes.

## 6. Cost

| Measure | Vector | PageIndex | Graph |
|---|---|---|---|
| Context per question (characters) | 13.3k | 63.6k mean | 13.3k |
| Index build | Embeddings only | 0.96M prompt and 0.25M completion tokens | 0.65M prompt and 0.37M completion tokens |
| Run, retrieval and generation | not logged | 1.77M prompt tokens | 0.27M prompt tokens |
| Run, judge | not logged | 3.12M prompt tokens | 0.85M prompt tokens |

PageIndex passes about 4.8 times more context than the chunk strategies. The PageIndex judge is the largest cost, because the judge reads the context again in three metrics.

## 7. Does PageIndex win because of context size?

No clear win. PageIndex leads by 0.02 correctness and 0.07 recall at 4.8 times the context. Neither is significant. Its one visible strength is cross-document questions (tier F, 0.89 against 0.78), where it also retrieves better (all@5 0.88 against 0.29). Chunk strategies cannot hold two documents in a short list. A larger top-k for Vector may close this gap. This is not tested.

The paper may say: "PageIndex matches Vector at 4.8 times the context cost, with a possible edge on cross-document questions." Do not write that the method is better.

## 8. Which metric to report

1. **Primary: answer correctness.** It is the only metric that compares the answer with the gold answer.
2. **Secondary: faithfulness and context recall.**
3. **Always add cost:** context characters and prompt tokens.
4. **Retrieval rows:** strict all@5 and the 8000-character budget row.
5. **Context precision:** report it with the unit-size warning. Do not use it to rank strategies.

## 9. Claims the data supports

1. No strategy is significantly better in correctness. The three scores are within 0.04.
2. PageIndex uses about 4.8 times more context than Vector and Graph.
3. PageIndex has the highest context recall (0.91 against 0.84) and the highest score in tier F (0.89). Both are suggestions, not significant results.
4. Graph RAG did not improve answer quality over Vector and ranks gold passages worse (all@5 0.49 against 0.59). Its build cost is 0.65M prompt and 0.37M completion tokens.
5. Vector is the cheapest strategy and stays competitive.

## 10. Limits

- The set has 80 questions. Intervals are wide.
- An LLM wrote the questions. I checked every gold quote against the PDFs, but a person has not reviewed all gold answers.
- Five questions are ambiguous or have 3 sub-questions (q59, q64, q85, q104, q119). The file is frozen, so they stay.
- Tier G has 12 questions.
- One judge scores all runs. The judge is an LLM. You hand-check 15 rows to test it (`eval/results/handcheck.md`). This is not done yet.
- The judge sanity test passes 10 of 11 checks. The failing check is "perfect: all metrics >= 0.85". Context recall is 0.74 for the first 10 questions, because several multi-hop gold answers contain derived statements that the short gold quotes do not spell out. The judge is not at fault, but this limits how far recall can be trusted on multi-hop questions.
- The design choices (header and footer removal at 40%, Graph RAG entity linking) were made while looking at an earlier question set that is no longer part of this project. State this in the paper.
- PageIndex returns at most 5 nodes to the generator. This cut was in the code for every run.
- PageIndex trees depend on the model. 7 of 14 trees match the old Ollama trees. The others have 1 to 33 more nodes.
- The page-level retrieval metric is unfair to PageIndex, because a node records only its start page.
- The PageIndex "99%" claim is really 98.7% on FinanceBench. The vendor (VectifyAI) reports it for its commercial system Mafin 2.5, which is built on PageIndex, and nobody has replicated it independently (sources: github.com/VectifyAI/PageIndex README, github.com/VectifyAI/Mafin2.5-FinanceBench). If the paper cites it, call it vendor-reported, and do not treat it as a figure for the open-source PageIndex used here.
