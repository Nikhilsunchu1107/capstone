# Results summary (n=80 questions, judge = deepseek-flash, temperature 0)

## Mean score with 95% bootstrap CI

| Strategy | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness | context chars |
|---|---|---|---|---|---|---|
| Vector RAG | 0.92 [0.86, 0.97] | 0.90 [0.83, 0.95] | 0.21 [0.18, 0.25] | 0.84 [0.79, 0.90] | 0.80 [0.73, 0.86] | 13344 |
| PageIndex RAG | 0.92 [0.86, 0.98] | 0.88 [0.80, 0.95] | 0.77 [0.72, 0.83] | 0.91 [0.86, 0.95] | 0.82 [0.75, 0.89] | 63558 |
| Graph RAG | 0.93 [0.88, 0.98] | 0.88 [0.81, 0.94] | 0.21 [0.18, 0.24] | 0.83 [0.77, 0.89] | 0.78 [0.71, 0.85] | 13342 |

## Answer correctness by tier (mean)

| Strategy | E_multi_hop (n=24) | F_cross_document (n=24) | G_unanswerable (n=12) | H_paraphrased (n=20) |
|---|---|---|---|---|
| Vector RAG | 0.78 | 0.78 | 0.92 | 0.78 |
| PageIndex RAG | 0.72 | 0.89 | 1.00 | 0.75 |
| Graph RAG | 0.76 | 0.76 | 1.00 | 0.71 |

## Paired differences vs Vector RAG (same questions)

Mean difference [95% bootstrap CI], Wilcoxon p. Positive = strategy higher than Vector RAG.

| Strategy | Metric | Diff | 95% CI | p |
|---|---|---|---|---|
| PageIndex RAG | faithfulness | +0.003 | [-0.050, +0.058] | 0.849 |
| PageIndex RAG | answer_relevancy | -0.015 | [-0.082, +0.049] | 0.769 |
| PageIndex RAG | context_precision | +0.563 | [+0.495, +0.628] | 0.000 |
| PageIndex RAG | context_recall | +0.062 | [+0.001, +0.122] | 0.111 |
| PageIndex RAG | answer_correctness | +0.021 | [-0.044, +0.082] | 0.555 |
| Graph RAG | faithfulness | +0.013 | [-0.047, +0.074] | 0.828 |
| Graph RAG | answer_relevancy | -0.018 | [-0.075, +0.043] | 0.319 |
| Graph RAG | context_precision | -0.004 | [-0.029, +0.021] | 0.482 |
| Graph RAG | context_recall | -0.014 | [-0.064, +0.039] | 0.549 |
| Graph RAG | answer_correctness | -0.017 | [-0.065, +0.032] | 0.485 |

Null scores: Vector RAG=0, PageIndex RAG=0, Graph RAG=0

Note: context precision rewards large retrieval units. Do not compare it across strategies as retrieval quality.
Note: for unanswerable questions (tier G) only answer correctness is scored. The other four metrics are not applicable and are left out of the means. A correct refusal scores low on faithfulness because the context cannot support it.