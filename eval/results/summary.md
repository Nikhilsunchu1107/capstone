# Results summary (n=40 questions, judge = deepseek-flash, temperature 0)

## Mean score with 95% bootstrap CI

| Strategy | faithfulness | answer_relevancy | context_precision | context_recall | answer_correctness | context chars |
|---|---|---|---|---|---|---|
| Vector RAG | 0.91 [0.82, 0.99] | 0.88 [0.78, 0.96] | 0.27 [0.22, 0.32] | 0.95 [0.89, 1.00] | 0.83 [0.71, 0.93] | 4463 |
| PageIndex RAG | 0.94 [0.87, 1.00] | 0.93 [0.86, 0.99] | 0.75 [0.67, 0.83] | 0.99 [0.96, 1.00] | 0.91 [0.82, 0.99] | 62920 |
| Graph RAG (original) | 0.96 [0.90, 1.00] | 0.82 [0.71, 0.92] | 0.21 [0.17, 0.28] | 0.80 [0.67, 0.90] | 0.75 [0.62, 0.87] | 4383 |
| Graph RAG (linked) | 0.90 [0.80, 0.97] | 0.86 [0.75, 0.95] | 0.26 [0.21, 0.31] | 0.84 [0.72, 0.94] | 0.78 [0.66, 0.90] | 4417 |

## Answer correctness by tier (mean)

| Strategy | A_single_fact (n=16) | B_multi_part (n=12) | C_apply_rule (n=6) | D_cross_document (n=6) |
|---|---|---|---|---|
| Vector RAG | 0.88 | 1.00 | 0.71 | 0.47 |
| PageIndex RAG | 1.00 | 0.96 | 0.67 | 0.83 |
| Graph RAG (original) | 0.81 | 0.79 | 0.67 | 0.57 |
| Graph RAG (linked) | 0.88 | 0.83 | 0.67 | 0.56 |

## Paired differences vs Vector RAG (same questions)

Mean difference [95% bootstrap CI], Wilcoxon p. Positive = strategy higher than Vector RAG.

| Strategy | Metric | Diff | 95% CI | p |
|---|---|---|---|---|
| PageIndex RAG | faithfulness | +0.029 | [-0.075, +0.138] | 0.605 |
| PageIndex RAG | answer_relevancy | +0.055 | [-0.037, +0.152] | 0.418 |
| PageIndex RAG | context_precision | +0.486 | [+0.394, +0.575] | 0.000 |
| PageIndex RAG | context_recall | +0.037 | [-0.025, +0.113] | 0.564 |
| PageIndex RAG | answer_correctness | +0.085 | [-0.019, +0.192] | 0.202 |
| Graph RAG (original) | faithfulness | +0.050 | [+0.000, +0.125] | 0.551 |
| Graph RAG (original) | answer_relevancy | -0.057 | [-0.128, -0.002] | 0.265 |
| Graph RAG (original) | context_precision | -0.050 | [-0.080, -0.020] | 0.021 |
| Graph RAG (original) | context_recall | -0.154 | [-0.267, -0.058] | 0.036 |
| Graph RAG (original) | answer_correctness | -0.079 | [-0.179, +0.007] | 0.285 |
| Graph RAG (linked) | faithfulness | -0.013 | [-0.087, +0.062] | 0.777 |
| Graph RAG (linked) | answer_relevancy | -0.020 | [-0.125, +0.080] | 0.960 |
| Graph RAG (linked) | context_precision | -0.005 | [-0.050, +0.045] | 0.781 |
| Graph RAG (linked) | context_recall | -0.113 | [-0.225, -0.013] | 0.259 |
| Graph RAG (linked) | answer_correctness | -0.044 | [-0.150, +0.062] | 0.445 |

Null scores: Vector RAG=0, PageIndex RAG=0, Graph RAG (original)=0, Graph RAG (linked)=0

Note: context precision rewards large retrieval units. Do not compare it across strategies as retrieval quality.