# Phase 3 results explained

This file explains the Vector RAG retrieval check in simple words.
It uses controlled English (ASD-STE100 style). One term has one meaning.

## 1. Words used in this file

| Term | Meaning |
|---|---|
| Chunk | A piece of text from a PDF. Each chunk is about 1000 characters long. |
| Question | One of the 40 frozen questions. |
| Gold passage | The place in the PDF that contains the answer. We wrote it down for each question. |
| Search | The step that finds chunks for a question. Vector RAG does this with embeddings. |
| Top 5 | The 5 chunks that the search ranks highest. Rank 1 is the best chunk. |
| Hit | A chunk in the results that contains the gold passage. |

## 2. What the check does

The check tests only the search step. It does not use DeepSeek. It does not write answers.

For each of the 40 questions, it does these steps:

1. It sends the question to the search.
2. It gets a ranked list of chunks.
3. It looks for the gold passage in that list.
4. It records the rank of the first hit.

If the search does not find the gold passage, the answer step cannot write a correct answer.
So a bad search gives bad scores later, even when the answer step is good.

## 3. The numbers

### 3.1 The letter k

"k" is the number of results that we look at.
"@5" means "look at the top 5 results only".
"@10" means "look at the top 10 results".

### 3.2 any@5

any@5 is the share of questions where **at least one** gold passage is in the top 5.

Example: A question has 2 gold passages. The top 5 contains 1 of them. This question counts for any@5.

### 3.3 all@5

all@5 is the share of questions where **every** gold passage is in the top 5.

Example: A question has 2 gold passages. The top 5 contains 1 of them. This question does not count for all@5.

Most questions have only 1 gold passage. For these questions, any@5 and all@5 are the same.
They differ only for the 6 cross-document questions (tier D), which have 2 gold passages.

all@5 is the stricter number. Use it as the main number.

### 3.4 MRR

MRR means Mean Reciprocal Rank. It tells you how **high** in the list the gold passage is.

For each question, the score is 1 divided by the rank of the first hit:

| Rank of the first hit | Score |
|---|---|
| 1 | 1.00 |
| 2 | 0.50 |
| 3 | 0.33 |
| 5 | 0.20 |
| 10 | 0.10 |
| Not found | 0.00 |

MRR is the average of these scores over all questions.

- MRR near 1.0 means the gold passage is usually at rank 1.
- MRR near 0.5 means the gold passage is usually at rank 2.
- all@5 does not show this. A hit at rank 1 and a hit at rank 5 both count as one hit.

### 3.5 Worked example

Three questions, each with one gold passage:

| Question | Rank of the gold passage | In top 5? | Reciprocal rank |
|---|---|---|---|
| Question 1 | 1 | Yes | 1.00 |
| Question 2 | 4 | Yes | 0.25 |
| Question 3 | 9 | No | 0.11 |

- all@5 = 2 of 3 = **0.67**
- MRR = (1.00 + 0.25 + 0.11) / 3 = **0.45**

## 4. The results

| Setup | all@5 | any@5 | MRR |
|---|---|---|---|
| Raw pages | 0.70 | 0.72 | 0.58 |
| Page headers and footers removed | **0.85** | 0.93 | 0.75 |

How to read the second row:

- all@5 = 0.85. For 34 of 40 questions, the search puts every gold passage in the top 5.
- any@5 = 0.93. For 37 of 40 questions, the search puts at least one gold passage in the top 5.
- MRR = 0.75. On average, the first gold passage is between rank 1 and rank 2.

## 5. Why the first row was bad

Every page of a PDF starts with a header. A header looks like this:

`HDFC Life Smart Pension Plus (UIN - 101N173V13) – Individual Policy Bond … F&U Dated:`

Each of our questions names its policy. So the header chunk matches the question well.
The search ranked these header chunks at rank 1. They contain no answer.
The real clause was pushed down to rank 21.

The fix: remove each line that appears on 40% or more of the pages in one PDF.
This removes headers and footers. It does not use the questions, so it is not tuned to them.
It changes the number of chunks from 2096 to 1878.

This is probably the main reason for the low context precision and context recall in your old results.

## 6. Results by question type (after the fix)

| Tier | Question type | Questions | all@5 |
|---|---|---|---|
| A | Single fact | 16 | 0.88 |
| B | Multi-part answer | 12 | 1.00 |
| C | Apply a rule to a case | 6 | 0.83 |
| D | Needs 2 documents | 6 | 0.50 |

Tier D is low for a structural reason.
A top 5 list holds the best 5 chunks from all 15 PDFs together.
For a question about two policies, one policy often takes most of the 5 places.

Tier D has only 6 questions. Do not draw strong conclusions from one tier.

## 7. What this means for the project

1. The search step is now good enough to test the full pipeline.
2. We must apply the same header removal to PageIndex and Graph RAG. Then the comparison is fair.
3. The paper should show both rows (raw and cleaned). The change shows the effect of preprocessing.
4. For tier D, a later step can search each named policy on its own. We have not done this.

## 8. Next step

Run the same check on PageIndex and Graph RAG.
PageIndex must build its document trees again with DeepSeek flash. This costs tokens.
I will run a test on 1 PDF first and report the cost before I run all 15 PDFs.
