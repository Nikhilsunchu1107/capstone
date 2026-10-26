# Phase 3 results explained

This file explains the Vector RAG retrieval check in simple words.
It uses controlled English (ASD-STE100 style). One term has one meaning.

## 1. Words used in this file

| Term | Meaning |
|---|---|
| Chunk | A piece of text from a PDF. Each chunk is about 1000 characters long. |
| Question | One of the 80 frozen questions. |
| Gold passage | The place in the PDF that contains the answer. We wrote it down for each question. |
| Search | The step that finds chunks for a question. Vector RAG does this with embeddings. |
| Top 5 | The 5 chunks that the search ranks highest. Rank 1 is the best chunk. |
| Hit | A chunk in the results that contains the gold passage. |

## 2. What the check does

The check tests only the search step. It does not use DeepSeek. It does not write answers.

For each of the 68 questions that have a gold passage, it does these steps. (The 12 unanswerable questions have none.)

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
They differ for questions that need more than one passage, such as the cross-document questions (tier F).

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

Vector RAG, page headers and footers removed, 68 questions with a gold passage:

| all@5 | any@5 | MRR | all@10 |
|---|---|---|---|
| **0.59** | 0.90 | 0.54 | 0.78 |

How to read this row:

- all@5 = 0.59. For 40 of 68 questions, the search puts every gold passage in the top 5.
- any@5 = 0.90. For 61 of 68 questions, the search puts at least one gold passage in the top 5.
- MRR = 0.54. On average, the first gold passage is at about rank 2.
- all@10 = 0.78. A longer list finds more gold passages. The answer step uses 15 chunks for this reason.

## 5. Why the first row was bad

Every page of a PDF starts with a header. A header looks like this:

`HDFC Life Smart Pension Plus (UIN - 101N173V13) – Individual Policy Bond … F&U Dated:`

Each of our questions names its policy. So the header chunk matches the question well.
The search ranked these header chunks at rank 1. They contain no answer.
The real clause was pushed far down the list.

The fix: remove each line that appears on 40% or more of the pages in one PDF.
This removes headers and footers. It does not use the questions, so it is not tuned to them.
It changes the number of chunks from 2096 to 1878.

Without this step, the header chunks fill the top of the list.

## 6. Results by question type

| Tier | Question type | Questions | all@5 |
|---|---|---|---|
| E | Multi-hop in one policy | 24 | 0.71 |
| F | Needs 2 or 3 documents | 24 | 0.29 |
| H | Paraphrased question | 20 | 0.80 |

Tier F is low for a structural reason.
A top 5 list holds the best 5 chunks from all 15 PDFs together.
For a question about two policies, one policy often takes most of the 5 places.

Tier G (unanswerable) has no gold passage. It is not in this check.

## 7. What this means for the project

1. The search step is good enough to test the full pipeline. Its limit is cross-document questions.
2. The same header removal applies to PageIndex and Graph RAG. Then the comparison is fair.
3. For tier F, a later step can search each named policy on its own. We have not done this.

## 8. The other strategies

The same check runs for PageIndex and Graph RAG (`python -m pageindex_rag.check`, `python -m graph_rag.check`).
`COMPARISON.md` section 5 shows the three results side by side.
