"""Phase 2 gate: the judge must separate good from bad inputs. No RAG needed.

Run: python -m eval.judge_sanity
"""

import json
import statistics

from eval import config, judge, llm

N = 10


def main():
    qs = json.loads(config.EVAL_SET.read_text())[:N]
    gold_ctx = [[e["quote"] for e in q["evidence"]] for q in qs]
    cases = {"perfect": [], "wrong_answer": [], "wrong_context": []}
    for i, q in enumerate(qs):
        other = qs[(i + N // 2) % N]
        j = (i + N // 2) % N
        cases["perfect"].append({"question": q["question"], "answer": q["answer"], "ground_truth": q["answer"], "contexts": gold_ctx[i]})
        cases["wrong_answer"].append({"question": q["question"], "answer": other["answer"], "ground_truth": q["answer"], "contexts": gold_ctx[i]})
        cases["wrong_context"].append({"question": q["question"], "answer": q["answer"], "ground_truth": q["answer"], "contexts": gold_ctx[j]})

    mean, allscores, nulls, total = {}, [], 0, 0
    for name, rows in cases.items():
        res = judge.score_rows(rows)
        mean[name] = {}
        for m in judge.METRICS:
            vals = [r[m]["score"] for r in res]
            total += len(vals)
            nulls += sum(v is None for v in vals)
            vals = [v for v in vals if v is not None]
            allscores += vals
            mean[name][m] = statistics.mean(vals) if vals else float("nan")
        print(f"{name:14s} " + "  ".join(f"{m[:12]}={mean[name][m]:.2f}" for m in judge.METRICS))

    # unanswerable questions (tier G): only answer_correctness applies (see judge.NOT_APPLICABLE).
    # A first run showed faithfulness scores a correct refusal 0.33, so it is not used for tier G.
    gq = [q for q in json.loads(config.EVAL_SET.read_text()) if q["tier"].startswith("G")][:3]
    abstain = {}
    if gq:
        ctx = [[e["quote"] for e in qs[i]["evidence"]] for i in range(len(gq))]  # related policy text that lacks the answer
        good = [{"question": q["question"], "answer": q["answer"], "ground_truth": q["answer"], "contexts": ctx[i], "unanswerable": True} for i, q in enumerate(gq)]
        bad = [{**r, "answer": "The policy states that the amount is 45 days and the insurer pays 100% of the sum insured."} for r in good]
        for name, rows in (("abstain_ok", good), ("abstain_wrong", bad)):
            res = judge.score_rows(rows)
            abstain[name] = {m: statistics.mean(r[m]["score"] for r in res) for m in ("answer_correctness",)}
            nulls += sum(judge.is_null(r[m]) for r in res for m in judge.METRICS)
            total += len(res) * len(judge.METRICS)
            print(f"{name:14s} " + "  ".join(f"{m[:12]}={v:.2f}" for m, v in abstain[name].items()))

    p, wa, wc = mean["perfect"], mean["wrong_answer"], mean["wrong_context"]
    checks = {
        "perfect: all metrics >= 0.85": all(v >= 0.85 for v in p.values()),
        "wrong answer: relevancy <= 0.4": wa["answer_relevancy"] <= 0.4,
        "wrong answer: correctness <= 0.2": wa["answer_correctness"] <= 0.2,
        "wrong answer: faithfulness <= 0.3": wa["faithfulness"] <= 0.3,
        "wrong context: precision <= 0.2": wc["context_precision"] <= 0.2,
        "wrong context: recall <= 0.2": wc["context_recall"] <= 0.2,
        "wrong context: faithfulness <= 0.3": wc["faithfulness"] <= 0.3,
        "score std > 0": statistics.pstdev(allscores) > 0,
        f"null rate < 2% ({nulls}/{total})": nulls / total < 0.02,
    }
    if abstain:
        checks["abstention: correct refusal correctness >= 0.85"] = abstain["abstain_ok"]["answer_correctness"] >= 0.85
        checks["abstention: invented answer correctness <= 0.2"] = abstain["abstain_wrong"]["answer_correctness"] <= 0.2
    for k, ok in checks.items():
        print("PASS" if ok else "FAIL", k)
    print("usage:", llm.usage_totals())
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
