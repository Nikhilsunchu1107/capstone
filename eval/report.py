"""Phase 7 report: tables with bootstrap CIs and paired tests, plus a hand-check sheet.

Run: python -m eval.report  ->  eval/results/summary.md, handcheck.md, final_comparison.csv
Also writes eval/results/handcheck.md and the paper table eval/results/final_comparison.csv.
"""

import csv
import json
import random

import numpy as np

from eval import config, judge

NAMES = {"vector": "Vector RAG", "pageindex": "PageIndex RAG", "graph": "Graph RAG"}
METRICS = judge.METRICS
RNG = np.random.default_rng(config.RANDOM_SEED)
B = 10000
HANDCHECK = ("vector", "pageindex", "graph")


def load() -> dict:
    """Result rows per strategy."""
    out = {}
    for k in NAMES:
        p = config.RESULTS_DIR / f"{k}_v2_rows.json"
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def vec(rows: dict, metric: str, ids: list[str]) -> np.ndarray:
    return np.array([rows[i]["scores"][metric]["score"] if rows[i]["scores"][metric]["score"] is not None else np.nan for i in ids])


def ci(x: np.ndarray) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    boots = RNG.choice(x, (B, len(x))).mean(axis=1)
    return x.mean(), *np.percentile(boots, [2.5, 97.5])


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.stats import wilcoxon

    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    boots = RNG.choice(d, (B, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="zsplit").pvalue)
    return d.mean(), lo, hi, p


def common_ids(data: dict) -> list[str]:
    return sorted(set.intersection(*[{i for i, r in rows.items() if "scores" in r} for rows in data.values()]))


def complete(data: dict, n: int) -> dict:
    """Keep only strategies that have scored rows for all n questions of the set."""
    return {k: rows for k, rows in data.items() if sum("scores" in r for r in rows.values()) == n}


def tier_mean(rows: dict, ids: list[str]) -> str:
    v = vec(rows, "answer_correctness", ids)
    return f"{np.nanmean(v):.2f}" if ids and not np.isnan(v).all() else "-"


def tables(data: dict, ids: list[str]) -> list[str]:
    md = ["## Mean score with 95% bootstrap CI\n"]
    md.append("| Strategy | " + " | ".join(METRICS) + " | context chars |")
    md.append("|---|" + "---|" * (len(METRICS) + 1))
    for k, rows in data.items():
        cells = []
        for m in METRICS:
            mean, lo, hi = ci(vec(rows, m, ids))
            cells.append(f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
        md.append(f"| {NAMES[k]} | " + " | ".join(cells) + f" | {int(np.mean([rows[i]['context_chars'] for i in ids]))} |")
    md.append("\n## Answer correctness by tier (mean)\n")
    first = next(iter(data.values()))
    tiers = sorted({first[i]["tier"] for i in ids})
    md.append("| Strategy | " + " | ".join(f"{t} (n={sum(first[i]['tier'] == t for i in ids)})" for t in tiers) + " |")
    md.append("|---|" + "---|" * len(tiers))
    for k, rows in data.items():
        md.append(f"| {NAMES[k]} | " + " | ".join(tier_mean(rows, [i for i in ids if rows[i]["tier"] == t]) for t in tiers) + " |")
    if "vector" in data and len(data) > 1:
        md.append("\n## Paired differences vs Vector RAG (same questions)\n")
        md.append("Mean difference [95% bootstrap CI], Wilcoxon p. Positive = strategy higher than Vector RAG.\n")
        md.append("| Strategy | Metric | Diff | 95% CI | p |")
        md.append("|---|---|---|---|---|")
        for k, rows in data.items():
            if k == "vector":
                continue
            for m in METRICS:
                d, lo, hi, p = paired(vec(rows, m, ids), vec(data["vector"], m, ids))
                md.append(f"| {NAMES[k]} | {m} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    md.append("\nNull scores: " + ", ".join(
        f"{NAMES[k]}={sum(judge.is_null(rows[i]['scores'][m]) for i in ids for m in METRICS)}" for k, rows in data.items()))
    return md


PAPER_METRICS = ("answer_correctness", "faithfulness", "answer_relevancy", "context_recall")


def write_csv(data: dict, name: str) -> None:
    """Paper table for one question set. Rows are metrics, columns are strategies, last column names the best mean.

    Context precision is left out on purpose: it rewards large retrieval units.
    """
    ids = common_ids(data)
    means = {m: {k: float(np.nanmean(vec(rows, m, ids))) for k, rows in data.items()} for m in PAPER_METRICS}
    with open(config.RESULTS_DIR / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Metric"] + [NAMES[k] for k in data] + ["Best"])
        for m in PAPER_METRICS:
            top = max(round(v, 2) for v in means[m].values())
            best = " / ".join(NAMES[k] for k, v in means[m].items() if round(v, 2) == top)
            w.writerow([m.replace("_", " ").capitalize()] + [f"{means[m][k]:.2f}" for k in data] + [best])


def handcheck(data: dict, ids: list[str], name: str) -> None:
    rnd = random.Random(config.RANDOM_SEED)
    hc = ["# Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n\n"]
    picks = [(k, i) for k in HANDCHECK if k in data for i in rnd.sample(ids, 5)]
    for n, (k, i) in enumerate(picks, 1):
        r = data[k][i]
        hc.append(f"## {n}. {NAMES[k]} / {i} ({r['tier']})\n**Q:** {r['question']}\n\n**Gold:** {r['ground_truth']}\n\n**Answer:** {r['answer']}\n")
        hc.append("Judge: " + ", ".join(f"{m}={r['scores'][m]['score']}" for m in METRICS) + "\n\n**Your verdict (answer correct? y/n):** \n")
    (config.RESULTS_DIR / name).write_text("\n".join(hc))


def main():
    data = complete(load(), len(json.loads(config.EVAL_SET.read_text())))
    ids = common_ids(data)
    md = [f"# Results summary (n={len(ids)} questions, judge = {config.LLM_MODEL}, temperature 0)\n"]
    md += tables(data, ids)
    md.append("\nNote: context precision rewards large retrieval units. Do not compare it across strategies as retrieval quality.")
    md.append("Note: for unanswerable questions (tier G) only answer correctness is scored. "
              "The other four metrics are not applicable and are left out of the means. "
              "A correct refusal scores low on faithfulness because the context cannot support it.")
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(md))
    write_csv(data, "final_comparison.csv")
    handcheck(data, ids, "handcheck.md")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
