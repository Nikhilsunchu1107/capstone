"""Phase 7 report: tables with bootstrap CIs and paired tests, plus a hand-check sheet.

Run: python -m eval.report  ->  eval/results/summary.md, eval/results/handcheck.md
"""

import json
import random

import numpy as np

from eval import config, judge

NAMES = {"vector": "Vector RAG", "pageindex": "PageIndex RAG", "graph": "Graph RAG (original)", "graph_linked": "Graph RAG (linked)"}
METRICS = judge.METRICS
RNG = np.random.default_rng(config.RANDOM_SEED)
B = 10000


def load():
    out = {}
    for k in NAMES:
        p = config.RESULTS_DIR / f"{k}_rows.json"
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


def main():
    data = load()
    ids = sorted(set.intersection(*[set(r) for r in data.values()]))
    md = [f"# Results summary (n={len(ids)} questions, judge = {config.LLM_MODEL}, temperature 0)\n"]
    md.append("## Mean score with 95% bootstrap CI\n")
    md.append("| Strategy | " + " | ".join(METRICS) + " | context chars |")
    md.append("|---|" + "---|" * (len(METRICS) + 1))
    for k, rows in data.items():
        cells = []
        for m in METRICS:
            mean, lo, hi = ci(vec(rows, m, ids))
            cells.append(f"{mean:.2f} [{lo:.2f}, {hi:.2f}]")
        md.append(f"| {NAMES[k]} | " + " | ".join(cells) + f" | {int(np.mean([rows[i]['context_chars'] for i in ids]))} |")
    md.append("\n## Answer correctness by tier (mean)\n")
    tiers = sorted({data['vector'][i]['tier'] for i in ids})
    md.append("| Strategy | " + " | ".join(f"{t} (n={sum(data['vector'][i]['tier'] == t for i in ids)})" for t in tiers) + " |")
    md.append("|---|" + "---|" * len(tiers))
    for k, rows in data.items():
        md.append(f"| {NAMES[k]} | " + " | ".join(
            f"{np.nanmean(vec(rows, 'answer_correctness', [i for i in ids if rows[i]['tier'] == t])):.2f}" for t in tiers) + " |")
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
        f"{NAMES[k]}={sum(rows[i]['scores'][m]['score'] is None for i in ids for m in METRICS)}" for k, rows in data.items()))
    md.append("\nNote: context precision rewards large retrieval units. Do not compare it across strategies as retrieval quality.")
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(md))

    rnd = random.Random(config.RANDOM_SEED)
    hc = ["# Hand-check sheet (15 rows). For each row write: is the answer correct vs the gold? y/n\n"]
    picks = [(k, i) for k in ("vector", "pageindex", "graph_linked") if k in data for i in rnd.sample(ids, 5)]
    for n, (k, i) in enumerate(picks, 1):
        r = data[k][i]
        hc.append(f"## {n}. {NAMES[k]} / {i} ({r['tier']})\n**Q:** {r['question']}\n\n**Gold:** {r['ground_truth']}\n\n**Answer:** {r['answer']}\n")
        hc.append("Judge: " + ", ".join(f"{m}={r['scores'][m]['score']}" for m in METRICS) + "\n\n**Your verdict (answer correct? y/n):** \n")
    (config.RESULTS_DIR / "handcheck.md").write_text("\n".join(hc))
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
