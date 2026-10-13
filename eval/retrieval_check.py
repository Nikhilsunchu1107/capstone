"""Phase 3a: retrieval-only metrics against the frozen eval set. No LLM call.

A retrieved item is a dict with "text", "file", "page" (page may be None).
Strict hit : the item text contains the start or end of a gold quote fragment.
Page hit   : the item comes from the gold file and page (lenient). Unfair to
             strategies whose items span many pages and carry only a start page.
Budget hit : strict hit, but only inside the first RETRIEVAL_BUDGET_CHARS characters of the
             retrieved text, which is all the generator will see. This is the fair metric.
"""

import json
import re
import unicodedata

from eval import config

KS = (1, 3, 5, 10)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).lower())


def _anchors(quote: str) -> list[str]:
    out = []
    for frag in quote.split(" ... "):
        n = norm(frag)
        if len(n) >= 20:
            out += [n[:30], n[-30:]]
    return out


def first_hit_rank(retrieved: list[dict], ev: dict) -> tuple[int | None, int | None]:
    """Return (strict_rank, page_rank), 1-based, or None."""
    anchors = _anchors(ev["quote"])
    strict = page = None
    for r, item in enumerate(retrieved, 1):
        if item["file"] != ev["filename"]:
            continue
        if page is None and item.get("page") == ev["page"]:
            page = r
        if strict is None and any(a in norm(item["text"]) for a in anchors):
            strict = r
    return strict, page


def budget_rank(retrieved: list[dict], ev: dict, budget: int = config.RETRIEVAL_BUDGET_CHARS) -> int | None:
    """Rank of the first hit when the generator only sees the first `budget` characters.

    Items are taken in rank order and the text is cut at the budget, as in generation.
    """
    anchors = _anchors(ev["quote"])
    left = budget
    for r, item in enumerate(retrieved, 1):
        if left <= 0:
            break
        text = item["text"][:left]
        left -= len(text)
        if item["file"] == ev["filename"] and any(a in norm(text) for a in anchors):
            return r
    return None


def evaluate(retrieve_fn, top_k: int = max(KS)) -> dict:
    qs = json.loads(config.EVAL_SET.read_text())
    rows = []
    for q in qs:
        got = retrieve_fn(q["question"], top_k)
        ranks = [first_hit_rank(got, ev) for ev in q["evidence"]]
        rows.append({"id": q["eval_id"], "tier": q["tier"], "question": q["question"],
                     "n_evidence": len(ranks), "strict": [r[0] for r in ranks], "page": [r[1] for r in ranks],
                     "budget": [budget_rank(got, ev) for ev in q["evidence"]],
                     "chars": sum(len(g["text"]) for g in got[:5]),
                     "top": [(g["file"][:25], g.get("page"), round(g.get("score", 0), 3)) for g in got[:5]]})
    return {"rows": rows, "summary": summarize(rows)}


def _agg(rows: list[dict], key: str) -> dict:
    out = {}
    for k in KS:
        any_ = [any(r is not None and r <= k for r in row[key]) for row in rows]
        all_ = [all(r is not None and r <= k for r in row[key]) for row in rows]
        out[f"any@{k}"] = sum(any_) / len(rows)
        out[f"all@{k}"] = sum(all_) / len(rows)
    rr = []
    for row in rows:
        vals = [1 / r for r in row[key] if r is not None]
        rr.append(sum(vals) / len(row[key]))
    out["mrr"] = sum(rr) / len(rr)
    return out


def summarize(rows: list[dict]) -> dict:
    s = {"n": len(rows), "mean_chars_top5": sum(r["chars"] for r in rows) / len(rows), "strict": _agg(rows, "strict"), "page": _agg(rows, "page"), "budget": _agg(rows, "budget"), "by_tier": {}}
    for t in sorted({r["tier"] for r in rows}):
        sub = [r for r in rows if r["tier"] == t]
        s["by_tier"][t] = {"n": len(sub), "strict": _agg(sub, "strict"), "budget": _agg(sub, "budget")}
    return s


def print_summary(name: str, res: dict) -> None:
    s = res["summary"]
    print(f"\n== {name}  (n={s['n']}) ==")
    print(f"mean text returned in top 5: {s['mean_chars_top5']:.0f} chars")
    for key in ("strict", "budget", "page"):
        a = s[key]
        print(f"{key:6s} " + "  ".join(f"all@{k}={a[f'all@{k}']:.2f}" for k in KS) + f"  any@5={a['any@5']:.2f}  MRR={a['mrr']:.2f}")
    for t, v in s["by_tier"].items():
        a, b = v["strict"], v["budget"]
        print(f"  {t:18s} n={v['n']:2d} strict all@5={a['all@5']:.2f} MRR={a['mrr']:.2f} | budget all@5={b['all@5']:.2f}")
    miss = [r for r in res["rows"] if not all(x is not None and x <= 5 for x in r["budget"])]
    print(f"misses at k=5 (budget): {len(miss)}")
    for r in miss:
        print(f"  {r['id']} {r['tier'][:1]} budget={r['budget']} strict={r['strict']} | {r['question'][:90]}")
