"""Run one strategy end to end on the frozen eval set: retrieve -> answer -> judge.

Run: python -m eval.run_strategy vector|pageindex|graph [--limit N] [--ids q41,q42]
Output: <strategy>_v2_rows.json.
Checkpoints after every question and every judge batch. Resumable.
"""

import argparse
import hashlib
import json
import time

from eval import config, generate, judge, llm


def _retriever(name: str):
    if name == "graph":
        from graph_rag import retrieve
    elif name == "vector":
        from vector_rag import retrieve
    elif name == "pageindex":
        from pageindex_rag import retrieve
    else:
        raise SystemExit(f"unknown strategy: {name}")
    return retrieve.retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", help="comma-separated eval ids (smoke runs)")
    ap.add_argument("--rescore-null", action="store_true", help="re-judge rows that have a null score")
    args = ap.parse_args()

    path = config.EVAL_SET
    if hashlib.sha256(path.read_bytes()).hexdigest() != config.EVAL_SET_SHA256:
        raise SystemExit(f"{path.name} does not match its frozen sha256. Do not edit a frozen eval set.")
    qs = json.loads(path.read_text())[: args.limit]
    if args.ids:
        qs = [q for q in qs if q["eval_id"] in args.ids.split(",")]
    out = config.RESULTS_DIR / f"{args.strategy}_v2_rows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text()) if out.exists() else {}
    retrieve = _retriever(args.strategy)

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        tmp.replace(out)

    if args.rescore_null:
        for r in rows.values():
            if "scores" in r and any(judge.is_null(v) for v in r["scores"].values()):
                del r["scores"]
    t0, u0 = time.time(), llm.usage_totals()
    for q in qs:
        if q["eval_id"] in rows and "answer" in rows[q["eval_id"]]:
            continue
        items = retrieve(q["question"])
        rows[q["eval_id"]] = {
            "tier": q["tier"], "question": q["question"], "ground_truth": q["answer"],
            "unanswerable": q["tier"].startswith("G"),
            "contexts": [i["text"] for i in items],
            "retrieved": [{"file": i["file"], "page": i.get("page")} for i in items],
            "context_chars": sum(len(i["text"]) for i in items),
            "answer": generate.answer(q["question"], items),
        }
        save()
        print(f"{q['eval_id']} {q['tier'][:1]} ctx={rows[q['eval_id']]['context_chars']:6d} chars  {rows[q['eval_id']]['answer'][:80]!r}", flush=True)
    u1 = llm.usage_totals()

    todo = [k for k in rows if k in {q["eval_id"] for q in qs} and "scores" not in rows[k]]
    if todo:
        scores = judge.score_rows([rows[k] for k in todo])
        for k, sc in zip(todo, scores):
            rows[k]["scores"] = sc
        save()
    u2 = llm.usage_totals()

    done = [rows[q["eval_id"]] for q in qs]
    print(f"\n== {args.strategy}: {len(done)} questions ==")
    for m in judge.METRICS:
        cells = [r["scores"][m] for r in done]
        ok = [c["score"] for c in cells if c["score"] is not None]
        print(f"{m:20s} mean={sum(ok) / max(1, len(ok)):.3f}  n={len(ok)}  null={sum(judge.is_null(c) for c in cells)}")
    print("mean context chars:", sum(r["context_chars"] for r in done) // len(done))
    print("tokens retrieval+generation:", {k: u1[k] - u0[k] for k in u1}, "| judge:", {k: u2[k] - u1[k] for k in u2})


if __name__ == "__main__":
    main()
