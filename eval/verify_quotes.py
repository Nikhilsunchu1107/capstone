"""Check an eval set against the PDFs: every gold quote must be on its stated page.

Run: python -m eval.verify_quotes
Tier G (unanswerable) has no evidence. Its "absent" terms must not occur in the named PDF,
and must occur in the "trap" PDF (the look-alike that holds a real answer).
"""

import collections
import json

from langchain_community.document_loaders import PyMuPDFLoader

from eval import config, textclean
from eval.retrieval_check import norm

_pages: dict[str, list[str]] = {}


def pages(filename: str) -> list[str]:
    """Normalised text of each page, after the same header/footer removal the strategies use."""
    if filename not in _pages:
        raw = [p.page_content for p in PyMuPDFLoader(str(config.PDF_DIR / filename)).load()]
        _pages[filename] = [norm(p) for p in textclean.strip_pages(raw)]
    return _pages[filename]


def check(q: dict) -> list[str]:
    errs = []
    for ev in q["evidence"]:
        text = pages(ev["filename"])[ev["page"] - 1]
        for frag in ev["quote"].split(" ... "):
            if len(norm(frag)) < 20:
                errs.append(f"fragment shorter than 20 characters: {frag!r}")
            elif norm(frag) not in text:
                errs.append(f"not on {ev['filename'][:25]} p{ev['page']}: {frag[:70]!r}")
    if q["tier"].startswith("G"):
        named, trap = "".join(pages(q["named_file"])), "".join(pages(q["trap"]))
        for term in q["absent"]:
            if norm(term) in named:
                errs.append(f"term {term!r} occurs in the named PDF")
            if norm(term) not in trap:
                errs.append(f"term {term!r} does not occur in the trap PDF")
    elif not q["evidence"]:
        errs.append("no evidence")
    return errs


def main():
    qs = json.loads(config.EVAL_SET.read_text())
    bad = 0
    for q in qs:
        for e in check(q):
            bad += 1
            print(q["eval_id"], e)
    files = collections.Counter(f for q in qs for f in {e["filename"] for e in q["evidence"]} | {q.get("named_file")} if f)
    print(f"\n{len(qs)} questions, tiers {dict(collections.Counter(q['tier'] for q in qs))}")
    print(f"PDFs used: {len(files)} of {len(list(config.PDF_DIR.glob('*.pdf')))}, most used {files.most_common(1)}")
    print(f"ids unique: {len({q['eval_id'] for q in qs}) == len(qs)}, problems: {bad}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
