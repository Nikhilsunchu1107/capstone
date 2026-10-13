"""Page header and footer removal, shared by every strategy that reads PDFs."""

SHARE = 0.4
MIN_LINE = 15
MIN_PAGES = 5


def repeated_lines(pages: list[str]) -> set[str]:
    """Lines that appear on at least SHARE of a PDF's pages (headers, footers)."""
    if len(pages) < MIN_PAGES:
        return set()
    counts: dict[str, int] = {}
    for text in pages:
        for line in {l.strip() for l in text.splitlines() if len(l.strip()) >= MIN_LINE}:
            counts[line] = counts.get(line, 0) + 1
    return {l for l, n in counts.items() if n >= SHARE * len(pages)}


def strip_pages(pages: list[str]) -> list[str]:
    drop = repeated_lines(pages)
    return ["\n".join(l for l in p.splitlines() if l.strip() not in drop) for p in pages]
