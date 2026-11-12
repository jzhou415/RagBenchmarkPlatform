"""Golden-page context: the text the faithfulness judge reads.

The golden set maps each question to graded chunks, but the chunk text
itself lives in the ingestion manifest, not the YAML. For agent eval we
do not need chunk granularity: the judge reads the full text of the
golden pages (relevance >= min_grade, grade 3 first), extracted from
the source PDF with PyMuPDF. Page numbers in the golden set are
1-indexed PDF pages -- verified against the ASVS PDF (the expected
answer's key phrases appear on the listed pages).

A char budget keeps the judge prompt bounded: when the pages exceed it,
grade-2 pages are dropped before grade-3 pages, and the remainder is
truncated with a note rather than silently cut.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MIN_GRADE = 2
DEFAULT_CHAR_BUDGET = 30_000


def load_pdf_pages(pdf_path: str | Path) -> list[str]:
    """Page texts, 1-indexed: pages[0] is PDF page 1.

    Uses PyMuPDF (`import pymupdf`). The platform parses with Docling,
    but agent eval only needs plain page text, and PyMuPDF is the
    lighter dependency for that job.
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    return [doc[i].get_text() for i in range(doc.page_count)]


def golden_pages(question: dict, min_grade: int = DEFAULT_MIN_GRADE
                 ) -> list[tuple[int, int]]:
    """Distinct (page, best_grade) pairs for a question, page-ascending.

    A page's grade is the best relevance among its ground-truth chunks.
    """
    best: dict[int, int] = {}
    for g in question.get("ground_truth", []):
        page = int(g["page"])
        rel = int(g["relevance"])
        if rel >= min_grade:
            best[page] = max(best.get(page, 0), rel)
    return sorted(best.items())


def golden_page_context(question: dict, pages: list[str],
                        min_grade: int = DEFAULT_MIN_GRADE,
                        char_budget: int = DEFAULT_CHAR_BUDGET) -> str:
    """Judge context: "[page N]"-marked text of the golden pages.

    Pages are added grade-3 first, then grade-2, so a budget cut drops
    the weakest evidence first. Returns "" when the question has no
    ground-truth pages at or above min_grade.
    """
    ranked = sorted(golden_pages(question, min_grade),
                    key=lambda pg: (-pg[1], pg[0]))
    chunks = []
    total = 0
    for page, _grade in ranked:
        if page < 1 or page > len(pages):
            continue
        text = f"[page {page}]\n{pages[page - 1].strip()}"
        if chunks and total + len(text) > char_budget:
            break
        chunks.append(text)
        total += len(text)
    if not chunks:
        return ""
    # Page-ascending order for the judge (build order was grade-first).
    chunks.sort(key=lambda c: int(c.split("]")[0][len("[page "):]))
    out = "\n\n---\n\n".join(chunks)
    if len(out) > char_budget:
        out = out[:char_budget] + "\n\n[... context truncated ...]"
    return out


def retrieved_context(cited_pages: dict, retrieved, char_budget: int
                      = DEFAULT_CHAR_BUDGET) -> str:
    """Judge context from what the agent actually retrieved.

    `retrieved` is the adapter's RetrievedDoc list (retrieval order);
    `cited_pages` maps doc index -> PDF page (from the citation-marker
    mapping, or {} when the markers did not parse). Snippets are taken
    cited-first, then the remaining retrieved docs in retrieval order,
    each marked "[retrieved doc i | page P]" (page omitted when
    unmapped), until the char budget fills. Returns "" when the agent
    retrieved nothing -- the runner then records faithfulness as
    not-run rather than judging against an empty context.
    """
    if not retrieved:
        return ""
    order = sorted(cited_pages.keys())
    order += [i for i in range(len(retrieved)) if i not in cited_pages]
    chunks = []
    total = 0
    for i in order:
        doc = retrieved[i]
        snippet = (doc.snippet or "").strip()
        if not snippet:
            continue
        page = cited_pages.get(i)
        label = (f"[retrieved doc {i} | page {page}]"
                 if page else f"[retrieved doc {i}]")
        text = f"{label}\n{snippet}"
        if chunks and total + len(text) > char_budget:
            break
        chunks.append(text)
        total += len(text)
    out = "\n\n---\n\n".join(chunks)
    if len(out) > char_budget:
        out = out[:char_budget] + "\n\n[... context truncated ...]"
    return out


def context_sanity(question: dict, context: str) -> bool:
    """Lenient self-check: does the context contain the expected answer?

    True when at least half of the expected answer's distinctive words
    (length >= 5, up to 10 words) appear in the context. A False result
    does not stop the run -- it warns that the page mapping may be off
    (or the expected answer is noisy; both happen with auto sets).
    Returns True when there is no expected answer to check against.
    """
    expected = (question.get("expected_answer") or "").strip()
    if not expected:
        return True
    words = []
    for w in expected.split():
        w = w.strip(".,;:()[]{}\"'").lower()
        if len(w) >= 5 and w not in words:
            words.append(w)
        if len(words) >= 10:
            break
    if not words:
        return True
    ctx = context.lower()
    hits = sum(1 for w in words if w in ctx)
    return hits * 2 >= len(words)
