"""Relevance: answer relevance plus two retrieval measures.

1. Answer relevance (robust, always available). An LLM-as-judge rates
   whether the answer addresses the question, using the golden
   `expected_answer` as the reference when the set has one (the ASVS
   dev set fills it on all 20 questions). Verdicts: RELEVANT / PARTIAL /
   IRRELEVANT, with a one-line reason. This is the relevance number to
   report. It needs no citations and no retrieval internals.

2. Retrieval relevance (real, MCP-based). The Foundry agent retrieves
   through an MCP tool (`knowledge_base_retrieve`): ~22 documents per
   question, each a ~2.3KB snippet of the source PDF (probe 2026-11-12,
   response_001). Each snippet is matched to a PDF page with
   `snippet_to_page` (multi-window normalized substring voting), and
   the pages -- in retrieval order, deduplicated -- are scored against
   the golden ground-truth pages with `metrics.evaluate_question`
   (hit@k / recall@k / nDCG@k at page level). This is the retrieval
   number to report: it measures what the agent was given to answer
   with.

3. Citation relevance (best-effort, marker-based). The agent cites
   with `url_citation` markers `【4:N†source】` at the end of its
   answer. Working hypothesis, supported by the q001 probe (the cited
   pages were 3 of the 4 grade->=2 golden pages): the second number is
   the index of the retrieved document. `cited_page_metrics` resolves
   markers to retrieved docs, docs to pages, and scores the cited
   pages the same way. When citations carry quotable spans instead
   (`file_citation`), `citation_page_metrics` maps quotes to pages.
   Either path returns None with a note when nothing maps.

The uid suffix `0_pages_N` on a retrieved document is a
knowledge-base chunk ordinal, NOT a PDF page (doc 0 says pages_16,
its text is PDF page 9) -- never use it as a page number.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

try:
    from src.metrics import evaluate_question as _evaluate_question
except ImportError:  # flat local working copy (~/workspace/user/rag/)
    from metrics import evaluate_question as _evaluate_question

from .faithfulness import call_with_retry, default_llm_fn

ANSWER_RELEVANCE_PROMPT = """You judge whether a model answer addresses the question. Reply with a JSON object and nothing else.

Question: {question}

Reference answer (from the source document; the model did not see this): {expected}

Model answer: {generated}

Procedure:
1. Compare the model answer to the reference answer on substance, not wording. A paraphrase that carries the same facts is a match.
2. "RELEVANT" if the model answer gives the substance of the reference answer, or a correct answer to the question that covers it.
3. "PARTIAL" if it addresses the question but omits a material part of the reference answer, or mixes correct substance with an unsupported addition.
4. "IRRELEVANT" if it does not answer the question, answers a different question, or says it does not know.

Respond with exactly this JSON shape:
{{"verdict": "RELEVANT", "reason": "<one line>"}}"""


def parse_relevance_verdict(text: str):
    """Return (verdict: str, reason: str). Unknown -> IRRELEVANT."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = str(obj.get("verdict", "")).strip().upper()
            if verdict in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
                return verdict, str(obj.get("reason", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
    up = text.upper()
    for v in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
        if v in up:
            return v, ""
    return "IRRELEVANT", ""


def judge_answer_relevance(question: str, answer: str,
                           expected_answer: str = "",
                           llm_fn: Optional[Callable[[str], str]] = None,
                           model: str = "gpt-4o-mini") -> dict:
    """Rate one answer against the question (+ expected answer).

    Returns {"verdict": str, "reason": str, "method": str}.
    An "I don't know" answer is IRRELEVANT by rule (no LLM call):
    it is faithful, but it does not answer the question.
    """
    if not (answer or "").strip():
        return {"verdict": "IRRELEVANT", "reason": "no answer",
                "method": "no-answer"}
    if answer.strip().lower().rstrip(".!?").strip() == "i don't know":
        return {"verdict": "IRRELEVANT",
                "reason": "abstained -- did not answer the question",
                "method": "abstention-rule"}
    reference = expected_answer.strip() or "(no reference answer in the golden set)"
    fn = llm_fn or (lambda p: default_llm_fn(p, model=model))
    raw = call_with_retry(
        lambda: fn(ANSWER_RELEVANCE_PROMPT.format(
            question=question, expected=reference,
            generated=answer))).strip()
    verdict, reason = parse_relevance_verdict(raw)
    return {"verdict": verdict, "reason": reason, "method": "judge",
            "raw": raw}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_to_page(quote: str, pages: list[str]) -> Optional[int]:
    """Best-effort: which 1-indexed PDF page contains this quote span.

    Returns None when the quote is empty or matches no page. Matching
    is normalized-substring; a long quote is tried whole, then on its
    first 200 characters (API quotes are often excerpts).
    """
    q = _normalize(quote)
    if len(q) < 20:
        return None
    norm_pages = [_normalize(p) for p in pages]
    for cand in (q, q[:200]):
        for i, pt in enumerate(norm_pages, start=1):
            if cand and cand in pt:
                return i
    return None


def snippet_to_page(snippet: str, pages: list[str]) -> Optional[int]:
    """Best-effort: which 1-indexed PDF page a retrieved snippet is from.

    A snippet is ~2.3KB and may straddle a page boundary, so one
    window is not enough: probes are taken at several offsets (120
    chars each), every page that contains a probe gets a vote, and the
    most-voted page wins (earliest on a tie). Returns None when no
    probe matches any page. Verified against the real probe dump:
    all 22 retrieved snippets mapped, zero unmapped.
    """
    sn = _normalize(snippet)
    if len(sn) < 40:
        return None
    norm_pages = [_normalize(p) for p in pages]
    votes: dict[int, int] = {}
    for off in (0, 200, 500, 1000, 1500):
        probe = sn[off:off + 120]
        if len(probe) < 40:
            break
        for i, pt in enumerate(norm_pages, start=1):
            if probe in pt:
                votes[i] = votes.get(i, 0) + 1
                break
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def retrieved_doc_pages(retrieved, pages: list[str]) -> dict[int, Optional[int]]:
    """{doc index: PDF page or None} for the retrieved documents."""
    return {i: snippet_to_page(getattr(d, "snippet", "") or "", pages)
            for i, d in enumerate(retrieved or [])}


def retrieved_page_metrics(retrieved, question: dict, pages: list[str],
                           ks=(3, 5, 10)) -> dict:
    """Page-level retrieval metrics from the retrieved documents.

    Pages in retrieval order, deduplicated (first occurrence), scored
    against the golden ground-truth pages. Returns the
    `evaluate_question` dict plus the ranked pages and a mapping
    count, or metrics None with a note when nothing mapped.
    """
    doc_pages = retrieved_doc_pages(retrieved, pages)
    ranked: list[str] = []
    for i in sorted(doc_pages):
        page = doc_pages[i]
        if page is not None:
            pid = f"p{page}"
            if pid not in ranked:
                ranked.append(pid)
    mapped = sum(1 for p in doc_pages.values() if p is not None)
    gt = page_ground_truth(question)
    if not ranked:
        return {"metrics": None,
                "note": "no retrieved document mapped to a page "
                        f"({len(retrieved or [])} retrieved, "
                        f"{mapped} mapped)"}
    return {"metrics": _evaluate_question(ranked, gt, ks),
            "ranked_pages": ranked,
            "mapped": mapped,
            "retrieved": len(retrieved or []),
            "note": f"{mapped}/{len(retrieved or [])} retrieved docs "
                    f"mapped to pages"}


_MARKER_RE = re.compile(r"【\d+:(\d+)†source】")
_MARKER_LOOSE_RE = re.compile(r"【\d+:(\d+)】")


def parse_citation_marker(marker: str) -> Optional[int]:
    """Doc index from a `【4:N†source】` marker, or None.

    Working hypothesis (probe 2026-11-12, q001): the second number is
    the index of the retrieved document in the mcp_call output. The
    cited pages it produced were 3 of the 4 grade->=2 golden pages,
    which is the evidence for the reading; confirm on a second probe
    before treating it as settled.
    """
    if not marker:
        return None
    m = _MARKER_RE.search(marker) or _MARKER_LOOSE_RE.search(marker)
    return int(m.group(1)) if m else None


def cited_page_metrics(citations, retrieved, question: dict,
                       pages: list[str], ks=(3, 5, 10)) -> dict:
    """Page-level metrics from the agent's citations, via markers.

    Each citation marker resolves to a retrieved document (by index),
    each document to a PDF page (snippet match); cited pages, in
    citation order and deduplicated, are scored against the golden
    ground-truth pages. Returns the `evaluate_question` dict plus the
    cited doc indices and ranked pages, or metrics None with a note
    when no marker resolved (wrong marker shape, index out of range,
    or snippet unmapped).
    """
    doc_pages = retrieved_doc_pages(retrieved, pages)
    ranked: list[str] = []
    cited_idx: list[int] = []
    unresolved = 0
    for c in citations or []:
        idx = parse_citation_marker(getattr(c, "marker", "") or "")
        if idx is None or idx >= len(retrieved or []):
            unresolved += 1
            continue
        cited_idx.append(idx)
        page = doc_pages.get(idx)
        if page is not None:
            pid = f"p{page}"
            if pid not in ranked:
                ranked.append(pid)
    gt = page_ground_truth(question)
    if not ranked:
        return {"metrics": None,
                "note": "no citation marker resolved to a page "
                        f"({len(cited_idx)} markers parsed, "
                        f"{unresolved} unresolved)"}
    return {"metrics": _evaluate_question(ranked, gt, ks),
            "ranked_pages": ranked,
            "cited_doc_indices": cited_idx,
            "note": f"{len(ranked)} cited page(s) mapped "
                    f"({unresolved} marker(s) unresolved)"}


def page_ground_truth(question: dict) -> dict:
    """{page_id: best relevance} for a question, page-level.

    Page ids are "p<page>" so they read like chunk ids in the metrics
    output. A page's grade is the best relevance among its chunks.
    """
    best: dict[str, int] = {}
    for g in question.get("ground_truth", []):
        pid = f"p{int(g['page'])}"
        best[pid] = max(best.get(pid, 0), int(g["relevance"]))
    return best


def citation_page_metrics(citations, question: dict, pages: list[str],
                          ks=(3, 5, 10)) -> Optional[dict]:
    """Page-level retrieval metrics from the agent's citations.

    Returns the `evaluate_question` dict, or None when no citation
    could be mapped to a page (citations file-level only, or no
    citations at all). The `note` key explains which case it was.
    """
    ranked: list[str] = []
    for c in citations or []:
        page = quote_to_page(getattr(c, "quote", "") or "", pages)
        if page is not None:
            pid = f"p{page}"
            if pid not in ranked:
                ranked.append(pid)
    gt = page_ground_truth(question)
    if not ranked:
        return {"metrics": None,
                "note": "no citation mapped to a page -- citations "
                        "carry no quotable spans (file-level only) or "
                        "the agent cited nothing"}
    return {"metrics": _evaluate_question(ranked, gt, ks),
            "ranked_pages": ranked,
            "note": f"{len(ranked)} cited page(s) mapped"}
