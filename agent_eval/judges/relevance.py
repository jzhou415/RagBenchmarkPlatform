"""Answer relevance: two signals, one robust, one best-effort.

1. Answer relevance (robust, always available). An LLM-as-judge rates
   whether the answer addresses the question, using the golden
   `expected_answer` as the reference when the set has one (the ASVS
   dev set fills it on all 20 questions). Verdicts: RELEVANT / PARTIAL /
   IRRELEVANT, with a one-line reason. This is the relevance number to
   report. It needs no citations and no retrieval internals.

2. Citation retrieval relevance (best-effort, probe-dependent). When
   the agent's citations carry quotable spans, each span is matched
   against the PDF page texts to recover the cited pages, and those
   pages -- in citation order -- are scored against the golden
   ground-truth pages with the platform's `metrics.evaluate_question`
   (hit@k / recall@k / nDCG@k at page level). Foundry citations are
   file-level, so for a single-PDF knowledge base they may carry no
   page-discriminating info at all; in that case this returns None with
   a note, and the answer-relevance verdict above is the relevance
   measure. The --probe dump shows what the live citations contain.
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
