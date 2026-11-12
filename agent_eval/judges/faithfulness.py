"""Faithfulness judge (LLM-as-judge v2).

This is the judge built for the ESPP statement-parser benchmark
(`rejudge_faithfulness.py`, judge v2), copied here so agent_eval is
self-contained in the repo. Same prompt, same parser, same rules:

* the judge splits the answer into individual factual claims and quotes
  ONE verbatim support span per claim from the context;
* number-label rule: a numeric claim is supported only if the quoted
  span shows that number attached to the SAME label the claim uses
  (number + label co-occurring in the same table row, line, or short
  passage);
* "overall" is UNFAITHFUL if any claim is unsupported;
* an "I don't know" answer makes no factual claims, so it is faithful
  by definition -- handled by a deterministic rule, no LLM call;
* an unparseable judge reply is scored UNFAITHFUL (conservative) and
  the raw reply is kept for inspection.

Judge model: gpt-4o-mini, temperature 0, one call per question per
context. The LLM call is injectable (`llm_fn`) so unit tests run with
a stub and no API key.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional

FAITHFULNESS_V2_PROMPT = """You audit whether a model answer is grounded in retrieved context. Reply with a JSON object and nothing else.

Context:
---
{context}
---

Question: {question}

Model answer: {generated}

Procedure:
1. Split the model answer into individual factual claims.
2. For each claim, quote the ONE span of the context that best supports it, verbatim. Use an empty string if no span supports it.
3. Number-label rule: when a claim states a number, the quoted span must show that number attached to the SAME label the claim uses -- the number and its label must co-occur within the same table row, line, or short passage. If the number appears in the context but attached to a different label (for example a subtotal where the claim says total, or a share count where the claim says a dollar price), mark the claim unsupported.
4. A claim is supported only if its quoted span backs the claim as stated, with every number attached to its correct label.

Respond with exactly this JSON shape:
{{"claims": [{{"claim": "<claim text>", "support_quote": "<verbatim span or empty string>", "supported": true}}], "overall": "FAITHFUL"}}

"overall" is "UNFAITHFUL" if any claim has "supported": false, otherwise "FAITHFUL"."""


def call_with_retry(fn, tries=3, delay=2):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as e:  # transient API hiccup  # noqa: BLE001
            last = e
            time.sleep(delay)
            delay *= 2
    raise last


def is_abstention(text: str) -> bool:
    return text.strip().lower().rstrip(".!?").strip() == "i don't know"


def parse_v2_verdict(text: str):
    """Return (faithful: bool, claims: list, raw: str)."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            overall = str(obj.get("overall", "")).strip().upper()
            if overall in ("FAITHFUL", "UNFAITHFUL"):
                return overall == "FAITHFUL", obj.get("claims", []), text
        except (json.JSONDecodeError, AttributeError):
            pass
    up = text.upper()
    if "UNFAITHFUL" in up:
        return False, [], text
    if "FAITHFUL" in up:
        return True, [], text
    return False, [], text  # unparseable: conservative, inspect raw


def default_llm_fn(prompt: str, model: str = "gpt-4o-mini") -> str:
    """One judge call via LangChain. Imported lazily: importing this
    module needs no langchain install and no API key."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=model, temperature=0)
    return llm.invoke(prompt).content


def judge_faithfulness(question: str, answer: str, context: str,
                       llm_fn: Optional[Callable[[str], str]] = None,
                       model: str = "gpt-4o-mini") -> dict:
    """Score one answer against one context.

    Returns {"faithful": bool|None, "method": str, "claims": [...],
    "raw": str}. `faithful` is None only when there is no answer to
    judge (empty answer, no error recorded elsewhere).
    """
    if not (answer or "").strip():
        return {"faithful": None, "method": "no-answer",
                "claims": [], "raw": ""}
    if is_abstention(answer):
        return {"faithful": True, "method": "abstention-rule",
                "claims": [], "raw": ""}
    fn = llm_fn or (lambda p: default_llm_fn(p, model=model))
    raw = call_with_retry(
        lambda: fn(FAITHFULNESS_V2_PROMPT.format(
            context=context, question=question, generated=answer))).strip()
    faithful, claims, _ = parse_v2_verdict(raw)
    return {"faithful": faithful, "method": "judge-v2",
            "claims": claims, "raw": raw}
