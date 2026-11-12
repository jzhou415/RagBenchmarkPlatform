"""Fake adapter: canned answers for smoke tests and unit tests.

Reads a JSON file of {question_id: answer} (or takes a dict) and
returns those answers with no network calls. The runner matches
questions to answers by question id; an id with no canned answer
returns "I don't know", which exercises the abstention path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import AgentResponse, Citation


class FakeAdapter:
    name = "fake"

    def __init__(self, answers: Optional[dict] = None,
                 answers_path: Optional[Path] = None,
                 citations: Optional[dict] = None):
        self._answers = dict(answers or {})
        if answers_path is not None:
            with open(answers_path, encoding="utf-8") as f:
                self._answers.update(json.load(f))
        # citations: {question_id: [{"file_id":..., "filename":...,
        #   "quote":...}]} -- lets tests exercise citation mapping.
        self._citations = citations or {}
        self._current_id = None

    def set_question_id(self, qid: str) -> None:
        """The runner calls this before ask() so answers map by id."""
        self._current_id = qid

    def ask(self, question: str) -> AgentResponse:
        qid = self._current_id
        answer = self._answers.get(qid, "I don't know") if qid else \
            "I don't know"
        cites = [Citation(**c) for c in self._citations.get(qid, [])] \
            if qid else []
        return AgentResponse(answer=answer, citations=cites, latency_s=0.01)
