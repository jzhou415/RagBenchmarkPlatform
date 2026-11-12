"""Adapter interface: the one method every system under test must offer.

An adapter turns a question string into an AgentResponse. The evaluator
(judges + metrics) only ever sees this interface, so the same scoring
code tests the Foundry agent today and the RagBenchmarkPlatform itself
tomorrow -- only the adapter changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class Citation:
    """One source the agent cited for its answer.

    The Foundry agent's live shape (probe 2026-11-12, response_001):
    citations are `url_citation` annotations on the output text --
    a marker span such as "[4:0 source]" (start/end indices into the
    answer) plus a URL, all pointing at the same PDF blob. No
    file_id, no filename, no quote. `file_citation` (file_id /
    filename / quote) is the other shape the API can return, kept for
    compatibility; `marker` is the visible text for url_citations.
    """

    file_id: str = ""
    filename: str = ""
    quote: str = ""
    marker: str = ""          # visible marker text, e.g. "[4:0 source]"
    url: str = ""             # url_citation target (blob URL for Foundry)
    start_index: Optional[int] = None
    end_index: Optional[int] = None


@dataclass
class RetrievedDoc:
    """One document the agent's retrieval returned for a question.

    For the Foundry agent (probe 2026-11-12): the MCP
    `knowledge_base_retrieve` call returns ~22 documents per question,
    each a ~2.3KB snippet of the source PDF. `page_hint` is the N in
    the uid suffix `0_pages_N` -- a knowledge-base chunk ordinal, NOT
    a PDF page number (doc 0 says pages_16, its text is PDF page 9).
    The PDF page is recovered by matching `snippet` against the PDF
    page texts (see judges/relevance.py `snippet_to_page`).
    """

    doc_id: str = ""
    uid: str = ""
    snippet: str = ""
    title: str = ""           # blob URL for Foundry
    page_hint: Optional[int] = None


@dataclass
class AgentResponse:
    """What one `ask()` returned, success or failure.

    `error` is set (and `answer` empty) when the call failed or could
    not be completed -- e.g. an MCP approval request the harness
    refuses to approve interactively. Errors are recorded per question,
    never raised out of the run.
    """

    answer: str = ""
    citations: List[Citation] = field(default_factory=list)
    retrieved: List[RetrievedDoc] = field(default_factory=list)
    latency_s: float = 0.0
    error: Optional[str] = None
    raw_path: Optional[str] = None  # raw response JSON dump, probe only
    model: str = ""                 # e.g. "gpt-4.1-mini-1" for Foundry
    usage: dict = field(default_factory=dict)  # input/output/total tokens


class AgentAdapter(Protocol):
    """A system under test. `ask` must not raise for a bad question --
    return an AgentResponse with `error` set instead."""

    name: str

    def ask(self, question: str) -> AgentResponse: ...
