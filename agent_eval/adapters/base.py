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

    Foundry's Responses API exposes citations as annotations on the
    output text: file-level (file_id / filename) plus, when the API
    carries it, the quoted span. `quote` is best-effort -- see
    adapters/foundry_agent.py and the --probe output for what the live
    API actually returns.
    """

    file_id: str = ""
    filename: str = ""
    quote: str = ""


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
    latency_s: float = 0.0
    error: Optional[str] = None
    raw_path: Optional[str] = None  # raw response JSON dump, probe only


class AgentAdapter(Protocol):
    """A system under test. `ask` must not raise for a bad question --
    return an AgentResponse with `error` set instead."""

    name: str

    def ask(self, question: str) -> AgentResponse: ...
