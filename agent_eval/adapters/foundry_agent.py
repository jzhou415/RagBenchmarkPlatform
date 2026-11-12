"""Foundry adapter: drives the Microsoft Foundry knowledge agent.

This is the interactive `foundry_agent_client.py` rebuilt as a library:
the same SDK calls, but no top-level side effects, no prints, and no
`input()` prompts.

Three deliberate changes from the CLI script:

1. Fresh conversation per question. The CLI keeps ONE conversation for
   the whole session, so earlier answers leak into later ones. An eval
   question must stand alone, so `ask()` creates a new conversation
   every call.
2. MCP approval requests fail fast. The CLI blocks on
   `input("Do you approve...")`. An unattended run must never hang on
   a prompt, so an approval request returns an AgentResponse with
   `error` set and the question is recorded, not retried.
3. Structured return. The CLI returns only the answer string. The
   adapter returns answer + citations + retrieved documents + model +
   token usage + latency, and optionally dumps the raw response JSON
   (probe mode) so the citation shape can be inspected -- see
   README.md. The live citation shape (probe 2026-11-12) is
   `url_citation` markers plus MCP retrieval documents, not
   `file_citation`; the extractors handle both.

Azure SDK imports are lazy (inside `build_foundry_adapter`), so
importing this module -- and running the unit tests -- needs no Azure
credentials and no azure-ai-projects install.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

from .base import AgentResponse, Citation, RetrievedDoc


def _extract_text(response) -> str:
    """Answer text, preferring the SDK's output_text shortcut."""
    text = getattr(response, "output_text", "") or ""
    if text:
        return text
    parts = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                parts.append(getattr(content, "text", "") or "")
    return "".join(parts)


def _extract_citations(response, answer_text: str = "") -> list[Citation]:
    """Citations off the output-text annotations.

    The Responses API carries citations as annotations, not as a
    top-level `citations` attribute (the CLI's `response.citations`
    check never fires). Two shapes occur:

    * `url_citation` -- what this agent returns (probe 2026-11-12):
      a marker span `【4:0†source】` (start/end indices into the
      answer text) plus a URL. No file_id, no quote.
    * `file_citation` -- file_id / filename and, when the API carries
      it, a `quote`.

    The marker text is recovered from the answer via the indices, so
    the record keeps the visible citation the user would see.
    """
    citations: list[Citation] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") != "output_text":
                continue
            for ann in getattr(content, "annotations", []) or []:
                kind = getattr(ann, "type", "")
                if kind == "file_citation":
                    citations.append(Citation(
                        file_id=getattr(ann, "file_id", "") or "",
                        filename=getattr(ann, "filename", "") or "",
                        quote=getattr(ann, "quote", "") or "",
                    ))
                elif kind == "url_citation":
                    start = getattr(ann, "start_index", None)
                    end = getattr(ann, "end_index", None)
                    marker = ""
                    if (isinstance(start, int) and isinstance(end, int)
                            and 0 <= start < end <= len(answer_text)):
                        marker = answer_text[start:end]
                    citations.append(Citation(
                        marker=marker,
                        url=getattr(ann, "url", "") or "",
                        start_index=start, end_index=end,
                    ))
    return citations


def _parse_page_hint(uid: str) -> Optional[int]:
    """The N in a uid suffix `0_pages_N`, or None.

    The suffix is a knowledge-base chunk ordinal, not a PDF page --
    do not use it as a page number. It is kept for debugging the
    retrieval order only.
    """
    import re

    m = re.search(r"0_pages_(\d+)\s*$", uid or "")
    return int(m.group(1)) if m else None


def _extract_retrieved(response) -> list[RetrievedDoc]:
    """Documents the agent's retrieval returned, from mcp_call items.

    The Foundry agent retrieves through an MCP tool
    (`knowledge_base_retrieve` on server kb-knowledgebase816-89xt9),
    not file search: the mcp_call output is a JSON string
    `{"documents": [...]}` where each document has an id and a
    `content` JSON string `{uid, blob_url, snippet}`. A non-JSON
    output, or a call with no documents, yields no records -- the
    eval continues on the answer alone.
    """
    docs: list[RetrievedDoc] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "mcp_call":
            continue
        raw = getattr(item, "output", None)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw, dict):
            parsed = raw
        else:
            continue
        for doc in (parsed or {}).get("documents", []) or []:
            if not isinstance(doc, dict):
                continue
            content = doc.get("content", "") or ""
            inner: dict = {}
            if isinstance(content, str):
                try:
                    inner = json.loads(content)
                except json.JSONDecodeError:
                    inner = {}
            elif isinstance(content, dict):
                inner = content
            uid = str(inner.get("uid", "") or "")
            docs.append(RetrievedDoc(
                doc_id=str(doc.get("id", "") or ""),
                uid=uid,
                snippet=str(inner.get("snippet", "") or ""),
                title=str(doc.get("title", "") or ""),
                page_hint=_parse_page_hint(uid),
            ))
    return docs


def _extract_usage(response) -> dict:
    """Token counts, if the response carries them."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        val = getattr(usage, key, None)
        if val is None and isinstance(usage, dict):
            val = usage.get(key)
        if isinstance(val, int):
            out[key] = val
    return out


def _find_approval_request(response):
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") == "mcp_approval_request":
            return item
    return None


class FoundryAgentAdapter:
    """Adapter over a live Foundry agent. Build via build_foundry_adapter."""

    name = "foundry"

    def __init__(self, openai_client, agent, raw_dir: Optional[Path] = None):
        self._client = openai_client
        self._agent = agent
        self._raw_dir = Path(raw_dir) if raw_dir else None
        self._n = 0

    def ask(self, question: str) -> AgentResponse:
        t0 = time.time()
        try:
            conversation = self._client.conversations.create(items=[])
            self._client.conversations.items.create(
                conversation_id=conversation.id,
                items=[{"type": "message", "role": "user",
                        "content": question}])
            response = self._client.responses.create(
                conversation=conversation.id,
                extra_body={"agent_reference": {
                    "name": self._agent.name, "type": "agent_reference"}},
                input="")
        except Exception as e:  # noqa: BLE001 -- recorded per question
            return AgentResponse(error=f"foundry call failed: {e}",
                                 latency_s=time.time() - t0)
        latency = time.time() - t0

        approval = _find_approval_request(response)
        if approval is not None:
            return AgentResponse(
                error=("mcp_approval_request for "
                       f"{getattr(approval, 'name', '?')} "
                       f"({getattr(approval, 'server_label', '?')}) -- "
                       "not auto-approved; question recorded, not retried"),
                latency_s=latency)

        raw_path = None
        if self._raw_dir is not None:
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            self._n += 1
            raw_path = str(self._raw_dir / f"response_{self._n:03d}.json")
            try:
                dump = response.model_dump(mode="json")
            except Exception:  # noqa: BLE001 -- SDK objects vary
                dump = {"repr": repr(response)}
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, ensure_ascii=False)

        answer = _extract_text(response)
        return AgentResponse(
            answer=answer,
            citations=_extract_citations(response, answer),
            retrieved=_extract_retrieved(response),
            latency_s=latency,
            raw_path=raw_path,
            model=getattr(response, "model", "") or "",
            usage=_extract_usage(response))


def build_foundry_adapter(
        endpoint: Optional[str] = None,
        agent_name: Optional[str] = None,
        raw_dir: Optional[Path] = None) -> FoundryAgentAdapter:
    """Connect to the Foundry project and return an adapter.

    Reads PROJECT_ENDPOINT / AGENT_NAME from the environment (or a
    .env file, same as the CLI script). Auth is DefaultAzureCredential
    with environment + managed-identity credentials excluded -- i.e.
    Azure CLI login on the machine running the eval.
    """
    from dotenv import load_dotenv  # lazy: Azure SDK not needed to import
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    load_dotenv()
    endpoint = endpoint or os.getenv("PROJECT_ENDPOINT")
    agent_name = agent_name or os.getenv("AGENT_NAME")
    if not endpoint or not agent_name:
        raise ValueError(
            "PROJECT_ENDPOINT and AGENT_NAME must be set in .env file")
    credential = DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True)
    project_client = AIProjectClient(endpoint=endpoint,
                                     credential=credential)
    openai_client = project_client.get_openai_client()
    agent = project_client.agents.get(agent_name=agent_name)
    return FoundryAgentAdapter(openai_client, agent, raw_dir=raw_dir)
