"""Platform adapter: the RagBenchmarkPlatform itself as a system under test.

`ask()` runs the platform's own pipeline, not an external agent:

1. Retrieve top-k chunks with the four-rung ladder (`retrieve.py`),
   default rung ``hybrid_rerank`` at candidate_k=20 -- the best
   nDCG@10 in the 2026-10-26 dev run. The rung is a parameter, so the
   same golden questions can score semantic / hybrid / hybrid_rerank
   against each other on faithfulness and relevance, not just on
   ranking metrics.
2. Generate with gpt-4o-mini, grounded ONLY in those chunks -- the
   corpus-agnostic version of `rag_answer.py`'s prompt ("I don't know"
   when the context lacks the answer).
3. Return an `AgentResponse`: answer + retrieved chunks (snippet plus
   the TRUE manifest page as `page_hint`) + latency + model + usage.
   Citations are v1-empty: the faithfulness and relevance judges do
   not need them, and citation metrics report None, which the runner
   already handles. (v2: have the generator write inline [pN] markers
   and parse them into `Citation` objects.)

Two things the Foundry adapter did not have to deal with:

* The harness recovers a retrieved doc's PDF page by matching its
  snippet text against the PDF pages (`judges/relevance.py`
  `snippet_to_page`) -- it does not trust `page_hint`. The manifest
  page is carried as the hint for debugging, but the snippet must
  stay verbatim chunk text for the mapping to work. The smoke run's
  "mapped X/Y" count shows whether Docling chunks match the PyMuPDF
  page text well enough.
* `retrieve.py` and friends import `from src.*`, but the flat working
  copy has no `src/` dir -- tests run with a `PYTHONPATH=/tmp/shim`
  shim. `build_platform_adapter()` registers the same shim in-process
  (`_ensure_src_shim`: a namespace package pointing at the flat
  directory, no files written, no symlinks), so the adapter works on
  the PC without a shim directory. When a real `src/` package is
  importable (repo layout), the shim is skipped.

Dependency injection: `embed_fn` / `retrieve_fn` / `generate_fn` /
`rerank_fn` are injectable, so unit tests run with no API keys and no
indexes. The real defaults are built lazily by
`build_platform_adapter()` -- importing this module needs no keys,
no indexes, and no OpenAI install.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path
from typing import Callable, Optional

from .base import AgentResponse, RetrievedDoc

GEN_PROMPT = """You answer questions using ONLY the context below. If the context does not contain the answer, reply exactly: I don't know.

Context:
---
{context}
---

Question: {question}

Answer concisely, with the key facts and numbers. Do not add information that is not in the context."""

EMBED_MODEL = "text-embedding-3-small"
DEFAULT_GENERATOR_MODEL = "gpt-4o-mini"
DEFAULT_RUNG = "hybrid_rerank"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
DEFAULT_DEPTH = 50
GENERATION_CONTEXT_BUDGET = 30_000


def _ensure_src_shim(flat_dir: Optional[Path] = None) -> None:
    """Make `src.*` importable from the flat working copy.

    The platform modules (`retrieve.py`, `rerank.py`, ...) import each
    other as `src.<module>`, but the flat working copy keeps them at
    the top level. When no real `src` package is importable, register
    a namespace package whose path is the flat directory, so
    `from src.retrieve import ...` resolves to the flat modules. No
    files are written; the registration lives in `sys.modules` only.
    """
    try:
        import src  # noqa: F401
        return
    except ImportError:
        pass
    if flat_dir is None:
        # agent_eval/adapters/platform.py -> working-copy root.
        flat_dir = Path(__file__).resolve().parent.parent.parent
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(flat_dir)]
    sys.modules["src"] = pkg


def _build_context(ranked, texts, budget: int = GENERATION_CONTEXT_BUDGET
                   ) -> str:
    """Generation context: "[page N]"-marked chunk texts, best-first.

    Chunks missing from `texts` (or empty) are skipped here -- they
    still appear in `retrieved`, with an empty snippet, so the
    retrieval order stays faithful. Truncated at the budget with a
    note rather than silently cut.
    """
    chunks = []
    total = 0
    for cid, _score in ranked:
        text, page = texts.get(cid, ("", None))
        text = (text or "").strip()
        if not text:
            continue
        label = f"[page {page}]" if page else "[page ?]"
        piece = f"{label}\n{text}"
        if chunks and total + len(piece) > budget:
            break
        chunks.append(piece)
        total += len(piece)
    out = "\n\n---\n\n".join(chunks)
    if len(out) > budget:
        out = out[:budget] + "\n\n[... context truncated ...]"
    return out


def _default_embed_fn(client, model: str = EMBED_MODEL) -> Callable:
    """query text -> embedding vector (mirrors retrieve.embed_query_text)."""

    def embed(text: str) -> list[float]:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return embed


def _default_generate_fn(llm) -> Callable:
    """(question, context) -> (answer, usage dict)."""

    def generate(question: str, context: str) -> tuple[str, dict]:
        msg = llm.invoke(
            GEN_PROMPT.format(context=context, question=question))
        usage: dict = {}
        meta = getattr(msg, "response_metadata", None) or {}
        tok = meta.get("token_usage") or {}
        if isinstance(tok, dict):
            usage = {k: v for k, v in tok.items() if isinstance(v, int)}
        return (msg.content or "").strip(), usage

    return generate


class PlatformAdapter:
    """Adapter over the platform's retrieval ladder + generator.

    Build via `build_platform_adapter()`; tests construct it directly
    with injected fns and in-memory indexes/texts. `ask()` never
    raises for a bad question -- failures return an `AgentResponse`
    with `error` set, per the adapter contract.
    """

    name = "platform"

    def __init__(self, *, indexes=None, texts=None,
                 rung: str = DEFAULT_RUNG, top_k: int = DEFAULT_TOP_K,
                 candidate_k: int = DEFAULT_CANDIDATE_K,
                 rrf_k: int = DEFAULT_RRF_K, depth: int = DEFAULT_DEPTH,
                 model: str = DEFAULT_GENERATOR_MODEL,
                 embed_fn: Optional[Callable] = None,
                 retrieve_fn: Optional[Callable] = None,
                 generate_fn: Optional[Callable] = None,
                 rerank_fn: Optional[Callable] = None):
        self._indexes = indexes
        self._texts = dict(texts or {})
        self._rung = rung
        self._top_k = top_k
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._depth = depth
        self._model = model
        self._embed_fn = embed_fn
        self._retrieve_fn = retrieve_fn
        self._generate_fn = generate_fn
        self._rerank_fn = rerank_fn

    def ask(self, question: str) -> AgentResponse:
        t0 = time.perf_counter()
        try:
            ranked = self._retrieve(question)
            context = _build_context(ranked, self._texts)
            answer, usage = self._generate(question, context)
        except Exception as e:  # noqa: BLE001 -- recorded per question
            return AgentResponse(
                error=f"platform adapter failed: {e}",
                latency_s=time.perf_counter() - t0)
        retrieved = [
            RetrievedDoc(
                doc_id=cid,
                snippet=(self._texts.get(cid, ("", None))[0] or ""),
                page_hint=self._texts.get(cid, ("", None))[1],
            )
            for cid, _score in ranked
        ]
        return AgentResponse(
            answer=answer,
            citations=[],  # v1: no citations; judges don't need them
            retrieved=retrieved,
            latency_s=time.perf_counter() - t0,
            model=self._model,
            usage=usage or {},
        )

    def _retrieve(self, question: str) -> list:
        if self._retrieve_fn is not None:
            return self._retrieve_fn(question)
        if self._indexes is None:
            raise ValueError(
                "no indexes: build via build_platform_adapter() "
                "or pass retrieve_fn=")
        _ensure_src_shim()
        from src.retrieve import retrieve as platform_retrieve

        return platform_retrieve(
            question, self._rung, self._indexes, texts=self._texts,
            top_k=self._top_k, rrf_k=self._rrf_k, depth=self._depth,
            candidate_k=self._candidate_k, rerank_fn=self._rerank_fn,
            embed_fn=self._embed_fn)

    def _generate(self, question: str, context: str) -> tuple[str, dict]:
        if self._generate_fn is not None:
            return self._generate_fn(question, context)
        raise ValueError(
            "no generator: build via build_platform_adapter() "
            "or pass generate_fn=")


def build_platform_adapter(
        indexes_dir=None, manifest_path=None, rung: str = DEFAULT_RUNG,
        top_k: int = DEFAULT_TOP_K, candidate_k: int = DEFAULT_CANDIDATE_K,
        rrf_k: int = DEFAULT_RRF_K, depth: int = DEFAULT_DEPTH,
        model: str = DEFAULT_GENERATOR_MODEL,
        reranker: str = "cross-encoder", reranker_model=None,
        embed_model: str = EMBED_MODEL) -> PlatformAdapter:
    """Load the platform indexes and return an adapter.

    `indexes_dir` holds the step-7 index dirs (`bm25_<hash>/` and
    `dense_<hash>/`, e.g. the PC's `data1/`). `manifest_path` is the
    chunk manifest jsonl (chunk_id -> text + page_no) the adapter
    needs for snippets and generation context. The cross-encoder
    reranker (default) loads its model on first use; use
    rung="hybrid" to skip the reranker entirely. Reads OPENAI_API_KEY
    from the environment (or a .env file, same as the judges).
    """
    _ensure_src_shim()
    from dotenv import load_dotenv  # lazy: no keys needed to import

    load_dotenv()
    if indexes_dir is None:
        raise ValueError(
            "--platform-indexes is required (directory holding "
            "bm25_<hash>/ and dense_<hash>/, e.g. data1)")
    if manifest_path is None:
        raise ValueError(
            "--platform-manifest is required (chunk manifest jsonl: "
            "chunk_id -> text + page_no)")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in .env file")

    from src.retrieve import load_indexes, load_texts

    indexes = load_indexes(str(indexes_dir))
    texts = load_texts(str(manifest_path))

    rerank_fn = None
    if rung == "hybrid_rerank":
        from src.rerank import make_reranker

        rerank_fn = make_reranker(reranker, model=reranker_model).rerank

    from openai import OpenAI
    from langchain_openai import ChatOpenAI

    client = OpenAI()
    llm = ChatOpenAI(model=model, temperature=0)
    return PlatformAdapter(
        indexes=indexes, texts=texts, rung=rung, top_k=top_k,
        candidate_k=candidate_k, rrf_k=rrf_k, depth=depth, model=model,
        embed_fn=_default_embed_fn(client, model=embed_model),
        generate_fn=_default_generate_fn(llm),
        rerank_fn=rerank_fn)
