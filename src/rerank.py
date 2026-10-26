#!/usr/bin/env python3
"""
rerank.py -- step 10: rerankers for rung 4 of the retrieval ladder.

Two rerankers, one interface:

    reranker.rerank(query, pairs, top_k=None) -> [(chunk_id, score)]
    pairs = [(chunk_id, text)]

The bound method plugs straight into retrieve.py's rung 4 as rerank_fn:

    retrieve(..., rung="hybrid_rerank", rerank_fn=reranker.rerank, ...)

- CrossEncoderReranker: local sentence-transformers cross-encoder
  (default cross-encoder/ms-marco-MiniLM-L-6-v2). No API key, fully
  reproducible; the default for offline evals.
- CohereReranker: Cohere Rerank API (default rerank-english-v3.0).
  Needs COHERE_API_KEY in the environment (never paste it in chat).

CLI (ad-hoc reranking):
    python rerank.py --query "..." --candidates cands.json --top-k 10 \
        --reranker cross-encoder
    cands.json: [{"chunk_id": ..., "text": ...}]

    # or straight from a retrieve.py run JSON (candidates joined from the
    # manifest -- retrieve run JSONs carry chunk_ids + scores, not text):
    python rerank.py --from-run run.json --manifest full/manifest_full.jsonl \
        --top-k 10 --reranker cross-encoder
    (query defaults to the run's query; --query overrides it)
"""
import argparse
import json
import os
import sys

DEFAULT_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_COHERE_MODEL = "rerank-english-v3.0"


def rerank_by_scores(query, pairs, scores, top_k=None):
    """Sort [(chunk_id, text)] pairs by aligned scores, best first.

    Pure function -- the shared tail of every reranker. Returns
    [(chunk_id, score)].
    """
    ranked = sorted(zip([cid for cid, _ in pairs], scores),
                    key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k] if top_k else ranked


class CrossEncoderReranker:
    """Local cross-encoder reranker (sentence-transformers)."""

    def __init__(self, model_name=DEFAULT_CE_MODEL):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            sys.exit("missing dep: pip install sentence-transformers")
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query, texts):
        """Raw cross-encoder scores for [(query, text)] pairs."""
        return self._model.predict([(query, t) for t in texts]).tolist()

    def rerank(self, query, pairs, top_k=None):
        """pairs: [(chunk_id, text)]. Returns [(chunk_id, score)]."""
        scores = self.score(query, [t for _, t in pairs])
        return rerank_by_scores(query, pairs, scores, top_k)


class CohereReranker:
    """Cohere Rerank API."""

    def __init__(self, model_name=DEFAULT_COHERE_MODEL, api_key=None,
                 _client=None):
        self.model_name = model_name
        if _client is not None:  # test seam
            self._client = _client
            return
        try:
            import cohere
        except ImportError:
            sys.exit("missing dep: pip install cohere")
        key = api_key or os.environ.get("COHERE_API_KEY")
        if not key:
            sys.exit("COHERE_API_KEY not set in environment "
                     "(or pass api_key=)")
        self._client = cohere.Client(key)

    def rerank(self, query, pairs, top_k=None):
        """pairs: [(chunk_id, text)]. Returns [(chunk_id, score)]."""
        texts = [t for _, t in pairs]
        n = top_k or len(texts)
        resp = self._client.rerank(
            model=self.model_name, query=query,
            documents=texts, top_n=n)
        scores = [0.0] * len(texts)
        for r in resp.results:
            scores[r.index] = r.relevance_score
        return rerank_by_scores(query, pairs, scores, top_k)


def make_reranker(kind="cross-encoder", model=None, api_key=None):
    """kind: "cross-encoder" | "cohere"."""
    if kind == "cross-encoder":
        return CrossEncoderReranker(model or DEFAULT_CE_MODEL)
    if kind == "cohere":
        return CohereReranker(model or DEFAULT_COHERE_MODEL,
                              api_key=api_key)
    raise ValueError(
        f"unknown reranker: {kind} (choose cross-encoder | cohere)")


def load_manifest_texts(manifest_path):
    """chunk_id -> text, from a step-4 manifest JSONL.

    Mirrors retrieve.load_texts (text only); kept local so this module
    stays free of retrieve.py's index/embedding imports.
    """
    texts = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            texts[c["chunk_id"]] = c["text"]
    return texts


def join_run_candidates(run, texts):
    """Build [(chunk_id, text)] from a retrieve.py run dict.

    Run results carry chunk_ids + scores, not text, so each id is joined
    against the manifest. Ids missing from the manifest are skipped with
    a stderr warning (a stale run or mismatched manifest).
    """
    pairs, missing = [], []
    for r in run["results"]:
        cid = r["chunk_id"]
        if cid in texts:
            pairs.append((cid, texts[cid]))
        else:
            missing.append(cid)
    if missing:
        shown = ", ".join(missing[:10])
        print(f"warning: {len(missing)} chunk(s) not in manifest, "
              f"skipped: {shown}", file=sys.stderr)
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description="rerank candidates (step 10)")
    ap.add_argument("--query", default=None,
                    help="query text (defaults to the run's query "
                         "with --from-run)")
    ap.add_argument("--candidates", default=None,
                    help='JSON list of {"chunk_id": ..., "text": ...}')
    ap.add_argument("--from-run", default=None,
                    help="retrieve.py run JSON; candidates are joined "
                         "from --manifest")
    ap.add_argument("--manifest", default=None,
                    help="chunk manifest JSONL (chunk_id -> text); "
                         "required with --from-run")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--reranker", default="cross-encoder",
                    choices=["cross-encoder", "cohere"])
    ap.add_argument("--model", default=None,
                    help="override the default reranker model")
    ap.add_argument("--out", default=None,
                    help="write the ranked list as JSON here")
    args = ap.parse_args(argv)

    if args.from_run:
        if args.candidates:
            ap.error("--candidates and --from-run are mutually exclusive")
        if not args.manifest:
            ap.error("--from-run requires --manifest")
        with open(args.from_run, encoding="utf-8") as f:
            run = json.load(f)
        query = args.query or run.get("query")
        if not query:
            ap.error("--from-run run has no query and --query was not given")
        pairs = join_run_candidates(run, load_manifest_texts(args.manifest))
        if not pairs:
            sys.exit("no candidates: none of the run's chunk_ids were "
                     "found in the manifest")
    else:
        if not args.candidates:
            ap.error("need --candidates or --from-run")
        if not args.query:
            ap.error("--query is required without --from-run")
        with open(args.candidates, encoding="utf-8") as f:
            cands = json.load(f)
        pairs = [(c["chunk_id"], c["text"]) for c in cands]
        query = args.query
    reranker = make_reranker(args.reranker, model=args.model)
    ranked = reranker.rerank(query, pairs, top_k=args.top_k)
    for cid, score in ranked:
        print(f"{cid}  {score:.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([{"chunk_id": cid, "score": score}
                       for cid, score in ranked], f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
