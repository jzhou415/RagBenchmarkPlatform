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


def main(argv=None):
    ap = argparse.ArgumentParser(description="rerank candidates (step 10)")
    ap.add_argument("--query", required=True)
    ap.add_argument("--candidates", required=True,
                    help='JSON list of {"chunk_id": ..., "text": ...}')
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--reranker", default="cross-encoder",
                    choices=["cross-encoder", "cohere"])
    ap.add_argument("--model", default=None,
                    help="override the default reranker model")
    ap.add_argument("--out", default=None,
                    help="write the ranked list as JSON here")
    args = ap.parse_args(argv)

    with open(args.candidates, encoding="utf-8") as f:
        cands = json.load(f)
    pairs = [(c["chunk_id"], c["text"]) for c in cands]
    reranker = make_reranker(args.reranker, model=args.model)
    ranked = reranker.rerank(args.query, pairs, top_k=args.top_k)
    for cid, score in ranked:
        print(f"{cid}  {score:.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump([{"chunk_id": cid, "score": score}
                       for cid, score in ranked], f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
