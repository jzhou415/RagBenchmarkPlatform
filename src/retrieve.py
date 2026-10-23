#!/usr/bin/env python3
"""
retrieve.py -- step 9: the four-rung retrieval ladder.

Rung 1 -- semantic:      dense Chroma index (step 7b)
Rung 2 -- bm25:          keyword index (step 7a)
Rung 3 -- hybrid:        Reciprocal Rank Fusion of rungs 1+2
Rung 4 -- hybrid_rerank: RRF candidates rescored by a reranker.
                         The reranker is injected as rerank_fn; see
                         src/rerank.py (step 10). Rung 4 without a
                         rerank_fn raises.

Each rung is a pure function: query in, ranked [(chunk_id, score)] out.
"Immutable named index + run": the CLI stamps the index dir names and
config hash into every output JSON, so a ranking can always be traced
back to the exact index that produced it.

Reads (step 7 outputs):
    <indexes>/bm25_<hash>/   (index_bm25.py)
    <indexes>/dense_<hash>/  (index_dense.py)
"""
import argparse
import datetime
import glob
import json
import os
import sys

import yaml

from src.index_bm25 import load_index as load_bm25
from src.index_bm25 import search as bm25_search_fn
from src.index_dense import query_dense

RUNGS = ("semantic", "bm25", "hybrid", "hybrid_rerank")
EMBED_MODEL = "text-embedding-3-small"


def load_indexes(indexes_dir="indexes", bm25_name=None, dense_name=None):
    """Load the step-7 indexes. Returns {bm25, dense, config_hash}.

    Auto-discovers the index dirs when exactly one bm25_* and one dense_*
    dir exist; otherwise pass --bm25-index / --dense-index explicitly.
    Refuses to proceed if the two indexes disagree on config_hash.
    """
    indexes_dir = str(indexes_dir)
    if bm25_name is None:
        found = sorted(d for d in glob.glob(os.path.join(indexes_dir, "bm25_*"))
                       if os.path.isdir(d))
        if len(found) != 1:
            sys.exit(f"found {len(found)} bm25 indexes in {indexes_dir}; "
                     f"pass --bm25-index explicitly")
        bm25_name = os.path.basename(found[0])
    if dense_name is None:
        found = sorted(d for d in glob.glob(os.path.join(indexes_dir, "dense_*"))
                       if os.path.isdir(d))
        if len(found) != 1:
            sys.exit(f"found {len(found)} dense indexes in {indexes_dir}; "
                     f"pass --dense-index explicitly")
        dense_name = os.path.basename(found[0])
    bm25_dir = os.path.join(indexes_dir, bm25_name)
    dense_dir = os.path.join(indexes_dir, dense_name)
    bm25, doc_ids, bm25_meta = load_bm25(bm25_dir)
    meta_path = dense_dir + ".meta.json"
    with open(meta_path, encoding="utf-8") as f:
        dense_meta = json.load(f)
    if bm25_meta.get("config_hash") != dense_meta.get("config_hash"):
        sys.exit(f"config hash mismatch: bm25={bm25_meta.get('config_hash')} "
                 f"dense={dense_meta.get('config_hash')}")
    tag = str(dense_meta["config_hash"])[:12]
    return {
        "bm25": {"dir": bm25_dir, "index": bm25, "doc_ids": doc_ids,
                 "meta": bm25_meta},
        "dense": {"persist_dir": dense_dir,
                  "collection": f"chunks_{tag}",
                  "meta": dense_meta},
        "config_hash": bm25_meta.get("config_hash"),
    }


def load_texts(manifest_path):
    """chunk_id -> (text, page_no), from the step-4 manifest."""
    texts = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                c = json.loads(line)
                texts[c["chunk_id"]] = (c["text"], c["page_no"])
    return texts


def semantic_search(query_vector, dense_info, top_k=10):
    """Dense rung. query_vector is a precomputed embedding."""
    return query_dense(dense_info["persist_dir"], dense_info["collection"],
                       query_vector, top_k=top_k)


def bm25_search(query_text, bm25_info, top_k=10):
    """Keyword rung."""
    return bm25_search_fn(bm25_info["index"], bm25_info["doc_ids"],
                          query_text, top_k=top_k)


def rrf_fuse(rankings, rrf_k=60):
    """Reciprocal Rank Fusion over rankings (each [(chunk_id, score)]).

    Only ranks are used; raw scores are ignored, which is what makes RRF
    safe across rungs with incomparable score scales. Returns
    [(chunk_id, fused_score)] sorted best-first.
    """
    fused = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(query_text, query_vector, indexes, top_k=10, rrf_k=60,
                  depth=50):
    """RRF of the top-`depth` dense + top-`depth` BM25 hits, cut to top_k."""
    sem = semantic_search(query_vector, indexes["dense"], top_k=depth)
    kw = bm25_search(query_text, indexes["bm25"], top_k=depth)
    return rrf_fuse([sem, kw], rrf_k=rrf_k)[:top_k]


def hybrid_rerank_search(query_text, query_vector, indexes, texts,
                         top_k=10, rrf_k=60, candidate_k=50, rerank_fn=None):
    """Hybrid candidates rescored by rerank_fn.

    rerank_fn(query_text, [(chunk_id, text)], top_k) -> [(chunk_id, score)].
    """
    if rerank_fn is None:
        raise ValueError(
            "hybrid_rerank needs a rerank_fn (see src/rerank.py, step 10)")
    candidates = hybrid_search(query_text, query_vector, indexes,
                               top_k=candidate_k, rrf_k=rrf_k,
                               depth=candidate_k)
    pairs = [(cid, texts[cid][0]) for cid, _ in candidates]
    return rerank_fn(query_text, pairs, top_k=top_k)


def retrieve(query_text, rung, indexes, texts=None, query_vector=None,
             top_k=10, rrf_k=60, depth=50, candidate_k=50,
             rerank_fn=None, embed_fn=None):
    """Dispatch one query to one rung. Returns [(chunk_id, score)]."""
    if rung not in RUNGS:
        raise ValueError(f"unknown rung: {rung} (choose from {RUNGS})")
    if rung in ("semantic", "hybrid", "hybrid_rerank") \
            and query_vector is None:
        if embed_fn is None:
            raise ValueError(f"rung {rung} needs query_vector or embed_fn")
        query_vector = embed_fn(query_text)
    if rung == "semantic":
        return semantic_search(query_vector, indexes["dense"], top_k=top_k)
    if rung == "bm25":
        return bm25_search(query_text, indexes["bm25"], top_k=top_k)
    if rung == "hybrid":
        return hybrid_search(query_text, query_vector, indexes,
                             top_k=top_k, rrf_k=rrf_k, depth=depth)
    if texts is None:
        raise ValueError("hybrid_rerank needs texts "
                         "(chunk_id -> (text, page_no))")
    return hybrid_rerank_search(query_text, query_vector, indexes, texts,
                                top_k=top_k, rrf_k=rrf_k,
                                candidate_k=candidate_k, rerank_fn=rerank_fn)


def embed_query_text(client, text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="four-rung retrieval ladder (step 9)")
    ap.add_argument("--indexes", default="indexes",
                    help="parent dir of the named step-7 indexes")
    ap.add_argument("--bm25-index", default=None,
                    help="e.g. bm25_<hash12> (auto-discovered if unambiguous)")
    ap.add_argument("--dense-index", default=None,
                    help="e.g. dense_<hash12> (auto-discovered if unambiguous)")
    ap.add_argument("--manifest", default=None,
                    help="full/manifest_full.jsonl (for texts + page numbers)")
    ap.add_argument("--rung", required=True, choices=RUNGS)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--depth", type=int, default=50,
                    help="per-rung depth fused by RRF")
    ap.add_argument("--out", default=None,
                    help="write the stamped run JSON here")
    args = ap.parse_args(argv)

    indexes = load_indexes(args.indexes, args.bm25_index, args.dense_index)
    texts = load_texts(args.manifest) if args.manifest else None

    query_vector = None
    if args.rung in ("semantic", "hybrid", "hybrid_rerank"):
        if "OPENAI_API_KEY" not in os.environ:
            sys.exit("OPENAI_API_KEY not set in environment")
        from src.index_dense import make_client
        query_vector = embed_query_text(make_client(), args.query)

    results = retrieve(args.query, args.rung, indexes, texts=texts,
                       query_vector=query_vector, top_k=args.top_k,
                       rrf_k=args.rrf_k, depth=args.depth)
    run = {
        "run": {"timestamp": _now_iso(), "rung": args.rung,
                "top_k": args.top_k, "rrf_k": args.rrf_k},
        "indexes": {
            "bm25": os.path.basename(indexes["bm25"]["dir"]),
            "dense": os.path.basename(indexes["dense"]["persist_dir"]),
            "config_hash": indexes["config_hash"],
        },
        "query": args.query,
        "results": [
            {"chunk_id": cid, "score": score,
             "page_no": texts[cid][1] if texts and cid in texts else None}
            for cid, score in results
        ],
    }
    for r in run["results"]:
        print(f"{r['chunk_id']}  p{r['page_no']}  {r['score']:.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
