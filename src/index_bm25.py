#!/usr/bin/env python3
"""
index_bm25.py -- step 7a: build the BM25 keyword index over the full corpus.

RagBenchmarkPlatform, Phase 1 (Data Setup backend).

Reads the full-manual chunk manifest (step 4, chunk.py) and its
chunk_manifest.yaml sidecar. Writes an immutable named index:

    <out-dir>/bm25_<config_hash12>/
        bm25.pkl      pickled BM25Okapi (corpus order == doc_ids order)
        doc_ids.json  chunk_id list aligned with the pickled corpus
        meta.json     {config_hash, n_chunks, created_at}

The directory name pins the ingestion config hash: rebuilding into an
existing non-empty index dir is refused (delete it first), so a stale
index can never silently shadow a chunking-config change. The eval
(Phase 2) loads the index via load_index() and MUST tokenize queries
with the same tokenize() below -- BM25 scores are meaningless if the
index-time and query-time tokenizers differ.

Reads:
    --manifest        full/manifest_full.jsonl (step 4 output)
    --chunk-manifest  full/chunk_manifest.yaml (step 4 sidecar)
Writes:
    <out-dir>/bm25_<config_hash12>/
"""
import argparse
import datetime
import json
import os
import pickle
import re
import sys

import yaml

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("missing dep: pip install rank_bm25")


def tokenize(text):
    """Lowercase alphanumeric tokens. Queries MUST use this same function."""
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks):
    """chunks: [{chunk_id, text, ...}]. Returns (BM25Okapi, doc_ids)."""
    doc_ids = [c["chunk_id"] for c in chunks]
    corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus), doc_ids


def index_dir_name(config_hash):
    return f"bm25_{str(config_hash)[:12]}"


def save_index(bm25, doc_ids, config_hash, out_dir):
    """Write the index dir. Refuses to overwrite a non-empty index dir."""
    target = os.path.join(out_dir, index_dir_name(config_hash))
    if os.path.exists(target) and os.listdir(target):
        sys.exit(f"index dir exists and is not empty: {target}\n"
                 f"delete it first if you really want to rebuild")
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    with open(os.path.join(target, "doc_ids.json"), "w",
              encoding="utf-8") as f:
        json.dump(doc_ids, f)
    meta = {
        "config_hash": str(config_hash),
        "n_chunks": len(doc_ids),
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(target, "meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return target


def load_index(index_dir):
    """Load (BM25Okapi, doc_ids, meta) from an index dir."""
    with open(os.path.join(index_dir, "bm25.pkl"), "rb") as f:
        bm25 = pickle.load(f)
    with open(os.path.join(index_dir, "doc_ids.json"),
              encoding="utf-8") as f:
        doc_ids = json.load(f)
    with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return bm25, doc_ids, meta


def search(bm25, doc_ids, query, top_k=10):
    """Return [(chunk_id, score)] for the query, best first."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i],
                    reverse=True)
    return [(doc_ids[i], float(scores[i])) for i in ranked[:top_k]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="build BM25 index (step 7a)")
    ap.add_argument("--manifest", required=True,
                    help="full/manifest_full.jsonl from step 4")
    ap.add_argument("--chunk-manifest", required=True,
                    help="full/chunk_manifest.yaml sidecar from step 4")
    ap.add_argument("--out-dir", default="indexes",
                    help="parent dir for the named index (default: indexes)")
    args = ap.parse_args(argv)

    with open(args.chunk_manifest, encoding="utf-8") as f:
        cm = yaml.safe_load(f)
    config_hash = cm.get("config_hash")
    if not config_hash:
        sys.exit("chunk_manifest.yaml has no config_hash -- "
                 "refusing to build an unnamed index")

    chunks = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        sys.exit(f"no chunks in {args.manifest}")
    print(f"building BM25 over {len(chunks)} chunks "
          f"(config_hash {str(config_hash)[:12]}...)")
    bm25, doc_ids = build_bm25(chunks)
    target = save_index(bm25, doc_ids, config_hash, args.out_dir)
    print(f"wrote {target} ({len(doc_ids)} docs)")


if __name__ == "__main__":
    main()
