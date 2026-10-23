#!/usr/bin/env python3
"""
index_dense.py -- step 7b: embed the full corpus, store in Chroma.

RagBenchmarkPlatform, Phase 1 (Data Setup backend).

Reads the full-manual chunk manifest (step 4, chunk.py) and its
chunk_manifest.yaml sidecar. Writes an immutable named Chroma index:

    <out-dir>/dense_<config_hash12>/          Chroma persistent dir
    <out-dir>/dense_<config_hash12>.meta.json
        {config_hash, embedding_model, n_chunks, created_at}

Collection name: chunks_<config_hash12>. ids=chunk_id,
metadatas={page_no, section}. Embeddings are computed with
text-embedding-3-small (the ingestion config's embedding model) and
supplied explicitly, so Chroma never needs its own embedding function.
They are cached by chunk_id in <out-dir>/.embed_cache_<hash12>.json,
making reruns safe without re-paying for embeddings.

Like index_bm25.py, rebuilding into an existing non-empty index dir is
refused -- the directory name pins the ingestion config hash.

Requires OPENAI_API_KEY in the environment (never paste it in chat).

Reads:
    --manifest        full/manifest_full.jsonl (step 4 output)
    --chunk-manifest  full/chunk_manifest.yaml (step 4 sidecar)
Writes:
    <out-dir>/dense_<config_hash12>/
"""
import argparse
import datetime
import json
import os
import sys

import yaml

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100
# Chroma rejects a single add() over its max batch size (observed 5461);
# stay comfortably under it.
ADD_BATCH = 5000


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("missing dep: pip install openai")
    return OpenAI()


def embed_texts(client, texts):
    """Embed a list of texts; returns list of vectors. Batched."""
    out = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def embed_with_cache(client, chunks, cache_path):
    """Embed chunk texts; cache maps chunk_id -> vector (rerun-safe)."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    missing = [c for c in chunks if c["chunk_id"] not in cache]
    if missing:
        print(f"embedding {len(missing)} chunks with {EMBED_MODEL}...")
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        vecs = embed_texts(client, [c["text"] for c in batch])
        for c, v in zip(batch, vecs):
            cache[c["chunk_id"]] = v
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        done = min(start + EMBED_BATCH, len(missing))
        print(f"  embedded {done}/{len(missing)}", flush=True)
    return [cache[c["chunk_id"]] for c in chunks]


def build_dense_index(chunks, persist_dir, collection_name, vectors):
    """Store chunks + precomputed vectors in a persistent Chroma collection.

    Returns the collection. Refuses to overwrite a non-empty persist dir.
    """
    try:
        import chromadb
    except ImportError:
        sys.exit("missing dep: pip install chromadb")
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        sys.exit(f"index dir exists and is not empty: {persist_dir}\n"
                 f"delete it first if you really want to rebuild")
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"})
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [{"page_no": c["page_no"], "section": c.get("section", "")}
             for c in chunks]
    for start in range(0, len(chunks), ADD_BATCH):
        end = start + ADD_BATCH
        col.add(ids=ids[start:end], documents=docs[start:end],
                metadatas=metas[start:end], embeddings=vectors[start:end])
    return col


def query_dense(persist_dir, collection_name, query_vector, top_k=10):
    """Return [(chunk_id, distance)] best first. Lower distance = closer."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(name=collection_name)
    res = col.query(query_embeddings=[query_vector], n_results=top_k)
    return list(zip(res["ids"][0], res["distances"][0]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="build dense Chroma index (step 7b)")
    ap.add_argument("--manifest", required=True,
                    help="full/manifest_full.jsonl from step 4")
    ap.add_argument("--chunk-manifest", required=True,
                    help="full/chunk_manifest.yaml sidecar from step 4")
    ap.add_argument("--out-dir", default="indexes",
                    help="parent dir for the named index (default: indexes)")
    args = ap.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set in environment")

    with open(args.chunk_manifest, encoding="utf-8") as f:
        cm = yaml.safe_load(f)
    config_hash = cm.get("config_hash")
    if not config_hash:
        sys.exit("chunk_manifest.yaml has no config_hash -- "
                 "refusing to build an unnamed index")
    tag = str(config_hash)[:12]

    chunks = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if not chunks:
        sys.exit(f"no chunks in {args.manifest}")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, f".embed_cache_{tag}.json")
    client = make_client()
    vectors = embed_with_cache(client, chunks, cache_path)

    persist_dir = os.path.join(args.out_dir, f"dense_{tag}")
    collection_name = f"chunks_{tag}"
    build_dense_index(chunks, persist_dir, collection_name, vectors)

    meta = {
        "config_hash": str(config_hash),
        "embedding_model": EMBED_MODEL,
        "n_chunks": len(chunks),
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    meta_path = os.path.join(args.out_dir, f"dense_{tag}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {persist_dir} ({len(chunks)} docs, {EMBED_MODEL})")


if __name__ == "__main__":
    main()
