#!/usr/bin/env python3
"""
review_report.py -- step 13 review report: automatic checks on a golden set.

RagBenchmarkPlatform, Phase 3 (Data Setup page support).

Reads a golden YAML (auto_map.py output) + the chunk manifest +
the shared embedding cache (auto_map's chunk_embeddings.json) and
writes a JSON report the Data Setup page renders as a flags table.

Advisory only: a flag means "look here before approving", never an
automatic reject -- the human approves the set.

Checks per question:
  1. retrievability: is any graded chunk in the top-K of BM25
     (rebuilt in-memory from the manifest) and, when question vectors
     can be embedded, of dense cosine over the cached chunk vectors?
     A question neither signal retrieves is either a weak question or
     a retrieval gap worth knowing about before approval.
  2. answer-support: every number and every underscore-token (config
     names like shared_buffers) in the expected answer must appear in
     at least one grade-3 chunk. Empty / "..." expected answers flagged.
  3. duplicates: normalized-text exact dupes + token Jaccard >= 0.8
     pairs. Token overlap, not embedding similarity -- labeled as such.
  4. coverage: counts per query_type, per content_tag, and per
     query_type x content_tag; empty query_type or content_tag strata
     flagged. No plan file exists, so "vs plan" is counts + zeros, not
     a pass/fail against a requested distribution.

The dense check embeds questions with text-embedding-3-small (the
ingestion config's model). Without OPENAI_API_KEY it is reported
skipped, not failed.

Reads:
  --golden         golden set YAML (auto_map.py output)
  --manifest       manifest.jsonl from chunk.py
  --cache          dir holding chunk_embeddings.json (auto_map's
                   --cache); omit to skip the dense check
Writes:
  --out            review report JSON

Deps: pip install pyyaml rank_bm25 numpy (openai only for the dense check)
Runs on a local machine.
"""

import argparse
import json
import os
import re
import sys

import yaml

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("missing dep: pip install rank_bm25")

try:
    import numpy as np
except ImportError:
    sys.exit("missing dep: pip install numpy")

EMBED_MODEL = "text-embedding-3-small"   # must match IngestionConfig
QUERY_TYPES = ("exact_term", "paraphrase", "multi_hop")
CONTENT_TAGS = ("lookup", "table", "policy", "aggregation")
DUP_JACCARD = 0.8


def tokenize(text):
    return re.findall(r"\w+", (text or "").lower())


def normalize_question(text):
    return " ".join(tokenize(text))


def load_manifest(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_golden(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_cache(cache_dir):
    """chunk_embeddings.json under cache_dir, or {} if absent."""
    if not cache_dir:
        return {}
    path = os.path.join(cache_dir, "chunk_embeddings.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bm25_top(bm25, chunks, query_text, top_k):
    scores = bm25.get_scores(tokenize(query_text))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return {chunks[i]["chunk_id"] for i in order[:top_k]}


def dense_top(chunk_ids, chunk_vecs, query_vec, top_k):
    """chunk_ids/vecs aligned lists; cosine top-k as a set of ids."""
    if not chunk_ids:
        return set()
    q = np.asarray(query_vec, dtype=np.float64)
    mat = np.asarray(chunk_vecs, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12
    sims = (mat @ q) / norms
    order = sorted(range(len(chunk_ids)), key=lambda i: sims[i], reverse=True)
    return {chunk_ids[i] for i in order[:top_k]}


def numbers_and_labels(text):
    """Numbers + underscore-tokens an expected answer asserts."""
    text = text or ""
    numbers = set(re.findall(r"\d[\d,]*\.?\d*[A-Za-z]*", text))
    labels = {t for t in tokenize(text) if "_" in t}
    return numbers, labels


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def grade3_chunks(question):
    return [g["chunk_id"] for g in question.get("ground_truth", [])
            if g.get("relevance") == 3]


def check_retrievability(question, bm25, chunks, cache_vecs, query_vec,
                         top_k):
    """Flags when no graded chunk is retrieved. query_vec None -> dense
    check skipped (caller reports the reason once, not per question)."""
    flags = []
    graded = {g["chunk_id"] for g in question.get("ground_truth", [])}
    if not graded:
        return ["no graded chunks at all"]
    if not (graded & bm25_top(bm25, chunks, question["question"], top_k)):
        flags.append(f"no graded chunk in BM25 top-{top_k}")
    if query_vec is not None and cache_vecs:
        ids = [cid for cid in cache_vecs]
        vecs = [cache_vecs[cid] for cid in ids]
        if not (graded & dense_top(ids, vecs, query_vec, top_k)):
            flags.append(f"no graded chunk in dense top-{top_k}")
    return flags


def check_answer_support(question, chunk_texts):
    """Numbers/labels in the expected answer must appear in a grade-3
    chunk. chunk_texts: {chunk_id: text}."""
    flags = []
    expected = (question.get("expected_answer") or "").strip()
    if not expected or expected == "...":
        return ["expected answer empty or placeholder"]
    g3 = grade3_chunks(question)
    if not g3:
        return ["no grade-3 chunk to check answer against"]
    joined = " ".join(chunk_texts.get(cid, "") for cid in g3).lower()
    numbers, labels = numbers_and_labels(expected)
    for n in sorted(numbers):
        if n.lower() not in joined:
            flags.append(f"number {n} not in any grade-3 chunk")
    for lab in sorted(labels):
        if lab not in joined:
            flags.append(f"label {lab} not in any grade-3 chunk")
    return flags


def check_duplicates(questions):
    """{qid: [flags]} for exact + near-duplicate question pairs."""
    flags = {q["id"]: [] for q in questions}
    normed = [(q["id"], normalize_question(q.get("question", "")))
              for q in questions]
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            id_i, ni = normed[i]
            id_j, nj = normed[j]
            if not ni or not nj:
                continue
            if ni == nj:
                flags[id_i].append(f"duplicate of {id_j}")
                flags[id_j].append(f"duplicate of {id_i}")
            else:
                sim = jaccard(ni.split(), nj.split())
                if sim >= DUP_JACCARD:
                    flags[id_i].append(
                        f"near-duplicate of {id_j} (jaccard {sim:.2f})")
                    flags[id_j].append(
                        f"near-duplicate of {id_i} (jaccard {sim:.2f})")
    return flags


def coverage(questions):
    """Counts + empty-marginal flags. Cross table reported, not flagged:
    a small set cannot fill every query_type x content_tag cell."""
    by_type = {t: 0 for t in QUERY_TYPES}
    by_tag = {t: 0 for t in CONTENT_TAGS}
    by_both = {}
    for q in questions:
        qt = q.get("query_type")
        if qt in by_type:
            by_type[qt] += 1
        for tag in q.get("content_tags") or []:
            if tag in by_tag:
                by_tag[tag] += 1
            key = f"{qt} x {tag}"
            by_both[key] = by_both.get(key, 0) + 1
    flags = ([f"no {t} questions" for t, n in by_type.items() if n == 0]
             + [f"no {t} questions" for t, n in by_tag.items() if n == 0])
    return {"by_query_type": by_type, "by_content_tag": by_tag,
            "by_both": by_both, "flags": flags}


def build_report(golden, chunks, cache, embed_fn, top_k=10):
    """embed_fn(texts) -> vectors, or None to skip the dense check."""
    questions = golden.get("questions", [])
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    chunk_texts = {c["chunk_id"]: c["text"] for c in chunks}
    dup_flags = check_duplicates(questions)

    dense_status = "skipped: no chunk cache"
    query_vecs = {}
    if cache and embed_fn is not None:
        dense_status = "ok"
        texts = [q.get("question", "") for q in questions]
        for q, vec in zip(questions, embed_fn(texts)):
            query_vecs[q["id"]] = vec
    elif cache:
        dense_status = "skipped: no OPENAI_API_KEY (questions not embedded)"

    per_question = []
    n_flagged = 0
    for q in questions:
        flags = []
        flags += check_retrievability(
            q, bm25, chunks, cache, query_vecs.get(q["id"]), top_k)
        flags += check_answer_support(q, chunk_texts)
        flags += dup_flags.get(q["id"], [])
        if flags:
            n_flagged += 1
        per_question.append({"id": q["id"], "flags": flags})

    cov = coverage(questions)
    return {
        "golden": golden.get("ingestion", {}).get("config_hash", ""),
        "n_questions": len(questions),
        "n_flagged": n_flagged,
        "top_k": top_k,
        "dense": dense_status,
        "questions": per_question,
        "coverage": cov,
    }


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("missing dep: pip install openai (dense check only)")
    return OpenAI()


def embed_texts(client, texts, batch=100):
    out = []
    for start in range(0, len(texts), batch):
        resp = client.embeddings.create(
            model=EMBED_MODEL, input=texts[start:start + batch])
        out.extend(d.embedding for d in resp.data)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="automatic review checks on a golden set (advisory)")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", default=None,
                    help="auto_map cache dir holding chunk_embeddings.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args(argv)

    golden = load_golden(args.golden)
    chunks = load_manifest(args.manifest)
    cache = load_cache(args.cache)
    embed_fn = None
    if cache and "OPENAI_API_KEY" in os.environ:
        client = make_client()
        embed_fn = lambda texts: embed_texts(client, texts)

    report = build_report(golden, chunks, cache, embed_fn, args.top_k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"questions: {report['n_questions']}  "
          f"flagged: {report['n_flagged']}  dense: {report['dense']}")
    for q in report["questions"]:
        for flag in q["flags"]:
            print(f"FLAG {q['id']}: {flag}")
    for flag in report["coverage"]["flags"]:
        print(f"COVERAGE: {flag}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
