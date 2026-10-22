#!/usr/bin/env python3
"""
auto_map.py -- step 5: LLM-assisted golden-set chunk mapping.

RagBenchmarkPlatform, Phase 1 (Data Setup backend).

Turns draft questions (step 3, question_gen.py) into a golden set by
mapping each question to chunks from the ingestion manifest (step 4,
chunk.py) with graded relevance 0-3:

  1. Candidate pools: per question, union of top-20 BM25 + top-20
     dense (text-embedding-3-small, the ingestion config's embedding
     model), PLUS every chunk on the question's source_pages (forced
     in, so a source page can never silently miss grading). Two
     different signals on purpose -- a pool built from one retriever
     inherits that retriever's blind spots.
  2. Verbatim shortcut: chunks containing the expected answer verbatim
     are pre-marked as grade-3 candidates for the judge to confirm.
  3. gpt-4o-mini grades every candidate 0-3 against the rubric.
     Incomplete grading batches (dropped candidates, unparseable JSON)
     are retried once -- without the retry, judge misses silently become
     grade 0.
  4. Emits golden YAML (schema v1.1) + review_queue.yaml: all
     multi_hop questions, every question with no grade-3 chunk, every
     question where the judge downgraded a pre-marked chunk, and a
     ~20% random sample of the rest -- the human spot-check list.
     Reviewed questions get their provenance upgraded by hand.

Provenance: the golden set's ingestion.config_hash is copied from
chunk_manifest.yaml (step 4's sidecar, computed by src/config.py).
That hash is the pin: validate_golden.py (step 6) refuses the set if
the chunking config has changed since.

Reads:
  --questions        draft questions YAML (question_gen.py output):
                     [{id, question, query_type, content_tags,
                       expected_answer, source_pages}]
  --manifest         manifest.jsonl from chunk.py:
                     {chunk_id, page_no, section, text, char_count}
  --chunk-manifest   chunk_manifest.yaml sidecar from chunk.py
  --split            dev | heldout (stamped on every question)
Writes:
  --out              golden set YAML (schema v1.1)
  --review-queue     review_queue.yaml (human spot-check list)
  --cache            dir for cached dense embeddings (rerun-safe,
                     keyed by chunk_id)

Deps: pip install pyyaml rank_bm25 numpy openai
Runs on a local machine; needs OPENAI_API_KEY in the environment.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

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
JUDGE_MODEL = "gpt-4o-mini"
TOP_N = 20            # candidates per retriever
GRADING_BATCH = 6     # candidates per LLM grading call
SPOT_CHECK_FRAC = 0.2
PARSER_NAME = "docling"   # descriptive only; config_hash is the real pin

RUBRIC = """\
3 = chunk contains the exact answer
2 = directly answers most of the question
1 = background / useful context only
0 = not relevant"""


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_manifest(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_chunk_manifest(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bm25(chunks):
    return BM25Okapi([tokenize(c["text"]) for c in chunks])


def get_embeddings(client, chunks, cache_path):
    """Embed chunk texts; cache maps chunk_id -> vector (rerun-safe)."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    missing = [c for c in chunks if c["chunk_id"] not in cache]
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        resp = client.embeddings.create(
            model=EMBED_MODEL, input=[c["text"] for c in batch])
        for c, emb in zip(batch, resp.data):
            cache[c["chunk_id"]] = emb.embedding
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return [cache[c["chunk_id"]] for c in chunks]


def embed_query(client, text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def candidate_pool(question, chunks, bm25, chunk_vecs, query_vec):
    """Union of top-N BM25 and top-N dense candidate indices."""
    q_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top = set(sorted(range(len(chunks)),
                          key=lambda i: bm25_scores[i],
                          reverse=True)[:TOP_N])

    q = np.asarray(query_vec, dtype=np.float64)
    mat = np.asarray(chunk_vecs, dtype=np.float64)
    sims = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-12)
    dense_top = set(sorted(range(len(chunks)),
                           key=lambda i: sims[i],
                           reverse=True)[:TOP_N])
    return sorted(bm25_top | dense_top)


def force_source_pages(pool_idx, chunks, source_pages):
    """Indices of chunks on the question's source pages not already pooled.

    The pool is only a pre-filter for the expensive judge; guaranteeing
    the source pages are judged cannot inflate grades (the judge still
    scores 0-3), but it closes the "source page never graded" miss class.
    """
    pooled = set(pool_idx)
    src = set(source_pages or [])
    return sorted(i for i, c in enumerate(chunks)
                  if c["page_no"] in src and i not in pooled)


def verbatim_hit(expected_answer, chunk_text):
    ans = (expected_answer or "").strip()
    if not ans or ans == "...":
        return False
    return ans.lower() in chunk_text.lower()


GRADING_SYSTEM = (
    "You grade how well a text chunk answers a question, using this rubric:\n"
    + RUBRIC + "\n"
    "Reply with JSON only: "
    '{"grades": [{"chunk_id": "...", "grade": 0-3, "rationale": "one sentence"}]}'
)


def _grade_batch(client, question, expected_answer, batch):
    """Grade one batch. Returns ({chunk_id: (grade, rationale)}, complete).

    complete is False when the judge's JSON was unparseable or omitted
    candidates -- both used to silently become grade 0.
    """
    user = (
        f"Question: {question}\n"
        f"Expected answer: {expected_answer or '(not provided)'}\n\n"
    )
    for cand in batch:
        tag = (" [PRE-GRADED 3: the expected answer appears verbatim in "
               "this chunk -- keep grade 3 unless the match is clearly "
               "incidental, e.g. the answer string is part of a longer "
               "unrelated word]") if cand["premarked"] else ""
        user += f"--- chunk {cand['chunk_id']}{tag}\n{cand['text'][:2000]}\n\n"
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": GRADING_SYSTEM},
                  {"role": "user", "content": user}])
    raw = resp.choices[0].message.content
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        grades = json.loads(raw)["grades"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"WARNING: grading call failed to parse ({e})", file=sys.stderr)
        return {}, False
    out, seen = {}, set()
    for g in grades:
        cid = g.get("chunk_id")
        try:
            grade = max(0, min(3, int(g.get("grade", 0))))
        except (TypeError, ValueError):
            grade = 0
        if cid and cid not in seen:
            out[cid] = (grade, g.get("rationale", ""))
            seen.add(cid)
    complete = all(c["chunk_id"] in seen for c in batch)
    return out, complete


def grade_candidates(client, question, expected_answer, candidates):
    """candidates: [{chunk_id, text, premarked}].
    Returns {chunk_id: (grade, rationale)}. Never raises on bad judge
    output: incomplete batches are retried once, then any still-missing
    candidates become grade 0.
    """
    out = {}
    for start in range(0, len(candidates), GRADING_BATCH):
        batch = candidates[start:start + GRADING_BATCH]
        grades, complete = _grade_batch(client, question, expected_answer,
                                       batch)
        if not complete:
            # The judge sometimes drops candidates from its JSON or emits
            # unparseable output. One retry usually completes the batch;
            # without it, drops silently become grade 0 (observed 2026-10-21:
            # a pre-marked verbatim chunk graded 0).
            print("  WARNING: incomplete grading batch, retrying once",
                  file=sys.stderr)
            retry_grades, _ = _grade_batch(client, question,
                                           expected_answer, batch)
            for cid, g in retry_grades.items():
                grades.setdefault(cid, g)
        for cand in batch:
            out[cand["chunk_id"]] = grades.get(
                cand["chunk_id"], (0, "not graded by judge"))
    return out


def build_review_queue(questions, no_grade3_ids, rng, overridden_ids=()):
    """All multi_hop + all no-grade-3 questions + pre-mark overrides +
    ~20% sample of the rest."""
    queued = {}
    for q in questions:
        if q.get("query_type") == "multi_hop":
            queued[q["id"]] = "multi_hop: full human review"
    for qid in no_grade3_ids:
        queued.setdefault(qid, "no grade-3 chunk found: needs human attention")
    for qid in overridden_ids:
        queued.setdefault(qid, "pre-marked chunk downgraded: verify by hand")
    rest = [q["id"] for q in questions if q["id"] not in queued]
    k = max(1, int(len(rest) * SPOT_CHECK_FRAC)) if rest else 0
    for qid in sorted(rng.sample(rest, min(k, len(rest)))):
        queued[qid] = "spot-check sample"
    return [{"id": qid, "reason": queued[qid]}
            for q in questions for qid in [q["id"]] if qid in queued]


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("missing dep: pip install openai")
    return OpenAI()


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(
        description="map draft questions to graded chunks (golden set)")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunk-manifest", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "heldout"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--review-queue", default="review_queue.yaml")
    ap.add_argument("--cache", default=".automap_cache")
    args = ap.parse_args(argv)

    if client is None:
        if "OPENAI_API_KEY" not in os.environ:
            print("OPENAI_API_KEY not set in environment", file=sys.stderr)
            return 2
        client = make_client()

    with open(args.questions, encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    cm = load_chunk_manifest(args.chunk_manifest)
    chunks = load_manifest(args.manifest)
    by_id = {c["chunk_id"]: c for c in chunks}
    print(f"loaded {len(questions)} questions, {len(chunks)} chunks "
          f"(config_hash {str(cm.get('config_hash'))[:12]}...)")

    pages_in_manifest = {c["page_no"] for c in chunks}
    for q in questions:
        for sp in q.get("source_pages") or []:
            if sp not in pages_in_manifest:
                print(f"WARNING {q['id']}: source page {sp} not in this "
                      f"manifest -- wrong split file?", file=sys.stderr)

    os.makedirs(args.cache, exist_ok=True)
    bm25 = build_bm25(chunks)
    chunk_vecs = get_embeddings(client, chunks,
                                os.path.join(args.cache, "chunk_embeddings.json"))
    print("candidate pools + grading...")

    golden_questions = []
    no_grade3_ids = []
    overridden_ids = []
    for q in questions:
        qtext = q["question"]
        q_vec = embed_query(client, qtext)
        pool_idx = candidate_pool(qtext, chunks, bm25, chunk_vecs, q_vec)
        forced = force_source_pages(pool_idx, chunks, q.get("source_pages"))
        if forced:
            print(f"  {q['id']}: +{len(forced)} source-page chunk(s) "
                  f"forced into pool")
            pool_idx = sorted(set(pool_idx) | set(forced))
        candidates = [{
            "chunk_id": chunks[i]["chunk_id"],
            "text": chunks[i]["text"],
            "premarked": verbatim_hit(q.get("expected_answer"),
                                      chunks[i]["text"]),
        } for i in pool_idx]
        grades = grade_candidates(client, qtext, q.get("expected_answer"),
                                  candidates)

        ground_truth = []
        for cand in candidates:
            grade, _ = grades[cand["chunk_id"]]
            if grade >= 1:
                ground_truth.append({
                    "chunk_id": cand["chunk_id"],
                    "page": by_id[cand["chunk_id"]]["page_no"],
                    "relevance": grade,
                })
        ground_truth.sort(key=lambda g: (-g["relevance"], g["chunk_id"]))

        golden_questions.append({
            "id": q["id"],
            "split": args.split,
            "provenance": "auto",   # review pass upgrades to spot-checked
            "question": qtext,
            "query_type": q.get("query_type"),
            "content_tags": q.get("content_tags", []),
            "expected_answer": q.get("expected_answer", ""),
            "answer_span": "",
            "ground_truth": ground_truth,
            "notes": "",
        })

        n3 = sum(1 for g in ground_truth if g["relevance"] == 3)
        print(f"  {q['id']}: pool={len(pool_idx)} "
              f"graded={len(ground_truth)} grade3={n3}")
        overridden = [cand["chunk_id"] for cand in candidates
                      if cand["premarked"] and grades[cand["chunk_id"]][0] < 3]
        if overridden:
            print(f"  WARNING {q['id']}: pre-mark overridden for "
                  f"{overridden} -- added to review queue", file=sys.stderr)
            overridden_ids.append(q["id"])
        if n3 == 0:
            no_grade3_ids.append(q["id"])
            print(f"  WARNING {q['id']}: no grade-3 chunk found -- "
                  f"added to review queue", file=sys.stderr)

    review = build_review_queue(questions, no_grade3_ids,
                                random.Random(42), overridden_ids)

    golden = {
        "schema_version": "1.1",
        "corpus": {
            "name": cm.get("corpus_name", ""),
            "source": "pdf",
            "pages": cm.get("n_pages", ""),
            "sha256": cm.get("corpus_sha256", ""),
        },
        "ingestion": {
            "config_hash": cm.get("config_hash", ""),
            "parser": PARSER_NAME,
            "chunk_size": cm.get("chunk_size", ""),
            "chunk_overlap": cm.get("chunk_overlap", ""),
        },
        "questions": golden_questions,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(golden, f, sort_keys=False, allow_unicode=True)
    with open(args.review_queue, "w", encoding="utf-8") as f:
        yaml.safe_dump({"review_queue": review}, f, sort_keys=False)

    print(f"wrote {args.out} ({len(golden_questions)} questions)")
    print(f"wrote {args.review_queue} ({len(review)} flagged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
