#!/usr/bin/env python3
"""
validate_golden.py -- step 6: schema + integrity check for a golden set.

RagBenchmarkPlatform, Phase 1 (Data Setup backend).

Fails the run BEFORE any index is built (step 7) if the golden set is
stale or malformed. Pure local check, no API calls.

Checks:
  - schema_version == "1.1"; top-level keys corpus/ingestion/questions
  - ingestion.config_hash equals the chunk manifest's config_hash
    (chunk.py's sidecar, computed by src/config.py) -- a changed
    parser/chunker invalidates the set loudly, never silently
  - every question: unique id, split in {dev, heldout},
    query_type in {exact_term, paraphrase, multi_hop},
    content_tags subset of {lookup, table, policy, aggregation},
    provenance in {auto, spot-checked, human-verified}
  - every question has >= 1 grade-3 chunk (else nDCG has nothing to
    rank at the top)
  - relevance in {1, 2, 3} (0s are omitted, never listed)
  - every ground_truth chunk_id exists in the manifest, and its
    recorded page equals the manifest's page_no for that chunk
  - no duplicate chunk ids within a question
  - chunk_id format p<page>_c<nn> (warning only)

Usage:
  python validate_golden.py --golden golden_dev.yaml \
      --manifest manifest_dev.jsonl --chunk-manifest chunk_manifest.yaml

Exit 0 = valid, 1 = invalid, 2 = usage/IO error.
"""

import argparse
import json
import re
import sys

import yaml

QUERY_TYPES = {"exact_term", "paraphrase", "multi_hop"}
CONTENT_TAGS = {"lookup", "table", "policy", "aggregation"}
PROVENANCE = {"auto", "spot-checked", "human-verified"}
SPLITS = {"dev", "heldout"}
CHUNK_ID_RE = re.compile(r"^p\d+_c\d+$")


def load_manifest_pages(path):
    """chunk_id -> page_no, from chunk.py's manifest.jsonl."""
    pages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                pages[row["chunk_id"]] = row["page_no"]
    return pages


def validate(golden, manifest_pages, expected_config_hash):
    """Returns (errors, warnings): lists of human-readable strings."""
    errors, warnings = [], []

    if golden.get("schema_version") != "1.1":
        errors.append("schema_version must be '1.1'")
    for key in ("corpus", "ingestion", "questions"):
        if key not in golden:
            errors.append(f"missing top-level key: {key}")

    ing = golden.get("ingestion", {})
    if ing.get("config_hash") != expected_config_hash:
        errors.append(
            "ingestion.config_hash does not match the chunk manifest -- "
            "the set was built against a different parser/chunker and "
            "must be rebuilt (re-run chunk.py + auto_map.py)")

    questions = golden.get("questions", [])
    seen_ids = set()
    for qi, q in enumerate(questions):
        qid = q.get("id", f"<index {qi}>")
        if q.get("id") in seen_ids:
            errors.append(f"{qid}: duplicate question id")
        seen_ids.add(q.get("id"))

        if q.get("split") not in SPLITS:
            errors.append(f"{qid}: bad split {q.get('split')!r} "
                          f"(want one of {sorted(SPLITS)})")
        if q.get("query_type") not in QUERY_TYPES:
            errors.append(f"{qid}: bad query_type {q.get('query_type')!r} "
                          f"(want one of {sorted(QUERY_TYPES)})")
        bad_tags = set(q.get("content_tags", [])) - CONTENT_TAGS
        if bad_tags:
            errors.append(f"{qid}: unknown content_tags {sorted(bad_tags)}")
        if q.get("provenance", "auto") not in PROVENANCE:
            errors.append(f"{qid}: bad provenance {q.get('provenance')!r} "
                          f"(want one of {sorted(PROVENANCE)})")

        gt = q.get("ground_truth", [])
        if not any(g.get("relevance") == 3 for g in gt):
            errors.append(f"{qid}: no grade-3 chunk -- "
                          f"eval has nothing to rank at the top")
        seen_chunks = set()
        for g in gt:
            cid = g.get("chunk_id")
            rel = g.get("relevance")
            if rel not in (1, 2, 3):
                errors.append(f"{qid}: {cid}: relevance {rel!r} "
                              f"(must be 1-3; 0s are omitted)")
            if cid in seen_chunks:
                errors.append(f"{qid}: duplicate chunk_id {cid}")
            seen_chunks.add(cid)
            if cid not in manifest_pages:
                errors.append(f"{qid}: chunk_id {cid} not in manifest")
            else:
                if g.get("page") != manifest_pages[cid]:
                    errors.append(
                        f"{qid}: {cid}: page {g.get('page')!r} != manifest "
                        f"page_no {manifest_pages[cid]}")
                if not CHUNK_ID_RE.match(cid or ""):
                    warnings.append(f"{qid}: chunk_id {cid} has unusual format")
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="validate a golden set against its chunk manifest")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--chunk-manifest", required=True)
    args = ap.parse_args(argv)

    with open(args.golden, encoding="utf-8") as f:
        golden = yaml.safe_load(f)
    with open(args.chunk_manifest, encoding="utf-8") as f:
        cm = yaml.safe_load(f)
    manifest_pages = load_manifest_pages(args.manifest)

    errors, warnings = validate(golden, manifest_pages,
                                 cm.get("config_hash", ""))

    if errors:
        print(f"INVALID: {len(errors)} error(s), "
              f"{len(warnings)} warning(s)\n")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  warn:  {w}")
        return 1
    print(f"VALID: {len(golden.get('questions', []))} questions, "
          f"{len(warnings)} warning(s)")
    for w in warnings:
        print(f"  warn:  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
