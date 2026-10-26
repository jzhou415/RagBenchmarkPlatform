#!/usr/bin/env python3
"""
metrics.py -- step 11: retrieval metrics for RagBenchmarkPlatform.

Pure functions, no API calls, no model loads. A ranked list in,
numbers out -- step 12 (eval runner) wires retrieve.py's four rungs
to these functions; this module never touches an index itself.

Inputs:
    ranked: [(chunk_id, score)] best-first, as retrieve.py returns.
            Scores are ignored here -- only order matters.
    ground_truth: {chunk_id: relevance} with relevance in {1, 2, 3}
            (0s are omitted, never listed -- see validate_golden.py).
            Relevance >= 1 counts as "relevant" for the binary metrics
            (hit, recall, precision, MRR); the grade itself only
            matters for nDCG, which is exactly why the golden set
            is graded instead of binary: nDCG is the metric that can
            show a reranker moving a grade-3 chunk above grade-1s.

Metrics (per question, at each k in ks):
    hit@k       1.0 if any relevant chunk is in the top k, else 0.0
    recall@k    relevant-in-top-k / total relevant
    precision@k relevant-in-top-k / chunks actually returned in top k
                (denominator is len(top-k), not k, so a ranking
                shorter than k is not penalised for missing slots)
    mrr         1 / rank of the first relevant chunk (0.0 if none)
    ndcg@k      graded DCG / ideal DCG, gains 2**rel - 1, log2 discount.
                0.0 when there is nothing relevant to rank.

Aggregation: mean_metrics() averages per-question dicts; evaluate_run()
does one question set end to end and returns per-question + mean.
Grouping (by query_type / content_tags) is step 12's job -- it owns
the golden-set rows, this module owns the arithmetic.

Run-level latency: latency_stats() summarizes per-query latencies
(p50/p95/p99 + mean + n) for step 12's quality-latency report; the
caller owns the unit (use one consistently, e.g. seconds or ms).

Edge cases, all defined rather than crashing:
    - empty ranking -> every metric 0.0
    - empty ground truth -> every metric 0.0 (nothing to find)
    - duplicate chunk_id in a ranking -> first occurrence only
    - k larger than the ranking -> metrics computed over the ranking

CLI (single-question scoring, ad-hoc use):
    python metrics.py --ranked ranked.json --ground-truth gt.json \
        --ks 3,5,10
    ranked.json: [{"chunk_id": ..., "score": ...}] or ["chunk_id", ...]
    gt.json: {"chunk_id": relevance, ...}
"""
import argparse
import json
import math
import sys


def _dedup_ids(ranked):
    """chunk_ids best-first, first occurrence only, scores dropped."""
    seen = set()
    ids = []
    for item in ranked:
        cid = item[0] if isinstance(item, (list, tuple)) else item
        if cid not in seen:
            seen.add(cid)
            ids.append(cid)
    return ids


def _relevant_set(ground_truth):
    """chunk_ids with relevance >= 1 (0s are omitted upstream anyway)."""
    return {cid for cid, rel in ground_truth.items() if rel >= 1}


def hit_at_k(ranked, ground_truth, k):
    ids = _dedup_ids(ranked)[:k]
    relevant = _relevant_set(ground_truth)
    return 1.0 if any(cid in relevant for cid in ids) else 0.0


def recall_at_k(ranked, ground_truth, k):
    relevant = _relevant_set(ground_truth)
    if not relevant:
        return 0.0
    found = sum(1 for cid in _dedup_ids(ranked)[:k] if cid in relevant)
    return found / len(relevant)


def precision_at_k(ranked, ground_truth, k):
    ids = _dedup_ids(ranked)[:k]
    if not ids:
        return 0.0
    relevant = _relevant_set(ground_truth)
    found = sum(1 for cid in ids if cid in relevant)
    return found / len(ids)


def reciprocal_rank(ranked, ground_truth):
    relevant = _relevant_set(ground_truth)
    for rank, cid in enumerate(_dedup_ids(ranked), start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(ids, ground_truth):
    """Sum over ranks of (2**rel - 1) / log2(rank + 1)."""
    total = 0.0
    for rank, cid in enumerate(ids, start=1):
        rel = ground_truth.get(cid, 0)
        if rel >= 1:
            total += (2 ** rel - 1) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked, ground_truth, k):
    ids = _dedup_ids(ranked)[:k]
    if not ids or not _relevant_set(ground_truth):
        return 0.0
    ideal = sorted((rel for rel in ground_truth.values() if rel >= 1),
                   reverse=True)[:k]
    # Ideal ranking: grades descending; only the grades matter to DCG.
    ideal_dcg = sum((2 ** rel - 1) / math.log2(rank + 1)
                    for rank, rel in enumerate(ideal, start=1))
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(ids, ground_truth) / ideal_dcg


def evaluate_question(ranked, ground_truth, ks=(3, 5, 10)):
    """Per-question metrics dict. Keys: hit@k, recall@k, precision@k
    for each k, plus mrr and ndcg@k for each k."""
    out = {"mrr": reciprocal_rank(ranked, ground_truth)}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, ground_truth, k)
        out[f"recall@{k}"] = recall_at_k(ranked, ground_truth, k)
        out[f"precision@{k}"] = precision_at_k(ranked, ground_truth, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, ground_truth, k)
    return out


def mean_metrics(per_question):
    """Mean of each metric over a list of per-question dicts.

    Returns {} for an empty list (no questions, no mean -- not 0.0,
    which would read as a measured zero).
    """
    if not per_question:
        return {}
    keys = per_question[0].keys()
    return {key: sum(m[key] for m in per_question) / len(per_question)
            for key in keys}


def evaluate_run(per_question_rankings, ks=(3, 5, 10)):
    """per_question_rankings: {qid: (ranked, ground_truth)}.

    Returns {"per_question": {qid: metrics}, "mean": {...}}.
    """
    per_q = {qid: evaluate_question(ranked, gt, ks)
             for qid, (ranked, gt) in per_question_rankings.items()}
    return {"per_question": per_q,
            "mean": mean_metrics(list(per_q.values()))}


def _percentile(sorted_vals, p):
    """Linear-interpolation percentile (numpy-style) over sorted values."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = p / 100 * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_vals[int(rank)])
    w = rank - lo
    return float(sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w)


def latency_stats(latencies):
    """p50/p95/p99 + mean + n over per-query latencies.

    Empty input gives None percentiles (no data, not a measured zero).
    """
    s = sorted(latencies)
    if not s:
        return {"p50": None, "p95": None, "p99": None,
                "mean": None, "n": 0}
    return {"p50": _percentile(s, 50), "p95": _percentile(s, 95),
            "p99": _percentile(s, 99),
            "mean": sum(s) / len(s), "n": len(s)}


def _load_ranked(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ranked = []
    for item in data:
        if isinstance(item, dict):
            ranked.append((item["chunk_id"], item.get("score", 0.0)))
        else:
            ranked.append((item, 0.0))
    return ranked


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="score one ranked list against a golden set (step 11)")
    ap.add_argument("--ranked", required=True,
                    help='JSON list of {"chunk_id": ..., "score": ...} '
                         'or plain chunk_id strings')
    ap.add_argument("--ground-truth", required=True,
                    help='JSON object {"chunk_id": relevance}')
    ap.add_argument("--ks", default="3,5,10",
                    help="comma-separated k values (default 3,5,10)")
    ap.add_argument("--out", default=None,
                    help="write the metrics JSON here")
    args = ap.parse_args(argv)

    try:
        ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    except ValueError:
        sys.exit("--ks must be comma-separated integers")
    if not ks:
        sys.exit("--ks must list at least one k")

    ranked = _load_ranked(args.ranked)
    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)
    result = evaluate_question(ranked, ground_truth, ks)
    for key in sorted(result):
        print(f"{key}  {result[key]:.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
