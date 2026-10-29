#!/usr/bin/env python3
"""eval_runner.py -- step 12: eval runner for RagBenchmarkPlatform.

Runs one golden set through retrieve.py's four rungs and scores every
ranking with step-11 metrics. One invocation = one golden set (one
split); run dev and held-out separately so the split discipline holds
(dev for iterating, held-out for the honest final number).

Per question, per rung:
    semantic, bm25, hybrid -- retrieve() directly
    hybrid_rerank@N -- retrieve() with candidate_k=N and the reranker
        injected (N sweep default 10/20/50: how many hybrid candidates
        the reranker sees. k is different: k is how many top results
        you score, see ks.)

Latency is end-to-end per query (including query embedding), measured
with perf_counter around each retrieve() call. Index-build cost is
not measured here -- it belongs to step 7, not per-query cost.

Scoring: rankings are cut to max(ks) before scoring, so MRR is strict
MRR@max(ks) (metrics.py computes full-list MRR; truncating to the
reporting cutoff makes it MRR@k). hit/recall/precision/nDCG at each k
are unaffected for k <= max(ks).

Outputs (designed so step 13's UI reads them without a re-run):
    --out results.json: run metadata + per-question rows (qid,
        question, query_type, content_tags, per-rung metrics,
        latency, ranked chunk ids+scores) + mean metrics +
        stratified means (by query_type; by content_tag -- a question
        with several tags counts in each tag's group) + latency
        p50/p95/p99 per rung.
    summary: printed to stdout, and to --summary-out if given.
"""
import argparse
import datetime
import json
import os
import sys
import time

import yaml

from src.metrics import evaluate_question, latency_stats, mean_metrics
from src.rerank import DEFAULT_CE_MODEL, DEFAULT_COHERE_MODEL, make_reranker
from src.retrieve import embed_query_text, load_indexes, load_texts, retrieve

RUNGS = ("semantic", "bm25", "hybrid")
DEFAULT_CANDIDATE_NS = (10, 20, 50)
DEFAULT_KS = (3, 5, 10)


def load_golden_questions(path):
    """Golden-set YAML -> [{id, split, question, query_type,
    content_tags, ground_truth}].

    ground_truth is collapsed to {chunk_id: relevance}, the shape
    evaluate_question takes (the YAML's per-entry page is dropped --
    metrics need only the chunk id and its grade). Parsed once here;
    the CLI's load_ground_truth (metrics.py) does the same collapse
    for a single question, but the runner already holds the parsed
    doc, so it does not re-read the file per question.
    """
    with open(path, encoding="utf-8") as f:
        golden = yaml.safe_load(f) or {}
    questions = []
    for q in golden.get("questions", []):
        questions.append({
            "id": q.get("id"),
            "split": q.get("split", ""),
            "question": q.get("question", ""),
            "query_type": q.get("query_type", ""),
            "content_tags": list(q.get("content_tags", [])),
            "ground_truth": {g["chunk_id"]: g["relevance"]
                             for g in q.get("ground_truth", [])},
        })
    return questions


def _stratified(per_question, key_fn):
    """{group: {rung: mean metrics}} over per-question rows.

    key_fn(row) -> list of groups the question belongs to (a question
    with several content_tags counts in each tag's group).
    """
    buckets = {}
    for row in per_question.values():
        for group in key_fn(row):
            buckets.setdefault(group, []).append(row)
    out = {}
    for group, rows in buckets.items():
        rungs = rows[0]["results"].keys()
        out[group] = {
            rung: mean_metrics([r["results"][rung]["metrics"]
                                for r in rows])
            for rung in rungs
        }
    return out


def run_eval(questions, search, ks=DEFAULT_KS,
             candidate_ns=DEFAULT_CANDIDATE_NS, rungs=None):
    """Core evaluation, no index or API access.

    search(question_text, rung_label) -> (ranked [(chunk_id, score)],
    latency_s). rung_label is "semantic" | "bm25" | "hybrid" |
    "hybrid_rerank@N". rungs (None = all) restricts the computation
    to a subset -- step 14's strategy-specific experiments run only
    their strategy, e.g. ("hybrid", "hybrid_rerank@10"). Unknown
    labels raise ValueError before any search runs. Returns
    {"per_question", "mean", "by_query_type", "by_content_tag",
    "latency"}.
    """
    max_k = max(ks)
    all_labels = list(RUNGS) + [f"hybrid_rerank@{n}"
                                for n in candidate_ns]
    if rungs is None:
        rung_labels = all_labels
    else:
        unknown = [r for r in rungs if r not in all_labels]
        if unknown:
            raise ValueError(
                f"unknown rung(s): {', '.join(unknown)} "
                f"(valid: {', '.join(all_labels)})")
        rung_labels = [label for label in all_labels if label in rungs]
    per_question = {}
    for q in questions:
        results = {}
        for label in rung_labels:
            ranked, latency = search(q["question"], label)
            ranked = ranked[:max_k]  # strict MRR@max_k semantics
            results[label] = {
                "metrics": evaluate_question(ranked, q["ground_truth"], ks),
                "latency_s": latency,
                "ranked": [{"chunk_id": cid, "score": s}
                           for cid, s in ranked],
            }
        per_question[q["id"]] = {
            "question": q["question"],
            "query_type": q["query_type"],
            "content_tags": q["content_tags"],
            "results": results,
        }
    mean = {
        label: mean_metrics([row["results"][label]["metrics"]
                             for row in per_question.values()])
        for label in rung_labels
    }
    by_query_type = _stratified(per_question,
                                lambda row: [row["query_type"]])
    by_content_tag = _stratified(per_question,
                                 lambda row: row["content_tags"])
    latency = {
        label: latency_stats([row["results"][label]["latency_s"]
                              for row in per_question.values()])
        for label in rung_labels
    }
    return {"per_question": per_question, "mean": mean,
            "by_query_type": by_query_type,
            "by_content_tag": by_content_tag, "latency": latency}


def summarize(output, ks):
    """Readable summary text over a run_eval results dict."""
    k_small, k_large = ks[0], ks[-1]
    run = output["run"]
    lines = [
        f"Eval summary: {len(output['per_question'])} questions "
        f"from {run['golden_set']} (split: {run['split']})",
        f"reranker: {run['reranker']} ({run['reranker_model']}), "
        f"candidate Ns: {run['candidate_ns']}, ks: {list(ks)}",
        f"{'rung':<18} {f'hit@{k_small}':>8} "
        f"{f'recall@{k_large}':>10} {f'ndcg@{k_large}':>9} {'p50_s':>7}",
    ]
    for rung, m in output["mean"].items():
        p50 = output["latency"][rung]["p50"]
        lines.append(
            f"{rung:<18} {m[f'hit@{k_small}']:>8.4f} "
            f"{m[f'recall@{k_large}']:>10.4f} "
            f"{m[f'ndcg@{k_large}']:>9.4f} "
            f"{f'{p50:.3f}' if p50 is not None else 'n/a':>7}")
    for axis, label in (("by_query_type", "query_type"),
                        ("by_content_tag", "content_tag")):
        for group, rungmap in output[axis].items():
            best_rung, best_m = max(
                rungmap.items(),
                key=lambda kv: kv[1][f"ndcg@{k_large}"])
            lines.append(
                f"Best by {label} (ndcg@{k_large}): {group} -> "
                f"{best_rung} ({best_m[f'ndcg@{k_large}']:.4f})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="eval runner (step 12): one golden set through "
                    "the four rungs")
    ap.add_argument("--golden-set", required=True,
                    help="golden-set YAML (one split per run)")
    ap.add_argument("--indexes", default="indexes",
                    help="parent dir of the named step-7 indexes")
    ap.add_argument("--bm25-index", default=None,
                    help="e.g. bm25_<hash12> (auto-discovered if "
                         "unambiguous)")
    ap.add_argument("--dense-index", default=None,
                    help="e.g. dense_<hash12> (auto-discovered if "
                         "unambiguous)")
    ap.add_argument("--manifest", required=True,
                    help="full/manifest_full.jsonl (texts for rung 4)")
    ap.add_argument("--reranker", default="cross-encoder",
                    choices=["cross-encoder", "cohere"],
                    help="default: local cross-encoder (no API key)")
    ap.add_argument("--reranker-model", default=None,
                    help="override the reranker default model")
    ap.add_argument("--candidate-ns", default="10,20,50",
                    help="comma-separated rerank candidate counts "
                         "(default 10,20,50)")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated subset of rungs to score "
                         "(e.g. 'semantic' or "
                         "'hybrid,hybrid_rerank@10'); default: all")
    ap.add_argument("--ks", default="3,5,10",
                    help="comma-separated k values (default 3,5,10)")
    ap.add_argument("--out", required=True,
                    help="write the results JSON here")
    ap.add_argument("--summary-out", default=None,
                    help="also write the readable summary here")
    args = ap.parse_args(argv)

    try:
        candidate_ns = tuple(int(x) for x in args.candidate_ns.split(",")
                             if x.strip())
    except ValueError:
        sys.exit("--candidate-ns must be comma-separated integers")
    if not candidate_ns:
        sys.exit("--candidate-ns must list at least one N")
    try:
        ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    except ValueError:
        sys.exit("--ks must be comma-separated integers")
    if not ks:
        sys.exit("--ks must list at least one k")
    rungs = (tuple(x.strip() for x in args.rungs.split(",") if x.strip())
             if args.rungs else None)
    if rungs is not None:
        valid = set(RUNGS) | {f"hybrid_rerank@{n}"
                              for n in candidate_ns}
        unknown = [r for r in rungs if r not in valid]
        if unknown:
            sys.exit(f"unknown rung(s): {', '.join(unknown)} "
                     f"(valid: {', '.join(sorted(valid))})")
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set in environment")

    questions = load_golden_questions(args.golden_set)
    if not questions:
        sys.exit(f"no questions in {args.golden_set}")
    splits = {q["split"] for q in questions if q["split"]}
    split = splits.pop() if len(splits) == 1 else "mixed"

    indexes = load_indexes(args.indexes, args.bm25_index, args.dense_index)
    texts = load_texts(args.manifest)
    from src.index_dense import make_client
    client = make_client()
    embed_fn = lambda text: embed_query_text(client, text)
    reranker = make_reranker(args.reranker, model=args.reranker_model,
                             api_key=os.environ.get("COHERE_API_KEY"))
    reranker_model = args.reranker_model or (
        DEFAULT_CE_MODEL if args.reranker == "cross-encoder"
        else DEFAULT_COHERE_MODEL)
    max_k = max(ks)

    def search(question_text, rung_label):
        t0 = time.perf_counter()
        if rung_label.startswith("hybrid_rerank@"):
            n = int(rung_label.split("@")[1])
            ranked = retrieve(question_text, "hybrid_rerank", indexes,
                              texts=texts, top_k=max_k, candidate_k=n,
                              rerank_fn=reranker.rerank, embed_fn=embed_fn)
        else:
            ranked = retrieve(question_text, rung_label, indexes,
                              top_k=max_k, embed_fn=embed_fn)
        return ranked, time.perf_counter() - t0

    results = run_eval(questions, search, ks=ks,
                       candidate_ns=candidate_ns, rungs=rungs)
    ran_ns = sorted({int(label.split("@")[1])
                     for label in results["mean"]
                     if label.startswith("hybrid_rerank@")})
    output = {
        "run": {
            "golden_set": args.golden_set,
            "split": split,
            "n_questions": len(questions),
            "config_hash": indexes["config_hash"],
            "indexes": {
                "bm25": os.path.basename(indexes["bm25"]["dir"]),
                "dense": os.path.basename(indexes["dense"]["persist_dir"]),
            },
            "reranker": args.reranker,
            "reranker_model": reranker_model,
            "candidate_ns": ran_ns,
            "ks": list(ks),
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        },
        **results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    summary = summarize(output, ks)
    print(summary)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write(summary + "\n")
        print(f"wrote {args.summary_out}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
