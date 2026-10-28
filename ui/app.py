"""ui/app.py -- RagBenchmarkPlatform Streamlit UI.

Pages (sidebar selector):
  Data Setup (step 13, new): pick one document, the indexes output dir,
    and the golden-set output dir, then Build runs the 7 pipeline stages
    in order through the FastAPI backend, with per-stage progress and a
    review-and-approve section for the golden set.
  Evaluation (step 14, placeholder): built by step 14 -- experiment
    registration, pending -> completed flow, runs through a new FastAPI
    endpoint. Only approved golden sets will be offered.
  Results (step 15 target, viewer moved here from the old step-13 app):
    reads a step-12 results JSON via upload or path. The cross-experiment
    comparison view is step 15's work.

Run from the repo root: streamlit run ui/app.py
The API must be running: uvicorn src.api.main:app (from the repo root).
The UI and the API run on the same machine, so the paths below are local
paths the backend reads and writes. OPENAI_API_KEY must be set in the
backend environment (question generation, auto-map judging, embeddings).

The JSON schema the Results page expects is exactly what eval_runner.py
writes:
    run: golden_set, split, n_questions, config_hash,
         indexes {bm25, dense}, reranker, reranker_model,
         candidate_ns, ks, timestamp
    per_question: {qid: {question, query_type, content_tags,
        results: {rung: {metrics, latency_s, ranked}}}}
    mean: {rung: {hit@k, recall@k, precision@k, mrr, ndcg@k}}
    by_query_type / by_content_tag: {group: {rung: metrics}}
    latency: {rung: {p50, p95, p99, mean, n}}
"""
import datetime
import json
import os
import time
import urllib.error
import urllib.request

import streamlit as st

# ---------------------------------------------------------------- Results
# Viewer helpers, unchanged from the old step-13 app.

REQUIRED_KEYS = ("run", "per_question", "mean", "latency")
METRIC_FAMILIES = ("hit", "recall", "precision", "ndcg")


def load_results(uploaded, path):
    """Results dict from the uploader (preferred) or a file path."""
    if uploaded is not None:
        return json.load(uploaded)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_schema(data):
    """Missing top-level keys -> error string, else None."""
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        return f"not a step-12 results file: missing {', '.join(missing)}"
    return None


def mean_rows(mean, latency, ks, family):
    """One row per rung: the family's @k columns + latency columns."""
    rows = []
    for rung, m in mean.items():
        row = {"rung": rung}
        for k in ks:
            row[f"{family}@{k}"] = round(m.get(f"{family}@{k}", 0.0), 4)
        lat = latency.get(rung, {})
        row["p50_s"] = lat.get("p50")
        row["p95_s"] = lat.get("p95")
        rows.append(row)
    return rows


def stratified_rows(grouped, ks, metric_key):
    """Rows = groups, columns = rungs, values = metric_key."""
    rungs = []
    for rungmap in grouped.values():
        for rung in rungmap:
            if rung not in rungs:
                rungs.append(rung)
    rows = []
    for group, rungmap in grouped.items():
        row = {"group": group}
        for rung in rungs:
            m = rungmap.get(rung, {})
            row[rung] = round(m.get(metric_key, 0.0), 4)
        rows.append(row)
    return rows


# ------------------------------------------------------------------- API
# Minimal stdlib client: the backend is the step-8 FastAPI app.


def api_get(base, path, timeout=10):
    req = urllib.request.Request(base.rstrip("/") + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def api_post_run(base, stage_id, args, timeout=10):
    body = json.dumps({"args": args}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + f"/stages/{stage_id}/run", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["run_id"]


def api_run_status(base, run_id, timeout=10):
    return api_get(base, f"/runs/{run_id}", timeout=timeout)


def api_run_log(base, run_id, tail=None, timeout=10):
    path = f"/runs/{run_id}/log"
    if tail is not None:
        path += f"?tail={tail}"
    return api_get(base, path, timeout=timeout).get("log", "")


def describe_api_error(err):
    """One readable line for a urllib failure."""
    if isinstance(err, urllib.error.HTTPError):
        try:
            detail = err.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - error path, keep it short
            detail = ""
        return f"HTTP {err.code}: {detail or err.reason}"
    return f"{type(err).__name__}: {err}"


# ------------------------------------------------------------ Data Setup
# Build chain: the 7 step-8 stages in pipeline order. All intermediate
# paths derive from the two output dirs the user picks, so a build is
# one self-contained directory pair. The embedding cache is shared:
# auto-map embeds the corpus and caches chunk_embeddings.json, and
# index-dense reuses that file via its --embed-cache flag instead of
# re-embedding -- vectors are paid for once.

# Base chain. build_chain() inserts "corpus-split" after parse when the
# question pool is split: the split files feed question-gen ONLY --
# chunk, auto-map and the indexes still use the full pages/manifest,
# so the retrieval corpus stays full-corpus either way.
BUILD_STAGES = ("parse", "chunk", "question-gen", "auto-map", "validate",
                "index-bm25", "index-dense")


def build_chain(pool):
    """Ordered stages for one build. pool: "full" or "split"."""
    stages = list(BUILD_STAGES)
    if pool == "split":
        stages.insert(1, "corpus-split")
    return stages


def derive_paths(indexes_dir, golden_dir, split):
    """All stage inputs/outputs for one build. split is dev|heldout."""
    parsed_dir = os.path.join(golden_dir, "parsed")
    pages = os.path.join(parsed_dir, "pages.jsonl")
    manifest = os.path.join(golden_dir, "manifest.jsonl")
    chunk_manifest = os.path.join(golden_dir, "chunk_manifest.yaml")
    drafts = os.path.join(golden_dir, "drafts.yaml")
    golden = os.path.join(golden_dir, f"golden_{split}.yaml")
    cache_dir = os.path.join(indexes_dir, "embed_cache")
    return {
        "parsed_dir": parsed_dir,
        "pages": pages,
        "pages_dev": os.path.join(golden_dir, "pages_dev.jsonl"),
        "pages_heldout": os.path.join(golden_dir, "pages_heldout.jsonl"),
        "manifest": manifest,
        "chunk_manifest": chunk_manifest,
        "drafts": drafts,
        "golden": golden,
        "review_queue": os.path.join(golden_dir, "review_queue.yaml"),
        "review_report": os.path.join(golden_dir, "review_report.json"),
        "cache_dir": cache_dir,
        "cache_file": os.path.join(cache_dir, "chunk_embeddings.json"),
        "marker": golden + ".approved",
    }


def stage_args(stage, p, inputs):
    """CLI args (as an args dict) for one stage of the build."""
    if stage == "parse":
        return {"pdf": inputs["doc"], "out": p["parsed_dir"]}
    if stage == "chunk":
        return {"pages": p["pages"], "out": p["manifest"],
                "chunk-size": inputs["chunk_size"],
                "chunk-overlap": inputs["chunk_overlap"]}
    if stage == "question-gen":
        if inputs["pool"] == "split":
            pages = (p["pages_dev"] if inputs["split"] == "dev"
                     else p["pages_heldout"])
        else:
            pages = p["pages"]
        return {"pages": pages, "out": p["drafts"],
                "n": inputs["n_questions"], "seed": inputs["seed"]}
    if stage == "corpus-split":
        args = {"pages": p["pages"], "out-dev": p["pages_dev"],
                "out-heldout": p["pages_heldout"]}
        # Never pass an empty list: args_to_cli would emit a bare flag
        # and argparse would swallow the next flag as its value.
        if inputs["dev_sections"]:
            args["dev-section"] = inputs["dev_sections"]
        if inputs["dev_ranges"]:
            args["dev-range"] = inputs["dev_ranges"]
        return args
    if stage == "auto-map":
        return {"questions": p["drafts"], "manifest": p["manifest"],
                "chunk-manifest": p["chunk_manifest"],
                "split": inputs["split"], "out": p["golden"],
                "review-queue": p["review_queue"], "cache": p["cache_dir"]}
    if stage == "validate":
        return {"golden": p["golden"], "manifest": p["manifest"],
                "chunk-manifest": p["chunk_manifest"]}
    if stage == "index-bm25":
        return {"manifest": p["manifest"],
                "chunk-manifest": p["chunk_manifest"],
                "out-dir": inputs["indexes_dir"]}
    if stage == "index-dense":
        return {"manifest": p["manifest"],
                "chunk-manifest": p["chunk_manifest"],
                "out-dir": inputs["indexes_dir"],
                "embed-cache": p["cache_file"]}
    if stage == "review-report":
        return {"golden": p["golden"], "manifest": p["manifest"],
                "cache": p["cache_dir"], "out": p["review_report"]}
    raise ValueError(f"unknown stage: {stage}")


def run_stage(base, stage, args, status_slot, log_slot):
    """POST a stage run and poll until it finishes. Returns
    (status, full_log). Scripts emit no percentages, so progress is the
    stage state plus a live log tail -- never a % bar."""
    try:
        run_id = api_post_run(base, stage, args)
    except Exception as err:  # noqa: BLE001 - surfaced, chain stops
        return "error", describe_api_error(err)
    status = "running"
    log = ""
    while status == "running":
        time.sleep(2)
        try:
            status = api_run_status(base, run_id).get("status", "unknown")
            log = api_run_log(base, run_id, tail=15)
        except Exception as err:  # noqa: BLE001 - surfaced, chain stops
            return "error", describe_api_error(err)
        status_slot.write(f"**{stage}**: {status}")
        log_slot.code(log or "(no log output yet)")
    try:
        full_log = api_run_log(base, run_id)
    except Exception:  # noqa: BLE001 - tail already shown
        full_log = log
    return status, full_log


def data_setup_page(base):
    st.header("Data Setup")
    st.caption("Builds indexes + golden-set questions for one document, "
               "through the FastAPI backend. One document per build.")
    doc = st.text_input("Document path", value="",
                        help="Local path to the PDF, as the backend sees "
                             "it (the UI and API run on the same machine).")
    indexes_dir = st.text_input("Indexes output dir", value="indexes")
    golden_dir = st.text_input("Golden-set output dir",
                               value=os.path.join("data", "golden_set"),
                               help="Drafts, the golden YAML, the review "
                                    "queue and the report land here.")
    chunk_size = st.number_input("Chunk size", value=1000, min_value=1,
                                 step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, min_value=0,
                                    step=50)
    strategy = st.selectbox("Chunk strategy", ["recursive"],
                            help="Recursive is the only strategy "
                                 "implemented; other strategies need new "
                                 "chunker code.")
    split_label = st.selectbox("Split", ["dev", "held-out"])
    split = "heldout" if split_label == "held-out" else "dev"
    n_questions = st.number_input("Question count", value=20, min_value=1,
                                  step=1)
    pool_label = st.selectbox(
        "Question pool",
        ["Full corpus", "Split pages (dev / held-out)"],
        help="Full corpus: question-gen reads all parsed pages. Split "
             "pages: corpus-split divides them into dev + held-out pools "
             "(disjoint by page_no) and question-gen reads only the file "
             "for the selected split. Chunking, auto-map and the indexes "
             "always use the full corpus.")
    pool = "split" if pool_label.startswith("Split") else "full"
    dev_sections = []
    dev_ranges = []
    if pool == "split":
        sections_raw = st.text_input(
            "Dev sections (comma-separated)", value="",
            help="Case-insensitive substrings matched against page "
                 "sections, e.g. SQL Commands, Data Types. A page joins "
                 "dev if it matches any section or range.")
        ranges_raw = st.text_input(
            "Dev page ranges (comma-separated)", value="",
            help="Inclusive page_no ranges, e.g. 2714-2900. Held-out is "
                 "the exact complement; overlap is verified by page_no.")
        dev_sections = [s.strip() for s in sections_raw.split(",")
                        if s.strip()]
        dev_ranges = [r.strip() for r in ranges_raw.split(",")
                      if r.strip()]
    seed = st.number_input("Question-gen seed", value=42, min_value=0,
                           step=1,
                           help="Same seed + same pages file samples the "
                                "same pages, so use a different seed per "
                                "split when the pool is shared.")
    st.caption("Backend needs OPENAI_API_KEY in its environment "
               "(question generation, auto-map judging, embeddings).")

    inputs = {"doc": doc, "indexes_dir": indexes_dir,
              "golden_dir": golden_dir, "chunk_size": chunk_size,
              "chunk_overlap": chunk_overlap, "split": split,
              "n_questions": n_questions, "pool": pool,
              "dev_sections": dev_sections, "dev_ranges": dev_ranges,
              "seed": seed}
    paths = derive_paths(indexes_dir, golden_dir, split)
    with st.expander("Derived build paths"):
        st.json({k: v for k, v in paths.items() if k != "marker"})

    build = st.session_state.setdefault("ds_build", {})
    pool_invalid = (pool == "split" and not dev_sections
                    and not dev_ranges)
    if pool_invalid:
        st.error("Split pool needs at least one dev section or dev "
                 "page range (corpus-split would refuse to run).")
    if st.button("Build", disabled=not doc or pool_invalid):
        build.clear()
        chain = build_chain(pool)
        build.update({"states": {}, "logs": {}, "paths": paths,
                      "inputs": inputs, "done": False, "report": None,
                      "chain": chain})
        status_slot = st.empty()
        log_slot = st.empty()
        failed = False
        for stage in chain:
            build["states"][stage] = "running"
            status, full_log = run_stage(
                base, stage, stage_args(stage, paths, inputs),
                status_slot, log_slot)
            build["states"][stage] = status
            build["logs"][stage] = full_log
            if status != "succeeded":
                failed = True
                break
        if not failed:
            # Advisory review report: a failed report must not fail a
            # build that already validated -- it only informs approval.
            try:
                status, full_log = run_stage(
                    base, "review-report",
                    stage_args("review-report", paths, inputs),
                    status_slot, log_slot)
                build["states"]["review-report"] = status
                build["logs"]["review-report"] = full_log
            except Exception as err:  # noqa: BLE001 - advisory only
                build["states"]["review-report"] = "error"
                build["logs"]["review-report"] = str(err)
            build["done"] = True
            st.success("Build finished. Review the golden set below, "
                       "then approve it.")

    states = build.get("states", {})
    if states:
        st.subheader("Build progress")
        st.dataframe([{"stage": s, "status": states.get(s, "waiting")}
                      for s in list(build.get("chain", BUILD_STAGES))
                      + ["review-report"]
                      if s in states], use_container_width=True)
        failed = [s for s, v in states.items()
                  if v in ("failed", "error", "unknown")]
        if failed:
            st.error(f"Stage {failed[0]} failed -- the chain stopped "
                     f"there. Its log:")
            st.code(build["logs"].get(failed[0], ""))

    if build.get("done"):
        paths = build["paths"]
        st.subheader("Review & approve")
        st.caption("Flags are advisory: a flag means look here, never an "
                   "automatic reject. Approval is your call.")
        try:
            with open(paths["review_report"], encoding="utf-8") as f:
                report = json.load(f)
            build["report"] = report
        except (OSError, ValueError):
            report = None
        if report is not None:
            st.write(f"Questions: {report['n_questions']} | flagged: "
                     f"{report['n_flagged']} | dense check: "
                     f"{report['dense']}")
            flagged = [q for q in report["questions"] if q["flags"]]
            if flagged:
                st.dataframe(
                    [{"question": q["id"],
                      "flags": "; ".join(q["flags"])} for q in flagged],
                    use_container_width=True)
            else:
                st.write("No question flags.")
            cov = report["coverage"]
            st.write("Coverage (counts; empty strata flagged below)")
            st.dataframe(
                [{"axis": "query_type", **cov["by_query_type"]},
                 {"axis": "content_tag", **cov["by_content_tag"]}],
                use_container_width=True)
            for flag in cov["flags"]:
                st.warning(f"Coverage: {flag}")
        else:
            st.info("No review report found -- see the review-report "
                    "stage log for its output.")
        with st.expander("Review queue (human spot-check list)"):
            try:
                with open(paths["review_queue"], encoding="utf-8") as f:
                    st.code(f.read())
            except OSError:
                st.info("No review queue file written.")
        with st.expander("Validate stage log"):
            st.code(build["logs"].get("validate", ""))
        if st.button("Approve this golden set"):
            try:
                with open(paths["marker"], "w", encoding="utf-8") as f:
                    f.write(f"approved: "
                            f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
                            f"golden: {paths['golden']}\n")
                st.success(f"Approved. Marker written: {paths['marker']} "
                           f"-- the Evaluation page (step 14) will only "
                           f"offer approved sets.")
            except OSError as err:
                st.error(f"could not write approval marker: {err}")


def evaluation_page():
    st.header("Evaluation")
    st.info("Step 14 builds this page. Planned: register an experiment "
            "by selecting an approved golden-set split, a retrieval "
            "strategy (dense, bm25, hybrid), reranking or not (rerank "
            "only with hybrid, pool N exposed), and Top-K. Experiments "
            "appear in a table with status pending/completed; selecting "
            "one runs it through a new FastAPI endpoint, and the Results "
            "page shows it when it completes.")


def results_page():
    st.header("Eval Results")
    uploaded = st.file_uploader("Upload a step-12 results JSON",
                                type=["json"])
    path = st.text_input("...or a path to one", value="")
    if uploaded is None and not path:
        st.info("Upload a step-12 results JSON (or give its path) to "
                "see the run.")
        return
    try:
        data = load_results(uploaded, path)
    except (OSError, ValueError) as e:
        st.error(f"could not read results file: {e}")
        return
    err = check_schema(data)
    if err:
        st.error(err)
        return

    run = data["run"]
    ks = run.get("ks", [3, 5, 10])
    k_large = ks[-1]
    overview, stratified, per_question = st.tabs(
        ["Overview", "Stratified", "Per question"])

    with overview:
        st.subheader("Run")
        st.json(run)
        st.subheader("Mean metrics per rung")
        family = st.selectbox("Metric family", METRIC_FAMILIES, index=3)
        st.dataframe(mean_rows(data["mean"], data["latency"], ks, family),
                     use_container_width=True)
        st.subheader("Quality vs latency")
        st.caption(f"Each point is a rung: p50 latency vs ndcg@{k_large}.")
        scatter = {"rung": [], "p50_s": [], f"ndcg@{k_large}": []}
        for rung, m in data["mean"].items():
            lat = data["latency"].get(rung, {})
            if lat.get("p50") is None:
                continue
            scatter["rung"].append(rung)
            scatter["p50_s"].append(lat["p50"])
            scatter[f"ndcg@{k_large}"].append(m.get(f"ndcg@{k_large}", 0.0))
        st.scatter_chart(scatter, x="p50_s", y=f"ndcg@{k_large}",
                         color="rung")

    with stratified:
        st.subheader("Stratified means")
        axis = st.selectbox("Group by", ["query_type", "content_tag"])
        grouped = data["by_query_type"] if axis == "query_type" \
            else data.get("by_content_tag", {})
        if not grouped:
            st.info(f"no {axis} groups in this results file.")
        else:
            st.dataframe(stratified_rows(grouped, ks, f"ndcg@{k_large}"),
                         use_container_width=True)

    with per_question:
        st.subheader("Per-question drill-down")
        qids = sorted(data["per_question"])
        qid = st.selectbox("Question", qids)
        row = data["per_question"][qid]
        st.write(f"**{row.get('question', '')}**")
        st.write(f"type: {row.get('query_type', '')} | tags: "
                 f"{', '.join(row.get('content_tags', []))}")
        rung = st.selectbox("Rung", list(row["results"].keys()))
        result = row["results"][rung]
        st.dataframe([{"metric": k, "value": round(v, 4)}
                      for k, v in sorted(result["metrics"].items())],
                     use_container_width=True)
        st.write(f"latency: {result.get('latency_s')} s")
        st.subheader("Ranked chunks")
        st.dataframe(result.get("ranked", []), use_container_width=True)


def main():
    st.title("RagBenchmarkPlatform")
    st.sidebar.header("Backend")
    base = st.sidebar.text_input("API URL", value="http://localhost:8000")
    try:
        health = api_get(base, "/health")
        st.sidebar.success(f"API ok: {', '.join(health.get('stages', []))}")
    except Exception as err:  # noqa: BLE001 - health is advisory
        st.sidebar.warning(f"API not reachable: "
                           f"{describe_api_error(err)} -- start it with "
                           f"uvicorn src.api.main:app from the repo root.")
    page = st.sidebar.radio("Page", ["Data Setup", "Evaluation", "Results"])
    if page == "Data Setup":
        data_setup_page(base)
    elif page == "Evaluation":
        evaluation_page()
    else:
        results_page()


if __name__ == "__main__":
    main()
