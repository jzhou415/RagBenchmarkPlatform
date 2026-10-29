"""ui/app.py -- RagBenchmarkPlatform Streamlit UI.

Pages (sidebar selector):
  Data Setup (step 13, new): pick one document, the indexes output dir,
    and the golden-set output dir, then Build runs the 7 pipeline stages
    in order through the FastAPI backend, with per-stage progress and a
    review-and-approve section for the golden set.
  Evaluation (step 14): register a strategy-specific experiment on
    an approved golden set (split, dense/bm25/hybrid, rerank only
    with hybrid, pool N exposed, Top-K fixed 3/5/10), watch it go
    pending -> running -> completed through a new FastAPI endpoint,
    one results JSON per experiment.
  Results (step 15): Experiments mode loads completed experiments
    from the step-14 registry and shows a cross-experiment comparison
    (mean nDCG@10 + p50 latency, grouped by split so dev and held-out
    are never compared against each other), plus per-experiment detail;
    Single file mode keeps the old upload-or-path viewer.

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


def api_post(base, path, body=None, timeout=10):
    """POST JSON to the API; body None sends an empty POST."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


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


def build_is_stale(build, inputs, paths):
    """True when a finished build's inputs/paths differ from current.

    Streamlit reruns the page on every widget change but session state
    persists, so without this the page would keep showing an old
    build's success after the user edits the inputs.
    """
    return bool(build.get("done")) and (
        build.get("inputs") != inputs or build.get("paths") != paths)


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
    with st.expander("Paths for the current inputs"):
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
        stale = build_is_stale(build, inputs, paths)
        paths = build["paths"]
        b_inputs = build.get("inputs", {})
        st.caption(f"Built with: doc={b_inputs.get('doc', '')} | "
                   f"indexes dir={b_inputs.get('indexes_dir', '')} | "
                   f"golden dir={b_inputs.get('golden_dir', '')} | "
                   f"split={b_inputs.get('split', '')} | "
                   f"pool={b_inputs.get('pool', '')}")
        if stale:
            st.warning("Inputs have changed since this build finished -- "
                       "the progress and review below are from the earlier "
                       "build. Click Build to run with the current inputs.")
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


def evaluation_page(base):
    st.header("Evaluation")
    st.write("Register a strategy-specific experiment on an approved "
             "golden set, then run it. Top-K is fixed at 3/5/10 and "
             "the reranker is the runner default (local cross-encoder), "
             "per the approved step-14 spec.")
    st.subheader("Register an experiment")
    golden = st.text_input("Golden-set path",
                           value="data/golden_set/golden_dev.yaml",
                           help="Must carry a <golden>.approved marker "
                                "from the Data Setup page.")
    if golden:
        marker = golden + ".approved"
        if os.path.exists(marker):
            st.caption(f"Approved marker found: {marker}")
        else:
            st.caption("Not approved yet -- run Data Setup and approve "
                       "this golden set first.")
    split_label = st.selectbox("Split", ["dev", "held-out"])
    split = "heldout" if split_label == "held-out" else "dev"
    strategy = st.selectbox("Retrieval strategy",
                            ["dense", "bm25", "hybrid"])
    rerank = False
    pool_n = 10
    if strategy == "hybrid":
        rerank = st.checkbox("Rerank the hybrid results", value=False)
        if rerank:
            pool_n = st.number_input("Rerank pool N", value=10,
                                     min_value=1,
                                     help="Hybrid candidates the "
                                          "reranker sees.")
    indexes_dir = st.text_input("Indexes dir", value="indexes")
    manifest = st.text_input(
        "Manifest path (empty = <golden dir>/manifest.jsonl)", value="")
    if st.button("Register experiment"):
        body = {"golden_set": golden, "split": split,
                "strategy": strategy, "rerank": rerank,
                "rerank_pool_n": int(pool_n),
                "indexes_dir": indexes_dir}
        if manifest.strip():
            body["manifest"] = manifest.strip()
        try:
            record = api_post(base, "/experiments", body)
            label = f"{split} / {strategy}"
            if rerank:
                label += f" + rerank@{pool_n}"
            st.success(f"Registered {record['id']} ({label}) "
                       f"-- status: pending.")
        except Exception as err:  # noqa: BLE001 - surfaced
            st.error(describe_api_error(err))
    st.subheader("Experiments")
    st.button("Refresh table")  # rerun reloads the table below
    try:
        experiments = api_get(base, "/experiments")
    except Exception as err:  # noqa: BLE001 - surfaced
        st.error(f"Could not load experiments: "
                 f"{describe_api_error(err)}")
        return
    if not experiments:
        st.info("No experiments yet -- register one above.")
        return
    rows = [{"id": e["id"], "split": e["split"],
             "strategy": e["strategy"],
             "rerank": (f"@{e['rerank_pool_n']}"
                        if e["rerank"] else "-"),
             "status": e["status"],
             "results_file": e["results_file"]}
            for e in experiments]
    st.table(rows)
    labels = [f"{e['id']} | {e['split']} | {e['strategy']} "
              f"{'+ rerank@' + str(e['rerank_pool_n']) if e['rerank'] else ''} "
              f"| {e['status']}" for e in experiments]
    choice = st.selectbox("Select an experiment", labels)
    selected = experiments[labels.index(choice)]
    if selected["status"] not in ("pending", "failed"):
        st.info(f"Experiment {selected['id']} is "
                f"{selected['status']} -- only pending or failed "
                f"experiments can run.")
        if selected["status"] == "completed":
            st.write(f"Results: {selected['results_file']} -- open the "
                     f"Results page and load this file.")
        return
    if st.button("Run selected experiment"):
        try:
            api_post(base, f"/experiments/{selected['id']}/run")
        except Exception as err:  # noqa: BLE001 - surfaced
            st.error(describe_api_error(err))
            return
        status_slot = st.empty()
        log_slot = st.empty()
        status = "running"
        log = ""
        while status == "running":
            time.sleep(2)
            try:
                status = api_get(
                    base, f"/experiments/{selected['id']}").get(
                        "status", "unknown")
                log = api_get(
                    base,
                    f"/experiments/{selected['id']}/log?tail=15").get(
                        "log", "")
            except Exception as err:  # noqa: BLE001 - surfaced
                st.error(describe_api_error(err))
                return
            status_slot.write(f"**{selected['id']}**: {status}")
            log_slot.code(log or "(no log output yet)")
        if status == "completed":
            st.success(f"Experiment {selected['id']} completed. "
                       f"Results: {selected['results_file']} -- open "
                       f"the Results page and load this file.")
        else:
            st.error(f"Experiment {selected['id']} failed -- see the "
                     f"log above.")


def experiment_label(record):
    """'abc123 | dev | hybrid + rerank@20' for registry records."""
    label = f"{record['id']} | {record['split']} | {record['strategy']}"
    if record.get("rerank"):
        label += f" + rerank@{record.get('rerank_pool_n')}"
    return label


def resolve_results_dir(user_input):
    """Absolute path as-is; repo-relative path against the repo root.

    The repo root is derived from this file's location, not the
    Streamlit working directory, so 'results' works regardless of
    where streamlit was launched from.
    """
    if os.path.isabs(user_input):
        return user_input
    repo_root = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    return os.path.join(repo_root, user_input)


def read_registry(results_dir):
    """Registry records, newest first; [] when no registry file."""
    path = os.path.join(results_dir, "experiments.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f).get("experiments", {})
    return sorted(records.values(),
                  key=lambda r: r.get("created_at", ""), reverse=True)


def comparison_rows(loaded):
    """{split: [rows]} for the comparison table.

    loaded: [(record, results_dict)]. One row per experiment rung --
    each experiment contributes only its own rungs (step-14 B
    subsets), so a dense experiment adds a semantic row, not empty
    bm25/hybrid cells. Metrics use the step-14 standard ks (3/5/10);
    files with other ks still load, missing metrics read as 0.0.
    """
    grouped = {}
    for record, data in loaded:
        mean = data.get("mean", {})
        latency = data.get("latency", {})
        n_questions = data.get("run", {}).get("n_questions")
        for rung, m in mean.items():
            lat = latency.get(rung, {})
            grouped.setdefault(record["split"], []).append({
                "experiment": experiment_label(record),
                "rung": rung,
                "hit@3": round(m.get("hit@3", 0.0), 4),
                "recall@10": round(m.get("recall@10", 0.0), 4),
                "ndcg@10": round(m.get("ndcg@10", 0.0), 4),
                "p50_s": lat.get("p50"),
                "n_questions": n_questions,
            })
    return grouped


def _experiments_mode_data():
    """Experiments-mode UI: comparison view; returns the detail-
    selected results dict, or None when nothing can be shown."""
    results_dir_in = st.text_input(
        "Results dir", value="results",
        help="Relative paths resolve against the repo root; the "
             "step-14 registry (experiments.json) lives here.")
    results_dir = resolve_results_dir(results_dir_in)
    try:
        records = read_registry(results_dir)
    except (OSError, ValueError) as e:
        st.error(f"could not read registry: {e}")
        return None
    completed = [r for r in records if r.get("status") == "completed"]
    if not records:
        st.info("No experiments registry found here -- register and run "
                "experiments on the Evaluation page first, or switch "
                "to Single file mode.")
        return None
    if not completed:
        st.info(f"{len(records)} experiment(s) registered, none "
                f"completed yet -- run one on the Evaluation page.")
        return None
    labels = [experiment_label(r) for r in completed]
    chosen = st.multiselect("Experiments to compare", labels,
                            default=labels)
    if not chosen:
        st.info("Select at least one experiment.")
        return None
    loaded = []
    for record in completed:
        if experiment_label(record) not in chosen:
            continue
        try:
            data = load_results(None, record["results_file"])
        except (OSError, ValueError) as e:
            st.warning(f"{record['id']}: could not load results: {e}")
            continue
        err = check_schema(data)
        if err:
            st.warning(f"{record['id']}: {err}")
            continue
        loaded.append((record, data))
    if not loaded:
        st.error("None of the selected experiments could be loaded.")
        return None
    st.subheader("Cross-experiment comparison")
    st.caption("Grouped by split -- dev and held-out are never compared "
               "against each other. Each experiment contributes only "
               "its own rungs.")
    grouped = comparison_rows(loaded)
    for split in sorted(grouped):
        st.write(f"**Split: {split}**")
        rows = grouped[split]
        st.dataframe(rows, use_container_width=True)
        chart = {"label": [], "ndcg@10": []}
        for row in rows:
            chart["label"].append(f"{row['experiment']} | {row['rung']}")
            chart["ndcg@10"].append(row["ndcg@10"])
        st.bar_chart(chart, x="label", y="ndcg@10")
    st.subheader("Per-experiment detail")
    detail_labels = [experiment_label(r) for r, _ in loaded]
    detail_choice = st.selectbox("Experiment", detail_labels)
    return loaded[detail_labels.index(detail_choice)][1]


def show_results_viewer(data):
    """Overview / Stratified / Per-question tabs for one results dict."""
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


def results_page():
    st.header("Eval Results")
    mode = st.selectbox("Load results from",
                        ["Experiments", "Single file"])
    if mode == "Experiments":
        data = _experiments_mode_data()
        if data is None:
            return
    else:
        uploaded = st.file_uploader("Upload a step-12 results JSON",
                                    type=["json"])
        path = st.text_input("...or a path to one", value="")
        if uploaded is None and not path:
            st.info("Upload a step-12 results JSON (or give its path) "
                    "to see the run.")
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
    show_results_viewer(data)


def main():
    st.title("RagBenchmarkPlatform")
    st.sidebar.header("Backend")
    base = st.sidebar.text_input("API URL", value="http://localhost:8000")
    try:
        health = api_get(base, "/health")
        st.sidebar.success(f"API ok: {', '.join(health.get('stages', []))}")
        if health.get("root"):
            st.sidebar.caption(f"API root: {health['root']}")
    except Exception as err:  # noqa: BLE001 - health is advisory
        st.sidebar.warning(f"API not reachable: "
                           f"{describe_api_error(err)} -- start it with "
                           f"uvicorn src.api.main:app from the repo root.")
    page = st.sidebar.radio("Page", ["Data Setup", "Evaluation", "Results"])
    if page == "Data Setup":
        data_setup_page(base)
    elif page == "Evaluation":
        evaluation_page(base)
    else:
        results_page()


if __name__ == "__main__":
    main()
