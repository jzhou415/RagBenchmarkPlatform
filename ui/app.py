"""ui/app.py -- step 13: Streamlit viewer for step-12 eval results.

Reads only a step-12 results JSON -- no index, model, or API access.
Run from the repo root: streamlit run ui/app.py

The JSON schema it expects is exactly what eval_runner.py writes:
    run: golden_set, split, n_questions, config_hash,
         indexes {bm25, dense}, reranker, reranker_model,
         candidate_ns, ks, timestamp
    per_question: {qid: {question, query_type, content_tags,
        results: {rung: {metrics, latency_s, ranked}}}}
    mean: {rung: {hit@k, recall@k, precision@k, mrr, ndcg@k}}
    by_query_type / by_content_tag: {group: {rung: metrics}}
    latency: {rung: {p50, p95, p99, mean, n}}
"""
import json

import streamlit as st

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


def main():
    st.title("RagBenchmarkPlatform — Eval Results")
    st.sidebar.header("Results file")
    uploaded = st.sidebar.file_uploader("Upload a step-12 results JSON",
                                        type=["json"])
    path = st.sidebar.text_input("...or a path to one", value="")
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


if __name__ == "__main__":
    main()
