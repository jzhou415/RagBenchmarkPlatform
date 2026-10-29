#!/usr/bin/env python3
"""Tests for ui/app.py wiring (derive_paths, stage_args, build_chain).

Streamlit is not installed in the test environment, so the module is
loaded with a stubbed streamlit: only the pure wiring functions are
exercised, never any st.* call. Run: python -m unittest test_ui
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _load_app():
    here = Path(__file__).resolve().parent
    for cand in (here / "ui" / "app.py", here.parent / "ui" / "app.py"):
        if cand.exists():
            path = cand
            break
    else:
        raise FileNotFoundError("ui/app.py not found")
    stub = types.ModuleType("streamlit")
    stub.__getattr__ = lambda name: (lambda *args, **kwargs: None)
    sys.modules["streamlit"] = stub
    spec = importlib.util.spec_from_file_location("rag_ui_app", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


app = _load_app()


def _inputs(**over):
    base = {"doc": "doc.pdf", "indexes_dir": "indexes",
            "golden_dir": "data/golden_set", "chunk_size": 1000,
            "chunk_overlap": 200, "split": "dev", "n_questions": 20,
            "pool": "full", "dev_sections": [], "dev_ranges": [],
            "seed": 42}
    base.update(over)
    return base


class TestDerivePaths(unittest.TestCase):
    def test_split_files_under_golden_dir(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        self.assertEqual(p["pages_dev"], "data/golden_set/pages_dev.jsonl")
        self.assertEqual(p["pages_heldout"],
                         "data/golden_set/pages_heldout.jsonl")
        self.assertEqual(p["pages"], "data/golden_set/parsed/pages.jsonl")


class TestBuildChain(unittest.TestCase):
    def test_full_has_no_split_stage(self):
        self.assertNotIn("corpus-split", app.build_chain("full"))
        self.assertEqual(app.build_chain("full"), list(app.BUILD_STAGES))

    def test_split_inserts_after_parse(self):
        chain = app.build_chain("split")
        self.assertEqual(chain[0], "parse")
        self.assertEqual(chain[1], "corpus-split")
        self.assertEqual(chain[2], "chunk")


class TestStageArgs(unittest.TestCase):
    def test_question_gen_full_pool_uses_full_pages(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        args = app.stage_args("question-gen", p, _inputs())
        self.assertEqual(args["pages"], p["pages"])
        self.assertEqual(args["seed"], 42)

    def test_question_gen_split_pool_uses_split_file(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        args = app.stage_args("question-gen", p,
                              _inputs(pool="split"))
        self.assertEqual(args["pages"], p["pages_dev"])
        p2 = app.derive_paths("indexes", "data/golden_set", "heldout")
        args = app.stage_args("question-gen", p2,
                              _inputs(pool="split", split="heldout"))
        self.assertEqual(args["pages"], p2["pages_heldout"])

    def test_chunk_still_full_pages_in_split_mode(self):
        # The split feeds question-gen ONLY: chunking, auto-map and the
        # indexes always use the full corpus.
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        args = app.stage_args("chunk", p, _inputs(pool="split"))
        self.assertEqual(args["pages"], p["pages"])

    def test_corpus_split_args(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        args = app.stage_args(
            "corpus-split", p,
            _inputs(pool="split", dev_sections=["SQL Commands"],
                    dev_ranges=["2714-2900"]))
        self.assertEqual(args["pages"], p["pages"])
        self.assertEqual(args["dev-section"], ["SQL Commands"])
        self.assertEqual(args["dev-range"], ["2714-2900"])
        self.assertEqual(args["out-dev"], p["pages_dev"])
        self.assertEqual(args["out-heldout"], p["pages_heldout"])

    def test_corpus_split_omits_empty_selector_lists(self):
        # An empty list would become a bare flag and argparse would
        # swallow the next flag as its value.
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        args = app.stage_args(
            "corpus-split", p,
            _inputs(pool="split", dev_sections=[],
                    dev_ranges=["2714-2900"]))
        self.assertNotIn("dev-section", args)
        self.assertEqual(args["dev-range"], ["2714-2900"])


class TestBuildIsStale(unittest.TestCase):
    def test_not_done_never_stale(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        self.assertFalse(app.build_is_stale({}, _inputs(), p))

    def test_same_inputs_not_stale(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        build = {"done": True, "inputs": _inputs(), "paths": p}
        self.assertFalse(app.build_is_stale(build, _inputs(), p))

    def test_changed_doc_is_stale(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        build = {"done": True, "inputs": _inputs(), "paths": p}
        self.assertTrue(
            app.build_is_stale(build, _inputs(doc="other.pdf"), p))

    def test_changed_paths_is_stale(self):
        p = app.derive_paths("indexes", "data/golden_set", "dev")
        p2 = app.derive_paths("indexes", "other/golden", "dev")
        build = {"done": True, "inputs": _inputs(), "paths": p}
        self.assertTrue(app.build_is_stale(build, _inputs(), p2))


def _results(mean, latency, n_questions=20):
    return {"run": {"ks": [3, 5, 10], "n_questions": n_questions},
            "per_question": {}, "mean": mean, "latency": latency}


def _record(exp_id, split, strategy, rerank=False, pool_n=None,
            created="2026-10-28T00:00:00"):
    return {"id": exp_id, "split": split, "strategy": strategy,
            "rerank": rerank, "rerank_pool_n": pool_n,
            "golden_set": "g.yaml", "indexes_dir": "indexes",
            "manifest": "m.jsonl", "results_file": "r.json",
            "status": "completed", "created_at": created,
            "updated_at": created, "returncode": 0}


class TestExperimentLabel(unittest.TestCase):
    def test_dense(self):
        self.assertEqual(app.experiment_label(
            _record("abc", "dev", "dense")), "abc | dev | dense")

    def test_rerank(self):
        self.assertEqual(
            app.experiment_label(_record("abc", "heldout", "hybrid",
                                         rerank=True, pool_n=20)),
            "abc | heldout | hybrid + rerank@20")


class TestResolveResultsDir(unittest.TestCase):
    def test_absolute_passthrough(self):
        self.assertEqual(app.resolve_results_dir("/tmp/results"),
                         "/tmp/results")

    def test_relative_against_repo_root(self):
        out = app.resolve_results_dir("results")
        self.assertTrue(os.path.isabs(out))
        self.assertTrue(out.endswith("results"))


class TestReadRegistry(unittest.TestCase):
    def test_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(app.read_registry(d), [])

    def test_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            recs = {"experiments": {
                "a": _record("a", "dev", "dense",
                             created="2026-10-28T00:00:00"),
                "b": _record("b", "dev", "bm25",
                             created="2026-10-29T00:00:00")}}
            with open(os.path.join(d, "experiments.json"), "w",
                      encoding="utf-8") as f:
                json.dump(recs, f)
            self.assertEqual([r["id"] for r in app.read_registry(d)],
                             ["b", "a"])


class TestComparisonRows(unittest.TestCase):
    def test_grouped_by_split_own_rungs_only(self):
        dense = _record("d1", "dev", "dense")
        hyb = _record("h1", "dev", "hybrid", rerank=True, pool_n=20)
        held = _record("h2", "heldout", "bm25")
        loaded = [
            (dense, _results({"semantic": {"hit@3": 1.0,
                                           "recall@10": 0.5,
                                           "ndcg@10": 0.58}},
                             {"semantic": {"p50": 0.39}})),
            (hyb, _results({"hybrid": {"hit@3": 0.9,
                                       "recall@10": 0.47,
                                       "ndcg@10": 0.59},
                            "hybrid_rerank@20": {"hit@3": 1.0,
                                                 "recall@10": 0.52,
                                                 "ndcg@10": 0.66}},
                           {"hybrid": {"p50": 0.41},
                            "hybrid_rerank@20": {"p50": 1.24}})),
            (held, _results({"bm25": {"hit@3": 0.79,
                                      "recall@10": 0.32,
                                      "ndcg@10": 0.48}},
                            {"bm25": {"p50": 0.08}}, n_questions=50)),
        ]
        grouped = app.comparison_rows(loaded)
        self.assertEqual(set(grouped), {"dev", "heldout"})
        dev_rungs = [(r["experiment"], r["rung"]) for r in grouped["dev"]]
        self.assertEqual(dev_rungs, [("d1 | dev | dense", "semantic"),
                                     ("h1 | dev | hybrid + rerank@20",
                                      "hybrid"),
                                     ("h1 | dev | hybrid + rerank@20",
                                      "hybrid_rerank@20")])
        sem = grouped["dev"][0]
        self.assertEqual(sem["ndcg@10"], 0.58)
        self.assertEqual(sem["p50_s"], 0.39)
        self.assertEqual(sem["n_questions"], 20)
        self.assertEqual(len(grouped["heldout"]), 1)
        self.assertEqual(grouped["heldout"][0]["n_questions"], 50)


class TestExperimentTableRows(unittest.TestCase):
    def test_rerank_blank_when_not_selected(self):
        rows = app.experiment_table_rows([
            {"id": "a", "split": "dev", "strategy": "hybrid",
             "rerank": True, "rerank_pool_n": 10,
             "status": "completed", "results_file": "r1.json"},
            {"id": "b", "split": "dev", "strategy": "dense",
             "rerank": False, "rerank_pool_n": None,
             "status": "completed", "results_file": "r2.json"},
        ])
        self.assertEqual(rows[0]["rerank"], "@10")
        self.assertEqual(rows[1]["rerank"], "")


if __name__ == "__main__":
    unittest.main()
