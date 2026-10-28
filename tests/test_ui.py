#!/usr/bin/env python3
"""Tests for ui/app.py wiring (derive_paths, stage_args, build_chain).

Streamlit is not installed in the test environment, so the module is
loaded with a stubbed streamlit: only the pure wiring functions are
exercised, never any st.* call. Run: python -m unittest test_ui
"""
import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
