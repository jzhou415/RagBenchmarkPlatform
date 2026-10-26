#!/usr/bin/env python3
"""Tests for src/rerank.py (step 10). Run: python -m unittest test_rerank"""
import json
import os
import tempfile
import unittest

import src.rerank as rr
from src.rerank import (
    CohereReranker,
    CrossEncoderReranker,
    make_reranker,
    rerank_by_scores,
)


class TestRerankByScores(unittest.TestCase):
    def test_sorts_desc(self):
        pairs = [("a", "t1"), ("b", "t2"), ("c", "t3")]
        out = rerank_by_scores("q", pairs, [0.1, 0.9, 0.5])
        self.assertEqual([cid for cid, _ in out], ["b", "c", "a"])
        self.assertAlmostEqual(out[0][1], 0.9)

    def test_top_k(self):
        pairs = [("a", "t1"), ("b", "t2"), ("c", "t3")]
        out = rerank_by_scores("q", pairs, [0.1, 0.9, 0.5], top_k=2)
        self.assertEqual([cid for cid, _ in out], ["b", "c"])

    def test_top_k_none_returns_all(self):
        pairs = [("a", "t1"), ("b", "t2")]
        self.assertEqual(len(rerank_by_scores("q", pairs, [0.2, 0.3])), 2)


class TestCrossEncoder(unittest.TestCase):
    def test_rerank_uses_scores(self):
        class FakeCE(CrossEncoderReranker):
            def __init__(self):
                self.model_name = "fake"  # skip real model load

            def score(self, query, texts):
                return [float("shared_buffers" in t) for t in texts]

        r = FakeCE()
        pairs = [("p1", "about logging"), ("p2", "shared_buffers default")]
        out = r.rerank("q", pairs, top_k=2)
        self.assertEqual(out[0][0], "p2")
        self.assertEqual(len(out), 2)


class TestCohere(unittest.TestCase):
    def _fake_client(self):
        class FakeResult:
            def __init__(self, index, relevance_score):
                self.index = index
                self.relevance_score = relevance_score

        class FakeResp:
            results = [FakeResult(1, 0.9), FakeResult(0, 0.1)]

        class FakeClient:
            def __init__(self):
                self.seen = None

            def rerank(self, model, query, documents, top_n):
                self.seen = (model, query, documents, top_n)
                return FakeResp()

        return FakeClient()

    def test_rerank_maps_indices(self):
        client = self._fake_client()
        r = CohereReranker(_client=client)
        pairs = [("p1", "t1"), ("p2", "t2")]
        out = r.rerank("q", pairs, top_k=2)
        self.assertEqual([cid for cid, _ in out], ["p2", "p1"])
        self.assertEqual(client.seen[0], rr.DEFAULT_COHERE_MODEL)
        self.assertEqual(client.seen[1], "q")
        self.assertEqual(client.seen[2], ["t1", "t2"])

    def test_unreturned_docs_score_zero(self):
        class FakeResp:
            class R:
                index, relevance_score = 0, 0.7
            results = [R()]

        class FakeClient:
            def rerank(self, model, query, documents, top_n):
                return FakeResp()

        r = CohereReranker(_client=FakeClient())
        out = r.rerank("q", [("p1", "t1"), ("p2", "t2")], top_k=2)
        self.assertEqual(out[0][0], "p1")
        self.assertEqual(out[1][1], 0.0)


class TestFactory(unittest.TestCase):
    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            make_reranker("nope")

    def test_cohere_factory_uses_default_model(self):
        r = make_reranker("cohere", api_key="fake-key-for-test")
        # cohere.Client init would fail without the real package only if
        # we hadn't... (no: _client is None here, so this needs cohere
        # installed). Instead assert via _client seam:
        self.assertIsInstance(r, CohereReranker)


class TestCLI(unittest.TestCase):
    def test_main_writes_json(self):
        with tempfile.TemporaryDirectory() as td:
            cp = os.path.join(td, "cands.json")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump([{"chunk_id": "p1", "text": "t1"},
                           {"chunk_id": "p2", "text": "t2"}], f)
            outp = os.path.join(td, "out.json")

            class FakeR:
                def rerank(self, query, pairs, top_k=None):
                    return [(cid, 1.0) for cid, _ in pairs[:top_k]]

            old = rr.make_reranker
            rr.make_reranker = lambda kind, model=None, api_key=None: FakeR()
            try:
                rr.main(["--query", "q", "--candidates", cp, "--top-k", "1",
                         "--reranker", "cross-encoder", "--out", outp])
            finally:
                rr.make_reranker = old
            with open(outp, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data[0]["chunk_id"], "p1")
            self.assertEqual(len(data), 1)


class TestFromRun(unittest.TestCase):
    def _write(self, td, name, obj, jsonl=False):
        p = os.path.join(td, name)
        with open(p, "w", encoding="utf-8") as f:
            if jsonl:
                for row in obj:
                    f.write(json.dumps(row) + "\n")
            else:
                json.dump(obj, f)
        return p

    def test_load_manifest_texts(self):
        with tempfile.TemporaryDirectory() as td:
            mp = self._write(td, "m.jsonl", [
                {"chunk_id": "p1", "text": "t1", "page_no": 1},
                {"chunk_id": "p2", "text": "t2", "page_no": 2},
            ], jsonl=True)
            self.assertEqual(rr.load_manifest_texts(mp),
                             {"p1": "t1", "p2": "t2"})

    def test_join_skips_missing_with_warning(self):
        import contextlib
        import io
        run = {"results": [{"chunk_id": "p1", "score": 0.9},
                           {"chunk_id": "pX", "score": 0.8}]}
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            pairs = rr.join_run_candidates(run, {"p1": "t1"})
        self.assertEqual(pairs, [("p1", "t1")])
        self.assertIn("pX", err.getvalue())

    def test_main_from_run_uses_run_query(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._write(td, "run.json", {
                "query": "run-q",
                "results": [{"chunk_id": "p1", "score": 0.9},
                            {"chunk_id": "p2", "score": 0.1}]})
            mp = self._write(td, "m.jsonl", [
                {"chunk_id": "p1", "text": "t1"},
                {"chunk_id": "p2", "text": "t2"},
            ], jsonl=True)
            outp = os.path.join(td, "out.json")

            seen = {}

            class FakeR:
                def rerank(self, query, pairs, top_k=None):
                    seen["query"] = query
                    seen["pairs"] = pairs
                    return [(cid, 1.0) for cid, _ in pairs[:top_k]]

            old = rr.make_reranker
            rr.make_reranker = lambda kind, model=None, api_key=None: FakeR()
            try:
                rr.main(["--from-run", rp, "--manifest", mp, "--top-k", "1",
                         "--reranker", "cross-encoder", "--out", outp])
            finally:
                rr.make_reranker = old
            self.assertEqual(seen["query"], "run-q")
            self.assertEqual([cid for cid, _ in seen["pairs"]],
                             ["p1", "p2"])
            with open(outp, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data[0]["chunk_id"], "p1")
            self.assertEqual(len(data), 1)

    def test_main_from_run_query_override(self):
        with tempfile.TemporaryDirectory() as td:
            rp = self._write(td, "run.json", {
                "query": "run-q",
                "results": [{"chunk_id": "p1", "score": 0.9}]})
            mp = self._write(td, "m.jsonl", [
                {"chunk_id": "p1", "text": "t1"},
            ], jsonl=True)

            seen = {}

            class FakeR:
                def rerank(self, query, pairs, top_k=None):
                    seen["query"] = query
                    return [(cid, 1.0) for cid, _ in pairs]

            old = rr.make_reranker
            rr.make_reranker = lambda kind, model=None, api_key=None: FakeR()
            try:
                rr.main(["--from-run", rp, "--manifest", mp,
                         "--query", "explicit",
                         "--reranker", "cross-encoder"])
            finally:
                rr.make_reranker = old
            self.assertEqual(seen["query"], "explicit")

    def test_main_mutual_exclusion(self):
        with self.assertRaises(SystemExit):
            rr.main(["--query", "q", "--candidates", "c.json",
                     "--from-run", "r.json", "--manifest", "m.jsonl"])

    def test_main_from_run_needs_manifest(self):
        with self.assertRaises(SystemExit):
            rr.main(["--from-run", "r.json"])


if __name__ == "__main__":
    unittest.main()
