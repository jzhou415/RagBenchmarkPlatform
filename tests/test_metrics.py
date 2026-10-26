#!/usr/bin/env python3
"""Tests for metrics.py (step 11). Run: python -m unittest test_metrics"""
import json
import os
import tempfile
import unittest

try:
    import src.metrics as mt
except ImportError:  # flat working copy (src/ layout on commit)
    import metrics as mt


class TestBinaryMetrics(unittest.TestCase):
    def test_hit_recall_precision_mrr(self):
        ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        gt = {"a": 3, "c": 1, "e": 2}
        self.assertEqual(mt.hit_at_k(ranked, gt, 3), 1.0)
        self.assertAlmostEqual(mt.recall_at_k(ranked, gt, 3), 2 / 3)
        self.assertAlmostEqual(mt.precision_at_k(ranked, gt, 3), 2 / 3)
        self.assertEqual(mt.reciprocal_rank(ranked, gt), 1.0)

    def test_mrr_second_rank_and_short_ranking_precision(self):
        ranked = [("x", 0.9), ("a", 0.8)]
        gt = {"a": 3}
        self.assertEqual(mt.reciprocal_rank(ranked, gt), 0.5)
        self.assertEqual(mt.hit_at_k(ranked, gt, 3), 1.0)
        self.assertEqual(mt.recall_at_k(ranked, gt, 3), 1.0)
        # denominator is chunks actually returned (2), not k (3)
        self.assertEqual(mt.precision_at_k(ranked, gt, 3), 0.5)

    def test_no_hit(self):
        ranked = [("x", 0.9), ("y", 0.8)]
        gt = {"a": 3}
        self.assertEqual(mt.hit_at_k(ranked, gt, 2), 0.0)
        self.assertEqual(mt.recall_at_k(ranked, gt, 2), 0.0)
        self.assertEqual(mt.precision_at_k(ranked, gt, 2), 0.0)
        self.assertEqual(mt.reciprocal_rank(ranked, gt), 0.0)
        self.assertEqual(mt.ndcg_at_k(ranked, gt, 2), 0.0)


class TestNdcg(unittest.TestCase):
    def test_graded_example_hand_computed(self):
        ranked = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        gt = {"a": 3, "c": 1, "e": 2}
        # DCG@3 = 7/1 + 0 + 1/2 = 7.5
        # IDCG@3 = 7/1 + 3/log2(3) + 1/2 = 9.3928...
        self.assertAlmostEqual(mt.ndcg_at_k(ranked, gt, 3),
                               7.5 / (7 + 3 / 1.584962500721156 + 0.5),
                               places=4)

    def test_grade_order_matters(self):
        # grade-3 second beats grade-1 first, but is not ideal
        ranked = [("low", 0.9), ("high", 0.8)]
        gt = {"low": 1, "high": 3}
        ndcg = mt.ndcg_at_k(ranked, gt, 2)
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.7)
        ideal = [("high", 0.9), ("low", 0.8)]
        self.assertAlmostEqual(mt.ndcg_at_k(ideal, gt, 2), 1.0)

    def test_second_rank_ndcg(self):
        ranked = [("x", 0.9), ("a", 0.8)]
        gt = {"a": 3}
        # DCG = 7/log2(3), IDCG = 7
        self.assertAlmostEqual(mt.ndcg_at_k(ranked, gt, 3),
                               1 / 1.584962500721156, places=4)


class TestEdgeCases(unittest.TestCase):
    def test_empty_ranking_and_empty_ground_truth(self):
        for ranked, gt in ([([], {"a": 3}), ([("a", 1.0)], {})]):
            m = mt.evaluate_question(ranked, gt, ks=(3,))
            self.assertEqual(m["hit@3"], 0.0)
            self.assertEqual(m["recall@3"], 0.0)
            self.assertEqual(m["precision@3"], 0.0)
            self.assertEqual(m["mrr"], 0.0)
            self.assertEqual(m["ndcg@3"], 0.0)

    def test_duplicate_chunk_counts_once(self):
        ranked = [("a", 0.9), ("a", 0.8), ("b", 0.7)]
        gt = {"b": 2}
        self.assertEqual(mt.reciprocal_rank(ranked, gt), 0.5)
        self.assertEqual(mt.precision_at_k(ranked, gt, 2), 0.5)

    def test_mean_and_empty_mean(self):
        run = mt.evaluate_run({
            "q1": ([("a", 1.0)], {"a": 3}),
            "q2": ([("x", 1.0)], {"a": 3}),
        }, ks=(3,))
        self.assertEqual(run["per_question"]["q1"]["hit@3"], 1.0)
        self.assertEqual(run["per_question"]["q2"]["hit@3"], 0.0)
        self.assertEqual(run["mean"]["hit@3"], 0.5)
        self.assertEqual(mt.mean_metrics([]), {})


class TestCLI(unittest.TestCase):
    def test_main_writes_json(self):
        with tempfile.TemporaryDirectory() as td:
            rp = os.path.join(td, "ranked.json")
            gp = os.path.join(td, "gt.json")
            op = os.path.join(td, "out.json")
            with open(rp, "w", encoding="utf-8") as f:
                json.dump([{"chunk_id": "a", "score": 0.9}], f)
            with open(gp, "w", encoding="utf-8") as f:
                json.dump({"a": 3}, f)
            self.assertEqual(
                mt.main(["--ranked", rp, "--ground-truth", gp,
                         "--ks", "3", "--out", op]), 0)
            with open(op, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["hit@3"], 1.0)
            self.assertEqual(data["ndcg@3"], 1.0)


class TestLatency(unittest.TestCase):
    def test_percentiles_hand_computed(self):
        stats = mt.latency_stats([10, 20, 30, 40])
        # linear interpolation: p50 rank 1.5 -> 25; p95 rank 2.85 -> 38.5;
        # p99 rank 2.97 -> 39.7
        self.assertAlmostEqual(stats["p50"], 25.0)
        self.assertAlmostEqual(stats["p95"], 38.5)
        self.assertAlmostEqual(stats["p99"], 39.7)
        self.assertAlmostEqual(stats["mean"], 25.0)
        self.assertEqual(stats["n"], 4)

    def test_unsorted_input_and_single_value(self):
        stats = mt.latency_stats([40, 10, 30, 20])
        self.assertAlmostEqual(stats["p50"], 25.0)
        single = mt.latency_stats([5.0])
        self.assertEqual(single["p50"], 5.0)
        self.assertEqual(single["p95"], 5.0)
        self.assertEqual(single["p99"], 5.0)
        self.assertEqual(single["n"], 1)

    def test_empty_latencies(self):
        stats = mt.latency_stats([])
        self.assertIsNone(stats["p50"])
        self.assertIsNone(stats["p95"])
        self.assertIsNone(stats["p99"])
        self.assertIsNone(stats["mean"])
        self.assertEqual(stats["n"], 0)


GOLDEN_YAML = """\
schema_version: "1.1"
corpus: test
ingestion:
  config_hash: abc
questions:
  - id: q001
    split: dev
    question: What is X?
    query_type: exact_term
    content_tags: [lookup]
    expected_answer: X is Y.
    ground_truth:
      - chunk_id: p1_c00
        page: 1
        relevance: 3
      - chunk_id: p1_c01
        page: 1
        relevance: 1
  - id: q002
    split: dev
    question: What is Z?
    query_type: paraphrase
    content_tags: [lookup]
    expected_answer: Z is W.
    ground_truth:
      - chunk_id: p2_c00
        page: 2
        relevance: 2
"""


class TestGoldenSetGroundTruth(unittest.TestCase):
    def _write(self, directory, name, text):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_load_ground_truth_from_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "golden.yaml", GOLDEN_YAML)
            self.assertEqual(mt.load_ground_truth(path, "q001"),
                             {"p1_c00": 3, "p1_c01": 1})
            self.assertEqual(mt.load_ground_truth(path, "q002"),
                             {"p2_c00": 2})

    def test_load_ground_truth_missing_question_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._write(d, "golden.yaml", GOLDEN_YAML)
            with self.assertRaises(ValueError):
                mt.load_ground_truth(path, "q999")

    def _run_cli(self, argv):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = mt.main(argv)
        return code, buf.getvalue()

    def test_cli_golden_set_matches_ground_truth_path(self):
        ranked = json.dumps([{"chunk_id": "p1_c00", "score": 0.9},
                             {"chunk_id": "zz", "score": 0.1}])
        with tempfile.TemporaryDirectory() as d:
            ranked_path = self._write(d, "ranked.json", ranked)
            golden_path = self._write(d, "golden.yaml", GOLDEN_YAML)
            gt_path = self._write(d, "gt.json",
                                  json.dumps({"p1_c00": 3, "p1_c01": 1}))
            code_gs, out_gs = self._run_cli(
                ["--ranked", ranked_path, "--golden-set", golden_path,
                 "--question-id", "q001"])
            code_gt, out_gt = self._run_cli(
                ["--ranked", ranked_path, "--ground-truth", gt_path])
            self.assertEqual(code_gs, 0)
            self.assertEqual(out_gs, out_gt)
            self.assertIn("hit@3", out_gs)

    def test_cli_flag_validation(self):
        with tempfile.TemporaryDirectory() as d:
            ranked_path = self._write(d, "ranked.json", "[]")
            golden_path = self._write(d, "golden.yaml", GOLDEN_YAML)
            gt_path = self._write(d, "gt.json", "{}")
            base = ["--ranked", ranked_path]
            # both truth sources
            with self.assertRaises(SystemExit):
                self._run_cli(base + ["--ground-truth", gt_path,
                                      "--golden-set", golden_path,
                                      "--question-id", "q001"])
            # neither truth source
            with self.assertRaises(SystemExit):
                self._run_cli(base)
            # golden-set without question-id
            with self.assertRaises(SystemExit):
                self._run_cli(base + ["--golden-set", golden_path])
            # question-id without golden-set
            with self.assertRaises(SystemExit):
                self._run_cli(base + ["--question-id", "q001"])
            # unknown question id fails loudly, not as measured zeros
            with self.assertRaises(SystemExit):
                self._run_cli(base + ["--golden-set", golden_path,
                                      "--question-id", "q999"])


if __name__ == "__main__":
    unittest.main()
