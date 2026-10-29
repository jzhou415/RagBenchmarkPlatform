#!/usr/bin/env python3
"""Tests for eval_runner.py (step 12). Run: python -m unittest test_eval_runner"""
import os
import tempfile
import unittest

try:
    import src.eval_runner as ev
except ImportError:  # flat working copy (src/ layout on commit)
    import eval_runner as ev


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
    content_tags: [table, lookup]
    expected_answer: Z is W.
    ground_truth:
      - chunk_id: p2_c00
        page: 2
        relevance: 2
"""

# (question_text, rung_label) -> (ranked, latency_s)
FAKE_RESULTS = {
    ("What is X?", "semantic"): ([("p1_c00", 0.9)], 0.01),
    ("What is X?", "bm25"): ([("zz", 0.8)], 0.02),
    ("What is X?", "hybrid"): ([("p1_c01", 0.7)], 0.03),
    ("What is X?", "hybrid_rerank@10"): ([("p1_c00", 0.9)], 0.04),
    ("What is X?", "hybrid_rerank@20"): ([("p1_c01", 0.7)], 0.05),
    ("What is Z?", "semantic"): ([("zz", 0.8)], 0.01),
    ("What is Z?", "bm25"): ([("p2_c00", 0.9)], 0.02),
    ("What is Z?", "hybrid"): ([("p2_c00", 0.9)], 0.03),
    ("What is Z?", "hybrid_rerank@10"): ([("zz", 0.8)], 0.04),
    ("What is Z?", "hybrid_rerank@20"): ([("p2_c00", 0.9)], 0.05),
}


def fake_search(question_text, rung_label):
    return FAKE_RESULTS[(question_text, rung_label)]


def _write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


class TestLoadGoldenQuestions(unittest.TestCase):
    def test_parse(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "golden.yaml", GOLDEN_YAML)
            qs = ev.load_golden_questions(path)
            self.assertEqual([q["id"] for q in qs], ["q001", "q002"])
            self.assertEqual(qs[0]["query_type"], "exact_term")
            self.assertEqual(qs[1]["content_tags"], ["table", "lookup"])
            self.assertEqual(qs[0]["ground_truth"],
                             {"p1_c00": 3, "p1_c01": 1})
            self.assertEqual(qs[1]["ground_truth"], {"p2_c00": 2})


class TestRunEval(unittest.TestCase):
    def _run(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "golden.yaml", GOLDEN_YAML)
            qs = ev.load_golden_questions(path)
        return ev.run_eval(qs, fake_search, ks=(3, 5, 10),
                           candidate_ns=(10, 20))

    def test_per_question_metrics_and_ranked_stored(self):
        res = self._run()
        q1 = res["per_question"]["q001"]
        self.assertEqual(q1["results"]["semantic"]["metrics"]["hit@3"], 1.0)
        self.assertEqual(q1["results"]["semantic"]["metrics"]["recall@3"],
                         0.5)
        self.assertEqual(q1["results"]["bm25"]["metrics"]["hit@3"], 0.0)
        self.assertEqual(q1["results"]["semantic"]["ranked"],
                         [{"chunk_id": "p1_c00", "score": 0.9}])
        # mean over the two questions
        self.assertEqual(res["mean"]["semantic"]["hit@3"], 0.5)

    def test_truncation_makes_mrr_strict(self):
        # only relevant chunk sits at rank 12; full-list MRR would be
        # 1/12, but truncated to max(ks)=10 it must be 0.0
        long_ranking = [(f"c{i:02d}", 1.0 - i * 0.01) for i in range(11)]
        long_ranking.append(("rel", 0.01))
        search = lambda q, label: (long_ranking, 0.01)  # noqa: E731
        qs = [{"id": "q1", "split": "dev", "question": "Q?",
               "query_type": "exact_term", "content_tags": ["lookup"],
               "ground_truth": {"rel": 3}}]
        res = ev.run_eval(qs, search, ks=(3, 5, 10),
                          candidate_ns=(10,))
        m = res["per_question"]["q1"]["results"]["semantic"]["metrics"]
        self.assertEqual(m["mrr"], 0.0)
        self.assertEqual(m["hit@10"], 0.0)

    def test_stratified_means(self):
        res = self._run()
        # query_type groups are singletons here
        self.assertEqual(
            res["by_query_type"]["exact_term"]["semantic"]["hit@3"], 1.0)
        self.assertEqual(
            res["by_query_type"]["paraphrase"]["semantic"]["hit@3"], 0.0)
        # q002 carries both table and lookup, so lookup averages q001+q002
        self.assertEqual(
            res["by_content_tag"]["lookup"]["semantic"]["hit@3"], 0.5)
        self.assertEqual(
            res["by_content_tag"]["table"]["semantic"]["hit@3"], 0.0)

    def test_latency_stats(self):
        res = self._run()
        sem = res["latency"]["semantic"]
        self.assertEqual(sem["n"], 2)
        self.assertEqual(sem["p50"], 0.01)
        self.assertEqual(res["per_question"]["q001"]["results"]["bm25"]
                         ["latency_s"], 0.02)

    def test_summarize(self):
        res = self._run()
        output = {"run": {"golden_set": "golden.yaml", "split": "dev",
                          "reranker": "cross-encoder",
                          "reranker_model": "m",
                          "candidate_ns": [10, 20]},
                  **res}
        text = ev.summarize(output, (3, 5, 10))
        self.assertIn("Eval summary: 2 questions", text)
        self.assertIn("hybrid_rerank@10", text)
        self.assertIn("Best by query_type (ndcg@10)", text)
        self.assertIn("Best by content_tag (ndcg@10)", text)


class TestRunEvalRungs(unittest.TestCase):
    def _qs(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, "golden.yaml", GOLDEN_YAML)
            return ev.load_golden_questions(path)

    def test_subset_scores_only_requested(self):
        qs = self._qs()
        res = ev.run_eval(qs, fake_search, ks=(3, 5, 10),
                          candidate_ns=(10, 20), rungs=("semantic",))
        for row in res["per_question"].values():
            self.assertEqual(set(row["results"]), {"semantic"})
        self.assertEqual(set(res["mean"]), {"semantic"})
        self.assertEqual(set(res["latency"]), {"semantic"})

    def test_hybrid_rerank_subset(self):
        qs = self._qs()
        res = ev.run_eval(qs, fake_search, ks=(3, 5, 10),
                          candidate_ns=(10, 20),
                          rungs=("hybrid", "hybrid_rerank@10"))
        self.assertEqual(set(res["mean"]),
                         {"hybrid", "hybrid_rerank@10"})

    def test_default_scores_all(self):
        qs = self._qs()
        res = ev.run_eval(qs, fake_search, ks=(3, 5, 10),
                          candidate_ns=(10, 20))
        self.assertEqual(set(res["mean"]),
                         {"semantic", "bm25", "hybrid",
                          "hybrid_rerank@10", "hybrid_rerank@20"})

    def test_unknown_rung_raises_before_search(self):
        qs = self._qs()
        calls = []

        def counting_search(question, label):
            calls.append(label)
            return fake_search(question, label)

        with self.assertRaises(ValueError):
            ev.run_eval(qs, counting_search, ks=(3, 5, 10),
                        candidate_ns=(10, 20), rungs=("nope",))
        self.assertEqual(calls, [])


class TestMainValidation(unittest.TestCase):
    def test_missing_required_args_exits(self):
        with self.assertRaises(SystemExit):
            ev.main([])

    def test_bad_ks_exits_before_any_file_access(self):
        with self.assertRaises(SystemExit):
            ev.main(["--golden-set", "nope.yaml",
                     "--manifest", "nope.jsonl",
                     "--out", "nope.json", "--ks", "3,x"])

    def test_bad_rungs_exits_before_any_file_access(self):
        with self.assertRaises(SystemExit):
            ev.main(["--golden-set", "nope.yaml",
                     "--manifest", "nope.jsonl",
                     "--out", "nope.json", "--rungs", "nope"])


if __name__ == "__main__":
    unittest.main()
