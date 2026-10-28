#!/usr/bin/env python3
"""Tests for review_report.py. Run: python -m unittest test_review_report"""
import unittest

from src.review_report import (
    build_report, check_answer_support, check_duplicates, coverage,
    dense_top, numbers_and_labels,
)


def make_chunks():
    return [
        {"chunk_id": "p0100_c00", "page_no": 100, "section": "S",
         "text": "The default value of shared_buffers is 128MB in this release."},
        {"chunk_id": "p0100_c01", "page_no": 100, "section": "S",
         "text": "shared_buffers sets the amount of memory for shared buffers."},
        {"chunk_id": "p0101_c00", "page_no": 101, "section": "T",
         "text": "Unrelated text about logging checkpoints and wal files."},
    ]


def make_golden():
    return {"ingestion": {"config_hash": "testhash"}, "questions": [
        {"id": "q001", "question": "What is the default shared_buffers?",
         "query_type": "exact_term", "content_tags": ["lookup"],
         "expected_answer": "128MB",
         "ground_truth": [{"chunk_id": "p0100_c00", "page": 100,
                           "relevance": 3}]},
        {"id": "q002", "question": "What is the default shared_buffers?",
         "query_type": "paraphrase", "content_tags": ["lookup"],
         "expected_answer": "128MB",
         "ground_truth": [{"chunk_id": "p0100_c00", "page": 100,
                           "relevance": 3}]},
        {"id": "q003", "question": "How do checkpoints interact with wal files?",
         "query_type": "exact_term", "content_tags": ["policy"],
         "expected_answer": "The wal_buffers setting is 256MB",
         "ground_truth": [{"chunk_id": "p0101_c00", "page": 101,
                           "relevance": 3}]},
    ]}


class TestAnswerSupport(unittest.TestCase):
    def test_number_and_label_present_pass(self):
        texts = {"p0100_c00": make_chunks()[0]["text"]}
        q = {"expected_answer": "shared_buffers is 128MB",
             "ground_truth": [{"chunk_id": "p0100_c00", "relevance": 3}]}
        self.assertEqual(check_answer_support(q, texts), [])

    def test_missing_number_and_label_flagged(self):
        texts = {"p0101_c00": make_chunks()[2]["text"]}
        q = {"expected_answer": "The wal_buffers setting is 256MB",
             "ground_truth": [{"chunk_id": "p0101_c00", "relevance": 3}]}
        flags = check_answer_support(q, texts)
        self.assertIn("number 256MB not in any grade-3 chunk", flags)
        self.assertIn("label wal_buffers not in any grade-3 chunk", flags)

    def test_empty_expected_answer_flagged(self):
        q = {"expected_answer": "...", "ground_truth": []}
        self.assertIn("expected answer empty or placeholder",
                      check_answer_support(q, {}))

    def test_numbers_and_labels_extraction(self):
        numbers, labels = numbers_and_labels(
            "Set shared_buffers to 128MB, not 64MB")
        self.assertEqual(numbers, {"128MB", "64MB"})
        self.assertEqual(labels, {"shared_buffers"})


class TestDuplicates(unittest.TestCase):
    def test_exact_duplicate_both_ways(self):
        qs = [{"id": "q001", "question": "What is X?"},
              {"id": "q002", "question": "what is x ?"},
              {"id": "q003", "question": "Something else entirely"}]
        flags = check_duplicates(qs)
        self.assertIn("duplicate of q002", flags["q001"])
        self.assertIn("duplicate of q001", flags["q002"])
        self.assertEqual(flags["q003"], [])

    def test_near_duplicate_threshold(self):
        qs = [{"id": "q001",
               "question": "What is the default value of shared_buffers"},
              {"id": "q002",
               "question": "What is the default value of shared_buffers today"}]
        flags = check_duplicates(qs)
        self.assertTrue(any("near-duplicate" in f for f in flags["q001"]))


class TestCoverage(unittest.TestCase):
    def test_counts_and_empty_marginals(self):
        cov = coverage(make_golden()["questions"])
        self.assertEqual(cov["by_query_type"]["exact_term"], 2)
        self.assertEqual(cov["by_query_type"]["paraphrase"], 1)
        self.assertIn("no multi_hop questions", cov["flags"])
        self.assertIn("no table questions", cov["flags"])
        self.assertIn("no aggregation questions", cov["flags"])


class TestRetrievability(unittest.TestCase):
    def test_bm25_retrieves_matching_grade3(self):
        # No cache, no embed_fn: dense skipped, BM25 decides.
        report = build_report(make_golden(), make_chunks(), {}, None,
                              top_k=10)
        by_id = {q["id"]: q["flags"] for q in report["questions"]}
        self.assertNotIn("no graded chunk in BM25 top-10", by_id["q001"])
        self.assertEqual(report["dense"], "skipped: no chunk cache")

    def test_bm25_miss_flagged_at_top1(self):
        # Question terms match the shared_buffers chunks, but the graded
        # chunk is the logging chunk: BM25 top-1 misses it.
        golden = {"questions": [
            {"id": "qX", "question": "What is the default shared_buffers?",
             "query_type": "exact_term", "content_tags": ["lookup"],
             "expected_answer": "128MB",
             "ground_truth": [{"chunk_id": "p0101_c00", "relevance": 3}]}]}
        report = build_report(golden, make_chunks(), {}, None, top_k=1)
        flags = report["questions"][0]["flags"]
        self.assertIn("no graded chunk in BM25 top-1", flags)

    def test_dense_top_uses_cosine(self):
        ids = ["a", "b", "c"]
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
        self.assertEqual(dense_top(ids, vecs, [1.0, 0.0], 1), {"a"})
        self.assertEqual(dense_top(ids, vecs, [0.0, 1.0], 2), {"b", "c"})

    def test_dense_check_with_fake_embed_fn(self):
        cache = {"p0100_c00": [1.0, 0.0], "p0100_c01": [0.0, 1.0],
                 "p0101_c00": [0.0, 0.0]}
        golden = {"questions": [
            {"id": "q001", "question": "What is the default shared_buffers?",
             "query_type": "exact_term", "content_tags": ["lookup"],
             "expected_answer": "128MB",
             "ground_truth": [{"chunk_id": "p0100_c00", "relevance": 3}]}]}
        # Question vector aligned with its grade-3 chunk: retrieved.
        report = build_report(golden, make_chunks(), cache,
                              lambda texts: [[1.0, 0.0]] * len(texts),
                              top_k=10)
        self.assertEqual(report["dense"], "ok")
        self.assertNotIn("no graded chunk in dense top-10",
                         report["questions"][0]["flags"])
        # Misaligned: flagged.
        report = build_report(golden, make_chunks(), cache,
                              lambda texts: [[0.0, 1.0]] * len(texts),
                              top_k=1)
        self.assertIn("no graded chunk in dense top-1",
                      report["questions"][0]["flags"])


if __name__ == "__main__":
    unittest.main()
