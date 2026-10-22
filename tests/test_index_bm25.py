#!/usr/bin/env python3
"""Tests for index_bm25.py (step 7a). Run: python -m unittest test_index_bm25"""
import json
import os
import tempfile
import unittest

from src.index_bm25 import (
    build_bm25,
    index_dir_name,
    load_index,
    save_index,
    search,
    tokenize,
)


def make_chunks():
    return [
        {"chunk_id": "p0100_c00", "page_no": 100, "section": "S",
         "text": "The default value of shared_buffers is 128MB in this release.",
         "char_count": 60},
        {"chunk_id": "p0100_c01", "page_no": 100, "section": "S",
         "text": "shared_buffers sets the amount of memory for shared buffers.",
         "char_count": 58},
        {"chunk_id": "p0101_c00", "page_no": 101, "section": "S",
         "text": "Unrelated text about logging and checkpoints.",
         "char_count": 45},
    ]


class TestTokenize(unittest.TestCase):
    def test_lowercase_alnum(self):
        self.assertEqual(tokenize("Shared_Buffers 128MB!"),
                         ["shared_buffers", "128mb"])


class TestBuildSearch(unittest.TestCase):
    def test_top_hit(self):
        bm25, doc_ids = build_bm25(make_chunks())
        hits = search(bm25, doc_ids, "default shared_buffers value", top_k=2)
        self.assertEqual(hits[0][0], "p0100_c00")
        self.assertEqual(len(hits), 2)

    def test_scores_descend(self):
        bm25, doc_ids = build_bm25(make_chunks())
        hits = search(bm25, doc_ids, "shared_buffers", top_k=3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        chunks = make_chunks()
        bm25, doc_ids = build_bm25(chunks)
        with tempfile.TemporaryDirectory() as td:
            target = save_index(bm25, doc_ids, "deadbeefcafe1234", td)
            self.assertTrue(target.endswith(index_dir_name("deadbeefcafe1234")))
            self.assertTrue(os.path.isfile(
                os.path.join(target, "bm25.pkl")))
            bm25b, doc_idsb, meta = load_index(target)
            self.assertEqual(doc_idsb, doc_ids)
            self.assertEqual(meta["config_hash"], "deadbeefcafe1234")
            self.assertEqual(meta["n_chunks"], 3)
            self.assertIn("created_at", meta)
            hits = search(bm25b, doc_idsb, "checkpoints logging", top_k=1)
            self.assertEqual(hits[0][0], "p0101_c00")

    def test_refuses_overwrite(self):
        chunks = make_chunks()
        bm25, doc_ids = build_bm25(chunks)
        with tempfile.TemporaryDirectory() as td:
            save_index(bm25, doc_ids, "deadbeefcafe1234", td)
            with self.assertRaises(SystemExit):
                save_index(bm25, doc_ids, "deadbeefcafe1234", td)

    def test_different_hash_different_dir(self):
        chunks = make_chunks()
        bm25, doc_ids = build_bm25(chunks)
        with tempfile.TemporaryDirectory() as td:
            t1 = save_index(bm25, doc_ids, "aaaa", td)
            t2 = save_index(bm25, doc_ids, "bbbb", td)
            self.assertNotEqual(t1, t2)


if __name__ == "__main__":
    unittest.main()
