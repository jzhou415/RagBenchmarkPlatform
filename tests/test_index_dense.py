#!/usr/bin/env python3
"""Tests for index_dense.py (step 7b). Run: python -m unittest test_index_dense"""
import os
import tempfile
import unittest

from src.index_dense import build_dense_index, query_dense


def make_chunks():
    return [
        {"chunk_id": "p0100_c00", "page_no": 100, "section": "S",
         "text": "The default value of shared_buffers is 128MB in this release.",
         "char_count": 60},
        {"chunk_id": "p0100_c01", "page_no": 100, "section": "S",
         "text": "shared_buffers sets the amount of memory for shared buffers.",
         "char_count": 58},
        {"chunk_id": "p0101_c00", "page_no": 101, "section": "T",
         "text": "Unrelated text about logging and checkpoints.",
         "char_count": 45},
    ]


def fake_vectors():
    # Deterministic unit vectors; no API needed.
    return [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]


class TestDenseIndex(unittest.TestCase):
    def test_build_query_roundtrip(self):
        chunks = make_chunks()
        with tempfile.TemporaryDirectory() as td:
            persist = os.path.join(td, "dense_test")
            build_dense_index(chunks, persist, "chunks_test", fake_vectors())
            hits = query_dense(persist, "chunks_test", [1.0, 0.0, 0.0],
                               top_k=1)
            self.assertEqual(hits[0][0], "p0100_c00")
            hits = query_dense(persist, "chunks_test", [0.0, 0.0, 1.0],
                               top_k=3)
            self.assertEqual([h[0] for h in hits][0], "p0101_c00")
            self.assertEqual(len(hits), 3)

    def test_metadatas_stored(self):
        import chromadb
        chunks = make_chunks()
        with tempfile.TemporaryDirectory() as td:
            persist = os.path.join(td, "dense_test")
            col = build_dense_index(chunks, persist, "chunks_test",
                                    fake_vectors())
            got = col.get(ids=["p0101_c00"])
            self.assertEqual(got["metadatas"][0]["page_no"], 101)
            self.assertEqual(got["metadatas"][0]["section"], "T")
            self.assertEqual(col.count(), 3)
            # Reopening the persist dir sees the same collection.
            col2 = chromadb.PersistentClient(
                path=persist).get_collection("chunks_test")
            self.assertEqual(col2.count(), 3)

    def test_refuses_overwrite(self):
        chunks = make_chunks()
        with tempfile.TemporaryDirectory() as td:
            persist = os.path.join(td, "dense_test")
            build_dense_index(chunks, persist, "chunks_test", fake_vectors())
            with self.assertRaises(SystemExit):
                build_dense_index(chunks, persist, "chunks_test",
                                  fake_vectors())

    def test_add_is_batched(self):
        # Chroma caps a single add() batch; build_dense_index must split
        # large adds (observed: 11198 chunks > max 5461).
        from src import index_dense
        chunks = [
            {"chunk_id": f"p0100_c{i:02d}", "page_no": 100, "section": "S",
             "text": f"chunk number {i} about shared_buffers",
             "char_count": 30}
            for i in range(5)
        ]
        vecs = [[float(i == j) for j in range(5)] for i in range(5)]
        old_batch, index_dense.ADD_BATCH = index_dense.ADD_BATCH, 2
        try:
            with tempfile.TemporaryDirectory() as td:
                persist = os.path.join(td, "dense_batched")
                col = build_dense_index(chunks, persist, "chunks_batched",
                                        vecs)
                self.assertEqual(col.count(), 5)
                hits = query_dense(persist, "chunks_batched",
                                   [0.0, 0.0, 0.0, 0.0, 1.0], top_k=1)
                self.assertEqual(hits[0][0], "p0100_c04")
        finally:
            index_dense.ADD_BATCH = old_batch


if __name__ == "__main__":
    unittest.main()
