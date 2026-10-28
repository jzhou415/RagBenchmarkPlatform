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


class _FakeEmbedding:
    def __init__(self, vector):
        self.embedding = vector


class _FakeEmbeddingsResp:
    def __init__(self, vectors):
        self.data = [_FakeEmbedding(v) for v in vectors]


class _FakeEmbeddings:
    """Records calls; returns deterministic one-hot vectors."""
    def __init__(self):
        self.calls = []
        self.seen = 0

    def create(self, model, input):
        self.calls.append(list(input))
        vecs = []
        for _ in input:
            v = [0.0, 0.0, 0.0]
            v[self.seen % 3] = 1.0
            vecs.append(v)
            self.seen += 1
        return _FakeEmbeddingsResp(vecs)


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


class TestEmbedCacheFlag(unittest.TestCase):
    def _write_inputs(self, td, chunks):
        import json
        manifest = os.path.join(td, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        cm = os.path.join(td, "chunk_manifest.yaml")
        with open(cm, "w", encoding="utf-8") as f:
            f.write("config_hash: abcdef1234567890\n")
        return manifest, cm

    def test_reuses_shared_cache_without_api_calls(self):
        # auto_map's cache, prefilled: index_dense must embed nothing.
        import json
        from unittest import mock
        from src import index_dense
        chunks = make_chunks()
        vecs = {c["chunk_id"]: v
                for c, v in zip(chunks, fake_vectors())}
        with tempfile.TemporaryDirectory() as td:
            manifest, cm = self._write_inputs(td, chunks)
            shared = os.path.join(td, "embed_cache", "chunk_embeddings.json")
            os.makedirs(os.path.dirname(shared), exist_ok=True)
            with open(shared, "w", encoding="utf-8") as f:
                json.dump(vecs, f)
            fake = _FakeClient()
            with mock.patch.dict(os.environ,
                                 {"OPENAI_API_KEY": "test"}), \
                 mock.patch.object(index_dense, "make_client",
                                   return_value=fake):
                index_dense.main(["--manifest", manifest,
                                  "--chunk-manifest", cm,
                                  "--out-dir", os.path.join(td, "indexes"),
                                  "--embed-cache", shared])
            self.assertEqual(fake.embeddings.calls, [])
            persist = os.path.join(td, "indexes", "dense_abcdef123456")
            self.assertTrue(os.path.isdir(persist))

    def test_embed_cache_flag_creates_parent_and_writes_there(self):
        import json
        from unittest import mock
        from src import index_dense
        chunks = make_chunks()
        with tempfile.TemporaryDirectory() as td:
            manifest, cm = self._write_inputs(td, chunks)
            fresh = os.path.join(td, "newdir", "sub", "cache.json")
            fake = _FakeClient()
            with mock.patch.dict(os.environ,
                                 {"OPENAI_API_KEY": "test"}), \
                 mock.patch.object(index_dense, "make_client",
                                   return_value=fake):
                index_dense.main(["--manifest", manifest,
                                  "--chunk-manifest", cm,
                                  "--out-dir", os.path.join(td, "indexes"),
                                  "--embed-cache", fresh])
            self.assertTrue(os.path.isfile(fresh))
            with open(fresh, encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(set(saved), {c["chunk_id"] for c in chunks})
            # Default cache location untouched: the flag, not the
            # default, was used.
            self.assertFalse(os.path.exists(
                os.path.join(td, "indexes", ".embed_cache_abcdef123456.json")))
            self.assertTrue(fake.embeddings.calls)


if __name__ == "__main__":
    unittest.main()
