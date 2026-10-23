#!/usr/bin/env python3
"""Tests for src/retrieve.py (step 9). Run: python -m unittest test_retrieve"""
import json
import os
import tempfile
import unittest

from src.retrieve import (
    RUNGS,
    bm25_search,
    hybrid_rerank_search,
    hybrid_search,
    load_indexes,
    retrieve,
    rrf_fuse,
    semantic_search,
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


def make_test_indexes(td):
    """Build a tiny bm25 + dense index pair under td/indexes. Returns dir."""
    from src.index_bm25 import build_bm25, save_index
    from src.index_dense import build_dense_index
    chunks = make_chunks()
    idx_dir = os.path.join(td, "indexes")
    os.makedirs(idx_dir)
    bm25, doc_ids = build_bm25(chunks)
    save_index(bm25, doc_ids, "testhash1234", idx_dir)
    dense_dir = os.path.join(idx_dir, "dense_testhash1234")
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    build_dense_index(chunks, dense_dir, "chunks_testhash1234", vecs)
    meta = {"config_hash": "testhash1234", "embedding_model": "fake",
            "n_chunks": 3, "created_at": "now"}
    with open(dense_dir + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return idx_dir


class TestRRF(unittest.TestCase):
    def test_fusion_order_and_scores(self):
        a = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        b = [("b", 1.0), ("c", 0.9), ("d", 0.8)]
        fused = rrf_fuse([a, b], rrf_k=60)
        order = [cid for cid, _ in fused]
        # b: 1/61+1/61 > c: 1/62+1/62 > a: 1/61 > d: 1/63
        self.assertEqual(order, ["b", "c", "a", "d"])
        scores = dict(fused)
        self.assertAlmostEqual(scores["b"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(scores["d"], 1 / 63)

    def test_single_ranking_preserves_order(self):
        r = [("x", 5.0), ("y", 1.0)]
        self.assertEqual([cid for cid, _ in rrf_fuse([r])], ["x", "y"])

    def test_empty_rankings(self):
        self.assertEqual(rrf_fuse([[], []]), [])


class TestDispatch(unittest.TestCase):
    def test_unknown_rung_raises(self):
        with self.assertRaises(ValueError):
            retrieve("q", "nope", {}, query_vector=[1.0])

    def test_hybrid_rerank_without_fn_raises(self):
        with self.assertRaises(ValueError):
            hybrid_rerank_search("q", [1.0], {}, {},
                                 rerank_fn=None)

    def test_rungs_tuple(self):
        self.assertEqual(RUNGS, ("semantic", "bm25", "hybrid",
                                 "hybrid_rerank"))


class TestLadderEndToEnd(unittest.TestCase):
    def test_bm25_rung(self):
        with tempfile.TemporaryDirectory() as td:
            indexes = load_indexes(make_test_indexes(td))
            hits = retrieve("default shared_buffers value", "bm25",
                            indexes, top_k=2)
            self.assertEqual(hits[0][0], "p0100_c00")

    def test_semantic_rung(self):
        with tempfile.TemporaryDirectory() as td:
            indexes = load_indexes(make_test_indexes(td))
            hits = retrieve("q", "semantic", indexes,
                            query_vector=[1.0, 0.0, 0.0], top_k=1)
            self.assertEqual(hits[0][0], "p0100_c00")

    def test_semantic_needs_vector_or_embed_fn(self):
        with tempfile.TemporaryDirectory() as td:
            indexes = load_indexes(make_test_indexes(td))
            with self.assertRaises(ValueError):
                retrieve("q", "semantic", indexes)
            hits = retrieve("q", "semantic", indexes,
                            embed_fn=lambda q: [0.0, 0.0, 1.0], top_k=1)
            self.assertEqual(hits[0][0], "p0101_c00")

    def test_hybrid_rung(self):
        with tempfile.TemporaryDirectory() as td:
            indexes = load_indexes(make_test_indexes(td))
            hits = retrieve("shared_buffers default 128MB", "hybrid",
                            indexes, query_vector=[1.0, 0.0, 0.0],
                            top_k=3, depth=3)
            ids = [cid for cid, _ in hits]
            # p0100_c00 wins both rungs -> fused first
            self.assertEqual(ids[0], "p0100_c00")
            self.assertEqual(len(ids), 3)

    def test_hybrid_rerank_uses_fn(self):
        with tempfile.TemporaryDirectory() as td:
            indexes = load_indexes(make_test_indexes(td))
            texts = {c["chunk_id"]: (c["text"], c["page_no"])
                     for c in make_chunks()}

            def fake_rerank(query, pairs, top_k):
                # Deterministic rule independent of candidate order:
                # chunk_id descending.
                ranked = sorted(pairs, key=lambda p: p[0], reverse=True)
                return [(cid, 1.0) for cid, _ in ranked[:top_k]]

            hits = retrieve("shared_buffers", "hybrid_rerank", indexes,
                            texts=texts, query_vector=[1.0, 0.0, 0.0],
                            top_k=2, candidate_k=3, rerank_fn=fake_rerank)
            # 3 chunks total, so candidates are all three; reranked desc.
            self.assertEqual([h[0] for h in hits],
                             ["p0101_c00", "p0100_c01"])

    def test_config_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as td:
            idx_dir = make_test_indexes(td)
            bad_meta = {"config_hash": "different!!!!!", "embedding_model":
                        "fake", "n_chunks": 3, "created_at": "now"}
            with open(os.path.join(idx_dir, "dense_testhash1234.meta.json"),
                      "w", encoding="utf-8") as f:
                json.dump(bad_meta, f)
            with self.assertRaises(SystemExit):
                load_indexes(idx_dir)


if __name__ == "__main__":
    unittest.main()
