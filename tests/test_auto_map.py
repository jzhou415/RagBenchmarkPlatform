"""Tests for src.auto_map (step 5). Stdlib unittest only.

The OpenAI client is a duck-typed fake: embeddings are deterministic
hash-based vectors, and the chat judge grades by rule (3 if the
expected answer appears in the chunk text, else 0). No API calls,
no OPENAI_API_KEY needed.
"""

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import yaml

from src.auto_map import (
    build_review_queue,
    candidate_pool,
    grade_candidates,
    get_embeddings,
    main,
    tokenize,
    verbatim_hit,
    build_bm25,
)


# ---------------------------------------------------------------------------
# fake OpenAI client
# ---------------------------------------------------------------------------

def _vec(text, dim=16):
    digest = hashlib.md5(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:dim]]


class _EmbObj:
    def __init__(self, embedding):
        self.embedding = embedding


class _EmbResp:
    def __init__(self, data):
        self.data = data


class FakeEmbeddings:
    def __init__(self):
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        texts = [input] if isinstance(input, str) else input
        return _EmbResp([_EmbObj(_vec(t)) for t in texts])


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _ChatResp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeChat:
    """Rule-based judge; `mode` can force fence-wrapped / broken output."""

    def __init__(self, mode="clean"):
        self.mode = mode
        self.calls = 0

    def create(self, model, temperature, response_format, messages):
        self.calls += 1
        if self.mode == "garbage":
            return _ChatResp("not json at all")
        user = messages[1]["content"]
        expected = ""
        for line in user.splitlines():
            if line.startswith("Expected answer:"):
                expected = line.split(":", 1)[1].strip()
        grades = []
        blocks = user.split("--- chunk ")[1:]
        for block in blocks:
            cid = block.split()[0].split("[")[0].strip()
            text = block.split("\n", 1)[1] if "\n" in block else ""
            grade = 3 if expected and expected.lower() in text.lower() else 0
            grades.append({"chunk_id": cid, "grade": grade,
                           "rationale": "fake judge"})
        if self.mode == "drop_one" and grades:
            grades = grades[1:]
        payload = json.dumps({"grades": grades})
        if self.mode == "fenced":
            payload = f"```json\n{payload}\n```"
        return _ChatResp(payload)


class FakeClient:
    def __init__(self, chat_mode="clean"):
        self.embeddings = FakeEmbeddings()
        self.chat = _ChatNs(FakeChat(chat_mode))


class _ChatNs:
    def __init__(self, completions):
        self.completions = completions


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def make_chunks():
    return [
        {"chunk_id": "p0100_c00", "page_no": 100, "section": "S",
         "text": "The default value of shared_buffers is 128MB in this release.",
         "char_count": 60},
        {"chunk_id": "p0100_c01", "page_no": 100, "section": "S",
         "text": "shared_buffers sets the amount of memory used for shared memory buffers.",
         "char_count": 70},
        {"chunk_id": "p0101_c00", "page_no": 101, "section": "S",
         "text": "Unrelated text about logging and checkpoints.",
         "char_count": 45},
    ]


class TestHelpers(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Hello, WORLD! x2"),
                         ["hello", "world", "x2"])

    def test_verbatim_hit(self):
        self.assertTrue(verbatim_hit("128MB", "the value is 128mb here"))
        self.assertFalse(verbatim_hit("128MB", "the value is 256MB"))
        self.assertFalse(verbatim_hit("", "anything"))
        self.assertFalse(verbatim_hit("...", "anything"))
        self.assertFalse(verbatim_hit(None, "anything"))


class TestCandidatePool(unittest.TestCase):
    def test_pool_is_union_within_bounds(self):
        chunks = make_chunks()
        bm25 = build_bm25(chunks)
        chunk_vecs = [_vec(c["text"]) for c in chunks]
        pool = candidate_pool("shared_buffers default", chunks, bm25,
                              chunk_vecs, _vec("shared_buffers default"))
        # fewer chunks than TOP_N -> pool covers everything
        self.assertEqual(pool, [0, 1, 2])

    def test_pool_deterministic(self):
        chunks = make_chunks()
        bm25 = build_bm25(chunks)
        chunk_vecs = [_vec(c["text"]) for c in chunks]
        a = candidate_pool("memory buffers", chunks, bm25, chunk_vecs,
                           _vec("memory buffers"))
        b = candidate_pool("memory buffers", chunks, bm25, chunk_vecs,
                           _vec("memory buffers"))
        self.assertEqual(a, b)


class TestEmbeddingsCache(unittest.TestCase):
    def test_cache_keyed_by_chunk_id_and_reused(self):
        chunks = make_chunks()
        client = FakeClient()
        with tempfile.TemporaryDirectory() as td:
            cache = str(Path(td) / "emb.json")
            v1 = get_embeddings(client, chunks, cache)
            calls_after_first = client.embeddings.calls
            self.assertGreater(calls_after_first, 0)
            v2 = get_embeddings(client, chunks, cache)
            self.assertEqual(client.embeddings.calls, calls_after_first)
            self.assertEqual(v1, v2)


class TestGrading(unittest.TestCase):
    def _candidates(self):
        return [
            {"chunk_id": "p0100_c00", "text": "value is 128MB", "premarked": True},
            {"chunk_id": "p0101_c00", "text": "unrelated", "premarked": False},
        ]

    def test_clean_grades(self):
        out = grade_candidates(FakeClient(), "q?", "128MB", self._candidates())
        self.assertEqual(out["p0100_c00"][0], 3)
        self.assertEqual(out["p0101_c00"][0], 0)

    def test_fenced_json_parsed(self):
        out = grade_candidates(FakeClient(chat_mode="fenced"),
                               "q?", "128MB", self._candidates())
        self.assertEqual(out["p0100_c00"][0], 3)

    def test_dropped_candidate_defaults_zero(self):
        out = grade_candidates(FakeClient(chat_mode="drop_one"),
                               "q?", "128MB", self._candidates())
        self.assertEqual(out["p0100_c00"][0], 0)
        self.assertEqual(out["p0100_c00"][1], "not graded by judge")

    def test_garbage_output_defaults_zero(self):
        out = grade_candidates(FakeClient(chat_mode="garbage"),
                               "q?", "128MB", self._candidates())
        self.assertTrue(all(g == 0 for g, _ in out.values()))


class TestReviewQueue(unittest.TestCase):
    def test_queue_composition(self):
        questions = (
            [{"id": f"q{i:03d}", "query_type": "multi_hop"} for i in range(2)]
            + [{"id": f"q{i:03d}", "query_type": "exact_term"}
               for i in range(2, 12)]
        )
        queue = build_review_queue(questions, ["q005"], random.Random(42))
        ids = [e["id"] for e in queue]
        self.assertEqual(len(ids), len(set(ids)))          # no dupes
        self.assertIn("q000", ids)                          # multi_hop
        self.assertIn("q001", ids)                          # multi_hop
        self.assertIn("q005", ids)                          # no grade-3
        # 2 multi_hop + 1 no-grade3 + max(1, int(7*0.2))=1 spot-check
        self.assertEqual(len(ids), 4)


class TestEndToEnd(unittest.TestCase):
    def test_main_writes_golden_and_queue(self):
        chunks = make_chunks()
        questions = [
            {"id": "q001", "question": "What is the default shared_buffers?",
             "query_type": "exact_term", "content_tags": ["lookup"],
             "expected_answer": "128MB", "source_pages": [100]},
            {"id": "q002", "question": "How do buffers and logging interact?",
             "query_type": "multi_hop", "content_tags": ["policy"],
             "expected_answer": "128MB", "source_pages": [100, 101]},
        ]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            manifest = td / "manifest.jsonl"
            with open(manifest, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c) + "\n")
            cm = td / "chunk_manifest.yaml"
            cm.write_text(yaml.safe_dump({
                "config_hash": "deadbeefcafe", "corpus_name": "test-corpus",
                "corpus_sha256": "abc123", "n_pages": 2,
                "chunk_size": 1000, "chunk_overlap": 200,
            }), encoding="utf-8")
            qfile = td / "questions.yaml"
            qfile.write_text(yaml.safe_dump(questions), encoding="utf-8")
            out = td / "golden.yaml"
            queue = td / "review_queue.yaml"

            rc = main(["--questions", str(qfile), "--manifest", str(manifest),
                       "--chunk-manifest", str(cm), "--split", "dev",
                       "--out", str(out), "--review-queue", str(queue),
                       "--cache", str(td / "cache")],
                      client=FakeClient())
            self.assertEqual(rc, 0)

            golden = yaml.safe_load(out.read_text(encoding="utf-8"))
            self.assertEqual(golden["schema_version"], "1.1")
            self.assertEqual(golden["ingestion"]["config_hash"],
                             "deadbeefcafe")
            q1 = golden["questions"][0]
            self.assertEqual(q1["split"], "dev")
            self.assertEqual(q1["provenance"], "auto")
            gt = {g["chunk_id"]: g for g in q1["ground_truth"]}
            self.assertEqual(gt["p0100_c00"]["relevance"], 3)
            self.assertEqual(gt["p0100_c00"]["page"], 100)

            review = yaml.safe_load(queue.read_text(encoding="utf-8"))
            review_ids = [e["id"] for e in review["review_queue"]]
            self.assertIn("q002", review_ids)   # multi_hop always queued


if __name__ == "__main__":
    unittest.main()
