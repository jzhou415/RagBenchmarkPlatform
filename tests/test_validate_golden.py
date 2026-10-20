"""Tests for src.validate_golden (step 6). Stdlib unittest only.

Each test mutates one aspect of a known-valid golden set and asserts
the validator rejects it (or, for the format check, only warns).
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from src.validate_golden import main, validate

CONFIG_HASH = "deadbeefcafe"
MANIFEST_PAGES = {"p0100_c00": 100, "p0100_c01": 100, "p0101_c00": 101}


def make_question(**overrides):
    q = {
        "id": "q001",
        "split": "dev",
        "provenance": "auto",
        "question": "What is the default value of shared_buffers?",
        "query_type": "exact_term",
        "content_tags": ["lookup"],
        "expected_answer": "128MB",
        "answer_span": "",
        "ground_truth": [
            {"chunk_id": "p0100_c00", "page": 100, "relevance": 3},
            {"chunk_id": "p0100_c01", "page": 100, "relevance": 1},
        ],
        "notes": "",
    }
    q.update(overrides)
    return q


def make_golden(**overrides):
    g = {
        "schema_version": "1.1",
        "corpus": {"name": "test-corpus", "source": "pdf", "pages": 2},
        "ingestion": {"config_hash": CONFIG_HASH, "parser": "docling",
                      "chunk_size": 1000, "chunk_overlap": 200},
        "questions": [make_question()],
    }
    g.update(overrides)
    return g


def check(golden):
    return validate(golden, MANIFEST_PAGES, CONFIG_HASH)


class TestValid(unittest.TestCase):
    def test_valid_set_passes(self):
        errors, warnings = check(make_golden())
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_unusual_chunk_id_warns_only(self):
        pages = dict(MANIFEST_PAGES)
        pages["weird-id"] = 100
        q = make_question(ground_truth=[
            {"chunk_id": "weird-id", "page": 100, "relevance": 3}])
        errors, warnings = validate(make_golden(questions=[q]), pages,
                                    CONFIG_HASH)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)


class TestRejections(unittest.TestCase):
    def assert_invalid(self, golden, needle):
        errors, _ = check(golden)
        self.assertTrue(any(needle in e for e in errors),
                        f"expected error containing {needle!r}, got {errors}")

    def test_bad_schema_version(self):
        self.assert_invalid(make_golden(schema_version="1.0"),
                            "schema_version")

    def test_missing_top_level_key(self):
        g = make_golden()
        del g["corpus"]
        self.assert_invalid(g, "missing top-level key: corpus")

    def test_config_hash_mismatch(self):
        g = make_golden()
        g["ingestion"]["config_hash"] = "someone-elses-hash"
        self.assert_invalid(g, "config_hash")

    def test_duplicate_question_id(self):
        g = make_golden(questions=[make_question(), make_question()])
        self.assert_invalid(g, "duplicate question id")

    def test_bad_split(self):
        self.assert_invalid(
            make_golden(questions=[make_question(split="test")]), "bad split")

    def test_bad_query_type(self):
        self.assert_invalid(
            make_golden(questions=[make_question(query_type="fuzzy")]),
            "bad query_type")

    def test_unknown_content_tag(self):
        self.assert_invalid(
            make_golden(questions=[make_question(content_tags=["bogus"])]),
            "unknown content_tags")

    def test_bad_provenance(self):
        self.assert_invalid(
            make_golden(questions=[make_question(provenance="guessed")]),
            "bad provenance")

    def test_no_grade3(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p0100_c01", "page": 100, "relevance": 2}])
        self.assert_invalid(make_golden(questions=[q]), "no grade-3")

    def test_relevance_zero_listed(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p0100_c00", "page": 100, "relevance": 3},
            {"chunk_id": "p0100_c01", "page": 100, "relevance": 0}])
        self.assert_invalid(make_golden(questions=[q]), "relevance")

    def test_relevance_out_of_range(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p0100_c00", "page": 100, "relevance": 4}])
        self.assert_invalid(make_golden(questions=[q]), "relevance")

    def test_chunk_not_in_manifest(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p9999_c00", "page": 9999, "relevance": 3}])
        self.assert_invalid(make_golden(questions=[q]), "not in manifest")

    def test_page_mismatch(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p0100_c00", "page": 101, "relevance": 3}])
        self.assert_invalid(make_golden(questions=[q]), "page")

    def test_duplicate_chunk_in_question(self):
        q = make_question(ground_truth=[
            {"chunk_id": "p0100_c00", "page": 100, "relevance": 3},
            {"chunk_id": "p0100_c00", "page": 100, "relevance": 2}])
        self.assert_invalid(make_golden(questions=[q]), "duplicate chunk_id")


class TestMainExitCodes(unittest.TestCase):
    def _write_fixtures(self, td, golden):
        td = Path(td)
        manifest = td / "manifest.jsonl"
        with open(manifest, "w", encoding="utf-8") as f:
            for cid, page in MANIFEST_PAGES.items():
                f.write(json.dumps({"chunk_id": cid, "page_no": page,
                                    "text": "x"}) + "\n")
        cm = td / "chunk_manifest.yaml"
        cm.write_text(yaml.safe_dump({"config_hash": CONFIG_HASH}),
                      encoding="utf-8")
        gfile = td / "golden.yaml"
        gfile.write_text(yaml.safe_dump(golden), encoding="utf-8")
        return ["--golden", str(gfile), "--manifest", str(manifest),
                "--chunk-manifest", str(cm)]

    def test_main_valid_returns_0(self):
        with tempfile.TemporaryDirectory() as td:
            argv = self._write_fixtures(td, make_golden())
            with redirect_stdout(io.StringIO()) as buf:
                rc = main(argv)
            self.assertEqual(rc, 0)
            self.assertIn("VALID", buf.getvalue())

    def test_main_invalid_returns_1(self):
        with tempfile.TemporaryDirectory() as td:
            argv = self._write_fixtures(td, make_golden(schema_version="9.9"))
            with redirect_stdout(io.StringIO()) as buf:
                rc = main(argv)
            self.assertEqual(rc, 1)
            self.assertIn("INVALID", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
