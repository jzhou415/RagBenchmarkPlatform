#!/usr/bin/env python3
"""Tests for corpus_split.py. Run: python -m unittest test_corpus_split"""
import json
import os
import tempfile
import unittest

from src.corpus_split import main, parse_range, select_dev


def make_pages():
    return [
        {"page_no": 1, "section": "SQL Commands", "text": "a",
         "char_count": 600, "table_count": 0},
        {"page_no": 2, "section": "SQL Commands", "text": "b",
         "char_count": 600, "table_count": 2},
        {"page_no": 3, "section": "Data Types", "text": "c",
         "char_count": 600, "table_count": 5},
        {"page_no": 4, "section": "Introduction", "text": "d",
         "char_count": 600, "table_count": 0},
        {"page_no": 5, "section": "Introduction", "text": "e",
         "char_count": 600, "table_count": 1},
    ]


class TestSelectDev(unittest.TestCase):
    def test_section_substring_case_insensitive(self):
        dev, matched = select_dev(make_pages(), ["sql commands"], [])
        self.assertEqual([p["page_no"] for p in dev], [1, 2])
        self.assertEqual(matched, {"SQL Commands"})

    def test_range_inclusive(self):
        dev, _ = select_dev(make_pages(), [], ["3-4"])
        self.assertEqual([p["page_no"] for p in dev], [3, 4])

    def test_section_or_range_union(self):
        dev, _ = select_dev(make_pages(), ["Data Types"], ["1-1"])
        self.assertEqual([p["page_no"] for p in dev], [1, 3])


class TestParseRange(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(parse_range("2714-2900"), (2714, 2900))

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            parse_range("abc")

    def test_start_after_end(self):
        with self.assertRaises(ValueError):
            parse_range("5-3")


class TestMainSplit(unittest.TestCase):
    def _write_pages(self, td):
        path = os.path.join(td, "pages.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in make_pages():
                f.write(json.dumps(p) + "\n")
        return path

    def _read_nos(self, path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line)["page_no"] for line in f if line.strip()]

    def test_split_disjoint_and_covering(self):
        with tempfile.TemporaryDirectory() as td:
            pages = self._write_pages(td)
            out_dev = os.path.join(td, "pages_dev.jsonl")
            out_held = os.path.join(td, "pages_heldout.jsonl")
            rc = main(["--pages", pages, "--dev-section", "SQL Commands",
                       "--out-dev", out_dev, "--out-heldout", out_held])
            self.assertEqual(rc, 0)
            dev_nos = set(self._read_nos(out_dev))
            held_nos = set(self._read_nos(out_held))
            self.assertEqual(dev_nos, {1, 2})
            self.assertEqual(dev_nos & held_nos, set())
            self.assertEqual(dev_nos | held_nos, {1, 2, 3, 4, 5})

    def test_no_selectors_refused(self):
        with tempfile.TemporaryDirectory() as td:
            pages = self._write_pages(td)
            rc = main(["--pages", pages,
                       "--out-dev", os.path.join(td, "d.jsonl"),
                       "--out-heldout", os.path.join(td, "h.jsonl")])
            self.assertEqual(rc, 2)

    def test_zero_matches_refused(self):
        with tempfile.TemporaryDirectory() as td:
            pages = self._write_pages(td)
            rc = main(["--pages", pages, "--dev-section", "No Such Section",
                       "--out-dev", os.path.join(td, "d.jsonl"),
                       "--out-heldout", os.path.join(td, "h.jsonl")])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
