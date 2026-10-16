"""Tests for src.parse (step 2). Stdlib unittest only -- no docling needed.

Docling is stubbed with duck-typed fakes: items need .text/.label/.prov,
tables need .data.table_cells. If the real Docling API drifts, these tests
pin the interface this module depends on.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.config import IngestionConfig
from src.parse import PageRecord, pages_from_document, parse_pdf, table_to_markdown


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------

class _Prov:
    def __init__(self, page_no):
        self.page_no = page_no


class _Label:
    def __init__(self, name):
        self.name = name


class FakeText:
    def __init__(self, text, page_no=1, label="TEXT", prov=True):
        self.text = text
        self.label = _Label(label)
        self.prov = [_Prov(page_no)] if prov else []


class FakeCell:
    def __init__(self, text, r, c):
        self.text = text
        self.start_row_offset_idx = r
        self.start_col_offset_idx = c


class FakeTable:
    """No export_to_markdown -> exercises the built-in grid renderer."""
    def __init__(self, cells, page_no=1):
        self.text = ""
        self.label = _Label("TABLE")
        self.prov = [_Prov(page_no)]
        self.data = type("D", (), {"table_cells": cells})()


class FakeTableWithExport(FakeTable):
    def export_to_markdown(self):
        return "EXPORTED TABLE"


class FakeDoc:
    def __init__(self, items):
        self._items = items

    def iterate_items(self):
        return [(it, 0) for it in self._items]

    @property
    def texts(self):
        return [it for it in self._items if isinstance(it, FakeText)]


class FakeResult:
    def __init__(self, doc):
        self.document = doc


class FakeConverter:
    def __init__(self, doc):
        self._doc = doc

    def convert(self, _path):
        return FakeResult(self._doc)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestPagesFromDocument(unittest.TestCase):
    def test_groups_text_by_page_in_order(self):
        doc = FakeDoc([
            FakeText("p2 text", page_no=2),
            FakeText("p1 text", page_no=1),
            FakeText("p1 more", page_no=1),
        ])
        pages = pages_from_document(doc)
        self.assertEqual([p.page_no for p in pages], [1, 2])
        self.assertIn("p1 text", pages[0].text)
        self.assertIn("p1 more", pages[0].text)
        self.assertEqual(pages[1].text, "p2 text")

    def test_tables_rendered_into_page_text_and_counted(self):
        doc = FakeDoc([
            FakeText("intro", page_no=1),
            FakeTable([FakeCell("a", 0, 0), FakeCell("b", 0, 1),
                       FakeCell("1", 1, 0), FakeCell("2", 1, 1)], page_no=1),
        ])
        pages = pages_from_document(doc)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].table_count, 1)
        self.assertIn("| a | b |", pages[0].text)
        self.assertIn("| 1 | 2 |", pages[0].text)
        self.assertIn("intro", pages[0].text)

    def test_table_export_preferred_when_available(self):
        doc = FakeDoc([FakeTableWithExport([], page_no=1)])
        pages = pages_from_document(doc)
        self.assertIn("EXPORTED TABLE", pages[0].text)

    def test_section_header_tracked_per_page(self):
        doc = FakeDoc([
            FakeText("SQL Commands", page_no=5, label="SECTION_HEADER"),
            FakeText("body", page_no=5),
            FakeText("more", page_no=6),
        ])
        pages = pages_from_document(doc)
        by_no = {p.page_no: p for p in pages}
        self.assertEqual(by_no[5].section, "SQL Commands")
        self.assertEqual(by_no[6].section, "SQL Commands")

    def test_missing_provenance_defaults_to_page_one(self):
        doc = FakeDoc([FakeText("orphan", prov=False)])
        pages = pages_from_document(doc)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].page_no, 1)

    def test_char_count_matches_text(self):
        doc = FakeDoc([FakeText("abc", page_no=1)])
        pages = pages_from_document(doc)
        self.assertEqual(pages[0].char_count, len(pages[0].text))


class TestTableToMarkdown(unittest.TestCase):
    def test_header_separator_row(self):
        t = FakeTable([FakeCell("h1", 0, 0), FakeCell("h2", 0, 1),
                       FakeCell("v", 1, 0)])
        md = table_to_markdown(t)
        lines = md.splitlines()
        self.assertEqual(lines[0], "| h1 | h2 |")
        self.assertEqual(lines[1], "| --- | --- |")
        self.assertEqual(lines[2], "| v |  |")

    def test_empty_table(self):
        self.assertEqual(table_to_markdown(FakeTable([])), "")


class TestParsePdfDriver(unittest.TestCase):
    def test_writes_pages_jsonl_and_manifest(self):
        doc = FakeDoc([
            FakeText("hello", page_no=1),
            FakeTable([FakeCell("x", 0, 0)], page_no=2),
        ])
        ingestion = IngestionConfig()
        ingestion.corpus.sha256 = "abc123"
        with tempfile.TemporaryDirectory() as tmp:
            dummy_pdf = Path(tmp) / "manual.pdf"
            dummy_pdf.write_bytes(b"%PDF-1.4 dummy")
            out = parse_pdf(dummy_pdf, Path(tmp) / "out",
                            ingestion, converter=FakeConverter(doc))
            pages = [json.loads(line) for line in
                     (out / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0]["page_no"], 1)
            manifest = json.loads((out / "parse_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["parser"], "docling")
            self.assertEqual(manifest["corpus_hash"], "abc123")
            self.assertEqual(manifest["config_hash"], ingestion.config_hash())
            self.assertEqual(manifest["pages"], 2)
            self.assertEqual(manifest["total_tables"], 1)
            self.assertIn("code_hash", manifest)

    def test_missing_pdf_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                parse_pdf("/no/such.pdf", tmp, converter=FakeConverter(FakeDoc([])))

    def test_page_record_schema(self):
        # contract with step 4 (chunk.py): every record carries page_no + text
        p = PageRecord(page_no=7, text="t", char_count=1, table_count=0)
        d = {"page_no": p.page_no, "text": p.text}
        self.assertEqual(set(d), {"page_no", "text"})


if __name__ == "__main__":
    unittest.main()
