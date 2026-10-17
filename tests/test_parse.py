"""Tests for src.parse v2 (slimmed interface). Stdlib unittest only -- no docling.

DRAFT for review -- NOT committed. Replaces tests/test_parse.py if the v2
parse.py is approved. Docling is stubbed with duck-typed fakes: the doc
needs .pages, .tables, .texts and export_to_markdown(page_no=..., ...).

If the real Docling API drifts, these tests pin the (much smaller) interface
this module now depends on.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.config import IngestionConfig
from src.parse import PageRecord, pages_from_document, parse_pdf


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


class FakeTable:
    def __init__(self, page_no=1, prov=True):
        self.prov = [_Prov(page_no)] if prov else []


class FakeDoc:
    """Duck-typed DoclingDocument: pages + tables + texts + page export."""

    def __init__(self, page_md, texts=(), tables=()):
        # page_md: {page_no: markdown string}
        self._page_md = dict(page_md)
        self._texts = list(texts)
        self._tables = list(tables)
        self.export_calls = []

    @property
    def pages(self):
        return self._page_md

    @property
    def texts(self):
        return self._texts

    @property
    def tables(self):
        return self._tables

    def export_to_markdown(self, page_no=None, compact_tables=False):
        self.export_calls.append((page_no, compact_tables))
        return self._page_md.get(page_no, "")


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
    def test_one_export_per_page_with_compact_tables(self):
        doc = FakeDoc({1: "p1 text", 2: "p2 text"})
        pages = pages_from_document(doc)
        self.assertEqual([p.page_no for p in pages], [1, 2])
        self.assertEqual(pages[0].text, "p1 text")
        self.assertEqual(pages[1].text, "p2 text")
        # every call used the proven one-liner: page_no + compact_tables=True
        self.assertEqual(
            doc.export_calls,
            [(1, True), (2, True)],
        )

    def test_text_is_stripped(self):
        doc = FakeDoc({3: "\n  padded  \n"})
        pages = pages_from_document(doc)
        self.assertEqual(pages[0].text, "padded")

    def test_tables_counted_from_provenance(self):
        doc = FakeDoc(
            {1: "a", 2: "b"},
            tables=[FakeTable(page_no=1), FakeTable(page_no=1), FakeTable(page_no=2)],
        )
        pages = pages_from_document(doc)
        by_no = {p.page_no: p for p in pages}
        self.assertEqual(by_no[1].table_count, 2)
        self.assertEqual(by_no[2].table_count, 1)

    def test_table_without_provenance_defaults_to_page_one(self):
        doc = FakeDoc({1: "a"}, tables=[FakeTable(prov=False)])
        pages = pages_from_document(doc)
        self.assertEqual(pages[0].table_count, 1)

    def test_section_header_carried_forward_across_pages(self):
        doc = FakeDoc(
            {5: "body", 6: "more"},
            texts=[FakeText("SQL Commands", page_no=5, label="SECTION_HEADER")],
        )
        pages = pages_from_document(doc)
        by_no = {p.page_no: p for p in pages}
        self.assertEqual(by_no[5].section, "SQL Commands")
        self.assertEqual(by_no[6].section, "SQL Commands")

    def test_latest_header_on_page_wins(self):
        doc = FakeDoc(
            {1: "x"},
            texts=[
                FakeText("First", page_no=1, label="SECTION_HEADER"),
                FakeText("Second", page_no=1, label="SECTION_HEADER"),
            ],
        )
        self.assertEqual(pages_from_document(doc)[0].section, "Second")

    def test_explicit_page_nos_subset(self):
        doc = FakeDoc({1: "a", 2: "b", 3: "c"})
        pages = pages_from_document(doc, page_nos=[2])
        self.assertEqual([p.page_no for p in pages], [2])
        self.assertEqual(pages[0].text, "b")

    def test_char_count_matches_text(self):
        doc = FakeDoc({1: "abc"})
        pages = pages_from_document(doc)
        self.assertEqual(pages[0].char_count, len(pages[0].text))


class TestParsePdfDriver(unittest.TestCase):
    def test_writes_pages_jsonl_and_manifest(self):
        doc = FakeDoc(
            {1: "hello", 2: "world"},
            tables=[FakeTable(page_no=2)],
        )
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
            self.assertEqual(pages[0]["table_count"], 0)
            self.assertEqual(pages[1]["table_count"], 1)
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
                parse_pdf("/no/such.pdf", tmp, converter=FakeConverter(FakeDoc({})))

    def test_page_record_schema(self):
        # contract with step 4 (chunk.py): every record carries page_no + text
        p = PageRecord(page_no=7, text="t", char_count=1, table_count=0)
        d = {"page_no": p.page_no, "text": p.text}
        self.assertEqual(set(d), {"page_no", "text"})


if __name__ == "__main__":
    unittest.main()
