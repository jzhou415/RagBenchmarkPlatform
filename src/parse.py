"""Step 2 (v2 DRAFT): parse the corpus PDF into per-page markdown (Docling).

DRAFT for review -- NOT committed. Needs John's per-write approval before it
replaces src/parse.py, and tests/test_parse.py must be rewritten against the
slimmer interface (see drafts/test_parse_v2_draft.py).

Contract (unchanged from committed src/parse.py)
------------------------------------------------
Input:  source PDF (+ optional ingestion config YAML from step 1).
Output: <out_dir>/pages.jsonl          one JSON object per line:
            {"page_no": 1, "text": "<markdown incl. tables>",
             "char_count": 1234, "table_count": 2, "section": "SELECT"}
        <out_dir>/parse_manifest.json  provenance for the run:
            config_hash, corpus_hash, code_hash, parser, docling_version,
            page count, timing.

The page records feed step 4 (chunk.py), which needs (page_no, text) per
page. Tables are rendered into the page markdown -- never dropped -- because
retrieval over table content is a first-class eval axis (content_tags=table).

What changed vs the committed version
-------------------------------------
The old pages_from_document hand-rolled per-page grouping via
iterate_items() + provenance bookkeeping and its own table_to_markdown
renderer. John's parse_docling (parse_chunk.py) showed export_to_markdown
accepts page_no, so Docling's own serializer now does reading order and
table placement -- simpler AND more faithful.

Kept deliberately:
- parse_manifest.json -- provenance (config_hash, corpus_hash, code_hash)
  that lets step 6 refuse to build indexes from a stale parse.
- table_count per page (from doc.tables + item provenance) -- cheap corpus stat.
- section per page (nearest preceding SECTION_HEADER from doc.texts) -- step 3
  uses it to scope question generation to the Part VI chapter for the dev split.
- compact_tables=True -- the measured ESPP fix (~47k chars of alignment
  padding) that the old version dropped.

Gone: the iterate_items grouping pass, _is_table, and table_to_markdown
(plus its TestTableToMarkdown tests). _page_no is now _prov_page_no and
_label_name is kept, both slimmed to header/table bookkeeping only.

Run:  python -m src.parse --pdf <file> [--config <ingestion.yaml>] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import IngestionConfig, code_hash, load_ingestion_config


# ---------------------------------------------------------------------------
# Docling access (lazy: the module must import without docling installed,
# so unit tests can run against stub documents)
# ---------------------------------------------------------------------------

def _require_converter():
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise RuntimeError(
            "docling is not installed. Install it with: pip install docling"
        ) from e
    return DocumentConverter()


def _docling_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "unknown")
    except ImportError:
        return "not-installed"


# ---------------------------------------------------------------------------
# page records
# ---------------------------------------------------------------------------

@dataclass
class PageRecord:
    page_no: int
    text: str            # markdown for the page, tables included
    char_count: int
    table_count: int
    section: str = ""    # nearest preceding section header (best effort)


def _prov_page_no(item) -> int:
    """1-based page number from Docling provenance; 1 when absent."""
    prov = getattr(item, "prov", None)
    if prov:
        pno = getattr(prov[0], "page_no", None)
        if pno:
            return int(pno)
    return 1


def _label_name(item) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "name", "") or ""


def pages_from_document(doc, page_nos=None) -> list[PageRecord]:
    """Per-page extraction via Docling's own serializer.

    text:        doc.export_to_markdown(page_no=p, compact_tables=True)
    table_count: number of doc.tables items whose provenance lands on the page
    section:     nearest preceding SECTION_HEADER in doc.texts (best effort,
                 carried forward across pages)
    """
    if page_nos is None:
        page_nos = sorted(doc.pages.keys())

    table_counts: dict[int, int] = {}
    for table in getattr(doc, "tables", []):
        pno = _prov_page_no(table)
        table_counts[pno] = table_counts.get(pno, 0) + 1

    headers: dict[int, str] = {}
    for item in getattr(doc, "texts", []):
        text = (getattr(item, "text", "") or "").strip()
        if "SECTION_HEADER" in _label_name(item) and text:
            headers[_prov_page_no(item)] = text  # last header on the page wins

    section, sections = "", {}
    for pno in page_nos:
        if pno in headers:
            section = headers[pno]
        sections[pno] = section

    records = []
    for pno in page_nos:
        # compact_tables=True: plain padded markdown inflates char counts with
        # alignment whitespace (measured ~47k chars of pure padding on the ESPP
        # PDF), which would bloat chunks and embeddings with spaces.
        text = (doc.export_to_markdown(page_no=pno, compact_tables=True) or "").strip()
        records.append(PageRecord(
            page_no=pno,
            text=text,
            char_count=len(text),
            table_count=table_counts.get(pno, 0),
            section=sections[pno],
        ))
    return records


# ---------------------------------------------------------------------------
# driver (unchanged contract)
# ---------------------------------------------------------------------------

def parse_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    ingestion: IngestionConfig | None = None,
    converter=None,
) -> Path:
    """Convert one PDF -> pages.jsonl + parse_manifest.json. Returns out_dir."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    ingestion = ingestion or IngestionConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if converter is None:
        converter = _require_converter()

    started = time.time()
    result = converter.convert(str(pdf_path))
    doc = result.document
    pages = pages_from_document(doc)
    duration_s = round(time.time() - started, 1)

    pages_path = out_dir / "pages.jsonl"
    with open(pages_path, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    manifest = {
        "parser": "docling",
        "docling_version": _docling_version(),
        "pdf_filename": pdf_path.name,
        "corpus_name": ingestion.corpus.name,
        "corpus_hash": ingestion.corpus.sha256,
        "config_hash": ingestion.config_hash(),
        "code_hash": code_hash(),
        "pages": len(pages),
        "total_chars": sum(p.char_count for p in pages),
        "total_tables": sum(p.table_count for p in pages),
        "duration_s": duration_s,
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "parse_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Step 2: PDF -> per-page markdown (Docling)")
    ap.add_argument("--pdf", required=True, help="source PDF path")
    ap.add_argument("--config", default=None, help="ingestion config YAML (step 1)")
    ap.add_argument(
        "--out", default=None,
        help="output dir (default: data/parsed/<corpus_name>/)",
    )
    args = ap.parse_args(argv)

    ingestion = load_ingestion_config(args.config) if args.config else IngestionConfig()
    out_dir = Path(args.out) if args.out else Path("data") / "parsed" / ingestion.corpus.name
    parse_pdf(args.pdf, out_dir, ingestion)
    manifest = json.loads((out_dir / "parse_manifest.json").read_text(encoding="utf-8"))
    print(f"pages: {manifest['pages']}  chars: {manifest['total_chars']:,}  "
          f"tables: {manifest['total_tables']}  ({manifest['duration_s']}s)")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()
