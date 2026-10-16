"""Step 2: parse the corpus PDF into per-page markdown with metadata (Docling).

Contract
--------
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


def _page_no(item) -> int:
    """1-based page number from Docling provenance; 1 when absent."""
    prov = getattr(item, "prov", None)
    if prov:
        pno = getattr(prov[0], "page_no", None)
        if pno:
            return int(pno)
    return 1


def _is_table(item) -> bool:
    return hasattr(item, "data") and hasattr(getattr(item, "data"), "table_cells")


def _label_name(item) -> str:
    label = getattr(item, "label", None)
    return getattr(label, "name", "") or ""


def table_to_markdown(table) -> str:
    """Render a Docling table as a markdown grid.

    Uses the table's own exporter when available, otherwise builds a simple
    grid from TableData cells. Spans beyond the start cell are not expanded;
    the text stays attached to its origin cell so number<->label association
    survives (this is what the ESPP benchmark showed matters).
    """
    export = getattr(table, "export_to_markdown", None)
    if callable(export):
        try:
            md = export()
            if md and md.strip():
                return md.strip()
        except Exception:
            pass
    data = getattr(table, "data", None)
    cells = getattr(data, "table_cells", None) or []
    grid: dict[int, dict[int, str]] = {}
    max_row, max_col = -1, -1
    for cell in cells:
        r = int(getattr(cell, "start_row_offset_idx", 0))
        c = int(getattr(cell, "start_col_offset_idx", 0))
        grid.setdefault(r, {})[c] = (getattr(cell, "text", "") or "").strip()
        max_row, max_col = max(max_row, r), max(max_col, c)
    if max_row < 0:
        return ""
    lines = []
    for r in range(max_row + 1):
        row = grid.get(r, {})
        lines.append("| " + " | ".join(row.get(c, "") for c in range(max_col + 1)) + " |")
        if r == 0:
            lines.append("| " + " | ".join("---" for _ in range(max_col + 1)) + " |")
    return "\n".join(lines)


def pages_from_document(doc) -> list[PageRecord]:
    """Single document-order pass: group text + tables per page, track sections.

    Works against the real DoclingDocument and against the stub documents
    used in tests (duck-typed: items need .text/.label/.prov; tables need
    .data.table_cells).
    """
    buckets: dict[int, dict] = {}
    current_section = ""

    def bucket(pno: int) -> dict:
        return buckets.setdefault(pno, {"parts": [], "tables": 0, "section": ""})

    try:
        ordered = list(doc.iterate_items())
    except Exception:
        ordered = [(t, 0) for t in getattr(doc, "texts", [])]

    for item, _level in ordered:
        pno = _page_no(item)
        b = bucket(pno)
        label = _label_name(item)
        text = (getattr(item, "text", "") or "").strip()

        if "SECTION_HEADER" in label and text:
            current_section = text
        if _is_table(item):
            md = table_to_markdown(item)
            if md:
                b["parts"].append(md)
            b["tables"] += 1
        elif text:
            b["parts"].append(text)
        if current_section:
            b["section"] = current_section  # latest header wins; best effort

    records = []
    for pno in sorted(buckets):
        b = buckets[pno]
        text = "\n\n".join(b["parts"])
        records.append(PageRecord(
            page_no=pno,
            text=text,
            char_count=len(text),
            table_count=b["tables"],
            section=b["section"],
        ))
    return records


# ---------------------------------------------------------------------------
# driver
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
