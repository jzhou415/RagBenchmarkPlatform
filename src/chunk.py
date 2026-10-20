#!/usr/bin/env python3
"""
chunk.py -- step 4: pages.jsonl -> manifest.jsonl (chunks).

RagBenchmarkPlatform, Phase 1 (Data Setup backend).

Uses LangChain's RecursiveCharacterTextSplitter with the ingestion
config's chunk_size / chunk_overlap (defaults 1000 / 200, see
src/config.py). Chunking is PER PAGE: no chunk ever crosses a page
boundary, so every chunk carries exactly one page_no -- that is what
keeps the (chunk_id, page, text) manifest schema honest and lets
auto_map (step 5) grade chunks against questions whose source pages
are known.

chunk_id format: p<page_no>_c<nn>  (e.g. p2714_c03 = 3rd chunk of
page 2714). Ids are stable for a fixed (pages.jsonl, chunk_size,
chunk_overlap): change any of those and ids shift, which is exactly
why the golden set pins the ingestion config_hash -- a sidecar
chunk_manifest.yaml next to the output records the config used here
(including config_hash, computed via src/config.py) so validate
(step 6) can refuse stale ground truth loudly.

Run on pages_dev.jsonl / pages_heldout.jsonl (from corpus_split.py)
or the full pages.jsonl -- chunk.py chunks whatever --pages names.

Example:
  python chunk.py --pages pages_dev.jsonl --out manifest_dev.jsonl
  python chunk.py --pages pages_heldout.jsonl --out manifest_heldout.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # works as `python src/chunk.py` and as `python -m src.chunk`
    from config import CorpusConfig, IngestionConfig
except ImportError:  # pragma: no cover
    from src.config import CorpusConfig, IngestionConfig


def load_pages(path):
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def chunk_pages(pages, splitter):
    """Split each page independently; yield manifest rows."""
    rows = []
    skipped_pages = 0
    for p in pages:
        text = (p.get("text") or "").strip()
        if not text:
            skipped_pages += 1
            continue
        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            rows.append({
                "chunk_id": f"p{p['page_no']}_c{i:02d}",
                "page_no": p["page_no"],
                "section": p.get("section"),
                "text": piece,
                "char_count": len(piece),
            })
    return rows, skipped_pages


def main(argv=None):
    ap = argparse.ArgumentParser(description="chunk pages.jsonl into manifest.jsonl")
    ap.add_argument("--pages", required=True, help="pages.jsonl (or a split subset)")
    ap.add_argument("--out", default="manifest.jsonl")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--corpus-manifest", default=None,
                    help="corpora/<name>/manifest.yaml (from config.py); "
                         "when given, its corpus sha256 feeds the config_hash")
    args = ap.parse_args(argv)

    if args.chunk_overlap >= args.chunk_size:
        print("error: --chunk-overlap must be < --chunk-size", file=sys.stderr)
        return 2

    pages = load_pages(args.pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    rows, skipped = chunk_pages(pages, splitter)

    corpus = (CorpusConfig.from_manifest(args.corpus_manifest)
              if args.corpus_manifest else CorpusConfig())
    cfg = IngestionConfig(chunk_size=args.chunk_size,
                          chunk_overlap=args.chunk_overlap,
                          corpus=corpus)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sidecar = {
        "pages_file": str(args.pages),
        "n_pages": len(pages),
        "n_pages_skipped_empty": skipped,
        "n_chunks": len(rows),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "config_hash": cfg.config_hash(),
        "corpus_name": corpus.name,
        "corpus_sha256": corpus.sha256,
    }
    sidecar_path = out_path.with_name("chunk_manifest.yaml")
    sidecar_path.write_text(
        yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True),
        encoding="utf-8")

    avg = sum(r["char_count"] for r in rows) / len(rows) if rows else 0
    print(f"pages:  {len(pages)} ({skipped} empty, skipped)")
    print(f"chunks: {len(rows)}  (avg {avg:.0f} chars)")
    print(f"config_hash: {cfg.config_hash()[:12]}...")
    print(f"wrote: {out_path} and {sidecar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
