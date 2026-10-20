#!/usr/bin/env python3
"""
corpus_split.py -- dev / held-out corpus split for golden-set generation.

RagBenchmarkPlatform, Phase 1 (step 3 tooling).

Runs on a local machine against the local pages.jsonl.
All file opens use encoding="utf-8" (pages.jsonl is UTF-8; Windows open()
defaults to cp1252 and chokes on the first non-Latin-1 byte).

Two modes:

  --analyze            print sections ranked by table count, so you can pick
                       the table-rich chapter to pair with Part VI.

  (default split)      write pages_dev.jsonl + pages_heldout.jsonl from
                       --dev-section and/or --dev-range selectors.

Why dev = Part VI + one table-rich chapter: Part VI pages carry no tables
(found 2026-10-19), so a Part-VI-only dev set cannot exercise the
table/aggregation content_tags -- the most discriminating axis of the
stratification grid (table questions are where parsers diverge, per the
ESPP benchmark). The held-out set is the exact complement of dev;
non-overlap is verified by page_no, not assumed.

Example:
  # 1. pick the table-rich chapter
  python corpus_split.py --pages pages.jsonl --analyze

  # 2. dev = Part VI ("SQL Commands") + the chosen chapter, e.g. "Data Types"
  python corpus_split.py --pages pages.jsonl ^
      --dev-section "SQL Commands" --dev-section "Data Types" ^
      --out-dev pages_dev.jsonl --out-heldout pages_heldout.jsonl

  # 3. generate the two draft sets (separate runs, non-overlapping pages)
  python question_gen.py --pages pages_dev.jsonl     --out draft_questions_dev.yaml     --n 20 --seed 42
  python question_gen.py --pages pages_heldout.jsonl --out draft_questions_heldout.yaml --n 50 --seed 7
"""

import argparse
import json
import re
import sys
from collections import defaultdict

NO_SECTION = "(no section)"


def load_pages(path):
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def section_of(p):
    return p.get("section") or NO_SECTION


def analyze(pages):
    tables = defaultdict(int)
    npages = defaultdict(int)
    for p in pages:
        s = section_of(p)
        tables[s] += p.get("table_count", 0)
        npages[s] += 1
    total_tables = sum(tables.values())
    print(f"{len(pages)} pages, {total_tables} tables total\n")
    print(f"{'tables':>6} / {'pages':>5}   section")
    for s, t in sorted(tables.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{t:6d} / {npages[s]:5d}   {s[:70]}")
    zero = [s for s, t in tables.items() if t == 0]
    if zero:
        print(f"\n{len(zero)} sections with zero tables "
              f"(incl. {[s[:40] for s in zero[:5]]}{'...' if len(zero) > 5 else ''})")


def parse_range(spec):
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"bad range {spec!r}; expected START-END like 2714-2900")
    a, b = int(m.group(1)), int(m.group(2))
    if a > b:
        raise ValueError(f"bad range {spec!r}: start > end")
    return a, b


def compact_ranges(sorted_nos):
    """[1,2,3,7,8] -> '1-3, 7-8'."""
    if not sorted_nos:
        return "-"
    out, start, prev = [], sorted_nos[0], sorted_nos[0]
    for n in sorted_nos[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = n
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def select_dev(pages, dev_sections, dev_ranges):
    matched_sections = set()
    ranges = [parse_range(r) for r in dev_ranges]
    dev = []
    for p in pages:
        s = section_of(p)
        hit_section = any(sub.lower() in s.lower() for sub in dev_sections)
        hit_range = any(a <= p["page_no"] <= b for a, b in ranges)
        if hit_section or hit_range:
            dev.append(p)
            if hit_section:
                matched_sections.add(s)
    return dev, matched_sections


def main(argv=None):
    ap = argparse.ArgumentParser(description="split pages.jsonl into dev + held-out")
    ap.add_argument("--pages", required=True, help="pages.jsonl from parse.py")
    ap.add_argument("--analyze", action="store_true",
                    help="print sections ranked by table count and exit")
    ap.add_argument("--dev-section", action="append", default=[],
                    help="case-insensitive substring matched against the page's "
                         "section; repeatable (e.g. --dev-section 'SQL Commands')")
    ap.add_argument("--dev-range", action="append", default=[],
                    help="inclusive page_no range START-END from PDF bookmarks; "
                         "repeatable (e.g. --dev-range 2714-2900)")
    ap.add_argument("--out-dev", default="pages_dev.jsonl")
    ap.add_argument("--out-heldout", default="pages_heldout.jsonl")
    args = ap.parse_args(argv)

    pages = load_pages(args.pages)

    if args.analyze:
        analyze(pages)
        return 0

    if not args.dev_section and not args.dev_range:
        print("error: need at least one --dev-section or --dev-range "
              "(or run --analyze first to pick the table-rich chapter)",
              file=sys.stderr)
        return 2

    dev, matched_sections = select_dev(pages, args.dev_section, args.dev_range)
    if not dev:
        print("error: selectors matched zero pages -- check section names / ranges",
              file=sys.stderr)
        return 2

    dev_nos = {p["page_no"] for p in dev}
    heldout = [p for p in pages if p["page_no"] not in dev_nos]
    heldout_nos = {p["page_no"] for p in heldout}
    assert len(dev) + len(heldout) == len(pages), "split lost pages!"
    overlap = dev_nos & heldout_nos
    assert not overlap, f"dev/held-out overlap on pages {sorted(overlap)}"

    for path, subset in ((args.out_dev, dev), (args.out_heldout, heldout)):
        with open(path, "w", encoding="utf-8") as f:
            for p in subset:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    dev_tables = sum(1 for p in dev if p.get("table_count", 0) > 0)
    held_tables = sum(1 for p in heldout if p.get("table_count", 0) > 0)

    print(f"dev:     {len(dev):5d} pages  ({dev_tables} with tables)  "
          f"pages {compact_ranges(sorted(dev_nos))}")
    print(f"heldout: {len(heldout):5d} pages  ({held_tables} with tables)")
    print(f"overlap: {len(overlap)} (verified by page_no)")
    if matched_sections:
        print("dev sections matched:")
        for s in sorted(matched_sections):
            print(f"  - {s[:80]}")

    # Stratification feasibility: per 20-question batch each content_tag
    # appears >= 2x, so the dev set needs a real table-page pool.
    if dev_tables < 4:
        print(f"WARNING: only {dev_tables} table pages in dev -- too thin for "
              f"the table/aggregation quotas. Add a more table-rich chapter.")
    if held_tables < 10:
        print(f"WARNING: only {held_tables} table pages in held-out -- the "
              f"50-question set needs table coverage too.")
    print("\nwrote:", args.out_dev, "and", args.out_heldout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
