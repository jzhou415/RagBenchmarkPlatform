#!/usr/bin/env python3
"""
question_gen.py -- stratified draft-question generation (step 3).

RagBenchmarkPlatform, Phase 1 (retrieval evaluation).

Runs on a local machine (needs OPENAI_API_KEY in the environment).

Implements step 3 of docs/implementation-plan.md: generate draft golden-set
questions from parsed pages, stratified so the benchmark has teeth for the
BM25-vs-semantic comparison:

  query_type quotas (per batch): 40% exact_term / 40% paraphrase / 20% multi_hop
  content_tags coverage (orthogonal): lookup / table / policy / aggregation,
      each appearing at least twice per 20-question batch.

One LLM call per question (gpt-4o-mini by default), using the stratified
prompt from question_gen_prompt.md. expected_answer is copied verbatim from
the page -- auto_map.py's grade-3 pre-marking only fires on verbatim matches,
so a sloppy expected_answer silently degrades mapping quality.

Reads:
  --pages      pages.jsonl from parse.py: {"page_no","text","char_count",
               "table_count","section"} per line
Writes:
  --out        draft questions YAML: list of
               {id, question, query_type, content_tags, expected_answer,
                provenance: auto, source_pages: [...]}
               (chunk ids come later -- auto_map.py maps these to chunks)

Dev and held-out sets are generated in separate runs (same quotas).
"""

import argparse
import json
import random
import sys

# ---------------------------------------------------------------------------
# Stratified prompt (from question_gen_prompt.md -- kept inline so the script
# is self-contained; the .md file remains the human-readable spec).
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are building a retrieval-evaluation benchmark for a document RAG system.
Given the source page text below, write ONE question of the assigned type,
following the rules for that type.

Source page (page {page_ref}):
{page_text}
{page2_block}
Assigned query_type: {query_type}
Required content_tags for this question: {tags}

Rules:
1. The question must be answerable ONLY from this page's text (for multi_hop,
   from the two pages given).
2. exact_term: reuse one rare exact term from the text verbatim
   (a function name, error code, parameter name, or quoted keyword).
3. paraphrase: restate the fact WITHOUT reusing its distinctive terms --
   a reader should not be able to match the question to the page by keyword.
4. multi_hop: the answer must combine two facts from different parts of the
   page(s). Put the two supporting facts in expected_answer, one per line --
   do not compress them into a single value.
5. For exact_term and paraphrase: write the expected_answer as it appears on
   the page -- copy the value verbatim. If the answer sits in a table, use
   content_tags: [table] and give the cell value plus its row/column labels.
6. Do NOT ask about page numbers, figure captions, or formatting artifacts.
7. Output YAML only, matching this shape:

- question: "..."
  query_type: {query_type}
  content_tags: [{tags}]
  expected_answer: "..."   # verbatim from the page; used for grade-3 pre-marking
  provenance: auto

Do not add commentary.
"""

# ---------------------------------------------------------------------------
# LLM plumbing (lazy import -- the script must import cleanly without it)
# ---------------------------------------------------------------------------

def _require_openai():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed: pip install openai "
                 "(OPENAI_API_KEY must be in the environment)")
    return OpenAI()


def _require_yaml():
    try:
        import yaml
    except ImportError:
        sys.exit("pyyaml not installed: pip install pyyaml")
    return yaml


# ---------------------------------------------------------------------------
# Planning: quota-compliant slot list before any LLM call
# ---------------------------------------------------------------------------

QUERY_TYPES = ("exact_term", "paraphrase", "multi_hop")
CONTENT_TAGS = ("lookup", "table", "policy", "aggregation")


def build_plan(n, seed):
    """Return [(query_type, content_tags), ...] honoring 40/40/20 quotas and
    tag coverage (>=2 appearances per tag per 20 questions)."""
    rng = random.Random(seed)
    n_exact = round(0.4 * n)
    n_para = round(0.4 * n)
    n_multi = n - n_exact - n_para
    slots = (["exact_term"] * n_exact + ["paraphrase"] * n_para
             + ["multi_hop"] * n_multi)
    rng.shuffle(slots)
    # Round-robin tags over the shuffled slots guarantees coverage.
    tags = [CONTENT_TAGS[i % len(CONTENT_TAGS)] for i in range(n)]
    rng.shuffle(tags)
    return list(zip(slots, tags))


def load_pages(path, min_chars=500):
    pages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p.get("char_count", len(p.get("text", ""))) >= min_chars:
                pages.append(p)
    if not pages:
        sys.exit(f"no usable pages in {path} (min_chars={min_chars})")
    return pages


# ---------------------------------------------------------------------------
# Generation: one LLM call per planned slot
# ---------------------------------------------------------------------------

MAX_PAGE_CHARS = 8000


def _clip(text):
    return text[:MAX_PAGE_CHARS]


def gen_one(client, model, yaml_mod, page, page2, query_type, tag):
    page2_block = ""
    if page2 is not None:
        page2_block = (f"\nSecond page (page {page2['page_no']}):\n"
                       f"{_clip(page2['text'])}\n")
    prompt = PROMPT_TEMPLATE.format(
        page_ref=page["page_no"],
        page_text=_clip(page["text"]),
        page2_block=page2_block,
        query_type=query_type,
        tags=tag,
    )
    last_err = None
    for _ in range(2):  # one retry on malformed YAML
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            data = yaml_mod.safe_load(resp.choices[0].message.content)
            item = data[0] if isinstance(data, list) else data
            return {
                "question": item["question"].strip(),
                "query_type": query_type,  # trust the plan, not the model
                "content_tags": [tag],
                "expected_answer": str(item["expected_answer"]).strip(),
                "provenance": "auto",
                "source_pages": ([page["page_no"], page2["page_no"]]
                                 if page2 else [page["page_no"]]),
            }
        except Exception as e:  # noqa: BLE001 -- retry then skip
            last_err = e
    print(f"  !! skipped {query_type}/{tag} on p{page['page_no']}: {last_err}",
          file=sys.stderr)
    return None


def print_coverage(questions):
    print("\ncoverage (query_type x content_tags):")
    header = " " * 12 + "".join(f"{t:>12}" for t in CONTENT_TAGS)
    print(header)
    for qt in QUERY_TYPES:
        row = f"{qt:>12}"
        for tag in CONTENT_TAGS:
            c = sum(1 for q in questions
                    if q["query_type"] == qt and tag in q["content_tags"])
            row += f"{c:>12}"
        print(row)
    print(f"total: {len(questions)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pages", required=True, help="pages.jsonl from parse.py")
    ap.add_argument("--out", required=True, help="output draft_questions.yaml")
    ap.add_argument("--n", type=int, default=20, help="questions to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--min-chars", type=int, default=500,
                    help="skip pages shorter than this")
    args = ap.parse_args(argv)

    yaml_mod = _require_yaml()
    client = _require_openai()
    pages = load_pages(args.pages, args.min_chars)
    rng = random.Random(args.seed)
    table_pages = [p for p in pages if p.get("table_count", 0) > 0]

    plan = build_plan(args.n, args.seed)
    questions = []
    for i, (query_type, tag) in enumerate(plan, 1):
        pool = table_pages if (tag == "table" and table_pages) else pages
        page = rng.choice(pool)
        page2 = rng.choice(pages) if query_type == "multi_hop" else None
        if page2 is not None and page2["page_no"] == page["page_no"]:
            page2 = rng.choice(pages)  # avoid pairing a page with itself
        print(f"[{i}/{args.n}] {query_type:11} {tag:11} "
              f"p{page['page_no']}" + (f"+p{page2['page_no']}" if page2 else ""))
        q = gen_one(client, args.model, yaml_mod, page, page2,
                    query_type, tag)
        if q is not None:
            q["id"] = f"q{i:03d}"
            questions.append(q)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Draft golden-set questions -- generated by question_gen.py\n"
                f"# model={args.model} seed={args.seed} "
                f"pages={args.pages}\n")
        yaml_mod.safe_dump(questions, f, allow_unicode=True, sort_keys=False)

    print(f"\nwrote {len(questions)}/{args.n} questions -> {args.out}")
    print_coverage(questions)


if __name__ == "__main__":
    main()
