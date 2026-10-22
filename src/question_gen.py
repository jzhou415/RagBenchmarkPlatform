#!/usr/bin/env python3
"""
question_gen.py -- stratified draft-question generation (step 3).

RagBenchmarkPlatform, Phase 1 (retrieval evaluation).

Runs on the Windows PC (needs OPENAI_API_KEY in the environment).

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
                provenance: auto, source_pages: [...],
                fact_pages: [...] (multi_hop only)}
               (chunk ids come later -- auto_map.py maps these to chunks)

Dev and held-out sets are generated in separate runs (same quotas).

Multi-hop hardening (2026-10-21): the first multi_hop batches showed the
model writing single-hop questions while attributing both pages (spurious
source_pages, caught by auto_map grading the extra page 0). Two fixes:
(1) page2 is paired by vocabulary overlap, not uniform random, so the two
pages are actually combinable; (2) the prompt forces per-page fact
extraction with a fact_pages attribution field, and check_multihop()
rejects any multi_hop question whose expected_answer lacks one verbatim
fact line per page (up to 3 attempts, then skip).
"""

import argparse
import json
import random
import re
import sys

# ---------------------------------------------------------------------------
# Stratified prompt (from question_gen_prompt.md -- kept inline so the script
# is self-contained; the .md file remains the human-readable spec).
# ---------------------------------------------------------------------------

MULTIHOP_BLOCK = """\
Multi-hop procedure (follow exactly):
a. Read BOTH pages. Pick ONE specific fact from page {page_ref} (fact_A) and
   ONE specific fact from page {page2_ref} (fact_B). The two facts must be
   about related subjects so that a single question can genuinely need both.
b. Write ONE question that CANNOT be answered from either page alone -- the
   reader must combine fact_A and fact_B. Test yourself: if your question is
   answerable using only one page, it is WRONG. Pick different facts and try
   again before outputting anything.
c. In expected_answer, put fact_A on line 1 and fact_B on line 2, each copied
   VERBATIM from its page (no rewording, no compression into a single value).
d. Attribute each fact to its page with fact_pages, in the same order as the
   expected_answer lines.
"""

PROMPT_TEMPLATE = """\
You are building a retrieval-evaluation benchmark for a document RAG system.
Given the source page text below, write ONE question of the assigned type,
following the rules for that type.

Source page (page {page_ref}):
{page_text}
{page2_block}
Assigned query_type: {query_type}
Required content_tags for this question: {tags}
{multihop_block}
Rules:
1. The question must be answerable ONLY from the given page's text (for
   multi_hop, from the two pages given -- and from BOTH of them, not one).
2. exact_term: reuse one rare exact term from the text verbatim
   (a function name, error code, parameter name, or quoted keyword).
3. paraphrase: restate the fact WITHOUT reusing its distinctive terms --
   a reader should not be able to match the question to the page by keyword.
4. multi_hop: follow the multi-hop procedure above. The two expected_answer
   lines must each be copied VERBATIM from their page -- no rewording.
5. For exact_term and paraphrase: write the expected_answer as it appears on
   the page -- copy the value verbatim. If the answer sits in a table, use
   content_tags: [table] and give the cell value plus its row/column labels.
6. Do NOT ask about page numbers, figure captions, or formatting artifacts.
7. Output YAML only, matching this shape:

- question: "..."
  query_type: {query_type}
  content_tags: [{tags}]
  expected_answer: "..."   # verbatim from the page(s); used for grade-3 pre-marking
{fact_pages_line}  provenance: auto

Do not add commentary. Do not wrap the output in code fences.
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


def _strip_fences(text):
    """Remove ```yaml code fences. Models add them despite the "YAML only"
    instruction, and yaml.safe_load chokes on the backticks."""
    m = re.search(r"```(?:yaml)?\s*\n(.*?)```", text.strip(), re.DOTALL)
    return m.group(1) if m else text


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
# Multi-hop quality gate + related-page pairing
# ---------------------------------------------------------------------------

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _verbatim_span(line, text, min_len=15):
    """True if the (normalized) line appears verbatim in the page text."""
    nline, ntext = _norm(line), _norm(text)
    return len(nline) >= min_len and nline in ntext


def check_multihop(q, pages_by_no):
    """Automated gate for multi_hop questions.

    Catches the observed failure mode: the model writes a single-hop
    question from one page while attributing both pages (spurious
    source_pages). Requires one verbatim fact line per source page,
    attributed via fact_pages in expected_answer line order.
    Returns (ok, reason). Pure function -- unit-tested.
    """
    if q.get("query_type") != "multi_hop":
        return True, ""
    src = list(q.get("source_pages") or [])
    if len(src) != 2 or src[0] == src[1]:
        return False, f"multi_hop needs 2 distinct source pages, got {src}"
    lines = [l for l in str(q.get("expected_answer") or "").splitlines()
             if _norm(l)]
    fact_pages = list(q.get("fact_pages") or [])
    if len(lines) < 2:
        return False, ("expected_answer must hold one fact line per page "
                       f"(got {len(lines)} non-blank lines)")
    if len(fact_pages) != len(lines):
        return False, (f"fact_pages {fact_pages} must align 1:1 with the "
                       f"{len(lines)} expected_answer lines")
    if set(fact_pages) != set(src):
        return False, (f"fact_pages {fact_pages} must cover "
                       f"source_pages {src}")
    for line, pg in zip(lines, fact_pages):
        if not _verbatim_span(line, pages_by_no.get(pg, "")):
            return False, (f"fact line not found verbatim on p{pg}: "
                            f"{line[:70]!r}...")
    return True, ""


def _tokens(text):
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def pick_related_page(page, pages, rng, top_k=20):
    """Pick a pairing page with high vocabulary overlap.

    Uniform-random multi_hop pairings (e.g. pg_checksums + CALL) give the
    model nothing to combine, so it writes a single-hop question and
    attributes both pages. Overlap pairing keeps the pages combinable.
    Falls back to uniform random when nothing overlaps. Never picks the
    page itself.
    """
    base = _tokens(page["text"])
    scored = []
    for p in pages:
        if p["page_no"] == page["page_no"]:
            continue
        scored.append((len(base & _tokens(p["text"])), p))
    scored.sort(key=lambda t: t[0], reverse=True)
    cands = [p for ov, p in scored[:top_k] if ov > 0]
    if not cands:  # degenerate vocabulary -- fall back to uniform
        cands = [p for p in pages if p["page_no"] != page["page_no"]]
    return rng.choice(cands)


# ---------------------------------------------------------------------------
# Generation: one LLM call per planned slot
# ---------------------------------------------------------------------------

MAX_PAGE_CHARS = 8000


def _clip(text):
    return text[:MAX_PAGE_CHARS]


MULTIHOP_ATTEMPTS = 3


def gen_one(client, model, yaml_mod, page, page2, query_type, tag):
    multihop = query_type == "multi_hop"
    page2_block = ""
    multihop_block = ""
    fact_pages_line = ""
    if page2 is not None:
        page2_block = (f"\nSecond page (page {page2['page_no']}):\n"
                       f"{_clip(page2['text'])}\n")
    if multihop and page2 is not None:
        multihop_block = MULTIHOP_BLOCK.format(page_ref=page["page_no"],
                                               page2_ref=page2["page_no"])
        fact_pages_line = (f"  fact_pages: [{page['page_no']}, "
                           f"{page2['page_no']}]   # multi_hop only: page "
                           f"per expected_answer line, same order\n")
    prompt = PROMPT_TEMPLATE.format(
        page_ref=page["page_no"],
        page_text=_clip(page["text"]),
        page2_block=page2_block,
        query_type=query_type,
        tags=tag,
        multihop_block=multihop_block,
        fact_pages_line=fact_pages_line,
    )
    pages_by_no = {page["page_no"]: page["text"]}
    if page2 is not None:
        pages_by_no[page2["page_no"]] = page2["text"]

    last_err = None
    feedback = ""
    attempts = MULTIHOP_ATTEMPTS if multihop else 2  # one retry on bad YAML
    for _ in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": prompt + feedback}],
                temperature=0.7,
            )
            text = _strip_fences(resp.choices[0].message.content)
            data = yaml_mod.safe_load(text)
            item = data[0] if isinstance(data, list) else data
            q = {
                "question": item["question"].strip(),
                "query_type": query_type,  # trust the plan, not the model
                "content_tags": [tag],
                "expected_answer": str(item["expected_answer"]).strip(),
                "provenance": "auto",
                "source_pages": ([page["page_no"], page2["page_no"]]
                                 if page2 else [page["page_no"]]),
            }
            if multihop:
                q["fact_pages"] = item.get("fact_pages") or []
                ok, reason = check_multihop(q, pages_by_no)
                if not ok:
                    last_err = f"multi_hop check failed: {reason}"
                    feedback = (f"\n\nYour previous output FAILED this "
                                f"automated check: {reason}\nFix it and "
                                f"output the corrected YAML only.")
                    continue
            return q
        except Exception as e:  # noqa: BLE001 -- retry then skip
            last_err = e
            feedback = ""
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
    ap.add_argument("--query-type", choices=list(QUERY_TYPES), default=None,
                    help="generate only this query type (for regenerating "
                         "one stratum, e.g. multi_hop)")
    ap.add_argument("--id-offset", type=int, default=0,
                    help="start question ids at q{offset+1:03d}")
    args = ap.parse_args(argv)

    yaml_mod = _require_yaml()
    client = _require_openai()
    pages = load_pages(args.pages, args.min_chars)
    rng = random.Random(args.seed)
    table_pages = [p for p in pages if p.get("table_count", 0) > 0]

    plan = build_plan(args.n, args.seed)
    if args.query_type:  # regenerate one stratum only
        tags = [CONTENT_TAGS[i % len(CONTENT_TAGS)] for i in range(args.n)]
        rng2 = random.Random(args.seed + 1)
        rng2.shuffle(tags)
        plan = [(args.query_type, t) for t in tags]
    questions = []
    for i, (query_type, tag) in enumerate(plan, 1):
        pool = table_pages if (tag == "table" and table_pages) else pages
        page = rng.choice(pool)
        page2 = None
        if query_type == "multi_hop":
            # Overlap pairing: uniform-random pairs (pg_checksums + CALL)
            # give the model nothing to combine, so it writes a single-hop
            # question and attributes both pages.
            page2 = pick_related_page(page, pages, rng)
        print(f"[{i}/{args.n}] {query_type:11} {tag:11} "
              f"p{page['page_no']}" + (f"+p{page2['page_no']}" if page2 else ""))
        q = gen_one(client, args.model, yaml_mod, page, page2,
                    query_type, tag)
        if q is not None:
            q["id"] = f"q{i + args.id_offset:03d}"
            questions.append(q)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Draft golden-set questions -- generated by question_gen.py\n"
                f"# model={args.model} seed={args.seed} "
                f"pages={args.pages}"
                + (f" query_type={args.query_type}" if args.query_type else "")
                + "\n")
        yaml_mod.safe_dump(questions, f, allow_unicode=True, sort_keys=False)

    print(f"\nwrote {len(questions)}/{args.n} questions -> {args.out}")
    print_coverage(questions)


if __name__ == "__main__":
    main()
