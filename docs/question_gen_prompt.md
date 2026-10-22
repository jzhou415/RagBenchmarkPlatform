# Stratified question-generation prompt — RagBenchmarkPlatform backend

Human-readable spec for the prompt inline in `src/question_gen.py`
(implements the "generating golden-set questions" stage of the
Data Setup pipeline). Drop-in prompt for the backend to call once per sampled
source page (or per sampled page pair for multi-hop), enforcing the
query_type × content_tags stratification so the golden set has teeth for the
BM25-vs-semantic comparison.

## Why quotas exist

Unconstrained LLM question generation skews lookup-style ("What is X?"), which
systematically flatters dense retrieval and starves the exact_term and
multi_hop cells of the stratification grid. The quotas below keep every cell
populated, so the Phase 1 report can show *when* each retrieval technique wins.

## Quota table (per generation batch)

| query_type | share | intent |
|---|---|---|
| exact_term | 40% | Question reuses a rare exact term from the page (function name, error code, config parameter, SQL keyword). BM25 should win these. |
| paraphrase | 40% | Same fact asked with synonyms / natural phrasing, term NOT copied verbatim. Dense should win these. |
| multi_hop | 20% | Answer needs facts from 2+ chunks (sample a page pair, or one page + a referenced section). Graded relevance needs both chunks at grade 3. |

## content_tags coverage (orthogonal to query_type — enforce per batch)

- `lookup`: the answer is a plain fact in prose
- `table`: the answer sits inside a table (row/column lookup)
- `policy`: the answer is a rule or behavior description ("what happens when…")
- `aggregation`: the answer combines values across rows or sections ("which N have…", "total of…")

Aim for every content_tag to appear at least twice per 20-question batch. The
backend should re-prompt for the under-filled cells rather than accept a batch
missing them.

## Prompt template (fill {{PAGE_TEXT}} / {{PAGE_REF}} per call)

```
You are building a retrieval-evaluation benchmark for a document RAG system.
Given the source page text below, write ONE question of the assigned type,
following the rules for that type.

Source page ({{PAGE_REF}}):
{{PAGE_TEXT}}

Assigned query_type: {{QUERY_TYPE}}
Allowed content_tags for this question: {{TAGS}}

Rules:
1. The question must be answerable ONLY from this page's text (for multi_hop,
   from the two pages given — and from BOTH of them, not just one).
2. exact_term: reuse one rare exact term from the text verbatim
   (a function name, error code, parameter name, or quoted keyword).
3. paraphrase: restate the fact WITHOUT reusing its distinctive terms —
   a reader should not be able to match the question to the page by keyword.
4. multi_hop — follow the multi-hop procedure exactly:
   a. Read BOTH pages. Pick ONE specific fact from page 1 (fact_A) and ONE
      specific fact from page 2 (fact_B); the two facts must be about related
      subjects so that one question can genuinely need both.
   b. Write ONE question that CANNOT be answered from either page alone — the
      reader must combine fact_A and fact_B. If the question is answerable
      using only one page, it is WRONG: pick different facts and try again.
   c. In expected_answer, put fact_A on line 1 and fact_B on line 2, each
      copied VERBATIM from its page (no rewording, no compression).
   d. Attribute each fact to its page with fact_pages, in the same order as
      the expected_answer lines.
   (The verbatim grade-3 shortcut in auto_map.py only fires when the whole
   expected_answer string appears in one chunk, so it will not fire here; the
   LLM judge grades these, and every multi_hop question gets full human
   review anyway.)
5. For exact_term and paraphrase: write the expected_answer as it appears on
   the page — copy the value verbatim. If the answer sits in a table, tag
   content_tags: [table] and give the cell value plus its row/column labels.
6. Do NOT ask about page numbers, figure captions, or formatting artifacts.
7. Output YAML only, matching this shape:

- question: "..."
  query_type: {{QUERY_TYPE}}
  content_tags: [{{TAGS}}]
  expected_answer: "..."   # verbatim from the page(s); used for grade-3 pre-marking
  fact_pages: [{{PAGE_REF}}, {{PAGE_REF_2}}]   # multi_hop only: page per expected_answer line, same order
  provenance: auto

Do not add commentary.

## Multi-hop hardening (2026-10-21)

First multi_hop batches showed the model writing single-hop questions while
attributing both pages (spurious source_pages — auto_map graded the extra
page's chunks 0). Three backend-side fixes, all in question_gen.py:

- **Overlap pairing**: page2 is picked by vocabulary overlap with page 1, not
  uniform random. Random pairs (pg_checksums + CALL) give the model nothing
  to combine, so it cheats.
- **Per-page fact extraction**: the prompt forces fact_A/fact_B identification
  before writing the question, with a self-test ("answerable from one page
  alone = wrong").
- **Automated gate**: check_multihop() rejects any multi_hop question whose
  expected_answer lacks one verbatim fact line per page (up to 3 attempts,
  then skip). The integrity check "multi_hop source page with zero graded
  chunks" doubles as a detector for this failure mode.
```

## Backend wiring notes

- `expected_answer` verbatim → auto_map.py pre-marks grade-3 candidates, so a
  sloppy or paraphrased expected_answer silently degrades mapping quality. The
  validator does not check verbatimness; spot-check it in the review queue.
- `provenance: auto` is the default for everything this prompt produces; the
  human review pass upgrades entries to `spot-checked` / `human-verified`.
- Dev/held-out split happens at generation time: generate the dev batch from
  the dev subset (SQL reference chapter) and the held-out batch from the full
  manual separately, enforcing the same quotas in each — never random-split
  after generation, or the dev set will not predict held-out.
- Generation itself costs an LLM call per question; prefer a cheap model here
  (gpt-4o-mini class) and spend the stronger judge on the auto-mapping grades
  for the held-out set.
