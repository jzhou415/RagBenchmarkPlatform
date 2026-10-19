# Implementation plan

We build in pipeline order. Each step is drafted, reviewed, committed, then
run on a local machine (API keys stay in `.env`, never in the repo).

## Phase 0 — Skeleton (this commit)

- [x] Repo skeleton: README, .gitignore, .env.example, requirements.txt,
      `src/`, `ui/`, `golden/`, `corpora/`, `data/`, `runs/`, `docs/`

## Phase 1 — Data Setup backend

1. `src/config.py` — ingestion/eval config dataclasses + sha256 hashing
   (config hash, corpus hash). Golden-set ground truth is pinned to the
   config hash.
2. `src/parse.py` — Docling: PDF → pages with metadata.
3. `src/question_gen.py` — gpt-4o-mini drafts questions with 40/40/20
   exact_term/paraphrase/multi_hop quotas and content_tags coverage
   (prompt: `src/question_gen_prompt.md`).
4. `src/chunk.py` — LangChain splitter (1000/200) → `manifest.jsonl`
   (chunk_id, page, text).
5. `src/auto_map.py` — per-question union of top-20 BM25 + top-20 dense,
   verbatim answer pre-marked grade 3, gpt-4o-mini grades 0–3 →
   golden YAML (schema v1.1) + review queue.
6. `src/validate_golden.py` — schema checks; fails the run before indexing.
7. `src/index_bm25.py` + `src/index_dense.py` — immutable named indexes;
   changed inputs mark an index stale, never silently overwritten.
8. `src/api/main.py` — FastAPI: one endpoint per stage + health check.

## Phase 2 — Retrieval evaluation

9. `src/retrieve.py` — four-rung ladder: semantic, BM25, hybrid (RRF),
   hybrid + rerank. Each rung is an immutable named index + run.
10. `src/rerank.py` — local cross-encoder and/or Cohere reranker.
11. `src/metrics.py` — hit@k, recall@k, MRR, nDCG; per-stage latency
    p50/p95/p99; index-build vs per-query cost split.
12. Eval runner — sweep rerank candidate N (10/20/50) → quality-vs-latency
    Pareto curve, stratified by query type.

## Phase 3 — UI

13. `ui/app.py` — Data Setup page wired to the FastAPI backend, per-stage
    progress, named-index table with stale reasons.
14. Evaluation page + Results page (design TBD).

## Known limitations

- Docling's `MatchingPostProcessor` logs warnings when PDF text cells can't
  be fit into a predicted table row/column grid (e.g. "13 of 35 pdf cells
  matched neither a row nor a column band ... and were dropped from the
  table"). Small drop counts (1–2 cells) are noise; tables with a large
  share of dropped cells may be mis-structured — spot-check those pages
  before sampling them for table-tagged questions.
  (Observed 2026-10-18 on the PostgreSQL 18 manual parse.)

## Working agreement

- Assistant drafts → John reviews → commit (per-write approval) → John runs
  on his PC and pastes output.
- Golden sets stay local until the public demo set is built; the committed
  schema is the contract.
