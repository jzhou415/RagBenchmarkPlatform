# RagBenchmarkPlatform

A platform for evaluating RAG performance — built to answer, with measurements
instead of opinions, which retrieval choices actually work.

Phase 1 evaluates the retrieval pipeline: semantic retrieval, BM25 keyword
retrieval, hybrid retrieval, and reranking — measuring result quality alongside
latency and cost.

## Layout

- `src/` — pipeline stages (parse → question_gen → chunk → auto_map → index →
  retrieve → rerank → metrics), plus `src/api/` (FastAPI backend)
- `ui/` — Streamlit frontend (Data setup / Ask / Evaluation / Results)
- `golden/` — golden-set schema and sets (public demo corpus only)
- `corpora/` — source PDFs (gitignored, never committed)
- `data/` — Chroma DBs, named indexes, manifests (gitignored)
- `runs/` — per-run manifests + metrics (gitignored)
- `docs/` — design docs and the implementation plan

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

See `docs/implementation-plan.md` for the build order.

## Privacy rule

This repo is public. Never commit source PDFs, `.env`, `data/`, or `runs/` —
they may carry private document content or API keys. The platform is
corpus-agnostic; only code, schema, and docs are committed.
