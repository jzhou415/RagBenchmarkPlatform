# agent_eval — test a black-box agent against a golden set

Scores an agent's answers for **faithfulness** and **answer relevance**
against a golden set, reusing the platform's own judges and metrics so
the numbers are comparable when the platform itself is tested through
this harness. First system under test: the Microsoft Foundry knowledge
agent built on OWASP ASVS 4.0.3.

## How it works

The evaluator never talks to an agent directly. It talks to an
**adapter** — one method, `ask(question) -> AgentResponse` (answer,
citations, latency, error). Swapping the system under test means
swapping the adapter; the scoring stays the same. A platform adapter
(wrapping `rag_answer.py`) is the planned second adapter.

Per question:

1. The adapter answers. A fresh Foundry conversation is created per
   question, so answers cannot leak across questions. An MCP approval
   request fails fast (recorded, not retried) — an unattended run never
   hangs on a prompt. The agent runs gpt-4.1-mini-1 (probe 2026-11-12)
   and retrieves through an MCP tool (`knowledge_base_retrieve`),
   returning ~22 documents per question, each a ~2.3KB snippet of the
   source PDF.
2. **Faithfulness** (primary) — LLM-as-judge v2 (the ESPP benchmark
   judge: claim decomposition, verbatim support quotes, number-label
   rule, abstention rule), gpt-4o-mini at temp 0, one call. The judge
   reads the retrieved snippets for that question (cited docs first,
   30k-char budget). A second call judges the same answer against
   the golden pages (relevance >= 2) as a cross-check — this is the
   `faithfulness_golden` number. Read the two together: a gap means
   the answer is grounded in what it retrieved but not in the true
   source material (or the reverse).
3. **Answer relevance** — LLM-as-judge against the golden
   `expected_answer` (RELEVANT / PARTIAL / IRRELEVANT + reason).
4. **Retrieval relevance** — each retrieved snippet is matched to a
   PDF page (multi-window normalized-substring voting; verified
   against the probe dump: all 22 snippets mapped), and the pages in
   retrieval order are scored against the golden pages with
   `metrics.evaluate_question` (hit@k / recall@k / nDCG@k, page
   level).
5. **Citation relevance** (best-effort, marker-based) — the agent
   cites with `url_citation` markers `【4:N†source】` at the end of
   its answer. Working hypothesis, supported by the q001 probe (the
   cited pages were 3 of the 4 grade->=2 golden pages): the second
   number is the index of the retrieved document. Markers resolve to
   retrieved docs, docs to pages, and the cited pages are scored the
   same way. When citations carry quotable spans instead
   (`file_citation`), quotes are matched to pages directly. Either
   path reports None with a note when nothing maps.

Results are the platform's result shape — `run / per_question / mean /
by_query_type / by_content_tag / latency` — so the Results page can
compare a Foundry run against a platform run side by side.

## Install (on the PC that runs the eval)

```bash
pip install langchain-openai azure-ai-projects azure-identity python-dotenv pymupdf pyyaml
```

`.env` (same file the Foundry client uses):

```text
PROJECT_ENDPOINT=https://<your-project>.services.ai.azure.com/api/projects/<name>
AGENT_NAME=<agent name>
OPENAI_API_KEY=<key for the gpt-4o-mini judge>
```

Azure login: `az login` (the adapter uses `DefaultAzureCredential`
with environment + managed-identity credentials excluded, as the
client does).

## Run

```bash
# 1. Probe first: one question, dump the raw Foundry response to
#    agent_eval/data/raw/ -- inspect it for the citation shape
#    (file_citation annotations? quotes?) before the full run.
python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf --probe

# 2. Smoke test: first 5 questions.
python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf --limit 5

# 3. Full dev run: 20 questions, ~60 judge calls.
python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf

# No Azure / no key: fake adapter, judges skipped (adapter smoke test).
python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf \
    --adapter fake --skip-judges --out /tmp/smoke.json
```

Default output: `agent_eval/data/golden_dev_results.json` next to the
golden file. `--out` overrides it. Per-question errors (agent
failures, judge failures) are recorded in the results, never fatal.

## Data

`agent_eval/data/golden_dev.yaml` — the ASVS dev set, 20 questions,
schema v1.1, `expected_answer` filled on all 20, ground truth graded
0–3 at chunk level with page numbers. Header corrected from the
uploaded copy (`corpus.name` was `postgresql-18-manual`, a copy-paste
leftover; sha256 of the PDF filled in). Provenance is `auto` throughout
— the set is unreviewed, and the numbers inherit that noise. Some
`expected_answer` values are noisy auto artifacts (q010 is a TOC
fragment, q011 is just a section header).

## Known limits

- The citation shape is verified by the probe (2026-11-12,
  response_001.json, one question): `url_citation` markers
  `【4:N†source】` plus MCP retrieval documents with ~2.3KB snippets.
  The marker's second number is read as the retrieved-document index
  — supported by the q001 probe (cited pages were 3 of the 4
  grade->=2 golden pages) but still a hypothesis; confirm on a second
  probe before treating it as settled. The retrieved docs' uid suffix
  `0_pages_N` is a knowledge-base chunk ordinal, NOT a PDF page.
  The raw probe dump is NOT in the repo (it contains the Foundry
  blob-storage URL).
- The judge itself can err. Every verdict keeps its claim-level
  breakdown (faithfulness) or reason (relevance) in the results JSON,
  so verdicts are auditable per question.
- The primary faithfulness judge reads the retrieved snippets, not
  the golden pages: it measures grounding in what the agent was
  given. The golden-pages judge (`faithfulness_golden`) is the
  cross-check against the true source. A faithful-to-the-wrong-source
  case shows up as faithful on the primary number but unfaithful on
  the golden number — read the two together.
- Per-question results now carry `retrieved` (doc ids, page hints,
  mapped pages), `retrieval_metrics`, `citation_metrics`, and
  `faithfulness_golden` alongside `faithfulness`. `page_metrics` is
  kept as an alias of `citation_metrics` for the probe's results
  shape. Note the rename: before this update `faithfulness` meant the
  golden-pages judge; now it means the retrieved-snippets judge, and
  the golden-pages number is `faithfulness_golden`. The probe run's
  single-question record uses the old meaning.
