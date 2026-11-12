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
   hangs on a prompt.
2. **Faithfulness** — LLM-as-judge v2 (the ESPP benchmark judge:
   claim decomposition, verbatim support quotes, number-label rule,
   abstention rule), gpt-4o-mini at temp 0, one call. The judge reads
   the golden pages for that question (relevance >= 2, 30k-char
   budget), extracted from the source PDF.
3. **Answer relevance** — LLM-as-judge against the golden
   `expected_answer` (RELEVANT / PARTIAL / IRRELEVANT + reason).
4. **Citation retrieval relevance** (best-effort) — when citations
   carry quotable spans, the spans are matched to PDF pages and scored
   against the golden pages with `metrics.evaluate_question`
   (hit@k / recall@k / nDCG@k, page level). Foundry citations are
   file-level, so for a single-PDF knowledge base they may carry no
   page-discriminating info; then this reports None with a note, and
   the answer-relevance verdict is the relevance measure.

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

# 3. Full dev run: 20 questions, ~40 judge calls.
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

- The citation shape is unverified until the probe runs. If the live
  citations carry no quotable spans, citation retrieval relevance
  reports None; faithfulness and answer relevance are unaffected.
- The judge itself can err. Every verdict keeps its claim-level
  breakdown (faithfulness) or reason (relevance) in the results JSON,
  so verdicts are auditable per question.
- The faithfulness judge reads golden pages, not the agent's retrieved
  chunks: it measures grounding in the true source material. A
  faithful-to-the-wrong-source case shows up as faithful here but
  irrelevant in the relevance verdict — read the two numbers together.
