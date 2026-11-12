"""Agent evaluation harness for RagBenchmarkPlatform.

Scores a black-box agent (first: the Microsoft Foundry knowledge agent
built on OWASP ASVS 4.0.3) against a golden set, using the same judges
and metrics as the platform's own retrieval eval so the numbers are
comparable when the platform itself is tested through this harness.

Layout:
  adapters/   one `ask(question) -> AgentResponse` implementation per
              system under test (Foundry now, the platform later)
  judges/     faithfulness (LLM-as-judge v2) and answer relevance
  context.py  golden-page context extraction from the source PDF
  eval_agent.py  the runner (CLI)

The evaluator never talks to an agent directly -- it talks to an
adapter. Swapping the system under test means swapping the adapter,
not the scoring.
"""

__version__ = "0.1.0"
