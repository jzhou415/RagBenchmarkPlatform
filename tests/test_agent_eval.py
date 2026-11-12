"""Tests for agent_eval. No Azure, no OpenAI calls: the adapter is
stubbed and the judge LLM is injected as a stub function.

Run from ~/workspace/user/rag/ (the flat working copy):
    PYTHONPATH=/tmp/shim:~/workspace/user/rag \\
      ~/.testvenv/bin/python -m unittest discover -s . -p "test_*.py"
"""

import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

import pymupdf  # noqa: E402
import yaml  # noqa: E402

from agent_eval.adapters.fake import FakeAdapter  # noqa: E402
from agent_eval.adapters.foundry_agent import (  # noqa: E402
    FoundryAgentAdapter,
    _extract_citations,
    _extract_text,
)
from agent_eval.context import (  # noqa: E402
    context_sanity,
    golden_page_context,
    load_pdf_pages,
)
from agent_eval.eval_agent import _load_golden, run_eval  # noqa: E402
from agent_eval.judges.faithfulness import (  # noqa: E402
    is_abstention,
    judge_faithfulness,
    parse_v2_verdict,
)
from agent_eval.judges.relevance import (  # noqa: E402
    citation_page_metrics,
    judge_answer_relevance,
    parse_relevance_verdict,
    quote_to_page,
)
from agent_eval.adapters.base import Citation  # noqa: E402

FAITHFUL_JSON = json.dumps({
    "claims": [{"claim": "tokens need 64 bits",
                "support_quote": "at least 64 bits of entropy",
                "supported": True}],
    "overall": "FAITHFUL"})
UNFAITHFUL_JSON = json.dumps({
    "claims": [{"claim": "show full stack traces",
                "support_quote": "", "supported": False}],
    "overall": "UNFAITHFUL"})


def stub_judge(prompt):
    """Judge stub: dispatches on the prompt, like the real judges do.

    Faithfulness prompts embed the generated answer; answers carrying
    MARK-BAD are judged UNFAITHFUL / PARTIAL, everything else
    FAITHFUL / RELEVANT.
    """
    if "You audit whether" in prompt:
        return UNFAITHFUL_JSON if "MARK-BAD" in prompt else FAITHFUL_JSON
    if "You judge whether" in prompt:
        if "MARK-BAD" in prompt:
            return json.dumps({"verdict": "PARTIAL",
                               "reason": "omits a material part"})
        return json.dumps({"verdict": "RELEVANT", "reason": "matches"})
    raise AssertionError(f"unexpected judge prompt: {prompt[:60]}")


def make_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


GOLDEN_YAML = textwrap.dedent("""\
    schema_version: "1.1"
    corpus:
      name: test-corpus
      source: pdf
      pages: 3
    questions:
    - id: q1
      split: dev
      provenance: auto
      question: What is the minimum entropy requirement for session tokens?
      query_type: exact_term
      content_tags: [lookup]
      expected_answer: "64 bits"
      ground_truth:
      - {chunk_id: p1_c00, page: 1, relevance: 3}
    - id: q2
      split: dev
      provenance: auto
      question: How must passwords be stored?
      query_type: paraphrase
      content_tags: [policy]
      expected_answer: Passwords must be stored using a secure key store.
      ground_truth:
      - {chunk_id: p2_c00, page: 2, relevance: 3}
    - id: q3
      split: dev
      provenance: auto
      question: What should error messages do?
      query_type: paraphrase
      content_tags: [table]
      expected_answer: Verify that a generic message is shown.
      ground_truth:
      - {chunk_id: p3_c00, page: 3, relevance: 3}
    """)

PAGE_TEXTS = [
    "The session token must have at least 64 bits of entropy. Tokens are random.",
    "Passwords must be stored using a secure key store or hardware module.",
    "Error messages must be generic and never expose sensitive information.",
]


class TestFaithfulnessJudge(unittest.TestCase):
    def test_parse_faithful(self):
        faithful, claims, _ = parse_v2_verdict(FAITHFUL_JSON)
        self.assertTrue(faithful)
        self.assertEqual(len(claims), 1)

    def test_parse_unfaithful(self):
        faithful, claims, _ = parse_v2_verdict(UNFAITHFUL_JSON)
        self.assertFalse(faithful)
        self.assertFalse(claims[0]["supported"])

    def test_parse_garbage_is_conservative(self):
        faithful, claims, raw = parse_v2_verdict("not json at all")
        self.assertFalse(faithful)
        self.assertEqual(claims, [])

    def test_abstention_variants(self):
        for text in ("I don't know", "I don't know.", "I don't know!"):
            self.assertTrue(is_abstention(text), text)
        self.assertFalse(is_abstention("I don't know, but probably 64 bits"))
        self.assertFalse(is_abstention("The answer is 64 bits."))

    def test_judge_with_stub(self):
        calls = []
        res = judge_faithfulness(
            "q", "Session tokens need 64 bits.", "context",
            llm_fn=lambda p: calls.append(p) or FAITHFUL_JSON)
        self.assertTrue(res["faithful"])
        self.assertEqual(res["method"], "judge-v2")
        self.assertEqual(len(calls), 1)

    def test_judge_unfaithful_with_stub(self):
        res = judge_faithfulness(
            "q", "MARK-BAD show stack traces", "context",
            llm_fn=lambda p: UNFAITHFUL_JSON)
        self.assertFalse(res["faithful"])

    def test_abstention_never_calls_llm(self):
        calls = []
        res = judge_faithfulness(
            "q", "I don't know", "context",
            llm_fn=lambda p: calls.append(p) or FAITHFUL_JSON)
        self.assertTrue(res["faithful"])
        self.assertEqual(res["method"], "abstention-rule")
        self.assertEqual(calls, [])

    def test_empty_answer_has_no_verdict(self):
        res = judge_faithfulness("q", "   ", "context",
                                 llm_fn=lambda p: FAITHFUL_JSON)
        self.assertIsNone(res["faithful"])
        self.assertEqual(res["method"], "no-answer")


class TestRelevanceJudge(unittest.TestCase):
    def test_parse_verdicts(self):
        for verdict in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
            v, _ = parse_relevance_verdict(
                json.dumps({"verdict": verdict, "reason": "r"}))
            self.assertEqual(v, verdict)

    def test_judge_with_stub(self):
        res = judge_answer_relevance(
            "q", "Session tokens need 64 bits.", "64 bits",
            llm_fn=stub_judge)
        self.assertEqual(res["verdict"], "RELEVANT")

    def test_abstention_is_irrelevant_not_unfaithful(self):
        res = judge_answer_relevance("q", "I don't know", "64 bits",
                                     llm_fn=stub_judge)
        self.assertEqual(res["verdict"], "IRRELEVANT")
        self.assertEqual(res["method"], "abstention-rule")


class TestContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.tmp.name) / "test.pdf"
        make_pdf(self.pdf, PAGE_TEXTS)
        self.pages = load_pdf_pages(self.pdf)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_pdf_pages_is_1_indexed(self):
        self.assertEqual(len(self.pages), 3)
        self.assertIn("64 bits", self.pages[0])
        self.assertIn("key store", self.pages[1])

    def test_golden_pages_min_grade(self):
        q = {"ground_truth": [
            {"chunk_id": "p1_c00", "page": 1, "relevance": 3},
            {"chunk_id": "p2_c00", "page": 2, "relevance": 1},
            {"chunk_id": "p3_c00", "page": 3, "relevance": 2}]}
        ctx = golden_page_context(q, self.pages, min_grade=2)
        self.assertIn("[page 1]", ctx)
        self.assertIn("[page 3]", ctx)
        self.assertNotIn("[page 2]", ctx)
        self.assertIn("64 bits", ctx)

    def test_char_budget_drops_weakest_first(self):
        q = {"ground_truth": [
            {"chunk_id": "p1_c00", "page": 1, "relevance": 3},
            {"chunk_id": "p3_c00", "page": 3, "relevance": 2}]}
        ctx = golden_page_context(q, self.pages, char_budget=60)
        self.assertIn("[page 1]", ctx)
        self.assertNotIn("[page 3]", ctx)

    def test_context_sanity(self):
        q = {"expected_answer": "Passwords must be stored using a "
                                "secure key store."}
        ctx = "\n".join(self.pages)
        self.assertTrue(context_sanity(q, ctx))
        self.assertFalse(context_sanity(q, "unrelated text here"))


class TestCitationMapping(unittest.TestCase):
    def test_quote_to_page(self):
        self.assertEqual(
            quote_to_page("at least 64 bits of entropy", PAGE_TEXTS), 1)
        self.assertEqual(
            quote_to_page("secure key store or hardware", PAGE_TEXTS), 2)

    def test_quote_to_page_rejects_short_and_missing(self):
        self.assertIsNone(quote_to_page("short", PAGE_TEXTS))
        self.assertIsNone(quote_to_page(
            "this sentence appears nowhere at all", PAGE_TEXTS))

    def test_citation_page_metrics(self):
        q = {"ground_truth": [
            {"chunk_id": "p1_c00", "page": 1, "relevance": 3},
            {"chunk_id": "p2_c00", "page": 2, "relevance": 2}]}
        cites = [Citation(file_id="f1", filename="doc.pdf",
                          quote="at least 64 bits of entropy"),
                 Citation(file_id="f1", filename="doc.pdf",
                          quote="secure key store or hardware")]
        out = citation_page_metrics(cites, q, PAGE_TEXTS)
        self.assertEqual(out["ranked_pages"], ["p1", "p2"])
        self.assertEqual(out["metrics"]["hit@3"], 1.0)
        self.assertGreater(out["metrics"]["ndcg@3"], 0.9)

    def test_citation_page_metrics_none_without_quotes(self):
        q = {"ground_truth": [
            {"chunk_id": "p1_c00", "page": 1, "relevance": 3}]}
        out = citation_page_metrics(
            [Citation(file_id="f1", filename="doc.pdf", quote="")],
            q, PAGE_TEXTS)
        self.assertIsNone(out["metrics"])
        self.assertIn("no citation mapped", out["note"])


class _FakeItems:
    def create(self, conversation_id=None, items=None):
        return None


class _FakeConversations:
    def __init__(self):
        self.items = _FakeItems()

    def create(self, items=None):
        return SimpleNamespace(id="conv-1")


class _FakeResponses:
    def __init__(self, response):
        self._response = response

    def create(self, conversation=None, extra_body=None, input=""):
        return self._response


class _FakeOpenAIClient:
    def __init__(self, response):
        self.conversations = _FakeConversations()
        self.responses = _FakeResponses(response)


class TestFoundryAdapter(unittest.TestCase):
    def _response(self, text="answer text", citations=(), approval=False):
        output = []
        if approval:
            output.append(SimpleNamespace(
                type="mcp_approval_request", name="tool",
                server_label="srv"))
        else:
            annotations = [
                SimpleNamespace(type="file_citation", file_id=c["file_id"],
                                filename=c["filename"], quote=c["quote"])
                for c in citations]
            output.append(SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text,
                                         annotations=annotations)]))
        return SimpleNamespace(output_text=text, output=output)

    def test_extract_text_prefers_output_text(self):
        resp = self._response(text="hello")
        self.assertEqual(_extract_text(resp), "hello")

    def test_extract_text_falls_back_to_items(self):
        resp = self._response(text="hello")
        del resp.output_text
        self.assertEqual(_extract_text(resp), "hello")

    def test_extract_citations(self):
        resp = self._response(citations=[{
            "file_id": "file-1", "filename": "doc.pdf",
            "quote": "some span"}])
        cites = _extract_citations(resp)
        self.assertEqual(len(cites), 1)
        self.assertEqual(cites[0].file_id, "file-1")
        self.assertEqual(cites[0].quote, "some span")

    def test_ask_happy_path(self):
        resp = self._response(
            text="Session tokens need 64 bits.",
            citations=[{"file_id": "file-1", "filename": "doc.pdf",
                        "quote": "at least 64 bits of entropy"}])
        adapter = FoundryAgentAdapter(
            _FakeOpenAIClient(resp), SimpleNamespace(name="agent"))
        out = adapter.ask("What is the minimum entropy?")
        self.assertEqual(out.answer, "Session tokens need 64 bits.")
        self.assertEqual(len(out.citations), 1)
        self.assertIsNone(out.error)
        self.assertGreaterEqual(out.latency_s, 0.0)

    def test_ask_approval_fails_fast(self):
        resp = self._response(approval=True)
        adapter = FoundryAgentAdapter(
            _FakeOpenAIClient(resp), SimpleNamespace(name="agent"))
        out = adapter.ask("question")
        self.assertEqual(out.answer, "")
        self.assertIn("mcp_approval_request", out.error)

    def test_ask_call_failure_recorded(self):
        class _FailingResponses:
            def create(self, **kwargs):
                raise RuntimeError("boom")

        client = _FakeOpenAIClient(None)
        client.responses = _FailingResponses()
        adapter = FoundryAgentAdapter(client, SimpleNamespace(name="agent"))
        out = adapter.ask("question")
        self.assertIn("boom", out.error)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pdf = self.dir / "test.pdf"
        make_pdf(self.pdf, PAGE_TEXTS)
        self.golden = self.dir / "golden.yaml"
        self.golden.write_text(GOLDEN_YAML, encoding="utf-8")
        self.answers = {
            "q1": "Session tokens must have at least 64 bits of entropy.",
            "q2": "I don't know",
            "q3": "MARK-BAD Error messages should include full stack "
                  "traces and database passwords.",
        }
        self.citations = {
            "q1": [{"file_id": "f1", "filename": "doc.pdf",
                    "quote": "at least 64 bits of entropy"}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_full_run_with_fake_adapter(self):
        adapter = FakeAdapter(answers=self.answers,
                              citations=self.citations)
        results = run_eval(self.golden, self.pdf, adapter,
                           llm_fn=stub_judge)
        self.assertEqual(
            set(results),
            {"run", "per_question", "mean", "by_query_type",
             "by_content_tag", "latency"})
        self.assertEqual(results["run"]["n"], 3)
        self.assertEqual(results["run"]["adapter"], "fake")
        per_q = {r["id"]: r for r in results["per_question"]}
        # q1 faithful + relevant; q2 abstention (faithful, irrelevant);
        # q3 unfaithful + partial.
        self.assertTrue(per_q["q1"]["faithfulness"]["faithful"])
        self.assertEqual(per_q["q1"]["relevance"]["verdict"], "RELEVANT")
        self.assertTrue(per_q["q2"]["faithfulness"]["faithful"])
        self.assertEqual(per_q["q2"]["faithfulness"]["method"],
                         "abstention-rule")
        self.assertEqual(per_q["q2"]["relevance"]["verdict"], "IRRELEVANT")
        self.assertFalse(per_q["q3"]["faithfulness"]["faithful"])
        self.assertEqual(per_q["q3"]["relevance"]["verdict"], "PARTIAL")
        # Citation mapping worked end to end for q1.
        self.assertEqual(
            per_q["q1"]["page_metrics"]["metrics"]["hit@3"], 1.0)
        self.assertIsNone(per_q["q2"]["page_metrics"]["metrics"])
        # Means divide by judged counts: faithfulness 2/3, relevance 1/3.
        self.assertAlmostEqual(results["mean"]["faithfulness"], 2 / 3)
        self.assertAlmostEqual(results["mean"]["relevance"], 1 / 3)
        self.assertAlmostEqual(results["mean"]["partial"], 1 / 3)
        self.assertEqual(results["latency"]["n"], 3)

    def test_limit(self):
        adapter = FakeAdapter(answers=self.answers)
        results = run_eval(self.golden, self.pdf, adapter,
                           llm_fn=stub_judge, limit=1)
        self.assertEqual(results["run"]["n"], 1)
        self.assertEqual(results["per_question"][0]["id"], "q1")

    def test_skip_judges_still_maps_citations(self):
        adapter = FakeAdapter(answers=self.answers,
                              citations=self.citations)
        results = run_eval(self.golden, self.pdf, adapter,
                           llm_fn=stub_judge, limit=1, skip_judges=True)
        rec = results["per_question"][0]
        self.assertEqual(rec["faithfulness"]["method"], "not-run")
        self.assertEqual(
            rec["page_metrics"]["metrics"]["hit@3"], 1.0)

    def test_adapter_error_recorded_not_raised(self):
        class _RaisingAdapter:
            name = "raising"

            def ask(self, question):
                raise RuntimeError("kaboom")

        results = run_eval(self.golden, self.pdf, _RaisingAdapter(),
                           llm_fn=stub_judge, limit=1)
        rec = results["per_question"][0]
        self.assertIn("kaboom", rec["error"])
        self.assertIsNone(rec["faithfulness"]["faithful"])
        self.assertEqual(results["mean"], {"n": 1})

    def test_load_golden_rejects_bad_file(self):
        bad = self.dir / "bad.yaml"
        bad.write_text("questions:\n- {id: q1}\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            _load_golden(bad)

    def test_shipped_golden_file(self):
        # Works whether this test lives flat next to agent_eval/ (local
        # working copy) or under tests/ (repo layout): walk up until
        # the golden file is found.
        here = Path(__file__).resolve().parent
        root = next(
            (c for c in (here, here.parent)
             if (c / "agent_eval" / "data" / "golden_dev.yaml").exists()),
            here)
        path = root / "agent_eval" / "data" / "golden_dev.yaml"
        with open(path, encoding="utf-8") as f:
            golden = yaml.safe_load(f)
        self.assertEqual(len(golden["questions"]), 20)
        self.assertEqual(golden["corpus"]["name"], "owasp-asvs-4.0.3")
        self.assertTrue(all(
            (q.get("expected_answer") or "").strip()
            for q in golden["questions"]))


if __name__ == "__main__":
    unittest.main()
