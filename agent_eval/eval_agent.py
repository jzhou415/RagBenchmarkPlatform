"""Runner: score an agent against a golden set.

Usage (on the PC that holds the Azure login, the .env, and the PDF):

    # 1. Probe first: one question, dump the raw Foundry response so the
    #    citation shape can be inspected before the full run.
    python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf --probe

    # 2. Smoke test: first 5 questions.
    python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf --limit 5

    # 3. Full dev run (20 questions, ~40 gpt-4o-mini judge calls).
    python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf

    # No Azure / no key: fake adapter + stub-free run (judges skipped).
    python -m agent_eval.eval_agent --pdf OWASP-ASVS-4.0.3-en.pdf \\
        --adapter fake --skip-judges --out /tmp/smoke.json

Output is the platform's result shape -- run / per_question / mean /
by_query_type / by_content_tag / latency -- so the Results page can
compare this run against the platform's own eval runs. Per-question
errors (agent failures, judge failures) are recorded, never fatal:
one bad question does not kill the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from src.metrics import latency_stats as _latency_stats
except ImportError:  # flat local working copy (~/workspace/user/rag/)
    from metrics import latency_stats as _latency_stats

from agent_eval.adapters.base import AgentResponse  # noqa: F401  (typing)
from agent_eval.adapters.fake import FakeAdapter
from agent_eval.adapters.foundry_agent import build_foundry_adapter
from agent_eval.context import (
    context_sanity,
    golden_page_context,
    load_pdf_pages,
)
from agent_eval.judges.faithfulness import judge_faithfulness
from agent_eval.judges.relevance import (
    citation_page_metrics,
    judge_answer_relevance,
)

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "data" / "golden_dev.yaml"


def _load_golden(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        golden = yaml.safe_load(f) or {}
    questions = golden.get("questions") or []
    if not questions:
        raise ValueError(f"no questions in {path}")
    for q in questions:
        for key in ("id", "question", "ground_truth"):
            if key not in q:
                raise ValueError(
                    f"question {q.get('id', '?')} in {path} "
                    f"is missing {key!r}")
    return golden


def _aggregate(per_question: list[dict]) -> dict:
    """Mean faithfulness / relevance overall and per stratum.

    Each rate divides by the number of JUDGED questions in that group,
    not by the group size -- an errored or unjudged question must not
    dilute the rate. A group with nothing judged reports no rate at
    all (absent key), not 0.0, which would read as a measured zero.
    """

    def rates(qs):
        if not qs:
            return {}
        out: dict = {"n": len(qs)}
        judged_f = [q for q in qs
                    if q["faithfulness"].get("faithful") is not None]
        if judged_f:
            out["faithfulness"] = sum(
                1 for q in judged_f if q["faithfulness"]["faithful"]
            ) / len(judged_f)
            out["faithfulness_n"] = len(judged_f)
        judged_r = [q for q in qs
                    if q["relevance"].get("verdict") is not None]
        if judged_r:
            out["relevance"] = sum(
                1 for q in judged_r
                if q["relevance"]["verdict"] == "RELEVANT"
            ) / len(judged_r)
            out["partial"] = sum(
                1 for q in judged_r
                if q["relevance"]["verdict"] == "PARTIAL"
            ) / len(judged_r)
            out["relevance_n"] = len(judged_r)
        return out

    by_type: dict[str, list] = {}
    for q in per_question:
        by_type.setdefault(q["query_type"], []).append(q)
    by_tag: dict[str, list] = {}
    for q in per_question:
        for tag in q.get("content_tags", []):
            by_tag.setdefault(tag, []).append(q)
    return {
        "mean": rates(per_question),
        "by_query_type": {k: rates(v) for k, v in by_type.items()},
        "by_content_tag": {k: rates(v) for k, v in by_tag.items()},
    }


def run_eval(golden_path: Path, pdf_path: Path, adapter,
             llm_fn=None, limit: int | None = None,
             min_grade: int = 2, char_budget: int = 30_000,
             skip_judges: bool = False) -> dict:
    """Score every question (or the first `limit`) and return results.

    `llm_fn` is the shared judge call (prompt -> text); None means the
    default gpt-4o-mini call. Tests pass a stub. Judge failures are
    recorded per question, never raised.
    """
    golden = _load_golden(Path(golden_path))
    pages = load_pdf_pages(pdf_path)
    questions = golden["questions"][:limit] if limit else golden["questions"]
    started = datetime.now(timezone.utc).isoformat()
    per_question: list[dict] = []
    latencies: list[float] = []

    for q in questions:
        qid = q["id"]
        print(f"[{qid}] {q['question'][:80]}...")
        if hasattr(adapter, "set_question_id"):
            adapter.set_question_id(qid)
        try:
            resp = adapter.ask(q["question"])
        except Exception as e:  # noqa: BLE001 -- recorded, run continues
            resp = AgentResponse(error=f"adapter raised: {e}")
        latencies.append(resp.latency_s)
        rec: dict = {
            "id": qid,
            "question": q["question"],
            "query_type": q.get("query_type", ""),
            "content_tags": q.get("content_tags", []),
            "answer": resp.answer,
            "citations": [{"file_id": c.file_id, "filename": c.filename,
                           "quote": c.quote} for c in resp.citations],
            "latency_s": round(resp.latency_s, 3),
            "error": resp.error,
            "faithfulness": {"faithful": None, "method": "not-run",
                             "claims": [], "raw": ""},
            "relevance": {"verdict": None, "reason": "",
                          "method": "not-run"},
            "page_metrics": None,
        }
        if resp.error:
            print(f"  error: {resp.error}")
            per_question.append(rec)
            continue
        context = golden_page_context(q, pages, min_grade, char_budget)
        if not context_sanity(q, context):
            print("  warning: expected answer not found in golden pages "
                  "-- page mapping or expected answer may be off")
            rec["context_warning"] = True
        # Deterministic, no LLM call: computed even when judges are skipped.
        try:
            rec["page_metrics"] = citation_page_metrics(
                resp.citations, q, pages)
        except Exception as e:  # noqa: BLE001
            rec["page_metrics"] = {"metrics": None,
                                   "note": f"mapping failed: {e}"}
        if skip_judges:
            per_question.append(rec)
            continue
        try:
            rec["faithfulness"] = judge_faithfulness(
                q["question"], resp.answer, context, llm_fn=llm_fn)
        except Exception as e:  # noqa: BLE001 -- judge failure, not fatal
            rec["faithfulness"] = {"faithful": None, "method": "judge-error",
                                   "claims": [], "raw": "", "error": str(e)}
        try:
            rec["relevance"] = judge_answer_relevance(
                q["question"], resp.answer,
                q.get("expected_answer", ""), llm_fn=llm_fn)
        except Exception as e:  # noqa: BLE001
            rec["relevance"] = {"verdict": None, "reason": "",
                                "method": "judge-error", "error": str(e)}
        f = rec["faithfulness"].get("faithful")
        r = rec["relevance"].get("verdict")
        print(f"  faithful={f} relevance={r} "
              f"latency={resp.latency_s:.2f}s "
              f"citations={len(resp.citations)}")
        per_question.append(rec)

    agg = _aggregate(per_question)
    results = {
        "run": {
            "adapter": getattr(adapter, "name", "?"),
            "golden": str(golden_path),
            "corpus": golden.get("corpus", {}).get("name", "?"),
            "pdf": str(pdf_path),
            "n": len(per_question),
            "judge_model": "gpt-4o-mini" if not skip_judges else None,
            "min_grade": min_grade,
            "started_at": started,
        },
        "per_question": per_question,
        "mean": agg["mean"],
        "by_query_type": agg["by_query_type"],
        "by_content_tag": agg["by_content_tag"],
        "latency": _latency_stats(latencies),
    }
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ap.add_argument("--pdf", type=Path, required=True,
                    help="source PDF (OWASP-ASVS-4.0.3-en.pdf)")
    ap.add_argument("--adapter", choices=["foundry", "fake"],
                    default="foundry")
    ap.add_argument("--fake-answers", type=Path, default=None,
                    help="JSON {question_id: answer} for --adapter fake")
    ap.add_argument("--fake-citations", type=Path, default=None,
                    help="JSON {question_id: [{file_id, filename, quote}]} "
                         "for --adapter fake")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="one question + dump the raw Foundry response")
    ap.add_argument("--out", type=Path, default=None,
                    help="results JSON (default: <stem>_results.json next "
                         "to the golden file)")
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--min-grade", type=int, default=2)
    ap.add_argument("--char-budget", type=int, default=30_000)
    ap.add_argument("--skip-judges", action="store_true",
                    help="record answers/citations only, no LLM judge calls")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv  # noqa: F401  (side effect: .env)
        load_dotenv()
    except ImportError:
        pass

    out = args.out or (args.golden.parent /
                       f"{args.golden.stem}_results.json")
    raw_dir = out.parent / "raw" if args.probe else None

    if args.adapter == "fake":
        citations = None
        if args.fake_citations is not None:
            with open(args.fake_citations, encoding="utf-8") as f:
                citations = json.load(f)
        adapter = FakeAdapter(answers_path=args.fake_answers,
                              citations=citations)
    else:
        try:
            adapter = build_foundry_adapter(raw_dir=raw_dir)
        except Exception as e:  # noqa: BLE001 -- fatal: no agent, no run
            print(f"cannot build Foundry adapter: {e}", file=sys.stderr)
            return 2

    limit = 1 if args.probe else args.limit
    try:
        results = run_eval(args.golden, args.pdf, adapter, limit=limit,
                           min_grade=args.min_grade,
                           char_budget=args.char_budget,
                           skip_judges=args.skip_judges)
    except (ValueError, FileNotFoundError) as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 2

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out}")
    print(f"mean: {json.dumps(results['mean'])}")
    print(f"latency: {json.dumps(results['latency'])}")
    if args.probe and raw_dir is not None:
        print("probe raw response dumped to "
              f"{raw_dir} -- inspect it for the citation shape "
              "(file_citation annotations, quotes?) before the full run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
