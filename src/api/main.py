#!/usr/bin/env python3
"""
api/main.py -- step 8: FastAPI backend for the Data Setup pipeline.

One endpoint per pipeline stage (each runs its src/*.py module as a
background subprocess and reports progress), plus a health check.

Endpoints:
    GET  /health                 -> {"status": "ok", ...}
    GET  /stages                 -> [{id, script, description}]
    POST /stages/{stage_id}/run  -> {"run_id": ...} (202 Accepted)
         body: {"args": {"manifest": "full/manifest_full.jsonl"}}
         (args become CLI flags: {"chunk_manifest": "x"} -> --chunk-manifest x)
    GET  /runs/{run_id}          -> {run_id, stage, status, returncode, ...}
    GET  /runs/{run_id}/log      -> {"log": "..."}  (?tail=N lines)
    POST /experiments            -> register an eval experiment (201)
    GET  /experiments            -> [experiment records, newest first]
    GET  /experiments/{id}       -> one experiment record
    POST /experiments/{id}/run   -> run it (202 Accepted)
    GET  /experiments/{id}/log   -> {"log": "..."}  (?tail=N lines)

Run logs live under runs/<run_id>.log (the repo's runs/ dir). Status is
"running" | "succeeded" | "failed".

Run with:  uvicorn src.api.main:app --reload   (from the repo root)
"""
import datetime
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"

STAGES = {
    "parse": {
        "script": "src/parse.py",
        "description": "Docling: PDF -> pages with metadata",
    },
    "corpus-split": {
        "script": "src/corpus_split.py",
        "description": "Split pages into dev + held-out pools (disjoint by page_no)",
    },
    "chunk": {
        "script": "src/chunk.py",
        "description": "Split pages into per-page chunks -> manifest.jsonl",
    },
    "question-gen": {
        "script": "src/question_gen.py",
        "description": "LLM drafts stratified eval questions",
    },
    "auto-map": {
        "script": "src/auto_map.py",
        "description": "Map questions to graded chunks -> golden YAML",
    },
    "validate": {
        "script": "src/validate_golden.py",
        "description": "Schema checks on the golden set",
    },
    "index-bm25": {
        "script": "src/index_bm25.py",
        "description": "Build the BM25 keyword index",
    },
    "index-dense": {
        "script": "src/index_dense.py",
        "description": "Embed the corpus, store in Chroma",
    },
    "review-report": {
        "script": "src/review_report.py",
        "description": "Automatic review checks on the golden set (advisory)",
    },
}

app = FastAPI(title="RagBenchmarkPlatform API")

_runs: dict = {}
_runs_lock = threading.Lock()


class RunRequest(BaseModel):
    args: dict = {}


def args_to_cli(args: dict) -> list:
    """{"chunk_manifest": "x", "verbose": True} -> [--chunk-manifest, x, --verbose]."""
    cli = []
    for key, value in args.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(flag)
        elif isinstance(value, (list, tuple)):
            cli.append(flag)
            cli.extend(str(v) for v in value)
        elif value is not None:
            cli.extend([flag, str(value)])
    return cli


def stage_command(script_path):
    """How to execute a stage script (argv prefix, before CLI args).

    Scripts inside the repo run as modules (python -m src.parse):
    parse.py uses a relative import (from .config import ...), which
    only works under module execution, and chunk.py already supports
    both modes. Anything else (ad-hoc scripts outside the repo) runs
    as a plain file, as before.
    """
    script = Path(script_path)
    try:
        rel = script.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return [sys.executable, str(script)]
    if rel.suffix != ".py":
        return [sys.executable, str(script)]
    module = ".".join(rel.with_suffix("").parts)
    return [sys.executable, "-m", module]


def _run_stage(run_id: str, stage_id: str, cli_args: list):
    script = ROOT / STAGES[stage_id]["script"]
    cmd = stage_command(script)
    log_path = RUNS_DIR / f"{run_id}.log"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd + cli_args)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd + cli_args,
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                text=True)
        returncode = proc.wait()
        status = "succeeded" if returncode == 0 else "failed"
    except Exception as exc:  # noqa: BLE001 - record and report, never crash the thread silently
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[runner error] {exc}\n")
        returncode, status = -1, "failed"
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _runs_lock:
        _runs[run_id].update(
            {"status": status, "returncode": returncode,
             "finished_at": finished})


@app.get("/health")
def health():
    return {"status": "ok", "stages": sorted(STAGES),
            "root": str(ROOT)}


@app.get("/stages")
def list_stages():
    return [{"id": sid, **info} for sid, info in STAGES.items()]


@app.post("/stages/{stage_id}/run", status_code=202)
def run_stage(stage_id: str, req: RunRequest):
    if stage_id not in STAGES:
        raise HTTPException(status_code=404, detail=f"unknown stage: {stage_id}")
    if not isinstance(req.args, dict):
        raise HTTPException(status_code=422, detail="args must be an object")
    run_id = uuid.uuid4().hex[:12]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _runs_lock:
        _runs[run_id] = {"run_id": run_id, "stage": stage_id,
                         "status": "running", "returncode": None,
                         "started_at": started, "finished_at": None}
    thread = threading.Thread(
        target=_run_stage, args=(run_id, stage_id, args_to_cli(req.args)),
        daemon=True)
    thread.start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}")
def run_status(run_id: str):
    with _runs_lock:
        run = _runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
    return dict(run)


@app.get("/runs/{run_id}/log")
def run_log(run_id: str, tail: int = 200):
    with _runs_lock:
        if run_id not in _runs:
            raise HTTPException(status_code=404,
                                detail=f"unknown run: {run_id}")
    log_path = RUNS_DIR / f"{run_id}.log"
    if not log_path.exists():
        return {"log": ""}
    with open(log_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return {"log": "\n".join(lines[-tail:])}


# ------------------------------------------------------------ Experiments
# Step 14: strategy-specific eval experiments. One registry file,
# RESULTS_DIR/experiments.json, holds every experiment (pending ->
# running -> completed, or failed with its log); one results JSON per
# experiment, named exp_<id>_<split>_<strategy>[_rerank<N>].json, so
# step 15's Results page loads them without a re-run. The registry is
# the source of truth (read from disk on every request), so the table
# survives an API restart. Results live under the repo root, not the
# API process's cwd, so a wrong-folder launch cannot scatter them.

RESULTS_DIR = ROOT / "results"
EXPERIMENT_KS = (3, 5, 10)
EXPERIMENT_STRATEGIES = ("dense", "bm25", "hybrid")
EXPERIMENT_SPLITS = ("dev", "heldout")

_experiments_lock = threading.Lock()


def _resolve(path_str: str) -> Path:
    """Absolute path as-is; repo-relative path against ROOT."""
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def strategy_rungs(strategy: str, rerank: bool, pool_n: int) -> tuple:
    """Experiment strategy -> eval_runner --rungs labels.

    UI "dense" is the runner's "semantic" rung. Hybrid+rerank scores
    the hybrid rung plus its reranked variant (pool N sets both the
    fusion depth and the reranker pool, as in step 12).
    """
    if strategy == "dense":
        return ("semantic",)
    if strategy == "bm25":
        return ("bm25",)
    if strategy == "hybrid":
        if rerank:
            return ("hybrid", f"hybrid_rerank@{pool_n}")
        return ("hybrid",)
    raise ValueError(f"unknown strategy: {strategy}")


def experiment_results_filename(exp_id: str, split: str, strategy: str,
                                rerank: bool, pool_n: int) -> str:
    base = f"exp_{exp_id}_{split}_{strategy}"
    if rerank:
        base += f"_rerank{pool_n}"
    return base + ".json"


def _registry_file() -> Path:
    return RESULTS_DIR / "experiments.json"


def _load_registry() -> dict:
    path = _registry_file()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("experiments", {})


def _save_registry(registry: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_registry_file(), "w", encoding="utf-8") as f:
        json.dump({"experiments": registry}, f, indent=2)


def _update_experiment(exp_id: str, **fields) -> dict:
    with _experiments_lock:
        registry = _load_registry()
        if exp_id not in registry:
            raise KeyError(exp_id)
        registry[exp_id].update(fields)
        _save_registry(registry)
        return dict(registry[exp_id])


def _experiment_command(record: dict) -> list:
    """eval_runner argv for one experiment record (absolute paths)."""
    rungs = strategy_rungs(record["strategy"], record["rerank"],
                           record["rerank_pool_n"])
    cmd = [sys.executable, "-m", "src.eval_runner",
           "--golden-set", record["golden_set"],
           "--indexes", record["indexes_dir"],
           "--manifest", record["manifest"],
           "--ks", ",".join(str(k) for k in EXPERIMENT_KS),
           "--rungs", ",".join(rungs),
           "--out", record["results_file"]]
    if record["rerank"]:
        cmd += ["--candidate-ns", str(record["rerank_pool_n"])]
    return cmd


def _run_experiment(exp_id: str):
    with _experiments_lock:
        record = _load_registry().get(exp_id)
    if record is None:
        return
    log_path = RESULTS_DIR / f"exp_{exp_id}.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = _experiment_command(record)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT), stdout=log,
                stderr=subprocess.STDOUT, text=True)
        returncode = proc.wait()
        status = "completed" if returncode == 0 else "failed"
    except Exception as exc:  # noqa: BLE001 - record and report
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[runner error] {exc}\n")
        returncode, status = -1, "failed"
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        _update_experiment(exp_id, status=status, returncode=returncode,
                           finished_at=finished)
    except KeyError:
        pass


class ExperimentRequest(BaseModel):
    golden_set: str
    split: str = "dev"
    strategy: str = "hybrid"
    rerank: bool = False
    rerank_pool_n: int = 10
    indexes_dir: str = "indexes"
    manifest: str | None = None


@app.post("/experiments", status_code=201)
def register_experiment(req: ExperimentRequest):
    if req.strategy not in EXPERIMENT_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"unknown strategy: {req.strategy} "
                   f"(valid: {', '.join(EXPERIMENT_STRATEGIES)})")
    if req.split not in EXPERIMENT_SPLITS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown split: {req.split} "
                   f"(valid: {', '.join(EXPERIMENT_SPLITS)})")
    if req.rerank and req.strategy != "hybrid":
        raise HTTPException(
            status_code=422,
            detail="rerank is only available with hybrid retrieval")
    if req.rerank and req.rerank_pool_n <= 0:
        raise HTTPException(
            status_code=422, detail="rerank_pool_n must be positive")
    if "OPENAI_API_KEY" not in os.environ:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY not set in the API environment -- "
                   "the runner needs it to embed queries")
    golden = _resolve(req.golden_set)
    if not golden.exists():
        raise HTTPException(
            status_code=422, detail=f"golden set not found: {golden}")
    if not Path(str(golden) + ".approved").exists():
        raise HTTPException(
            status_code=422,
            detail=f"golden set not approved: {golden} -- run Data "
                   f"Setup and approve it first")
    manifest = (_resolve(req.manifest) if req.manifest is not None
                else golden.parent / "manifest.jsonl")
    if not manifest.exists():
        raise HTTPException(
            status_code=422, detail=f"manifest not found: {manifest}")
    indexes = _resolve(req.indexes_dir)
    if not indexes.exists():
        raise HTTPException(
            status_code=422, detail=f"indexes dir not found: {indexes}")
    exp_id = uuid.uuid4().hex[:12]
    filename = experiment_results_filename(
        exp_id, req.split, req.strategy, req.rerank, req.rerank_pool_n)
    record = {
        "id": exp_id,
        "split": req.split,
        "strategy": req.strategy,
        "rerank": req.rerank,
        "rerank_pool_n": req.rerank_pool_n if req.rerank else None,
        "golden_set": str(golden),
        "indexes_dir": str(indexes),
        "manifest": str(manifest),
        "results_file": str(RESULTS_DIR / filename),
        "status": "pending",
        "returncode": None,
        "created_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
    }
    with _experiments_lock:
        registry = _load_registry()
        registry[exp_id] = record
        _save_registry(registry)
    return dict(record)


@app.get("/experiments")
def list_experiments():
    with _experiments_lock:
        registry = _load_registry()
    records = sorted(registry.values(),
                     key=lambda r: r.get("created_at", ""),
                     reverse=True)
    return [dict(r) for r in records]


@app.get("/experiments/{exp_id}")
def get_experiment(exp_id: str):
    with _experiments_lock:
        record = _load_registry().get(exp_id)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown experiment: {exp_id}")
    return dict(record)


@app.post("/experiments/{exp_id}/run", status_code=202)
def run_experiment(exp_id: str):
    with _experiments_lock:
        record = _load_registry().get(exp_id)
    if record is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown experiment: {exp_id}")
    if record["status"] not in ("pending", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"experiment is {record['status']}; only pending "
                   f"or failed experiments can run")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _update_experiment(exp_id, status="running", started_at=started,
                       finished_at=None, returncode=None)
    thread = threading.Thread(target=_run_experiment, args=(exp_id,),
                              daemon=True)
    thread.start()
    with _experiments_lock:
        return dict(_load_registry()[exp_id])


@app.get("/experiments/{exp_id}/log")
def experiment_log(exp_id: str, tail: int = 200):
    with _experiments_lock:
        if exp_id not in _load_registry():
            raise HTTPException(
                status_code=404, detail=f"unknown experiment: {exp_id}")
    log_path = RESULTS_DIR / f"exp_{exp_id}.log"
    if not log_path.exists():
        return {"log": ""}
    with open(log_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return {"log": "\n".join(lines[-tail:])}
