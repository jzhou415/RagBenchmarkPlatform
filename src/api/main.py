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
    return {"status": "ok", "stages": sorted(STAGES)}


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
