#!/usr/bin/env python3
"""Tests for src/api/main.py (step 8). Run: python -m unittest test_api"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import main as api_main


def make_client():
    return TestClient(api_main.app)


class TestHealthAndStages(unittest.TestCase):
    def test_health(self):
        r = make_client().get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("parse", body["stages"])

    def test_list_stages(self):
        r = make_client().get("/stages")
        self.assertEqual(r.status_code, 200)
        ids = [s["id"] for s in r.json()]
        for expected in ("parse", "corpus-split", "chunk", "question-gen",
                         "auto-map", "validate", "index-bm25",
                         "index-dense"):
            self.assertIn(expected, ids)

    def test_unknown_stage_404(self):
        r = make_client().post("/stages/nope/run", json={"args": {}})
        self.assertEqual(r.status_code, 404)

    def test_unknown_run_404(self):
        c = make_client()
        self.assertEqual(c.get("/runs/deadbeef").status_code, 404)
        self.assertEqual(c.get("/runs/deadbeef/log").status_code, 404)


class TestArgsToCli(unittest.TestCase):
    def test_conversion(self):
        cli = api_main.args_to_cli({
            "manifest": "full/manifest_full.jsonl",
            "chunk_manifest": "full/chunk_manifest.yaml",
            "top_k": 10,
            "verbose": True,
            "quiet": False,
            "nothing": None,
        })
        self.assertEqual(cli, ["--manifest", "full/manifest_full.jsonl",
                               "--chunk-manifest", "full/chunk_manifest.yaml",
                               "--top-k", "10", "--verbose"])


class TestRunLifecycle(unittest.TestCase):
    def test_run_succeeds_and_log_captured(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "fake_stage.py"
            script.write_text(
                "import sys\nprint('hello from fake stage')\n"
                "print('arg was: ' + sys.argv[1])\n",
                encoding="utf-8")
            fake_stages = {"fake": {"script": str(script),
                                    "description": "test only"}}
            old_stages, api_main.STAGES = api_main.STAGES, fake_stages
            old_runs = api_main.RUNS_DIR
            api_main.RUNS_DIR = Path(td) / "runs"
            try:
                c = make_client()
                r = c.post("/stages/fake/run",
                           json={"args": {"greeting": "hi"}})
                self.assertEqual(r.status_code, 202)
                run_id = r.json()["run_id"]
                # Poll until done (fake script exits immediately).
                for _ in range(100):
                    s = c.get(f"/runs/{run_id}").json()
                    if s["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(s["status"], "succeeded")
                self.assertEqual(s["returncode"], 0)
                self.assertEqual(s["stage"], "fake")
                log = c.get(f"/runs/{run_id}/log").json()["log"]
                self.assertIn("hello from fake stage", log)
                self.assertIn("--greeting hi", log)
            finally:
                api_main.STAGES = old_stages
                api_main.RUNS_DIR = old_runs

    def test_run_failure_reported(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "fail_stage.py"
            script.write_text("import sys; sys.exit(3)\n", encoding="utf-8")
            fake_stages = {"boom": {"script": str(script),
                                    "description": "test only"}}
            old_stages, api_main.STAGES = api_main.STAGES, fake_stages
            old_runs = api_main.RUNS_DIR
            api_main.RUNS_DIR = Path(td) / "runs"
            try:
                c = make_client()
                run_id = c.post("/stages/boom/run",
                                json={"args": {}}).json()["run_id"]
                for _ in range(100):
                    s = c.get(f"/runs/{run_id}").json()
                    if s["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(s["status"], "failed")
                self.assertEqual(s["returncode"], 3)
            finally:
                api_main.STAGES = old_stages
                api_main.RUNS_DIR = old_runs


class TestStageCommand(unittest.TestCase):
    def test_repo_script_runs_as_module(self):
        cmd = api_main.stage_command(api_main.ROOT / "src" / "parse.py")
        self.assertEqual(cmd, [sys.executable, "-m", "src.parse"])

    def test_outside_repo_runs_as_file(self):
        with tempfile.TemporaryDirectory() as td:
            script = Path(td) / "fake_stage.py"
            cmd = api_main.stage_command(script)
            self.assertEqual(cmd, [sys.executable, str(script)])

    def test_non_python_inside_repo_runs_as_file(self):
        script = api_main.ROOT / "data" / "notes.txt"
        cmd = api_main.stage_command(script)
        self.assertEqual(cmd, [sys.executable, str(script)])


class TestModuleExecution(unittest.TestCase):
    def test_relative_import_stage_runs_via_module(self):
        # Regression test for the Data Setup page's first Build: the
        # runner used to execute src/parse.py as a file, and parse.py's
        # relative import (from .config import ...) died with
        # "attempted relative import with no known parent package".
        # Deliberately no __init__.py here: module execution must work
        # for namespace packages too (the real repo has src/__init__.py).
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "pkg"
            pkg.mkdir()
            (pkg / "sibling.py").write_text(
                'VALUE = "from-sibling"\n', encoding="utf-8")
            (pkg / "mod.py").write_text(
                "import sys\nfrom .sibling import VALUE\n"
                "print('saw ' + VALUE + ' args: ' + ' '.join(sys.argv[1:]))\n",
                encoding="utf-8")
            mod_stages = {"mod": {"script": "pkg/mod.py",
                                  "description": "test only"}}
            old_stages, api_main.STAGES = api_main.STAGES, mod_stages
            old_runs = api_main.RUNS_DIR
            api_main.RUNS_DIR = Path(td) / "runs"
            old_root = api_main.ROOT
            api_main.ROOT = Path(td)
            try:
                c = make_client()
                r = c.post("/stages/mod/run",
                           json={"args": {"greeting": "hi"}})
                self.assertEqual(r.status_code, 202)
                run_id = r.json()["run_id"]
                for _ in range(100):
                    s = c.get(f"/runs/{run_id}").json()
                    if s["status"] != "running":
                        break
                    time.sleep(0.05)
                self.assertEqual(s["status"], "succeeded")
                log = c.get(f"/runs/{run_id}/log").json()["log"]
                self.assertIn("saw from-sibling args: --greeting hi", log)
                self.assertIn("-m pkg.mod", log.splitlines()[0])
            finally:
                api_main.STAGES = old_stages
                api_main.RUNS_DIR = old_runs
                api_main.ROOT = old_root


if __name__ == "__main__":
    unittest.main()
