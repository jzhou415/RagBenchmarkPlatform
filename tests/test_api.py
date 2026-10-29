#!/usr/bin/env python3
"""Tests for src/api/main.py (step 8). Run: python -m unittest test_api"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertEqual(body["root"], str(api_main.ROOT))

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


class TestExperiments(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._old_results = api_main.RESULTS_DIR
        api_main.RESULTS_DIR = self.dir / "results"
        self._env = mock.patch.dict(os.environ,
                                    {"OPENAI_API_KEY": "test-key"})
        self._env.start()
        self.golden = self.dir / "golden_dev.yaml"
        self.golden.write_text("questions: []\n", encoding="utf-8")
        (self.dir / "golden_dev.yaml.approved").write_text(
            "", encoding="utf-8")
        self.manifest = self.dir / "manifest.jsonl"
        self.manifest.write_text("", encoding="utf-8")
        self.indexes = self.dir / "indexes"
        self.indexes.mkdir()
        self.client = make_client()

    def tearDown(self):
        self._env.stop()
        api_main.RESULTS_DIR = self._old_results
        self._tmp.cleanup()

    def _body(self, **over):
        body = {"golden_set": str(self.golden), "split": "dev",
                "strategy": "hybrid", "rerank": False,
                "indexes_dir": str(self.indexes),
                "manifest": str(self.manifest)}
        body.update(over)
        return body

    def _register(self, **over):
        r = self.client.post("/experiments", json=self._body(**over))
        self.assertEqual(r.status_code, 201, r.text)
        return r.json()

    def test_register_success(self):
        record = self._register()
        self.assertEqual(record["status"], "pending")
        self.assertTrue(record["results_file"].endswith(
            f"exp_{record['id']}_dev_hybrid.json"))
        self.assertTrue((api_main.RESULTS_DIR
                         / "experiments.json").exists())
        listed = self.client.get("/experiments").json()
        self.assertEqual([e["id"] for e in listed], [record["id"]])
        got = self.client.get(f"/experiments/{record['id']}").json()
        self.assertEqual(got["id"], record["id"])

    def test_register_rejects_unapproved(self):
        (self.dir / "golden_dev.yaml.approved").unlink()
        r = self.client.post("/experiments", json=self._body())
        self.assertEqual(r.status_code, 422)

    def test_register_rejects_missing_golden(self):
        r = self.client.post(
            "/experiments",
            json=self._body(golden_set=str(self.dir / "nope.yaml")))
        self.assertEqual(r.status_code, 422)

    def test_register_rejects_rerank_without_hybrid(self):
        r = self.client.post(
            "/experiments",
            json=self._body(strategy="dense", rerank=True))
        self.assertEqual(r.status_code, 422)

    def test_register_rejects_bad_strategy_and_split(self):
        r = self.client.post("/experiments",
                             json=self._body(strategy="nope"))
        self.assertEqual(r.status_code, 422)
        r = self.client.post("/experiments",
                             json=self._body(split="nope"))
        self.assertEqual(r.status_code, 422)

    def test_strategy_rungs_and_filename(self):
        self.assertEqual(api_main.strategy_rungs("dense", False, 10),
                         ("semantic",))
        self.assertEqual(api_main.strategy_rungs("bm25", False, 10),
                         ("bm25",))
        self.assertEqual(api_main.strategy_rungs("hybrid", False, 10),
                         ("hybrid",))
        self.assertEqual(api_main.strategy_rungs("hybrid", True, 20),
                         ("hybrid", "hybrid_rerank@20"))
        self.assertEqual(api_main.experiment_results_filename(
            "abc", "dev", "hybrid", False, 10),
            "exp_abc_dev_hybrid.json")
        self.assertEqual(api_main.experiment_results_filename(
            "abc", "heldout", "hybrid", True, 20),
            "exp_abc_heldout_hybrid_rerank20.json")

    def test_experiment_command(self):
        record = {"strategy": "hybrid", "rerank": True,
                  "rerank_pool_n": 20,
                  "golden_set": str(self.golden),
                  "indexes_dir": str(self.indexes),
                  "manifest": str(self.manifest),
                  "results_file": str(self.dir / "out.json")}
        cmd = api_main._experiment_command(record)
        self.assertIn("-m", cmd)
        self.assertIn("src.eval_runner", cmd)
        self.assertEqual(cmd[cmd.index("--ks") + 1], "3,5,10")
        self.assertEqual(cmd[cmd.index("--rungs") + 1],
                         "hybrid,hybrid_rerank@20")
        self.assertEqual(cmd[cmd.index("--candidate-ns") + 1], "20")
        record["rerank"] = False
        cmd = api_main._experiment_command(record)
        self.assertNotIn("--candidate-ns", cmd)
        self.assertEqual(cmd[cmd.index("--rungs") + 1], "hybrid")

    def test_run_completes_with_fake_popen_then_conflicts(self):
        record = self._register(strategy="hybrid", rerank=True,
                                rerank_pool_n=20)
        exp_id = record["id"]

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                out = cmd[cmd.index("--out") + 1]
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_text(json.dumps({"run": {}}),
                                     encoding="utf-8")

            def wait(self):
                return 0

        with mock.patch.object(api_main.subprocess, "Popen", FakePopen):
            r = self.client.post(f"/experiments/{exp_id}/run")
            self.assertEqual(r.status_code, 202, r.text)
            for _ in range(100):
                status = self.client.get(
                    f"/experiments/{exp_id}").json()["status"]
                if status != "running":
                    break
                time.sleep(0.05)
            self.assertEqual(status, "completed")
            log = self.client.get(
                f"/experiments/{exp_id}/log").json()["log"]
            self.assertIn("-m src.eval_runner", log.splitlines()[0])
            self.assertTrue(Path(record["results_file"]).exists())
        r = self.client.post(f"/experiments/{exp_id}/run")
        self.assertEqual(r.status_code, 409)

    def test_unknown_experiment_404s(self):
        self.assertEqual(self.client.get("/experiments/nope").status_code,
                         404)
        self.assertEqual(self.client.get(
            "/experiments/nope/log").status_code, 404)
        self.assertEqual(self.client.post(
            "/experiments/nope/run").status_code, 404)


if __name__ == "__main__":
    unittest.main()
