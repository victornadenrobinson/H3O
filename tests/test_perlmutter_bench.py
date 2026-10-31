"""Real tiny toy benchmarks exercise process synchronization and aggregation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_aggregation_uses_concurrent_makespan(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import perlmutter_bench as bench
    summary = bench.timed_summary([
        dict(timed_started=10, timed_ended=12, lambda_steps=10),
        dict(timed_started=10.5, timed_ended=14, lambda_steps=20)])
    assert summary["lambda_steps_s"] == 7.5  # 30 / 4, not sum of individual rates.
    with pytest.raises(ValueError):
        bench.timed_summary([])


def test_quick_toy_screens_and_validates_all_slots(tmp_path):
    out = tmp_path / "bench"
    env = dict(os.environ, PY=sys.executable, TOYICE_FIT=str(ROOT / "docs/data/toyfit_t4.json"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    command = ["bash", str(ROOT / "mlp_pipeline/perlmutter_bench.sh"),
        "--cpu", "--model", "toyice", "--processes", "1", "2",
        "--replicas", "1", "2", "--steps", "2", "--warm", "1", "--threads", "1",
        "--out", str(out)]
    result = subprocess.run(command, cwd="/tmp", env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete_node_validation"]
    assert summary["whole_node"]["lambda_steps_s"] > 0
    assert len(summary["validation"]) == 4
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 8 and all(r["status"] == "complete" for r in rows)
    assert "local CPU toy check" in (out / "RUN_ORIGIN.txt").read_text()


def test_benchmark_deadline_reaps_children(tmp_path):
    out = tmp_path / "deadline"
    env = dict(os.environ, PY=sys.executable, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    command = ["bash", str(ROOT / "mlp_pipeline/perlmutter_bench.sh"), "--cpu", "--model", "toyice",
        "--processes", "1", "--replicas", "1", "--stage-seconds", ".01", "--minutes", "1", "--out", str(out)]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["complete_node_validation"]
    assert "whole_node" not in summary
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert rows and all(r["status"] == "failed" for r in rows)
    for row in rows:
        for pid in row["worker_pids"]:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
