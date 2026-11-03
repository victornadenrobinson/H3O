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


def test_memory_guard_skips_before_starting_workers(monkeypatch,tmp_path):
    import argparse,time
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import perlmutter_bench as bench
    profile=dict(process_MiB=20000,tensor_MiB=10000,context_MiB=10000,graphs_per_lambda=2)
    assert bench.memory_projection("pwmd",8,profile,1.5) == 240000
    assert bench.memory_projection("pwsp",4,profile,1.5) == 75000
    monkeypatch.setattr(bench,"telemetry",lambda:{"0":dict(memory_MiB=0,total_MiB=81920,utilization=0)})
    def no_worker(*args,**kwargs):
        raise AssertionError("An unsafe layout must not start a process")
    monkeypatch.setattr(bench.subprocess,"Popen",no_worker)
    args=argparse.Namespace(cpu=False,_memory_profile=profile,memory_safety=1.5,reserve_gib=12,
                           stage_seconds=10)
    rows=bench.phase(args,"memory_test",[("0","pwmd",8)],tmp_path,[],time.monotonic()+10,lambda:False)
    assert rows[0]["status"] == "skipped" and rows[0]["worker_pids"] == []
    assert "work_units_s" not in rows[0]


def test_quick_toy_screens_and_validates_all_slots(tmp_path):
    out = tmp_path / "bench"
    env = dict(os.environ, PY=sys.executable, TOYICE_FIT=str(ROOT / "docs/data/toyfit_t4.json"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    command = ["bash", str(ROOT / "mlp_pipeline/perlmutter_bench.sh"),
        "--workflow","tdi","--cpu", "--model", "toyice", "--processes", "1", "2",
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


def test_pwidom_benchmark_keeps_md_and_energy_rates_separate(tmp_path):
    out=tmp_path/"pwidom_bench"
    env=dict(os.environ,PY=sys.executable,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    env.pop("CUDA_MPS_PIPE_DIRECTORY",None)
    result=subprocess.run(["bash",str(ROOT/"mlp_pipeline/perlmutter_bench.sh"),
        "--cpu","--model","toyice","--processes","1","--replicas","2",
        "--steps","2","--warm","1","--out",str(out)],cwd=ROOT,env=env,
        capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr
    summary=json.loads((out/"summary.json").read_text())
    assert summary["workflow"] == "pwidom" and summary["complete_node_validation"]
    assert set(summary["whole_node_by_mode"]) == {"pwmd","pwsp"}
    assert summary["whole_node_by_mode"]["pwmd"]["unit"] == "MD steps"
    assert summary["whole_node_by_mode"]["pwsp"]["unit"] == "structures"


def test_single_structure_benchmark_cannot_time_cached_results(monkeypatch):
    import argparse
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(ROOT/"mlp_pipeline/tdi"))
    import bench
    import pwidom
    instances=[]
    class FakeModel:
        def __init__(self,*args):
            self.resets=self.calls=0
            self.calc=SimpleNamespace(reset=self.reset)
            self.gb=None; instances.append(self)
        def reset(self): self.resets+=1
        def energies(self,structs):
            assert self.resets == self.calls+1
            self.calls+=1
            return [0]*len(structs)
    monkeypatch.setattr(pwidom,"Model",FakeModel)
    output=[]
    monkeypatch.setattr(bench,"result",lambda **kwargs:output.append(kwargs))
    bench.mode_pwsp(argparse.Namespace(system="h2o",vpa=3,cell_nrep=1,nrep=1,
        model="toyice",device="cpu",warm=2,steps=3,barrier=None))
    assert instances[0].calls == 5 and output[0]["work_units"] == 3
