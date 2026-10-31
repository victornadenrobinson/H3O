"""Exercise the actual shell entry points with tiny CPU toy hosts; no Slurm/GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(tmp_path):
    states = tmp_path / "states"
    states.mkdir()
    for name, phase, hp in (("ice", "fcc", 2), ("h225", "fcc", 2.25), ("H", "H", 2)):
        data = dict(phase=phase, P_target=500, T=3000, vpa=3.0, B_GPa=1400,
                    args=dict(model="toyice", hper=hp, nrep=1,
                              natoms=8 if phase == "H" else 0))
        (states / f"{name}_stage0.json").write_text(json.dumps(data))
    env = dict(os.environ, PY=sys.executable, TOYICE_FIT="docs/data/toyfit_t4.json",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    args = ["--cpu", "--model", "toyice", "--roots", str(states),
            "--ice-proxy", "docs/data/toyfit_t4.json", "--h-proxy", "docs/data/toyfit_t4.json",
            "--ps", "0.1", "--equil-ps", "0", "--K", "16", "--sel", "1", "--ndel", "1", "--batch", "1"]
    return env, args, tmp_path / "output"


def shell(name, args, env):
    return subprocess.run(["bash", str(ROOT / "mlp_pipeline" / name), *args],
                          cwd="/tmp", env=env, text=True, capture_output=True, timeout=60)


def test_toy_scan_and_readonly_status(campaign):
    env, args, out = campaign
    queue = out.parent / "candidates"
    r = shell("perlmutter_pwidom.sh", [*args, "--out", str(out),
              "--candidate-dir", str(queue), "--candidate-limit", "2"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[H2O]" in r.stdout and "[pureH]" in r.stdout
    assert "proxy candidate" in r.stdout
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "complete" and len(m["jobs"]) == 4
    requests = [json.loads(line) for line in (queue / "queue.jsonl").read_text().splitlines()]
    assert len(requests) == 8
    assert all(r["reference_status"] == "unlabeled" for r in requests)
    for job in m["jobs"]:
        result = json.loads(Path(job["result"]).read_text())
        assert job["exit_code"] == 0 and result["n_ins"] == 5
        assert result["proxy_gap_std_kT_ins"] < 1e-8
        progress = json.loads(Path(job["progress"]).read_text())
        assert progress["stage"] == "complete" and progress["fraction"] == 1
    before = {p: p.read_bytes() for p in out.iterdir()}
    status = shell("perlmutter_pwidom_status.sh", [str(out)], env)
    assert status.returncode == 0, status.stderr
    assert "Pure-H minimum" in status.stdout and "H2O: dg" in status.stdout
    assert before == {p: p.read_bytes() for p in out.iterdir()}
    follow = shell("perlmutter_pwidom_status.sh", [str(out), "--follow"], env)
    assert follow.returncode == 0, follow.stderr
    assert "[H2O]" in follow.stdout and "proxy candidate" in follow.stdout
    assert before == {p: p.read_bytes() for p in out.iterdir()}
    again = shell("perlmutter_pwidom.sh", [*args, "--out", str(out)], env)
    assert again.returncode != 0 and "Output already exists" in again.stderr
    assert before == {p: p.read_bytes() for p in out.iterdir()}


def test_preflight_dry_run_and_missing_state(campaign):
    env, args, out = campaign
    r = shell("perlmutter_pwidom.sh", [*args, "--out", str(out), "--dry-run"], env)
    assert r.returncode == 0 and not out.exists(), r.stderr
    states = Path(args[args.index("--roots") + 1])
    (states / "H_stage0.json").unlink()
    r = shell("perlmutter_pwidom.sh", [*args, "--out", str(out)], env)
    assert r.returncode != 0 and "Missing H stage0" in r.stderr
    assert not out.exists()


def test_deadline_reaps_every_child(campaign):
    env, args, out = campaign
    r = shell("perlmutter_pwidom.sh", [*args, "--out", str(out), "--minutes", "0.00001"], env)
    assert r.returncode == 124, r.stdout + r.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "wall-time cap" and len(m["jobs"]) == 4
    for job in m["jobs"]:
        with pytest.raises(ProcessLookupError):
            os.kill(job["pid"], 0)


def test_explicit_eos_without_prior_tdi(campaign):
    env, args, out = campaign
    config = out.parent / "eos.json"
    eos = dict(P=500, T=3000,
               ice=dict(vpa=3, B_GPa=1400, nrep=1, source="toy test"),
               h225=dict(vpa=3, B_GPa=1400, nrep=1, source="toy test"),
               H=dict(vpa=3, B_GPa=1400, natoms=8, source="toy test"))
    config.write_text(json.dumps(eos))
    args = args[:args.index("--roots")] + args[args.index("--roots") + 2:]
    r = shell("perlmutter_pwidom.sh", [*args, "--state-json", str(config),
              "--out", str(out)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stage0"] == {} and manifest["eos_inputs"] == eos
    eos["T"] = 5000
    config.write_text(json.dumps(eos))
    r = shell("perlmutter_pwidom.sh", [*args, "--state-json", str(config),
              "--out", str(out.parent / "other")], env)
    assert r.returncode != 0 and "P/T do not match" in r.stderr


def test_inspect_eos_reports_values_without_running(campaign):
    import csv
    import math
    env, _, out = campaign
    inv = out.parent / "inventory.csv"
    with inv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["Path", "True_T_K", "True_P_GPa",
            "True_V_A3", "N_ions", "N_H", "N_O", "Other", "P_kin_GPa", "P_ext_GPa"])
        writer.writeheader()
        for name, nh, no in (("fcc-h2o", 8, 4), ("fcc-h2.25o", 9, 4), ("pure-h", 8, 0)):
            for i in range(21):
                v = 1.8 + i * .02
                p = 500 * math.exp(-(v - 2))
                writer.writerow(dict(Path=f"/{name}/{i}", True_T_K=3000,
                    True_P_GPa=p, True_V_A3=v * (nh + no), N_ions=nh + no,
                    N_H=nh, N_O=no, Other="", P_kin_GPa=0, P_ext_GPa=p))
    r = shell("perlmutter_pwidom.sh", ["--inspect-eos", str(inv)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("vpa=") == 3 and "Read-only report" in r.stdout
    assert not out.exists()
    config = out.parent / "saved_eos.json"
    args = ["--inspect-eos", str(inv), "--save-eos", str(config)]
    saved = shell("perlmutter_pwidom.sh", args, env)
    assert saved.returncode == 0, saved.stdout + saved.stderr
    eos = json.loads(config.read_text())
    assert eos["P"] == 500 and eos["ice"]["nrep"] == 3 and eos["H"]["natoms"] == 250
    assert eos["ice"]["vpa"] == pytest.approx(2, abs=.001)
    before = config.read_bytes()
    again = shell("perlmutter_pwidom.sh", args, env)
    assert again.returncode != 0 and config.read_bytes() == before


def discovery_env(tmp_path):
    """A fresh login-like shell with no active venv or python on PATH."""
    env = dict(os.environ, HOME=str(tmp_path / "home"), SCRATCH=str(tmp_path / "scratch"),
               PATH="/usr/bin:/bin")
    for key in ("PY", "VIRTUAL_ENV", "CONDA_PREFIX", "GRACE_ENV", "H3O_ENV_INIT"):
        env.pop(key, None)
    Path(env["HOME"]).mkdir()
    Path(env["SCRATCH"]).mkdir()
    return env


def interpreter_wrapper(path):
    import shlex
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    path.chmod(0o755)


def test_discovers_scratch_environment_without_activation(tmp_path):
    env = discovery_env(tmp_path)
    interpreter_wrapper(Path(env["SCRATCH"]) / "h3o/grace-venv/bin/python")
    r = shell("perlmutter_pwidom.sh", ["--help"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Perlmutter Python:" in r.stderr and "--inspect-eos" in r.stdout
    # Status uses the same discovery before validating its run path.
    r = shell("perlmutter_pwidom_status.sh", [str(tmp_path / "missing")], env)
    assert r.returncode != 0 and "Perlmutter Python:" in r.stderr
    assert "manifest.json" in r.stderr


def test_explicit_bad_python_does_not_fall_back(tmp_path):
    env = discovery_env(tmp_path)
    env["PY"] = str(tmp_path / "missing-python")
    interpreter_wrapper(Path(env["SCRATCH"]) / "h3o/grace-venv/bin/python")
    r = shell("perlmutter_pwidom.sh", ["--help"], env)
    assert r.returncode != 0 and "PY is explicit" in r.stderr
    assert "usage:" not in r.stdout


def test_explicit_initialization_recipe(tmp_path):
    import shlex
    env = discovery_env(tmp_path)
    init = tmp_path / "activate.sh"
    init.write_text(f"export PY={shlex.quote(sys.executable)}\n")
    env["H3O_ENV_INIT"] = str(init)
    r = shell("perlmutter_pwidom.sh", ["--help"], env)
    assert r.returncode == 0 and "Perlmutter Python:" in r.stderr, r.stderr


def test_conda_named_environment_fallback(tmp_path):
    import shlex
    env = discovery_env(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    conda = bin_dir / "conda"
    conda.write_text('#!/bin/bash\n'
                     '[[ "$1 $2 $3 $4 $5" == "run --no-capture-output -n grace-env python" ]] || exit 1\n'
                     f'shift 5\nexec {shlex.quote(sys.executable)} "$@"\n')
    conda.chmod(0o755)
    env["PATH"] = str(bin_dir) + ":" + env["PATH"]
    r = shell("perlmutter_pwidom.sh", ["--help"], env)
    assert r.returncode == 0 and "Perlmutter Python:" in r.stderr, r.stderr


def test_missing_environment_has_actionable_error(tmp_path):
    env = discovery_env(tmp_path)
    r = shell("perlmutter_pwidom.sh", ["--help"], env)
    assert r.returncode != 0 and "Set GRACE_ENV=" in r.stderr
    assert "No environment was installed" in r.stderr


def test_gpu_metrics_and_progress_text(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import perlmutter_pwidom as pm
    def query(cmd, **kwargs):
        assert "--id=0,GPU-bbbb" in cmd and kwargs["timeout"] == 3
        return "0, GPU-aaaa, 87, 10240, 81920\n1, GPU-bbbb-full, 52, 20480, 81920\n"
    monkeypatch.setattr(pm.subprocess, "check_output", query)
    metrics = pm.gpu_metrics([dict(gpu="0"), dict(gpu="GPU-bbbb")])
    assert "87% util, 10.0/80.0 GiB" in metrics["0"]
    assert "52% util, 20.0/80.0 GiB" in metrics["GPU-bbbb"]
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(dict(stage="MD host sampling", fraction=.5,
        done=100, total=200, unit="steps", eta_seconds=90, updated=0)))
    assert "50%" in pm.progress_text(dict(progress=str(path)))
    assert "stage ETA 1.5 min" in pm.progress_text(dict(progress=str(path)))


def test_progress_eta_uses_stage_timing(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import pwidom_progress as pp
    clock = [100.0]
    monkeypatch.setattr(pp, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]))
    path = tmp_path / "progress.json"
    reporter = pp.Progress(path)
    reporter.update("MD host sampling", 0, 100)
    assert json.loads(path.read_text())["eta_seconds"] is None
    clock[0] += 20
    reporter.update("MD host sampling", 25, 100)
    assert json.loads(path.read_text())["eta_seconds"] == 60
    assert json.loads(path.read_text())["elapsed_seconds"] == 20
    reporter.update("single points", 0, 40)
    assert json.loads(path.read_text())["eta_seconds"] is None


def test_progress_does_not_change_toy_md(monkeypatch, tmp_path):
    import argparse
    import numpy as np
    from ase import Atoms
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    monkeypatch.setenv("TOYICE_FIT", str(ROOT / "docs/data/toyfit_t4.json"))
    from pwidom import Model, md_frames
    from pwidom_progress import Progress
    host = Atoms("OH2", positions=[[1, 1, 1], [2, 1, 1], [1, 2, 1]], cell=[8]*3, pbc=True)
    settings = dict(T=3000, dt=.5, equil_ps=.01, ps=.1, every_fs=20)
    old = md_frames(Model("toyice", "cpu", 1), host, argparse.Namespace(**settings), 3, "host")
    new = md_frames(Model("toyice", "cpu", 1), host,
        argparse.Namespace(**settings, _progress=Progress(tmp_path / "progress.json")), 3, "host")
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)


def test_old_run_log_eta(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import perlmutter_pwidom as pm
    log = tmp_path / "host.log"
    log.write_text("  MD host + 1 H: 4/20 frames, P so far 514 GPa, 1.5 min\n"
                   "  MD host + 1 H: 8/20 frames, P so far 514 GPa, 1.9 min\n")
    text = pm.progress_text(dict(log=str(log)))
    assert "elapsed 00:01:54 / remaining ~00:01:12" in text
    assert "this MD only" in text


def test_extend_run_adds_frames_without_modifying_parent(campaign):
    env, args, out = campaign
    first = shell("perlmutter_pwidom.sh", [*args, "--out", str(out)], env)
    assert first.returncode == 0, first.stdout + first.stderr
    before = {p: p.read_bytes() for p in out.iterdir()}
    extension = out.parent / "extended"
    # Extension must not need original TDI roots; its exact state is inherited.
    r = shell("perlmutter_pwidom.sh", ["--cpu", "--extend-run", str(out),
        "--ps", "0.1", "--equil-ps", "0", "--out", str(extension)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    manifest = json.loads((extension / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["extends"] == str(out)
    for job in manifest["jobs"]:
        result = json.loads(Path(job["result"]).read_text())
        assert result["n_frames_host"] == 10 and result["n_ins"] == 10
        oldjob = next(j for j in json.loads((out/"manifest.json").read_text())["jobs"] if j["name"] == job["name"])
        oldtag = oldjob["command"][oldjob["command"].index("--tag")+1]
        with np.load(out / f"{oldtag}_frames.npz") as old, np.load(extension / f"{oldtag}_frames.npz") as new:
            assert new["segments"] == 2 and new["segment_seed"] > old["segment_seed"]
            np.testing.assert_array_equal(old["host"], new["host"][:5])
    assert before == {p: p.read_bytes() for p in out.iterdir()}
