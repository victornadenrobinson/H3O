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
              "--candidate-dir", str(queue), "--candidate-limit", "2", "--verbose"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "╔" in r.stdout and "Elapsed / left ETA" in r.stdout
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
        with np.load(result["work_file"]) as work:
            assert work["forward_work_eV"].shape == (5,1)
            assert work["reverse_work_eV"].shape == (5,1)
        assert result["proxy_gap_std_kT_ins"] < 1e-8
        progress = json.loads(Path(job["progress"]).read_text())
        assert progress["stage"] == "complete" and progress["fraction"] == 1
    before = {p: p.read_bytes() for p in out.iterdir()}
    status = shell("perlmutter_pwidom_status.sh", [str(out)], env)
    assert status.returncode == 0, status.stderr
    assert "Pure-H minimum" in status.stdout and "H2O: dg" in status.stdout
    assert before == {p: p.read_bytes() for p in out.iterdir()}
    eos = out.parent / "composition_eos.json"
    eos.write_text(json.dumps(dict(P=500, T=3000,
        ice=dict(vpa=3., B_GPa=1400., nrep=3, source="test"),
        h225=dict(vpa=3., B_GPa=1400., nrep=3, source="test"),
        H=dict(vpa=3., B_GPa=1400., natoms=8, source="test"))))
    custom_args = args[:args.index("--roots")] + args[args.index("--roots")+2:]
    custom = shell("perlmutter_pwidom.sh", [*custom_args,
        "--state-json", str(eos), "--reference-run", str(out),
        "--extra-h", "5", "11", "16", "22", "--dry-run",
        "--out", str(out.parent/"custom")], env)
    assert custom.returncode == 0, custom.stderr
    assert "Reusing completed pure-H reference" in custom.stdout
    for name, atoms in (("Hplus5",329),("Hplus11",335),("Hplus16",340),("Hplus22",346)):
        assert f"{name}: GPU CPU, {atoms} host atoms" in custom.stdout
    assert not (out.parent/"custom").exists()
    bad = shell("perlmutter_pwidom.sh", [*custom_args,
        "--state-json", str(eos), "--reference-run", str(out),
        "--extra-h", "5", "11", "16", "28", "--dry-run"], env)
    assert bad.returncode != 0 and "H2.25O anchor" in bad.stderr
    config = json.loads(eos.read_text())
    config["ice"]["nrep"] = config["h225"]["nrep"] = 2
    eos.write_text(json.dumps(config))
    denseout = out.parent / "dense"
    dense = shell("perlmutter_pwidom.sh", [*custom_args,
        "--state-json", str(eos), "--reference-run", str(out),
        "--extra-h", "1", "3", "5", "6", "--out", str(denseout)], env)
    assert dense.returncode == 0, dense.stdout + dense.stderr
    densemanifest = json.loads((denseout/"manifest.json").read_text())
    assert densemanifest["status"] == "complete"
    assert all(j["name"] != "pureH" for j in densemanifest["jobs"])
    report = shell("perlmutter_pwidom_status.sh", [str(denseout)], env)
    assert report.returncode == 0, report.stderr
    assert "Shared pure-H reference" in report.stdout and "Hplus3: dg" in report.stdout
    extended = shell("perlmutter_pwidom.sh", ["--cpu", "--extend-run", str(denseout),
        "--ps", "0.1", "--equil-ps", "0", "--dry-run",
        "--out", str(out.parent/"dense_extension")], env)
    assert extended.returncode == 0, extended.stdout + extended.stderr
    assert extended.stdout.count("--extend-from") == 4
    comparison = out.parent / "comparison"
    curves = shell("perlmutter_proxy_compare.sh", ["--run-dir", str(denseout),
        "--out", str(comparison), "--ice", "docs/data/toyfit_t4.json",
        "--H", "docs/data/toyfit_t4.json"], env)
    assert curves.returncode == 0, curves.stderr
    caches = {p: p.read_bytes() for p in denseout.glob("*_frames.npz")}
    replay = subprocess.run(["bash", str(comparison/"replay_old.sh")],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)
    assert replay.returncode == 0, replay.stdout + replay.stderr
    assert caches == {p: p.read_bytes() for p in caches}
    for job in densemanifest["jobs"]:
        new = json.loads(Path(job["result"]).read_text())
        old = json.loads(Path(job["result"].replace("_candidate_K", "_old_dft_K")).read_text())
        assert old["kish_frac_ins"] == new["kish_frac_ins"]
    convergence = out.parent/"convergence"
    plotted = subprocess.run([sys.executable,str(ROOT/"mlp_pipeline/tdi/pwidom_convergence.py"),
        str(denseout),"--out",str(convergence)],cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
    assert plotted.returncode == 0, plotted.stdout + plotted.stderr
    assert convergence.with_suffix(".png").stat().st_size > 1000
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
    assert json.loads(path.read_text())["eta_seconds"] is None  # first interval may include compilation
    assert json.loads(path.read_text())["elapsed_seconds"] == 20
    clock[0] += 20
    reporter.update("MD host sampling", 50, 100)
    assert json.loads(path.read_text())["eta_seconds"] == 40
    reporter.update("single points", 0, 40)
    assert json.loads(path.read_text())["eta_seconds"] is None


def test_progress_does_not_change_toy_md(monkeypatch, tmp_path):
    import argparse
    import numpy as np
    from ase import Atoms
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline"))
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


def test_terminal_dashboard_plain_and_color(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    import pwidom_dashboard as display
    rows = [dict(name="H2O", phase="Host MD", fraction=.5, timing="[02:00/02:00 ETA]",
                 gpu="GPU0 62% 2.5/80.0 GiB", free_energy="prior dg -1147 +/-90 meV/H")]
    rendered = display.render("Perlmutter pwidom", "Elapsed 02:00", rows)
    assert "╔" in rendered and "████░░░░" in rendered
    assert "2.5/80.0 GiB" in rendered and "prior dg" in rendered
    assert "\033[" not in rendered
    assert "\033[" in display.render("Title", "Time", rows, color=True)
    plain = display.render("Title", "Time", rows, plain=True)
    assert "╔" not in plain and "####----" in plain
    assert "\033[" not in plain
    narrow = display.render("Title", "Time", rows, width=80)
    assert "prior dg -1147" in narrow


def test_in_place_extension_archives_and_locks(campaign):
    import fcntl
    env, args, out = campaign
    initial = shell("perlmutter_pwidom.sh", [*args, "--out", str(out)], env)
    assert initial.returncode == 0, initial.stderr
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    extargs = ["--cpu", "--extend-run", str(out), "--in-place",
               "--ps", "0.1", "--equil-ps", "0"]
    dry = shell("perlmutter_pwidom.sh", [*extargs, "--dry-run"], env)
    assert dry.returncode == 0, dry.stderr
    assert before == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    with open(out/".pwidom_extend.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = shell("perlmutter_pwidom.sh", extargs, env)
        assert blocked.returncode != 0 and "already holds" in blocked.stderr
    for iteration in (1,2):
        result = shell("perlmutter_pwidom.sh", extargs, env)
        assert result.returncode == 0, result.stdout + result.stderr
        manifest = json.loads((out/"manifest.json").read_text())
        history = Path(manifest["parent_archive"])
        assert manifest["status"] == "complete" and manifest["extends"] == str(history)
        assert history.parent == out/"_history"
        oldmanifest = json.loads((history/"manifest.json").read_text())
        assert oldmanifest["status"] == "complete"
        for job in manifest["jobs"]:
            data = json.loads(Path(job["result"]).read_text())
            assert data["n_frames_host"] == 5*(iteration+1)
            oldjob = next(j for j in oldmanifest["jobs"] if j["name"] == job["name"])
            assert Path(oldjob["result"]).parent == history
            tag = job["command"][job["command"].index("--tag")+1]
            with np.load(history/f"{tag}_frames.npz") as old, np.load(out/f"{tag}_frames.npz") as new:
                np.testing.assert_array_equal(old["host"], new["host"][:len(old["host"])])
            if iteration == 1:
                assert (history/f"{tag}_frames.npz").read_bytes() == before[f"{tag}_frames.npz"]
        report = shell("perlmutter_pwidom_status.sh", [str(out)], env)
        assert report.returncode == 0, report.stderr


def test_resume_run_from_md_checkpoints_preserves_parent(campaign):
    import shutil
    env, args, out = campaign
    first = shell('perlmutter_pwidom.sh', [*args, '--out', str(out)], env)
    assert first.returncode == 0, first.stdout + first.stderr
    manifest = json.loads((out/'manifest.json').read_text())
    interrupted = out.parent/'interrupted'; interrupted.mkdir()
    manifest['status'] = 'wall-time cap'
    for job in manifest['jobs']:
        cmd=job['command']; tag=cmd[cmd.index('--tag')+1]
        cmd[cmd.index('--out')+1]=str(interrupted)
        for suffix in ('_host_md.npz','_full_md.npz'):
            shutil.copy2(out/(tag+suffix),interrupted/(tag+suffix))
    (interrupted/'manifest.json').write_text(json.dumps(manifest))
    before={p:p.read_bytes() for p in interrupted.iterdir()}
    resumed=out.parent/'resumed'
    result=shell('perlmutter_pwidom.sh', ['--cpu','--resume-run',str(interrupted),
                 '--out',str(resumed),'--verbose'],env)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('no velocity reset or repeated equilibration')==8
    assert before=={p:p.read_bytes() for p in interrupted.iterdir()}
    new=json.loads((resumed/'manifest.json').read_text())
    assert new['status']=='complete' and new['resumed_from']==str(interrupted)
    for job in new['jobs']:
        tag=job['command'][job['command'].index('--tag')+1]
        with np.load(out/(tag+'_frames.npz')) as a, np.load(resumed/(tag+'_frames.npz')) as b:
            for key in ('host','full','P_host','P_full'): np.testing.assert_array_equal(a[key],b[key])
        expected=json.loads((out/Path(job['result']).name).read_text())
        actual=json.loads(Path(job['result']).read_text())
        assert actual['mu_ex_bar_meV']==expected['mu_ex_bar_meV']


def test_in_place_resume_appends_logs_preserves_history_and_locks(campaign):
    import fcntl
    env,args,out=campaign
    first=shell('perlmutter_pwidom.sh',[*args,'--out',str(out)],env)
    assert first.returncode==0,first.stdout+first.stderr
    manifest_before=(out/'manifest.json').read_bytes()
    old=json.loads(manifest_before)
    logs={Path(j['log']):Path(j['log']).read_bytes() for j in old['jobs']}
    results={Path(j['result']):Path(j['result']).read_bytes() for j in old['jobs']}
    caches={p:p.read_bytes() for p in out.glob('*_frames.npz')}
    command=['--cpu','--resume-run',str(out),'--in-place','--verbose']
    with open(out/'.pwidom_extend.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        blocked=shell('perlmutter_pwidom.sh',command,env)
        assert blocked.returncode!=0 and 'lock' in blocked.stderr
    resumed=shell('perlmutter_pwidom.sh',command,env)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    new=json.loads((out/'manifest.json').read_text())
    assert new['status']=='complete' and new['in_place']
    history=Path(new['resume_history'])
    assert (history/'manifest.json').read_bytes()==manifest_before
    for p,before in logs.items():
        assert p.read_bytes().startswith(before)
        assert b'IN-PLACE RESUME' in p.read_bytes()[len(before):]
    for p,before in results.items(): assert (history/p.name).read_bytes()==before
    assert caches=={p:p.read_bytes() for p in caches}
    for job in new['jobs']:
        actual=json.loads(Path(job['result']).read_text())
        expected=json.loads(results[Path(job['result'])])
        assert actual['mu_ex_bar_meV']==expected['mu_ex_bar_meV']


def test_legacy_in_place_restart_without_checkpoint(campaign):
    env,args,out=campaign
    # Plan only; emulate old interrupted run with manifest and logs but no state.
    dry=shell('perlmutter_pwidom.sh',[*args,'--out',str(out),'--dry-run'],env)
    assert dry.returncode==0,dry.stderr
    # Build the old manifest through the same preflight, without starting MD.
    import importlib.util
    spec=importlib.util.spec_from_file_location('pm_legacy',ROOT/'mlp_pipeline/tdi/perlmutter_pwidom.py')
    pm=importlib.util.module_from_spec(spec);spec.loader.exec_module(pm)
    from argparse import Namespace
    a=Namespace(P=500,T=3000,ps=.1,equil_ps=0,minutes=1,K=16,sel=1,ndel=1,batch=1,
        roots=[args[args.index('--roots')+1]],state_json=None,extend_run=None,in_place=False,
        extra_h=None,reference_run=None,model='toyice',ice_proxy='docs/data/toyfit_t4.json',
        h_proxy='docs/data/toyfit_t4.json',cpu=True,dry_run=True,out=str(out),candidate_dir=None)
    _,manifest=pm.plan(a)
    out.mkdir();manifest['status']='running'
    (out/'manifest.json').write_text(json.dumps(manifest))
    for j in manifest['jobs']: Path(j['log']).write_text('old interrupted log\n')
    result=shell('perlmutter_pwidom.sh',['--cpu','--resume-run',str(out),'--in-place'],env)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('no checkpoint/cache; starting this host')==4
    assert json.loads((out/'manifest.json').read_text())['status']=='complete'
    for j in manifest['jobs']: assert Path(j['log']).read_text().startswith('old interrupted log\n')


def test_hosts_subset_runs_one_host_and_checks_names(campaign):
    env, args, out = campaign
    one = shell("perlmutter_pwidom.sh", [*args, "--hosts", "pureH", "--out", str(out)], env)
    assert one.returncode == 0, one.stdout + one.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "complete" and [j["name"] for j in m["jobs"]] == ["pureH"]
    eos = out.parent / "composition_eos.json"
    eos.write_text(json.dumps(dict(P=500, T=3000, ice=dict(vpa=3., B_GPa=1400., nrep=3, source="test"),
        h225=dict(vpa=3., B_GPa=1400., nrep=3, source="test"), H=dict(vpa=3., B_GPa=1400., natoms=8, source="test"))))
    custom_args = args[:args.index("--roots")] + args[args.index("--roots")+2:]
    sub = shell("perlmutter_pwidom.sh", [*custom_args, "--state-json", str(eos), "--reference-run", str(out),
        "--extra-h", "5", "11", "16", "22", "--hosts", "Hplus11", "--dry-run", "--out", str(out.parent/"sub")], env)
    assert sub.returncode == 0, sub.stderr
    assert "Hplus11: GPU CPU" in sub.stdout and "Hplus5" not in sub.stdout and "Hplus22" not in sub.stdout
    bad = shell("perlmutter_pwidom.sh", [*custom_args, "--state-json", str(eos), "--reference-run", str(out),
        "--extra-h", "5", "11", "16", "22", "--hosts", "Hplus99", "--dry-run"], env)
    assert bad.returncode != 0 and "not among this run" in bad.stderr
