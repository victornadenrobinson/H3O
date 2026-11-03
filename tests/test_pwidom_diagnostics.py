"""CPU-only checks for proxy comparisons and the unlabeled DFT geometry queue."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.io import read
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "mlp_pipeline/tdi" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_candidate_queue_preserves_trial_pair_and_deduplicates(tmp_path):
    candidates = module("pwidom_candidates")
    host = Atoms("OH2", positions=[[1,1,1],[2,1,1],[1,2,1]], cell=[8]*3, pbc=True)
    full = host + Atoms("H", positions=[[2,2,2]])
    trial = dict(direction="insertion", frame=0, trial=0, dU_mlp_eV=1.2,
                 phi_eV=.4, gap_kT=3.1, log_weight=2., structures=[host, full])
    provenance = dict(model="toyice", model_fp="test", tag="test", T_K=3000)
    assert candidates.export_trials(tmp_path, [trial], 1, provenance) == 1
    assert candidates.export_trials(tmp_path, [trial], 1, provenance) == 0
    rows = [json.loads(line) for line in (tmp_path / "queue.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["reference_status"] == "unlabeled"
    a, b = (read(tmp_path / p) for p in rows[0]["structures"])
    assert len(b) == len(a) + 1 and a.calc is None and b.calc is None
    np.testing.assert_array_equal(a.positions, host.positions)
    np.testing.assert_array_equal(b.positions[-1], [2,2,2])
    assert "energy" not in a.info and "not a measured GRACE error" in rows[0]["diagnosis"]


def test_cached_queue_does_not_invent_insertion_pairs(tmp_path):
    candidates = module("pwidom_candidates")
    archive = tmp_path / "frames.npz"
    np.savez(archive, host=np.zeros((3,2,3)), full=np.zeros((3,3,3)),
             numbers=np.array([1,1]), cell=np.eye(3)*4, V=64, T=3000,
             model="toyice", model_fp="test")
    out = tmp_path / "queue"
    candidates.cached_frames(argparse.Namespace(frames=[str(archive)], out=str(out), per_source=2))
    rows = [json.loads(l) for l in (out / "queue.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(len(r["structures"]) == 1 for r in rows)
    assert all(r["reasons"] == ["representative_cached_MD"] for r in rows)


def test_reported_formation_integral():
    plot = module("pwidom_formation")
    n = np.array([0, 9/108, .25]); mu = np.array([-1146.5,-738.5,185.7])
    values, grid, integral = plot.integrate(n, mu)
    np.testing.assert_allclose(values/(3+n), [0,-25.47297297297,-38.34102564103])
    assert integral[-1] == pytest.approx(values[-1])
    bound = plot.statistical_error_bound(n, np.array([89.6,105,109.4]))
    np.testing.assert_allclose(bound, [0, (89.6+105)*(.08333333333333333/2)/ (3+9/108),
        (.0416666666666667*89.6 + .125*105 + .0833333333333333*109.4)/3.25])
    with pytest.raises(ValueError):
        plot.integrate(np.array([.1,.25]), mu[:2])


def test_pair_curve_plot_and_no_overwrite(tmp_path):
    base = str(ROOT / "docs/data/toyfit_t4.json")
    out = tmp_path / "curves"
    cmd = [sys.executable, str(ROOT / "mlp_pipeline/tdi/proxy_compare.py"),
           "--old", base, "--ice", base, "--H", base, "--out", str(out)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "proxy_pairs.png").stat().st_size > 1000
    assert (out / "pair_curves.csv").stat().st_size > 1000
    before = (out / "proxy_pairs.png").read_bytes()
    again = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert again.returncode != 0 and before == (out / "proxy_pairs.png").read_bytes()


def test_baseline_replay_plan_is_cached_only_and_not_executed(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "mlp_pipeline/tdi"))
    compare = module("proxy_compare")
    run = tmp_path / "run"; run.mkdir()
    np.savez(run / "pw_test_frames.npz", host=np.zeros((1,2,3)),
             full=np.zeros((1,3,3)), model_fp="test")
    command = [sys.executable, "mlp_pipeline/tdi/pwidom.py", "run", "--tag", "pw_test",
        "--out", str(run), "--proxy", "new.json", "--label", "candidate", "--K", "2000"]
    (run / "manifest.json").write_text(json.dumps(dict(status="complete",
        jobs=[dict(name="test", gpu="0", command=command)])))
    text = compare.replay_plan(run, ROOT / "docs/data/toyfit_t4.json")
    assert "--label old_dft" in text and "test -s" in text and "test ! -e" in text
    script = tmp_path / "replay.sh"; script.write_text(text)
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    assert not (run / "test_old_dft.log").exists()
