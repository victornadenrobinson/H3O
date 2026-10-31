import numpy as np
import pandas as pd
from ase.build import molecule

from h3o_mlp.benchmark import metrics
from h3o_mlp.extract import pick_indices
from h3o_mlp.finetune import mace_command, split
from h3o_mlp.inventory import (bin_inventory, coverage,
                               hydrogen_fraction_from_path)


def test_composition():
    f = hydrogen_fraction_from_path
    assert f("a/bcc-h2o/350gpa") == 0.0
    assert abs(f("a/h3o-128mol/x") - 1 / 3) < 1e-9
    assert f("a/pure-h2/x") == 1.0
    assert np.isnan(f("a/10.20"))


def test_bins_and_coverage(tmp_path):
    inv = pd.DataFrame({"Path": ["a/h2o/1", "a/h2o/2"], "True_T_K": [5100, 4900],
                        "True_P_GPa": [310, 290], "x": [0.0, 0.0]})
    inv = bin_inventory(inv)
    assert set(inv.T_bin) == {5000.0} and set(inv.P_bin) == {300.0}
    h = tmp_path / "hull.csv"
    pd.DataFrame({"x": [0.0, 1.0], "P": [300, 300], "T": [5000, 5000]}).to_csv(h, index=False)
    cov = coverage(inv, h, "x", "P", "T")
    assert dict(zip(cov.x_bin, cov.n_runs)) == {0.0: 2, 1.0: 0}


def test_pick_indices_includes_last():
    assert pick_indices(100, 5)[-1] == 99 and len(pick_indices(3, 10)) == 3


def test_metrics():
    m = metrics([0.1, -0.1], [np.array([1.0, -1.0])], [np.ones(6)])
    assert m["force_RMSE_eV_per_A"] == 1.0 and m["stress_MAE_GPa"] == 1.0


def test_split_and_cmd(tmp_path):
    from ase.io import write
    p = tmp_path / "a.extxyz"
    write(p, [molecule("H2O") for _ in range(10)], format="extxyz")
    assert split(p, tmp_path / "t.xyz", tmp_path / "v.xyz") == (8, 2)
    assert "--forces_weight=100.0" in mace_command("t", "v")


def test_insertion_G_per_added_H_same_in_both_modes(tmp_path):
    """run_real.py insertion (toy model, 8 host H + 1 inserted H): the all-in-one process and the separate stage0 / node / collect route must report the same G per added H
    (the all-in-one path once divided by all 9 atoms instead of the added H: -186 instead of -1677 meV)"""
    import json
    import os
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); rr = os.path.join(root, "mlp_pipeline", "tdi", "run_real.py"); col = os.path.join(root, "mlp_pipeline", "tdi", "collect.py")
    args = ["toyice", "--phase", "insert", "--host-phase", "H", "--natoms", "8", "--vpa", "1.3", "--T", "3000", "--P", "400", "--bulk-gpa", "1000", "--nodes", "2", "--ps", "0.1",
            "--chunk-ps", "0.05", "--equil-ps", "0.05", "--ps1", "0.5", "--device", "cpu"]
    env = dict(os.environ, ALLOW_NOGPU="1"); one, nodes = str(tmp_path / "one"), str(tmp_path / "nodes")
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True, text=True, env=env)
    run(rr, *args, "--outdir", one)
    run(rr, *args, "--outdir", nodes, "--stage0")
    for k in (0, 1):
        run(rr, *args, "--outdir", nodes, "--node", str(k))
    tag = [f[:-len("_stage0.json")] for f in os.listdir(nodes) if f.endswith("_stage0.json")][0]
    run(col, nodes, tag, "--every-ps", "0.05")
    a, b = (json.load(open(os.path.join(d, f"tdi_{tag}.json"))) for d in (one, nodes))
    assert a["N"] == b["N"] == 1 and a["N_atoms_total"] == 9   # per added H, not per atom
    assert abs(a["G_meV_atom"] - b["G_meV_atom"]) < 1e-6 * abs(b["G_meV_atom"]), (a["G_meV_atom"], b["G_meV_atom"])
