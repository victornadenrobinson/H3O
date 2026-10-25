"""Adjust a gracemaker input.yaml for a first smoke fine-tune (keeps a .orig copy; comments are not preserved).
usage: ~/grace-venv/bin/python mlp_pipeline/grace_ft_set.py runs/grace/input.yaml --updates 400 --batch 2 [--lr 1e-3]
"""
import argparse, shutil, yaml

ap = argparse.ArgumentParser()
ap.add_argument("yaml"); ap.add_argument("--updates", type=int, default=400); ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--lr", type=float, default=None)
a = ap.parse_args()
shutil.copy(a.yaml, a.yaml + ".orig")
d = yaml.safe_load(open(a.yaml))
f = d.setdefault("fit", {})
f["target_total_updates"] = a.updates
f["batch_size"] = a.batch
f["test_batch_size"] = a.batch
if a.lr is not None and isinstance(f.get("opt_params"), dict):
    f["opt_params"]["learning_rate"] = a.lr
lr = (f.get("opt_params") or {}).get("learning_rate")
sp = f.get("scheduler_params")
if lr and isinstance(sp, dict) and sp.get("minimum_learning_rate", 1e-4) >= lr:
    sp["minimum_learning_rate"] = lr / 12   # the schedule must decrease (docs)
yaml.safe_dump(d, open(a.yaml, "w"), sort_keys=False, default_flow_style=False)
print("potential:", d.get("potential"))
print("data:", {k: d["data"][k] for k in d.get("data", {}) if k in ("filename", "test_filename", "reference_energy", "stress_units")})
print("fit:", {k: f.get(k) for k in ("loss", "batch_size", "target_total_updates", "optimizer", "opt_params", "scheduler", "scheduler_params", "trainable_variable_names", "eval_init_stats")})
