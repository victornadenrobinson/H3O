"""How big a cell fits on this GPU, for a given model? One frame per distinct atom count, ascending; records peak GPU memory and time per
energy+forces+stress call, plus neighbour/triplet counts so memory can be predicted for any frame.

usage (inside a GPU job, with the model's own python): python 13_memory_scaling.py mattersim valid_m.extxyz [--min-box 9] [--cut 5 --cut3 4]
Inference only. FORCE TRAINING needs more (second derivatives + optimizer state: expect ~2-4x), so a model that barely fits here will not train.
cut/cut3 are the two-body and three-body cutoffs used only for the triplet COUNT (assumed MatterSim/M3GNet defaults 5.0/4.0 A: check yours).
"""
import argparse, gc, time
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from common import inp
from smoke_test import get_calc


def counts(f, cut, cut3):
    i = neighbor_list("i", f, cut); n2 = np.bincount(i, minlength=len(f))
    i3 = neighbor_list("i", f, cut3); n3 = np.bincount(i3, minlength=len(f))
    return int(n2.sum()), int((n3 * np.maximum(n3 - 1, 0)).sum())


def main(a):
    try:
        import torch
    except ImportError:
        torch = None
    frames = read(inp(a.data), index=":")
    if a.min_box:
        frames = [f for f in frames if f.get_volume() ** (1 / 3) >= a.min_box]
    by = {}
    for f in frames:
        by.setdefault(len(f), []).append(f)
    calc = get_calc(a.model.split(":")[0], a.model.split(":")[1] if ":" in a.model else None, a.device)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch and torch.cuda.is_available() else float("nan")
    print(f"GPU memory {total:.1f} GiB; {len(frames)} frames, {len(by)} distinct sizes")
    print(f"{'atoms':>6} {'A3/atom':>8} {'pairs':>9} {'triplets':>11} {'peak_GiB':>9} {'sec':>6}  status")
    rows = []
    for nat in sorted(by):
        f = min(by[nat], key=lambda x: x.get_volume())   # densest frame of this size = worst case
        pairs, trip = counts(f, a.cut, a.cut3)
        g = f.copy(); g.calc = calc
        status, peak, dt = "ok", float("nan"), float("nan")
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            g.get_potential_energy(); g.get_forces(); g.get_stress(voigt=False)       # warm-up
            g.positions = f.positions + 1e-6
            t0 = time.time(); g.get_potential_energy(); g.get_forces(); g.get_stress(voigt=False); dt = time.time() - t0
            if torch and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
        except Exception as e:
            status = "OOM" if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError" else f"error: {str(e)[:60]}"
        g.calc = None; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{nat:6d} {f.get_volume() / nat:8.2f} {pairs:9d} {trip:11d} {peak:9.2f} {dt:6.2f}  {status}", flush=True)
        rows.append((nat, trip, peak, status))
    ok = [(t, p) for _, t, p, s in rows if s == "ok" and np.isfinite(p)]
    if len(ok) >= 2:
        t, p = np.array(ok).T
        k = float(np.sum(t * p) / np.sum(t * t))          # peak ~= k * triplets (through the origin)
        print(f"\nfit: peak_GiB ~= {k * 1e6:.2f} per million triplets (assumed cutoffs {a.cut}/{a.cut3} A)")
        print(f"=> on this GPU ({total:.0f} GiB) inference fits up to ~{0.9 * total / k / 1e6:.0f} million triplets; "
              f"force-training at 3x memory: ~{0.9 * total / (3 * k) / 1e6:.0f} million")
    bad = [r for r in rows if r[3] != "ok"]
    if bad:
        print(f"first failing size: {bad[0][0]} atoms ({bad[0][3]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--min-box", type=float, default=0.0)
    ap.add_argument("--cut", type=float, default=5.0); ap.add_argument("--cut3", type=float, default=4.0)
    main(ap.parse_args())
