"""How big a cell fits on this GPU, for a given model? One frame per distinct atom count, ascending; records peak GPU memory and time per
energy+forces+stress call, plus neighbour/triplet counts so memory can be predicted for any frame.

usage (inside a GPU job, with the model's own python): python 13_memory_scaling.py mattersim valid_m.extxyz [--min-box 9] [--cut 5 --cut3 4]
Inference only. MLP_DTYPE=float32 halves MACE memory. GRACE (TensorFlow): memory comes from tf.config get_memory_info (growth mode); use --cut 6. FORCE TRAINING needs more (second derivatives + optimizer state: expect ~2-4x), so a model that barely fits here will not train.
cut/cut3 are the two-body and three-body cutoffs used only for the triplet COUNT (assumed MatterSim/M3GNet defaults 5.0/4.0 A: check yours).
"""
import argparse, gc, os, subprocess, threading, time
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")   # GRACE (TensorFlow) otherwise preallocates the whole GPU and memory cannot be read
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from common import inp
from smoke_test import get_calc


def counts(f, cut, cut3):
    i = neighbor_list("i", f, cut); n2 = np.bincount(i, minlength=len(f))
    i3 = neighbor_list("i", f, cut3); n3 = np.bincount(i3, minlength=len(f))
    return int(n2.sum()), int((n3 * np.maximum(n3 - 1, 0)).sum())


class SmiPoller:
    """Peak GPU memory used on device 0 (all processes, any framework) by polling nvidia-smi in a thread: the safety net for TF/XLA."""
    def __init__(self):
        self.peak, self.run = float("nan"), False

    def _read(self):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=5).stdout
            return float(o.strip().split("\n")[0]) / 1024
        except Exception:
            return float("nan")

    def __enter__(self):
        self.base = self._read(); self.peak = self.base; self.run = True
        def loop():
            while self.run:
                v = self._read()
                if v == v and (self.peak != self.peak or v > self.peak):
                    self.peak = v
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.run = False; self.t.join(timeout=5)

    @property
    def used(self):
        return self.peak - self.base


def model_dtype(calc):
    """Best-effort dtype report for torch or TF models behind an ASE calculator."""
    out = set()
    def scan(o, depth=0):
        if depth > 2 or o is None:
            return
        try:
            import torch
            if isinstance(o, torch.nn.Module):
                out.add(str(next(o.parameters()).dtype)); return
        except Exception:
            pass
        for k in ("models", "model", "potential"):
            v = getattr(o, k, None)
            if isinstance(v, (list, tuple)):
                [scan(x, depth + 1) for x in v]
            else:
                scan(v, depth + 1)
        for k in dir(o):
            if "dtype" in k.lower() and not k.startswith("__"):
                try:
                    out.add(f"{k}={getattr(o, k)}")
                except Exception:
                    pass
    scan(calc)
    return ", ".join(sorted(out)) or "unknown"


def tf_mem(reset=False):
    try:
        import tensorflow as tf
        if reset:
            tf.config.experimental.reset_memory_stats("GPU:0")
            return 0.0
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
    except Exception:
        return float("nan")


def main(a):
    try:
        import torch
    except ImportError:
        torch = None
    is_tf = a.model.startswith("grace")
    frames = read(inp(a.data), index=":")
    if a.min_box:
        frames = [f for f in frames if f.get_volume() ** (1 / 3) >= a.min_box]
    by = {}
    for f in frames:
        by.setdefault(len(f), []).append(f)
    calc = get_calc(a.model.split(":")[0], a.model.split(":")[1] if ":" in a.model else None, a.device)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch and torch.cuda.is_available() else float("nan")
    if is_tf and total != total:
        total = float(os.environ.get("GPU_GIB", 40))   # no torch in the GRACE env: assume the 40 GB A100 (override with GPU_GIB)
    print(f"GPU memory {total:.1f} GiB; {len(frames)} frames, {len(by)} distinct sizes; model dtype: {model_dtype(calc)}")
    print(f"{'atoms':>6} {'A3/atom':>8} {'pairs':>9} {'triplets':>11} {'peak_GiB':>9} {'smi_abs_GiB':>12} {'sec':>6}  status")
    rows = []
    for nat in sorted(by):
        f = min(by[nat], key=lambda x: x.get_volume())   # densest frame of this size = worst case
        pairs, trip = counts(f, a.cut, a.cut3)
        g = f.copy(); g.calc = calc
        status, peak, dt, smi = "ok", float("nan"), float("nan"), float("nan")
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            if is_tf:
                tf_mem(reset=True)
            g.get_potential_energy(); g.get_forces(); g.get_stress(voigt=False)       # warm-up
            g.positions = f.positions + 1e-6
            with SmiPoller() as sp:
                t0 = time.time(); g.get_potential_energy(); g.get_forces(); g.get_stress(voigt=False); dt = time.time() - t0
            smi = sp.peak   # absolute GPU memory in use by everything on the device (includes the framework's cache): an upper bound
            if is_tf:
                peak = tf_mem()
            elif torch and torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 2**30
        except Exception as e:
            status = "OOM" if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError" else f"error: {str(e)[:60]}"
        g.calc = None; gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{nat:6d} {f.get_volume() / nat:8.2f} {pairs:9d} {trip:11d} {peak:9.2f} {smi:12.2f} {dt:6.2f}  {status}", flush=True)
        rows.append((nat, trip, peak if peak == peak else smi, status))
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
