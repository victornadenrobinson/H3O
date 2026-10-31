"""Smoke test of the proxy-fit label path (seconds): load the real model, evaluate one cached host frame and the same frame + 1 H, print dU, timing and GPUs.

usage (repo root, in the GPU env):  python mlp_pipeline/tdi/proxy_smoke.py [grace-ft:runs/grace/ft_m2] [runs/pwidom/pw_fcc_500GPa_6000K_frames.npz]
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE)]
model = sys.argv[1] if len(sys.argv) > 1 else "grace-ft:runs/grace/ft_m2"
src = sys.argv[2] if len(sys.argv) > 2 else "runs/pwidom/pw_fcc_500GPa_6000K_frames.npz"

import tensorflow as tf
print("tensorflow", tf.__version__, "GPUs:", [d.name for d in tf.config.list_physical_devices("GPU")], flush=True)
from ase import Atoms
from pwidom import Model

z = np.load(src); pos = z["host"][0]; nums = z["numbers"]; L = float(z["V"]) ** (1 / 3); cell = np.eye(3) * L
print(f"{src}: {len(nums)} atoms ({int((nums == 1).sum())} H), cell {L:.3f} A, T {float(z['T']):g} K", flush=True)
t0 = time.time(); m = Model(model, "cuda", 4); print(f"model loaded in {time.time() - t0:.1f} s ({type(m.calc).__name__})", flush=True)
rng = np.random.default_rng(0)
host = Atoms(numbers=nums, positions=pos, cell=cell, pbc=True)
ins = [Atoms(numbers=np.r_[nums, 1], positions=np.vstack([pos, rng.random(3) @ cell]), cell=cell, pbc=True) for _ in range(3)]
for rep in range(2):
    t0 = time.time(); E = m.energies([host] + ins); dt = time.time() - t0
    print(f"call {rep + 1}: {len(E)} structures in {dt:.2f} s; E0 {E[0]:.4f} eV, dU of 3 random points {np.round(E[1:] - E[0], 3)} eV", flush=True)
assert np.all(np.isfinite(E)), "non-finite energies"
print("SMOKE OK")
