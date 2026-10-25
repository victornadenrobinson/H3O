"""Extract decorrelated DFT snapshots (positions, E, F, stress) to extxyz."""
import gzip
import os
import tempfile

import numpy as np
from ase.io import read, write


def pick_indices(n_frames, n_snap):
    """Evenly spaced frame indices ending at the final frame (index -1)."""
    n_snap = min(n_snap, n_frames)
    if n_snap <= 0:
        return []
    return sorted(set(np.linspace(n_frames - 1, 0, n_snap).round().astype(int)))


def _read_outcar(path):
    if str(path).endswith(".gz"):
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, "OUTCAR")
            with gzip.open(path, "rb") as f, open(tmp, "wb") as g:
                g.write(f.read())
            return read(tmp, index=":", format="vasp-out")
    return read(path, index=":", format="vasp-out")


def extract_snapshots(path, n_snap=10, skip_frac=0.5, meta=None):
    """Return up to n_snap frames from the equilibrated part of a run."""
    frames = _read_outcar(path)
    start = int(len(frames) * skip_frac)
    frames = frames[start:] or frames[-1:]
    out = []
    for i in pick_indices(len(frames), n_snap):
        a = frames[i]
        res = a.calc.results
        at = a.copy()
        at.info["energy"] = float(res["energy"])
        at.arrays["forces"] = np.asarray(res["forces"])
        if "stress" in res:
            at.info["stress"] = np.asarray(res["stress"])
        at.info["source"] = str(path)
        at.info.update(meta or {})
        out.append(at)
    return out


def build_dataset(inv, out_path, n_snap=10, **filters):
    """Extract snapshots for each row of a (filtered) binned inventory."""
    for k, v in filters.items():
        inv = inv[inv[k] == v]
    frames = []
    for _, r in inv.iterrows():
        try:
            frames += extract_snapshots(
                r["Path"], n_snap,
                meta={"P_GPa": r["True_P_GPa"], "T_K": r["True_T_K"], "x": r["x"]},
            )
        except Exception as e:  # corrupt/truncated OUTCAR
            print(f"skip {r['Path']}: {e}")
    write(out_path, frames, format="extxyz")
    return len(frames)
