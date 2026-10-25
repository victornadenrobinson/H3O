"""Label the dynamical phase of each run from its trajectory (MSD of O and H over the 2nd half of the run).

usage: python 09_label_phase.py inventory_binned.csv phase.csv --root ~/work [--workers 64] [--tmin 2000] [--sample N]
NVT, fixed cell: positions are unwrapped via fractional minimum-image steps. Rates are A^2 per 1000 MD steps over the window
(2nd half, >= MIN_WINDOW steps). Labels: O rate > O_LIQ -> liquid; else H rate > H_SUPER -> superionic; else solid.
Pure-H runs: H rate > H_LIQ -> liquid else solid. Print percentiles first and tune the thresholds against known runs.
"""
import argparse, os
from multiprocessing import Pool
from functools import partial
import numpy as np
import pandas as pd
from ase.io import read
from common import progress, out, inp

MIN_WINDOW = 100


def msd_rates(frames):
    """O and H MSD rates (A^2 / 1000 steps) over the second half; fixed-cell, mic-unwrapped."""
    L = len(frames)
    if L < 2 * MIN_WINDOW:
        return None
    cell = np.array(frames[0].cell)
    frac = np.array([f.get_scaled_positions(wrap=False) for f in frames])
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    unw = np.concatenate([np.zeros((1,) + frac.shape[1:]), np.cumsum(d, axis=0)]) @ cell
    sym = np.array(frames[0].get_chemical_symbols())
    i0 = L // 2
    out = {}
    for el in ("O", "H"):
        m = sym == el
        out[el] = float(np.mean(np.sum((unw[-1, m] - unw[i0, m]) ** 2, axis=1)) / (L - 1 - i0) * 1000) if m.any() else np.nan
    out["window"] = L - 1 - i0
    return out


def label(r, o_liq, h_super, h_liq):
    if not np.isnan(r["O"]):
        return "liquid" if r["O"] > o_liq else ("superionic" if r["H"] > h_super else "solid")
    return "liquid" if r["H"] > h_liq else "solid"


def one(path, root):
    try:
        r = msd_rates(read(os.path.join(root, path), index=":", format="vasp-out"))
        return dict(Path=path, O_rate=np.nan, H_rate=np.nan, window=0) if r is None else dict(Path=path, O_rate=r["O"], H_rate=r["H"], window=r["window"])
    except Exception as e:
        return dict(Path=path, O_rate=np.nan, H_rate=np.nan, window=-1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out")
    ap.add_argument("--root", default="~/work"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tmin", type=float, default=2000); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--o-liq", type=float, default=2.0, help="A^2 per 1000 steps")
    ap.add_argument("--h-super", type=float, default=20.0); ap.add_argument("--h-liq", type=float, default=20.0)
    a = ap.parse_args()
    df = pd.read_csv(inp(a.inv))
    df = df[df.good & (df.True_T_K >= a.tmin) & (df.comp_snap != "unknown")]
    if a.sample:
        df = df.sample(min(a.sample, len(df)), random_state=0)
    root = os.path.abspath(os.path.expanduser(a.root))
    with Pool(a.workers) as p:
        rows = list(progress(p.imap(partial(one, root=root), list(df.Path), chunksize=4), total=len(df), desc="phase MSD"))
    x = pd.DataFrame(rows)
    x["phase"] = [label(r, a.o_liq, a.h_super, a.h_liq) if r["window"] > 0 else "unknown" for r in x.to_dict("records")]
    x.to_csv(out(a.out), index=False)
    print(x.phase.value_counts().to_string())
    print("O rate percentiles (A^2/1000 steps):", x.O_rate.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
    print("H rate percentiles:", x.H_rate.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
