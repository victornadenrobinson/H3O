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
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from common import progress, out, inp, path_phase_tag

MIN_WINDOW = 100


def o_structure(pos, cell):
    """O-sublattice type from TIME-AVERAGED O positions (thermal noise averaged out): bcc has 8 neighbours inside 1.08 d1 and 14 inside 1.22 d1,
    fcc 12 and 12 (d1 = median nearest-O distance). Only meaningful when O does not diffuse (not liquid). Returns (label, CN1, CN2)."""
    at = Atoms("O" * len(pos), positions=pos, cell=cell, pbc=True); at.wrap()
    if len(at) < 8:
        return "other", np.nan, np.nan, np.nan
    edge = abs(np.linalg.det(np.array(cell))) ** (1 / 3)
    rc = min(0.45 * edge, 1.6 * (abs(np.linalg.det(np.array(cell))) / len(at)) ** (1 / 3))
    i, d = neighbor_list("id", at, rc)
    nn = np.array([d[i == k].min() if (i == k).any() else np.nan for k in range(len(at))]); d1 = np.nanmedian(nn)
    cn1 = np.bincount(i[d < 1.08 * d1], minlength=len(at)); cn2 = np.bincount(i[d < 1.22 * d1], minlength=len(at))
    c1, c2 = float(np.median(cn1)), float(np.median(cn2))
    lab = "fcc" if abs(c1 - 12) <= 1 and abs(c2 - 12) <= 1 else "bcc" if abs(c1 - 8) <= 1 and 12 <= c2 <= 14.5 else "other"   # bcc: distorted cells split the 6 second neighbours
    return lab, c1, c2, float(d1)


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
    mO = sym == "O"
    if mO.any():
        start = frac[0, mO] @ cell                     # unw is the displacement from frame 0: add the start positions
        pos_O = unw[i0:, mO] + start
        out["struct"], out["CN1"], out["CN2"], d1 = o_structure(pos_O.mean(0), cell)
        # Lindemann-type ratio: rms O displacement about its own time average in the 2nd half, over the nearest-O distance.
        # Unlike the start-to-end MSD it does not depend on the window length (if the window spans several vibration periods)
        out["delta_L"] = float(np.sqrt(np.mean(np.sum((pos_O - pos_O.mean(0)) ** 2, axis=2)))) / d1
    else:                                              # pure-H runs: no O sublattice
        out["struct"], out["CN1"], out["CN2"], out["delta_L"] = "none", np.nan, np.nan, np.nan
    return out


def label(r, o_liq, h_super, h_liq, lind=0.0):
    if not np.isnan(r["O_rate"]):
        dl = r.get("delta_L", np.nan)
        liquid = (dl > lind) if (lind > 0 and not np.isnan(dl)) else r["O_rate"] > o_liq
        return "liquid" if liquid else ("superionic" if r["H_rate"] > h_super else "solid")
    return "liquid" if r["H_rate"] > h_liq else "solid"


def one(path, root):
    try:
        r = msd_rates(read(os.path.join(root, path), index=":", format="vasp-out"))
        return dict(Path=path, O_rate=np.nan, H_rate=np.nan, window=0) if r is None else dict(Path=path, O_rate=r["O"], H_rate=r["H"], window=r["window"], O_struct=r["struct"], CN1=r["CN1"], CN2=r["CN2"], delta_L=r["delta_L"])
    except Exception as e:
        return dict(Path=path, O_rate=np.nan, H_rate=np.nan, window=-1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out")
    ap.add_argument("--root", default="~/work"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tmin", type=float, default=2000); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="reuse rows already in the output file (skip re-reading those OUTCARs); relabels with the current thresholds")
    ap.add_argument("--o-liq", type=float, default=2.0, help="A^2 per 1000 steps")
    ap.add_argument("--lindemann", type=float, default=0.0, help="if > 0: liquid when the O Lindemann ratio exceeds this (typical melting value 0.12-0.2); 0 = use --o-liq on the O rate")
    ap.add_argument("--h-super", type=float, default=20.0); ap.add_argument("--h-liq", type=float, default=20.0)
    a = ap.parse_args()
    df = pd.read_csv(inp(a.inv))
    df = df[df.good & (df.True_T_K >= a.tmin) & (df.comp_snap != "unknown")]
    if a.sample:
        df = df.sample(min(a.sample, len(df)), random_state=0)
    root = os.path.abspath(os.path.expanduser(a.root))
    BASE = ["Path", "O_rate", "H_rate", "window", "O_struct", "CN1", "CN2", "delta_L"]
    old = pd.DataFrame(columns=BASE)
    fn = out(a.out)
    if a.reuse and os.path.exists(fn):   # keep rows mined earlier; only read OUTCARs that are not in the file yet
        o = pd.read_csv(fn); old = o[[c for c in BASE if c in o]]
        old = old[old.Path.isin(df.Path) & (old.window != -1)]
    todo = [p for p in df.Path if p not in set(old.Path)]
    print(f"{len(old)} runs reused from {fn}, {len(todo)} to mine")
    rows = []
    if todo:
        with Pool(a.workers) as p:
            rows = list(progress(p.imap(partial(one, root=root), todo, chunksize=4), total=len(todo), desc="phase MSD"))
    x = pd.concat([d for d in (old, pd.DataFrame(rows)) if len(d)], ignore_index=True)   # skip the empty frame: it would turn every column into dtype object
    for c in ("O_rate", "H_rate", "window", "CN1", "CN2", "delta_L"):
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    keep = [c for c in ("Path", "comp_snap", "True_T_K", "T_bin", "True_P_GPa", "N_ions") if c in df]
    x = x.merge(df[keep], on="Path", how="left")
    x["H_msd"] = x.H_rate * x.window / 1000; x["O_msd"] = x.O_rate * x.window / 1000   # total displacement^2 over the window (A^2)
    x["phase"] = [label(r, a.o_liq, a.h_super, a.h_liq, a.lindemann) if r["window"] > 0 else "unknown" for r in x.to_dict("records")]
    x.to_csv(out(a.out), index=False)
    print(x.phase.value_counts().to_string())
    print("O rate percentiles (A^2/1000 steps):", x.O_rate.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
    print("H rate percentiles:", x.H_rate.quantile([.05, .25, .5, .75, .95]).round(2).tolist())
    # thresholds should be judged against total displacement over the window, per temperature (H diffuses slower at 2000 K than at 7000 K)
    if "T_bin" in x:
        ok = x[x.window > 0]
        q = lambda c: ok.groupby("T_bin")[c].quantile([.1, .5, .9]).unstack().round(1)
        print("\nH_msd (A^2 over the window) by T bin: 10/50/90th percentile"); print(pd.concat([ok.groupby("T_bin").size().rename("n"), q("H_msd")], axis=1).to_string())
        print("\nO_msd (A^2 over the window) by T bin: 10/50/90th percentile"); print(pd.concat([ok.groupby("T_bin").size().rename("n"), q("O_msd")], axis=1).to_string())
        if "delta_L" in ok:
            print("\nO Lindemann ratio delta_L by T bin: 10/50/90th percentile (a gap between a low and a high cluster = crystal / liquid)")
            dq = ok.groupby("T_bin")["delta_L"].quantile([.1, .5, .9]).unstack().round(3)
            print(pd.concat([ok.groupby("T_bin").size().rename("n"), dq], axis=1).to_string())
            h, e = np.histogram(ok.delta_L.dropna(), bins=np.arange(0.0, 0.62, 0.02))
            print("histogram of delta_L over all runs (bin start: count) - a dip separates crystal from liquid:")
            print("  " + "  ".join(f"{b:.2f}:{c}" for b, c in zip(e[:-1], h) if c))
        print("\nwindow (MD steps) percentiles 10/50/90:", ok.window.quantile([.1, .5, .9]).round(0).tolist())
        nl = ok[ok.phase.isin(["solid", "superionic"]) & (ok.O_struct == "other")]
        print(f"\nnon-liquid runs whose O sublattice is neither bcc nor fcc: {len(nl)}; O_msd median {nl.O_msd.median():.2f}, CN1/CN2 value counts:")
        print(nl.groupby(["CN1", "CN2"]).size().sort_values(ascending=False).head(8).to_string())
    # calibration: how the path-word tags map onto the measured rates and O-sublattice types (tune --o-liq / --h-super from this table)
    x["tag"] = x.Path.map(path_phase_tag)
    print("\nby path tag: median / 90th percentile rate (A^2 per 1000 steps) and O-sublattice counts")
    g = x.groupby("tag")
    print(pd.DataFrame({"n": g.size(), "O_med": g.O_rate.median().round(2), "O_p90": g.O_rate.quantile(.9).round(2),
                        "H_med": g.H_rate.median().round(2), "H_p90": g.H_rate.quantile(.9).round(2)}).to_string())
    if "O_struct" in x:
        print("\nO sublattice by tag x phase (non-liquid runs only):")
        print(x[x.phase.isin(["solid", "superionic"])].pivot_table(index="tag", columns=["O_struct", "phase"], values="Path", aggfunc="count", fill_value=0).to_string())
