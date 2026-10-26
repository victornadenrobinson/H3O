"""Label the dynamical phase of each run from its trajectory (MSD of O and H over the 2nd half of the run).
The restart segments of a run (<run>/out1/OUTCAR, <run>/out2/OUTCAR, ...) are joined into one trajectory first (--per-file turns this off);
a segment that does not continue the previous one (different cell/atoms or a jump > 1.5 A) starts a new piece and the longest piece is used.

usage: python 09_label_phase.py inventory_binned.csv phase.csv --root ~/work [--workers 64] [--tmin 2000] [--sample N]
NVT, fixed cell: positions are unwrapped via fractional minimum-image steps. Rates are A^2 per 1000 MD steps over the window
(2nd half, >= MIN_WINDOW steps). Labels: O rate > O_LIQ -> liquid; else H rate > H_SUPER -> superionic; else solid.
Pure-H runs: H rate > H_LIQ -> liquid else solid. Print percentiles first and tune the thresholds against known runs.
"""
import argparse, os, re
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
    return msd_arrays(np.array(frames[0].cell), np.array(frames[0].get_chemical_symbols()), np.array([f.get_scaled_positions(wrap=False) for f in frames]))


def msd_arrays(cell, sym, frac):
    L = len(frac)
    if L < 2 * MIN_WINDOW:
        return None
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    unw = np.concatenate([np.zeros((1,) + frac.shape[1:]), np.cumsum(d, axis=0)]) @ cell
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


def run_key(path):
    """Run directory: the folder above out<N>/ (restart segments out1, out2, ... of one MD run share it)."""
    parts = path.split("/")[:-1]
    if parts and re.fullmatch(r"out\d*", parts[-1]):
        parts = parts[:-1]
    return "/".join(parts)


def seg_no(path):
    m = re.search(r"/out(\d*)/[^/]*$", path)
    return int(m.group(1)) if m and m.group(1) else 0


def read_arrays(path, root):
    fr = read(os.path.join(root, path), index=":", format="vasp-out")
    return np.array(fr[0].cell), np.array(fr[0].get_chemical_symbols()), np.array([f.get_scaled_positions(wrap=False) for f in fr])


def join_segments(segs, max_jump=1.5, vol_tol=0.005):
    """Concatenate consecutive (cell, sym, frac) segments into one trajectory. A new piece starts where the atoms/cell differ or the first frame
    is more than max_jump A (minimum image) from the previous last frame; the longest piece is returned with the piece count."""
    pieces = [[segs[0]]]
    for s in segs[1:]:
        c0, y0, f0 = pieces[-1][-1]; c1, y1, f1 = s
        ok = len(y0) == len(y1) and bool((y0 == y1).all()) and abs(abs(np.linalg.det(c1)) / abs(np.linalg.det(c0)) - 1) < vol_tol
        if ok:
            d = f1[0] - f0[-1]; d -= np.round(d)
            ok = float(np.linalg.norm(d @ c0, axis=1).max()) < max_jump
        if ok:
            pieces[-1].append(s)
        else:
            pieces.append([s])
    best = max(pieces, key=lambda p: sum(len(x[2]) for x in p))
    return best[0][0], best[0][1], np.concatenate([x[2] for x in best]), len(pieces)


def one_chain(args):
    """All restart segments of one run, joined, analysed once; one result row per member OUTCAR path (same numbers)."""
    paths, root, max_steps = args
    segs = []
    for p in paths:
        try:
            segs.append(read_arrays(p, root))
        except Exception:
            pass
    if not segs:
        return [dict(Path=p, O_rate=np.nan, H_rate=np.nan, window=-1) for p in paths]
    cell, sym, frac, npieces = join_segments(segs)
    if max_steps and len(frac) > max_steps:
        frac = frac[-max_steps:]
    r = msd_arrays(cell, sym, frac)
    rows = []
    for p in paths:
        row = dict(Path=p, chain_steps=len(frac), n_seg=len(paths), n_pieces=npieces)
        if r is None:
            row.update(O_rate=np.nan, H_rate=np.nan, window=0)
        else:
            row.update(O_rate=r["O"], H_rate=r["H"], window=r["window"], O_struct=r["struct"], CN1=r["CN1"], CN2=r["CN2"], delta_L=r["delta_L"])
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out")
    ap.add_argument("--root", default="~/work"); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tmin", type=float, default=2000); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--reuse", action="store_true", help="reuse rows already in the output file (skip re-reading those OUTCARs); relabels with the current thresholds")
    ap.add_argument("--per-file", action="store_true", help="analyse every OUTCAR on its own instead of joining the restart segments (out1, out2, ...) of a run")
    ap.add_argument("--max-steps", type=int, default=20000, help="use at most the last N MD steps of a joined run (memory)")
    ap.add_argument("--o-liq", type=float, default=2.0, help="A^2 per 1000 steps")
    ap.add_argument("--lindemann", type=float, default=0.0, help="if > 0: liquid when the O Lindemann ratio exceeds this (typical melting value 0.12-0.2); 0 = use --o-liq on the O rate")
    ap.add_argument("--h-super", type=float, default=20.0); ap.add_argument("--h-liq", type=float, default=20.0)
    a = ap.parse_args()
    df = pd.read_csv(inp(a.inv))
    df = df[df.good & (df.True_T_K >= a.tmin) & (df.comp_snap != "unknown")]
    df = df.assign(run=df.Path if a.per_file else df.Path.map(run_key))
    if a.sample:   # sample whole runs, not single OUTCARs
        runs = pd.Series(df.run.unique()).sample(min(a.sample, df.run.nunique()), random_state=0)
        df = df[df.run.isin(runs)]
    root = os.path.abspath(os.path.expanduser(a.root))
    BASE = ["Path", "O_rate", "H_rate", "window", "O_struct", "CN1", "CN2", "delta_L", "chain_steps", "n_seg", "n_pieces"]
    old = pd.DataFrame(columns=BASE)
    fn = out(a.out)
    if a.reuse and os.path.exists(fn):   # keep rows mined earlier; only read runs that are not in the file yet
        o = pd.read_csv(fn); old = o[[c for c in BASE if c in o]]
        old = old[old.Path.isin(df.Path) & (old.window != -1)]
        if "chain_steps" not in old or old.chain_steps.isna().any():   # rows from the old per-file version are not joined-run results: drop them
            n0 = len(old); old = old[old.get("chain_steps", pd.Series(dtype=float)).notna()]
            print(f"dropped {n0 - len(old)} old per-file rows without chain information (they will be re-mined)")
    have = set(old.Path)
    todo = df[~df.Path.isin(have)]
    chains = [(sorted(g.Path, key=lambda p: (seg_no(p), p)), root, a.max_steps) for _, g in todo.groupby("run")]
    print(f"{len(old)} OUTCARs reused from {fn}; {len(chains)} runs ({len(todo)} OUTCARs) to mine; {df.run.nunique()} runs selected")
    rows = []
    if chains:
        with Pool(a.workers) as p:
            for r in progress(p.imap(one_chain, chains, chunksize=2), total=len(chains), desc="phase MSD"):
                rows += r
    x = pd.concat([d for d in (old, pd.DataFrame(rows)) if len(d)], ignore_index=True)   # skip the empty frame: it would turn every column into dtype object
    for c in ("O_rate", "H_rate", "window", "CN1", "CN2", "delta_L", "chain_steps", "n_seg", "n_pieces"):
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
        if "chain_steps" in ok:
            print("joined run length (MD steps) percentiles 10/50/90:", ok.chain_steps.quantile([.1, .5, .9]).round(0).tolist(), "; segments per run 50/90/max:", ok.n_seg.quantile([.5, .9, 1]).tolist(),
                  "; runs with a broken (discontinuous) chain:", int((ok.n_pieces > 1).sum()))
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
