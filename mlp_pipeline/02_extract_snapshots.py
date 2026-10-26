"""Extract decorrelated frames from selected OUTCARs to NESTED extxyz datasets (small <= medium <= large).

usage: python 02_extract_snapshots.py inventory_binned.csv snapshots [--tiers s=1x1,m=3x3,l=6x5] [--workers 32] [--root ~/work]
Writes snapshots_s.extxyz, snapshots_m.extxyz, snapshots_l.extxyz. A tier "RxF" = up to R runs per (composition, T bin,
100 GPa bin) and up to F frames per run. Runs/frames are ranked by a spread-out order (middle first, then fill the gaps),
so every smaller tier is a subset of the larger ones. Frames: 2nd half of each run, >= MIN_STEP_GAP MD steps apart.
Energy label = VASP free energy (force-consistent); E(sigma->0) kept in info['energy_sigma0'].
Stress is stored as full 3x3 (ASE convention, eV/A^3, excludes the ions' kinetic term).
"""
import argparse
import hashlib
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from ase.io import read, write
from common import progress, out, inp, phase_bucket

MIN_STEP_GAP = 50
CACHE_VERSION = "v2-free-energy"   # bump when the frame content/label convention changes: invalidates all cached runs
CACHE_DIR = None                   # set from --cache-dir; None disables caching
MIN_WIDTH = 0.0                    # set from --min-box: minimum perpendicular cell width (A), applied per run after reading the cell


def spread_order(n):
    """Order 0..n-1 so that every prefix is spread out (van der Corput, starting mid-range)."""
    order, i = [], 0
    while len(order) < n and i < 64 * n:
        f, k, w = 0.0, i, 0.5
        while k:
            f += w * (k & 1); k >>= 1; w /= 2
        j = int(((f + 0.5) % 1.0) * n)
        if j not in order:
            order.append(j)
        i += 1
    return order + [j for j in range(n) if j not in order]


def cache_file(path, n, root):
    """Cache entry for one OUTCAR: keyed by path, size, mtime, frame-selection params and label convention."""
    if not CACHE_DIR:
        return None
    try:
        st = os.stat(os.path.join(root, path))
    except OSError:
        return None
    key = hashlib.sha1(f"{CACHE_VERSION}|{path}|{st.st_size}|{st.st_mtime_ns}|{n}|{MIN_STEP_GAP}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, key[:2], key + ".extxyz")


def min_width(cell):
    m = np.array(cell); V = abs(np.linalg.det(m))
    return min(V / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3))


def pick(path, n, root=""):
    path, fr, err = _pick_cached(path, n, root)
    if not err and fr and MIN_WIDTH and min_width(fr[0].cell) < MIN_WIDTH:   # volume edge V^(1/3) overstates sheared/orthorhombic cells
        return path, [], f"narrow cell (width {min_width(fr[0].cell):.2f} A < {MIN_WIDTH})"
    return path, fr, err


def _pick_cached(path, n, root=""):
    cf = cache_file(path, n, root)
    if cf and os.path.exists(cf):
        try:
            return path, read(cf, index=":"), None
        except Exception:
            pass  # corrupt cache entry: fall through and recompute
    res = _pick(path, n, root)
    if cf and res[2] is None and res[1]:
        os.makedirs(os.path.dirname(cf), exist_ok=True)
        tmp = f"{cf}.{os.getpid()}.tmp"
        write(tmp, res[1], format="extxyz"); os.replace(tmp, cf)
    return res


def _pick(path, n, root=""):
    try:
        frames = read(os.path.join(root, path), index=":", format="vasp-out")
    except Exception as e:
        return path, [], str(e)
    L = len(frames)
    lo = L // 2
    if L - lo < 1:
        return path, [], "empty"
    k = min(n, max(1, (L - lo) // MIN_STEP_GAP))
    idx = np.unique(np.linspace(lo, L - 1, k).astype(int))
    out = []
    for rank, j in enumerate(spread_order(len(idx))):
        i = idx[j]
        a = frames[i]
        a.info.update(source=path, frame=int(i), frame_rank=rank)
        res = a.calc.results
        if res.get("free_energy") is not None:  # forces are derivatives of the free energy (TOTEN), not of E(sigma->0)
            a.info["energy_sigma0"] = float(res["energy"])
            res["energy"] = float(res["free_energy"])
        res["stress"] = a.get_stress(voigt=False)
        out.append(a)
    return path, out, None


def _set_cache(d, minw=0.0):
    global CACHE_DIR, MIN_WIDTH
    CACHE_DIR, MIN_WIDTH = d, minw


def pick_star(args):
    return pick(*args)  # (path, n, root)


def select(df, a):
    """Quality-filtered, state-balanced run selection. Uses the P-corrected inventory from 01 (inventory_binned.csv)."""
    d = df[df.good & (df.comp_snap != "unknown")]
    d = d[d.True_P_GPa.between(a.pmin, a.pmax) & (d.True_T_K >= a.tmin) & (d.P_std_GPa.fillna(0) <= a.max_pstd)]
    if a.min_box:
        box = d.True_V_A3 ** (1 / 3)
        print(f"--min-box {a.min_box} A: dropping {int((box < a.min_box).sum())} of {len(d)} runs (cells smaller than the 9-12 A rule); "
              f"box edge range kept {box[box >= a.min_box].min():.1f}-{box.max():.1f} A")
        d = d[box >= a.min_box]
    # production-only: composition whitelist + path exclusion rules (exclude_paths.txt, --exclude)
    if a.comps != "all":
        keep_c = set(a.comps.split(","))
        print(f"composition whitelist: dropping {int((~d.comp_snap.isin(keep_c)).sum())} runs ({sorted(set(d.comp_snap) - keep_c)})")
        d = d[d.comp_snap.isin(keep_c)]
    rules = list(a.exclude or [])
    if not a.no_default_exclusions and os.path.exists(a.exclude_file):
        for ln in open(a.exclude_file):
            ln = ln.split("#", 1)[0].strip()
            if ln:
                rules.append(ln)
    for rx in rules:
        hit = d.Path.str.contains(rx, regex=True, case=False)
        if hit.any():
            print(f"exclusion rule {rx!r}: dropping {int(hit.sum())} runs, e.g. {d.Path[hit].iloc[0]}")
        d = d[~hit]
    lam = d.Path.str.contains("lambda_", case=False)
    keep = d.Path.str.contains(a.lambda_keep) if a.lambda_keep else pd.Series(False, index=d.index)
    d = d[~lam | keep]
    d = d.assign(run=d.Path.map(lambda p: p.rsplit("/", 2)[0]), P100=(d.True_P_GPa // 100 * 100).astype(int))
    d = d.sort_values("Path").drop_duplicates("run")  # one OUTCAR per run directory (out1/out2... are a chain)
    # --by-phase: h2o runs are balanced separately per path-tag bucket (fcc / bcc / liquid / other) so fcc and bcc each get their own runs per state
    d = d.assign(phase_bucket=[phase_bucket(p) if (a.by_phase and c == "h2o") else "all" for p, c in zip(d.Path, d.comp_snap)])
    out = []
    for _, g in d.groupby(["comp_snap", "T_bin", "P100", "phase_bucket"]):
        g = g.sort_values("True_P_GPa")
        order = spread_order(len(g))[:a.max_runs]
        out.append(g.iloc[order].assign(run_rank=range(len(order))))
    return pd.concat(out) if out else d.iloc[:0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("out", help="output prefix, e.g. snapshots")
    ap.add_argument("--tiers", default="s=1x1,m=3x3,l=6x5", help="name=RUNSxFRAMES, nested (small first)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--root", default="~/work", help="dir that inventory paths are relative to")
    ap.add_argument("--pmin", type=float, default=200.0); ap.add_argument("--pmax", type=float, default=1450.0)
    ap.add_argument("--tmin", type=float, default=2000.0)
    ap.add_argument("--max-pstd", type=float, default=20.0)
    ap.add_argument("--cache-dir", default=None, help="per-run frame cache (default: runs/cache); reruns only extract new/changed OUTCARs")
    ap.add_argument("--no-cache", action="store_true", help="ignore and do not write the cache")
    ap.add_argument("--comps", default="h2o,h2.25o,h2.5o,h2.75o,h3o,h4o,h5o,h5.5o,h6o,h7o,h",
                    help="comma list of snapped compositions to keep ('all' = no filter); default = production H:O mixtures + pure H")
    ap.add_argument("--exclude", action="append", help="extra regex on the OUTCAR path to exclude (repeatable)")
    ap.add_argument("--exclude-file", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "exclude_paths.txt"))
    ap.add_argument("--no-default-exclusions", action="store_true")
    ap.add_argument("--min-box", type=float, default=0.0, help="drop runs whose cell edge V^(1/3) is below this many A (e.g. 9.0); 0 = keep all")
    ap.add_argument("--by-phase", action="store_true", help="h2o only: balance runs per (T, 100 GPa) bin separately for fcc / bcc / liquid / other path tags (adds info['phase_tag'])")
    ap.add_argument("--lambda-keep", default=None, help="regex of lambda_* paths to keep (default: drop all lambda_* runs)")
    a = ap.parse_args()
    tiers = [(n, *map(int, v.split("x"))) for n, v in (t.split("=") for t in a.tiers.split(","))]
    a.max_runs = max(t[1] for t in tiers); a.per_run = max(t[2] for t in tiers)
    root = os.path.abspath(os.path.expanduser(a.root))
    CACHE_DIR = None if a.no_cache else (os.path.expanduser(a.cache_dir) if a.cache_dir else out("cache"))
    df = select(pd.read_csv(inp(a.inv)), a)
    meta = df.set_index("Path")
    if CACHE_DIR:
        have = sum(1 for p in df.Path if (cf := cache_file(p, a.per_run, root)) and os.path.exists(cf))
        print(f"cache {CACHE_DIR}: {have} of {len(df)} selected runs already cached")
    print(f"{len(df)} runs selected in {df.groupby(['comp_snap', 'T_bin']).ngroups} (composition, T) states")
    print(df.pivot_table(index="comp_snap", columns="T_bin", values="Path", aggfunc="count", fill_value=0).to_string())
    allf = []; n_narrow = 0
    with Pool(a.workers, initializer=_set_cache, initargs=(CACHE_DIR, a.min_box)) as p:
        for path, fr, err in progress(p.imap(pick_star, [(x, a.per_run, root) for x in df.Path]), total=len(df), desc="extracting"):
            if err:
                if err.startswith("narrow"):
                    n_narrow += 1
                else:
                    print("skip", path, err)
                continue
            m = meta.loc[path]
            for f in fr:
                f.info.update(config_type=f"{m.comp_snap}_{int(m.T_bin)}K", T_K=float(m.True_T_K), P_GPa=float(m.True_P_GPa), box_A=round(float(m.True_V_A3) ** (1 / 3), 3),
                              run_rank=int(m.run_rank), phase_tag=str(m.phase_bucket))
            allf += fr
    if a.min_box:
        print(f"{n_narrow} runs dropped because the minimum perpendicular cell width is below {a.min_box} A (sheared/orthorhombic cells)")
    for name, nr, nf in tiers:
        sub = [f for f in allf if f.info["run_rank"] < nr and f.info["frame_rank"] < nf]
        fn = out(f"{a.out}_{name}.extxyz")
        write(fn, sub, format="extxyz")
        print(f"tier {name} ({nr} runs x {nf} frames): {len(sub)} frames -> {fn}")
