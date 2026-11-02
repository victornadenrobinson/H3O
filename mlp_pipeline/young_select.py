"""Pick DFT-labelling candidates out of the Young inventory (young_inventory.csv from gen_inventory.py) and write ONE small extxyz to copy to Perlmutter.
Reads pwidom *_frames.npz (host / full ensembles, cubic cells) and extxyz trajectories. Excludes toy-potential runs (toyscan, toychecks) and lambda-node TDI trajectories (tdi*/), whose frames are not physical ensembles, unless --allow-tdi.
usage (Young, repo root):
  python mlp_pipeline/young_select.py young_inventory.csv --runs runs --kind pwidom-host pwidom-full --P 400 600 --T 4000 6000 --ratio 2.0 2.6 --per-file 4 --out sel_pwidom.extxyz
  python mlp_pipeline/young_select.py young_inventory.csv --runs runs --kind extxyz --path-re 'melt_comp/.*_(free|hold)_' --T 4000 6000 --P 300 600 --per-file 2 --out sel_meltcomp.extxyz
  python mlp_pipeline/young_select.py young_inventory.csv --runs runs --kind extxyz --path-re 'eos_scan/' --ratio 2.2 3.1 --per-file 2 --max-vpa 2.4 --out sel_eos.extxyz   (v/atom from the file name _v<vpa>)
Frames per file are evenly spaced (extxyz: over the second half of the trajectory; npz: over all cached frames). Split: whole files go to train (2/3) or valid (1/3) by a hash of the path.
Output frame info: source, src_frame, kind, P_target (name), T_K, split, nO, nH, config_type; add --dry to list the chosen files and frame counts only.
"""
import argparse, hashlib, os, re, sys
import numpy as np, pandas as pd

EXCL_ALWAYS = ("toyscan", "toychecks"); EXCL_TDI = ("tdi/", "tdi_", "/tdi")


def ratio(comp):
    m = re.match(r"O(\d+)H(\d+)", str(comp)); return (int(m.group(2)) / int(m.group(1))) if m and int(m.group(1)) else np.inf


def pick(n, k, second_half):
    lo = n // 2 if second_half else 0; return sorted(set(np.linspace(lo, n - 1, k).round().astype(int))) if n > 0 else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv"); ap.add_argument("--runs", default="runs"); ap.add_argument("--kind", nargs="+", default=["extxyz"]); ap.add_argument("--P", nargs=2, type=float); ap.add_argument("--T", nargs=2, type=float)
    ap.add_argument("--ratio", nargs=2, type=float, help="H/O range (pure H = inf is excluded unless --ratio 0 inf)"); ap.add_argument("--path-re"); ap.add_argument("--max-vpa", type=float); ap.add_argument("--per-file", type=int, default=3)
    ap.add_argument("--allow-tdi", action="store_true"); ap.add_argument("--as-hosts", action="store_true", help="write host frames for build_insertion_set.py: config_type like the tiers (h2o_5000K), P_GPa, T_K"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv); df = df[df.kind.isin(a.kind)]
    df = df[~df.path.str.contains("|".join(EXCL_ALWAYS))]
    if not a.allow_tdi: df = df[~df.path.apply(lambda p: any(x in p for x in EXCL_TDI))]
    if a.P: df = df[df.P.between(*a.P)]
    if a.T: df = df[df["T"].between(*a.T)]
    if a.path_re: df = df[df.path.str.contains(a.path_re, regex=True)]
    if a.ratio: df = df[df.comp.apply(ratio).between(*a.ratio)]
    if a.max_vpa: df = df[df.path.apply(lambda p: (lambda m: m is not None and float(m.group(1)) <= a.max_vpa)(re.search(r"_v(\d+\.\d+)", p)))]
    df = df[(df.frames > 0) & ~df.path.str.endswith("final.extxyz")]
    print(f"{len(df)} files selected, {int(df.frames.sum())} frames available; taking up to {a.per_file} per file", file=sys.stderr)
    if a.dry:
        print(df.groupby(["kind", "comp"]).agg(files=("path", "size"), frames=("frames", "sum")).to_string()); return
    from ase import Atoms
    from ase.io import iread, write
    if os.path.exists(a.out): os.remove(a.out)
    nout = 0
    for r in df.itertuples():
        path = os.path.join(a.runs, r.path); split = "valid" if int(hashlib.md5(r.path.encode()).hexdigest(), 16) % 3 == 0 else "train"
        base = dict(source=r.path, kind=r.kind, P_target=(float(r.P) if np.isfinite(r.P) else np.nan), T_K=(float(r.T) if np.isfinite(r.T) else np.nan), split=split)
        frames = []
        if path.endswith(".npz"):
            z = np.load(path); ens = "full" if r.kind.endswith("full") else "host"; pos = z[ens]; nums = z["numbers"] if ens == "host" else np.r_[z["numbers"], 1]
            cell = z["cell"] if "cell" in z.files else np.eye(3) * float(z["V"]) ** (1 / 3)
            for i in pick(len(pos), a.per_file, False): frames.append((i, Atoms(numbers=nums, positions=pos[i], cell=cell, pbc=True)))
        else:
            fr = list(iread(path, index=":")); 
            for i in pick(len(fr), a.per_file, True): frames.append((i, fr[i]))
        for i, at in frames:
            at = at.copy(); at.calc = None; at.info = {}; at.arrays.pop("forces", None)
            nO, nH = int((at.numbers == 8).sum()), int((at.numbers == 1).sum())
            at.info.update(base, src_frame=int(i), nO=nO, nH=nH, config_type=f"{r.kind.split('-')[0]}_{nH}H{nO}O_{base['T_K']:.0f}K")
            if a.as_hosts:
                rr = nH / nO if nO else np.inf; lab = "h" if not nO else "h%go" % min((2.0, 2.25, 2.5, 2.75, 3.0), key=lambda x: abs(x - rr)) if min(abs(rr - x) / x for x in (2.0, 2.25, 2.5, 2.75, 3.0)) < 0.04 else "h%.3fo" % rr
                at.info.update(config_type=f"{lab}_{int(round(base['T_K'] / 1000.0) * 1000)}K", P_GPa=base["P_target"])
            write(a.out, at, append=True); nout += 1
    print(f"wrote {nout} frames to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
