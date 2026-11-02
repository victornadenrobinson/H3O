"""Guess the volume per atom at low pressures (default 50, 100, 200 GPa) for every (H/O, branch, T) of the MLP EOS scan, and print the 17_eos_md.py commands that run GRACE-MD there.
The EOS scans (runs/eos_scan, runs/eos_map: eos_*.csv written by 17_eos_md.py) mostly start at ~100-200 GPa; this extrapolates ln P(ln V) (quadratic fit to the lowest-pressure half of the runs, at least 4 runs)
and flags every target below the measured range as EXTRAPOLATED. The MLP pressure of the new runs is in their thermo / csv: the guess only has to be close, the snapshots are labelled with DFT anyway.
usage (repo root; Young or anywhere the csvs are):  python mlp_pipeline/eos_low_p.py runs/eos_scan runs/eos_map [--P 50 100 200] [--model grace-ft:runs/grace/ft_m2/seed/1/final_model] [--ps 1.5] > low_p_runs.sh
Output: a table to stderr, one `17_eos_md.py` command per (H/O, branch, T) to stdout (volumes of the 3 pressures in one call).
Solid branch = lattice start (hold-O 0), liquid = premelt. Pure H (phase H) is listed too; for the 729-atom pure-H cells use build_toy_cells.py --sys H --vpa <v> instead.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--P", type=float, nargs="+", default=[50, 100, 200]); ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2/seed/1/final_model"); ap.add_argument("--ps", type=float, default=1.5)
    a = ap.parse_args(); rows = []
    for d in a.dirs:
        for f in sorted(glob.glob(os.path.join(d, "eos_*.csv"))):
            try: rows.append(pd.read_csv(f))
            except Exception: pass
    if not rows: sys.exit("no eos_*.csv found")
    df = pd.concat(rows, ignore_index=True)
    for c, v in (("branch", "solid"), ("hper", 2.0)):
        if c not in df: df[c] = v
    df = df[np.isfinite(df.vpa) & (df.P_GPa > 1)]; out = []
    print(f"{'phase':5s} {'H/O':>5s} {'branch':7s} {'T':>5s} {'runs':>4s} {'Pmeas':>12s} " + " ".join(f"v@{p:g}".rjust(9) for p in a.P) + "  flag", file=sys.stderr)
    for (ph, h, b, T), g in df.groupby(["phase", "hper", "branch", "T_set"]):
        if len(g) < 4: continue
        g = g.sort_values("P_GPa").iloc[: max(4, len(g) // 2 + 1)]                    # lowest-pressure half: the part closest to the extrapolation
        x, y = np.log(g.vpa.values), np.log(g.P_GPa.values); c = np.polyfit(y, x, 2 if len(g) >= 5 else 1)   # ln V as a function of ln P: monotone, extrapolates smoothly
        v = [float(np.exp(np.polyval(c, np.log(P)))) for P in a.P]; ex = ["EXTRAPOLATED" if P < g.P_GPa.min() * 0.97 else "" for P in a.P]
        flag = ",".join(f"{P:g}" for P, e in zip(a.P, ex) if e) or "interpolated"
        print(f"{ph:5s} {h:5.2f} {b:7s} {T:5.0f} {len(g):4d} {g.P_GPa.min():5.0f}-{g.P_GPa.max():5.0f} " + " ".join(f"{x_:9.3f}" for x_ in v) + f"  {flag}", file=sys.stderr)
        nrep = 5 if ph == "H" else (3 if ph == "fcc" else 4)
        print(f"python mlp_pipeline/17_eos_md.py {a.model} --phase {ph} --hper {h:g} --branch {b} --T {T:.0f} --nrep {nrep} --ps {a.ps} --equil-ps 0.5 --dt 1.0 --vpa {' '.join(f'{x_:.4f}' for x_ in v)} --tag lowP_{ph}_h{h:g}_{b}_{T:.0f}K --outdir runs/eos_lowp")


if __name__ == "__main__":
    main()
