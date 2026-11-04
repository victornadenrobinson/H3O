#!/usr/bin/env python3
"""Finite-size table for PWidom pure-H (or any phase H) results: G_H(N) = mu_P(N) - fs(N) should not depend on N (fs = kT [ln(N+1) - ln(N!)/N - 1], the ideal-gas finite-size term used by dg).
   python3 mlp_pipeline/pwidom_finite_size.py runs/fs_pureH/*/pw_pureH_*_candidate_K*.json [--ref 729]
Pressure correction: robust by default (--pressure-mode measured for the old behaviour); the mode is printed. Prints per result: N, cell edge, P_host, mu_ex (BAR), mu_P, fs, G_H and its error, Widom - deletion gap, Kish, and G_H relative to the --ref size (default: the largest N); runs of equal N (seeds) are also averaged."""
import argparse, collections, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdi"))
import pwidom_scan as scan          # the same pressure correction as the scan / launcher / hull tools
KB = 8.617333262e-5
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--ref", type=int, default=0); ap.add_argument("--pressure-mode", choices=("robust", "measured"), default="robust", help="pressure correction to P_target (default robust, as everywhere else; measured is noisy in short runs)"); a = ap.parse_args()
rows = []
for g in a.files:
    for f in sorted(glob.glob(g)):
        w = json.load(open(f))
        if w.get("phase") != "H": continue
        N = int(w["N_H"]); kT = KB * w["T"] * 1e3; fs = kT * (math.log(N + 1) - math.lgamma(N + 1) / N - 1)
        muP, muPe = scan.mu_P(w, a.pressure_mode); rows.append(dict(N=N, L=w["V"] ** (1 / 3), P=w["P_host_GPa"], mex=w["mu_ex_bar_meV"], muP=muP, fs=fs, G=muP - fs, e=muPe, gap=w["mu_ex_widom_meV"] - w["mu_ex_delete_meV"], kish=w["kish_frac_ins"], nf=w["n_frames_host"], f=f))
if not rows: raise SystemExit("no phase-H results found")
rows.sort(key=lambda r: (r["N"], r["f"]))
print(f"pressure correction: {a.pressure_mode}")
print(f"{'N':>5} {'L(A)':>6} {'P host':>7} {'mu_ex':>8} {'mu_P':>8} {'fs':>6} {'G_H':>8} {'+-':>5} {'W-D':>6} {'Kish':>5} {'frames':>6}  file")
for r in rows: print(f"{r['N']:>5d} {r['L']:>6.2f} {r['P']:>7.1f} {r['mex']:>8.1f} {r['muP']:>8.1f} {r['fs']:>6.1f} {r['G']:>8.1f} {r['e']:>5.1f} {r['gap']:>6.0f} {r['kish']:>5.2f} {r['nf']:>6d}  {r['f'].split('/')[-2] if '/' in r['f'] else r['f']}")
by = collections.defaultdict(list)
for r in rows: by[r["N"]].append(r)
ref = a.ref if a.ref in by else max(by)
if a.ref and a.ref not in by: print(f"(no N = {a.ref} result yet: comparing with N = {ref})")
gref = np.mean([r["G"] for r in by[ref]])
print(f"\nG_H per size (mean over runs) relative to N = {ref}: should be ~0 (within the errors) if N = {ref} and below are converged")
for N in sorted(by):
    g = np.array([r["G"] for r in by[N]]); sd = g.std(ddof=1) if len(g) > 1 else float("nan")
    print(f"  N {N:>5d}  L {by[N][0]['L']:5.2f} A  runs {len(g)}  G_H {g.mean():8.1f} meV  vs ref {g.mean() - gref:+7.1f}  (run-to-run sd {sd:5.1f})")
