#!/usr/bin/env python3
"""Test A2 on DFT-ensemble frames: composition error c per added H, from a water host, a water + H host and the pure-H reservoir at the same (P, T).
   c = [ Ferr(host + n H) - Ferr(host) - n * Ferr_per_H(pure H) ] / n,   Ferr = free-energy error of the model relative to DFT.
Frames sampled from the DFT ensemble (dft_md_labelled.py), D = E_DFT - E_model per cell, Zwanzig (model -> DFT, sampled from DFT):  F_DFT - F_model = <D> + var(D)/(2kT) + ...
 mean-only:   Ferr = <D>             with var/2kT:  Ferr = <D> + var(D)/(2kT)     (pure H per atom = Ferr_cell / natoms)
Error bars: bootstrap over frames (2000 resamples). Pass (docs/MODEL_TEST_SUITE.md A2): |c| <= 50 meV/H.
   python3 mlp_pipeline/composition_check.py runs/Dmd_<model>.csv [more.csv]   (baseline_D.py output on runs/dft_md_*.extxyz; sets M400, M500, M600)"""
import argparse, collections, csv, os
import numpy as np
KB = 8.617333e-5
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args(); rng = np.random.default_rng(1)
for p in a.files:
    g = collections.defaultdict(list); T = {}
    for r in csv.DictReader(open(p)):
        g[(r["set"], int(r["nO"]), int(r["nH"]), int(r["natoms"]))].append(float(r["D_F"])); T[r["set"]] = float(r["T_K"])
    print(f"\n{os.path.basename(p)}   c per added H [meV] relative to the lowest-H water host and the pure-H reservoir (mean-only | with var/2kT), bootstrap sd")
    for st in sorted({k[0] for k in g}):
        kT = KB * T[st]; hosts = sorted(k for k in g if k[0] == st and k[1] > 0); pure = [k for k in g if k[0] == st and k[1] == 0]
        if len(hosts) < 2 or not pure: continue
        base = hosts[0]; pk = pure[0]
        def ferr(x, var): x = np.asarray(x); return x.mean() + (x.var(ddof=1) / (2 * kT) if var else 0.0)
        def c_of(sample, var):
            fb = ferr(sample[base], var); fh = ferr(sample[pk], var) / pk[3]; out = {}
            for k in hosts[1:]:
                n = k[2] - base[2]
                if n > 0: out[k] = (ferr(sample[k], var) - fb - n * fh) / n
            return out
        res = {}
        for var in (False, True):
            c0 = c_of(g, var); boots = collections.defaultdict(list)
            for _ in range(a.boot):
                s = {k: rng.choice(np.asarray(v), len(v)) for k, v in g.items() if k[0] == st}
                for k, v in c_of(s, var).items(): boots[k].append(v)
            res[var] = {k: (c0[k], np.std(boots[k])) for k in c0}
        print(f"  {st} (T {T[st]:.0f} K): pure-H reservoir {pk[2]} H, {len(g[pk])} frames; base host {base[2]} H / {base[1]} O, {len(g[base])} frames")
        for k in sorted(res[False]):
            m, e = res[False][k]; v, ev = res[True][k]; n = k[2] - base[2]
            print(f"     + {n:3d} H -> H{k[2] / k[1]:.3f}O ({len(g[k])} frames):  c = {m * 1000:+7.0f} +- {e * 1000:3.0f}   |   {v * 1000:+7.0f} +- {ev * 1000:3.0f}   {'PASS' if abs(m) <= 0.05 and abs(v) <= 0.05 else ''}")
