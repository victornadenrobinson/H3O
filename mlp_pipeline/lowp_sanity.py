#!/usr/bin/env python3
"""Sanity table + RDF plot of the geometries chosen for DFT (CPU, login node, ~1 min).
   python mlp_pipeline/lowp_sanity.py [runs/lowp3/dft_frames.extxyz] [-o runs/lowp3/sanity]
Table (one row per geometry): set, source, atoms, density (g/cc), shortest H-H / O-H / O-O contact, molecular H2 fraction (isolated H pair: r1 < 1.0 A and r2 > 1.6 r1), H2O-like O (O with exactly 2 H within 1.25 A), MD T / P.
Plot: g_HH, g_OH, g_OO per set (liquid / cold / hydrate / pureH), one thin line per geometry, so an odd geometry stands out."""
import argparse, os, sys
import numpy as np
from ase.io import read
ap = argparse.ArgumentParser(); ap.add_argument("frames", nargs="?", default="runs/lowp3/dft_frames.extxyz"); ap.add_argument("-o", default="runs/lowp3/sanity"); a = ap.parse_args()
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fr = read(a.frames, ":"); R = np.linspace(0.3, 4.0, 150); mid = 0.5 * (R[1:] + R[:-1]); shell = 4 / 3 * np.pi * (R[1:] ** 3 - R[:-1] ** 3)


def rdf(d, A, B, V):
    nA, nB = len(A), len(B)
    if nA == 0 or nB == 0: return None
    x = d[np.ix_(A, B)].ravel(); x = x[(x > 1e-6) & (x < R[-1])]
    return np.histogram(x, R)[0] / (nA * (nB - (1 if A is B or np.array_equal(A, B) else 0)) / V * shell)


rows = []; curves = {}
for at in fr:
    d = at.get_all_distances(mic=True); np.fill_diagonal(d, 99.0); Z = at.numbers; H, O = np.where(Z == 1)[0], np.where(Z == 8)[0]; V = at.get_volume()
    mn = lambda A, B: float(d[np.ix_(A, B)].min()) if len(A) and len(B) and (len(A) > 1 or A is not B) else float("nan")
    dh = d[np.ix_(H, H)]; s = np.sort(dh, axis=1) if len(H) > 2 else None
    molf = float(((s[:, 0] < 1.0) & (s[:, 1] > 1.6 * s[:, 0])).mean()) if s is not None else float("nan")
    h2o = float(((d[np.ix_(O, H)] < 1.25).sum(1) == 2).mean()) if len(O) else float("nan")
    rho = float(sum({1: 1.008, 8: 15.999}[z] for z in Z) * 1.66054 / V)
    st = at.info.get("set", "?"); rows.append((st, str(at.info.get("src", ""))[-34:], len(at), rho, mn(H, H), mn(O, H), mn(O, O), molf, h2o, at.info.get("T_set", float("nan")), at.info.get("P_md", float("nan"))))
    for k, (A, B) in {"HH": (H, H), "OH": (O, H), "OO": (O, O)}.items():
        g = rdf(d, A, B, V)
        if g is not None: curves.setdefault((st, k), []).append(g)
os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
hdr = f"{'set':8s} {'source':34s} {'N':>4s} {'g/cc':>5s} {'minHH':>6s} {'minOH':>6s} {'minOO':>6s} {'H2frac':>6s} {'H2Oish':>6s} {'T':>6s} {'P_md':>6s}"
lines = [hdr] + [f"{r[0]:8s} {r[1]:34s} {r[2]:4d} {r[3]:5.2f} {r[4]:6.2f} {r[5]:6.2f} {r[6]:6.2f} {r[7]:6.2f} {r[8]:6.2f} {r[9]:6.0f} {r[10]:6.1f}" for r in sorted(rows)]
open(a.o + "_table.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
sets = sorted({k[0] for k in curves}); fig, ax = plt.subplots(len(sets), 3, figsize=(12, 2.6 * len(sets)), squeeze=False, dpi=110)
for i, st in enumerate(sets):
    for j, k in enumerate(("HH", "OH", "OO")):
        for g in curves.get((st, k), []): ax[i, j].plot(mid, g, lw=0.6, alpha=0.6)
        ax[i, j].set(title=f"{st}  g_{k}  ({len(curves.get((st, k), []))} geometries)", xlim=(0.3, 4)); ax[i, j].axhline(1, color="gray", lw=0.4)
        if k == "HH": ax[i, j].axvline(0.74, color="r", lw=0.5, ls=":")
        if k == "OH": ax[i, j].axvline(1.0, color="r", lw=0.5, ls=":")
plt.tight_layout(); plt.savefig(a.o + "_rdf.png"); print(f"\nwrote {a.o}_table.txt and {a.o}_rdf.png   (red dotted: H2 0.74 A, O-H 1.0 A)")
