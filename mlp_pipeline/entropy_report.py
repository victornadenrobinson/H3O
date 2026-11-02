#!/usr/bin/env python3
"""How large is the electronic-entropy term in the new labels, and how much does it vary? (a T-independent potential has to average it)
  python mlp_pipeline/entropy_report.py snapdb/labelled.extxyz snapdb_v2/labelled.extxyz
VASP prints the free energy F = TOTEN and energy(sigma->0) = F + TS/2, so TS_el = 2 (E_sigma0 - F). snapdb.py collect stores F as the label and E_sigma0_eV in info.
The table gives -T S_el per cell and per atom by set and sampling temperature. If it is eV-scale per cell, differs between compositions and grows with T, then a model without an electronic
temperature input cannot reproduce D = E_DFT - E_model per cell to better than the spread of this term at fixed geometry class."""
import sys, collections
import numpy as np
from ase.io import read

rows = []
for p in sys.argv[1:]:
    for f in read(p, index=":"):
        if "E_sigma0_eV" not in f.info: continue
        F = f.get_potential_energy(); ts = 2.0 * (float(f.info["E_sigma0_eV"]) - F)
        nO = int((f.numbers == 8).sum()); nH = int((f.numbers == 1).sum())
        rows.append((str(f.info.get("db_set", "?")), float(f.info.get("T_K", np.nan)), "H" if nO == 0 else "H%.2gO" % (nH / nO), len(f), ts))
if not rows: sys.exit("no frames with E_sigma0_eV: run snapdb.py collect first")
g = collections.defaultdict(list)
for s, T, comp, n, ts in rows: g[(s, int(round(T / 1000.0)) * 1000 if np.isfinite(T) else -1)].append((ts, n))
print(f"{len(rows)} labelled frames;  TS_el = 2*(E_sigma0 - F)  [eV]")
print(f"{'set':8s}{'T_K':>7s}{'n':>6s}{'TS/cell mean':>14s}{'sd':>8s}{'min':>9s}{'max':>9s}{'TS/atom meV':>13s}")
for (s, T), v in sorted(g.items()):
    ts = np.array([x for x, _ in v]); n = np.array([m for _, m in v])
    print(f"{s:8s}{T:7d}{len(v):6d}{ts.mean():14.2f}{ts.std():8.2f}{ts.min():9.2f}{ts.max():9.2f}{1000 * (ts / n).mean():13.1f}")
c = collections.defaultdict(list)
for s, T, comp, n, ts in rows: c[comp].append(ts / n * 1000)
print("\nby composition (meV/atom):", {k: (round(float(np.mean(v)), 1), round(float(np.std(v)), 1), len(v)) for k, v in sorted(c.items(), key=lambda kv: -len(kv[1]))[:8]})
Ts = np.array([r[1] for r in rows]); tsa = np.array([r[4] / r[3] * 1000 for r in rows]); m = np.isfinite(Ts)
if m.sum() > 10: print(f"correlation of TS/atom with T_K: {np.corrcoef(Ts[m], tsa[m])[0, 1]:.2f};  mean TS/atom: {tsa.mean():.1f} meV, sd {tsa.std():.1f} meV")
