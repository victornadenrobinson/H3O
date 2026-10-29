"""Electronic entropy in the DFT training frames, and its share of mu_H - G_H. No MLP, no MD: reads extxyz frames written by 02_extract_snapshots.py
(energy = VASP free energy TOTEN, info['energy_sigma0'] = E(sigma->0)).
usage (repo root, a python with ASE):  python mlp_pipeline/tdi/elec_entropy.py valid_m.extxyz [train_m.extxyz ...] [--pmin 300 --pmax 650] [--xmax 3]
Per frame, for Fermi (or Gaussian) smearing VASP's E(sigma->0) = (E + F) / 2, so the smearing (electronic entropy) term is -T S_el = F - E = 2 (TOTEN - E_sigma0).
GRACE was fitted to TOTEN, so this term is inside the model, at the SIGMA of each AIMD run. Two questions:
 1. Did SIGMA follow the ionic temperature? For a metal (pure H at these P) -T S_el ~ -(pi^2/6) g(E_F) (kT)^2: per atom it should grow ~ T^2 across the T bins.
    If it is flat in T, all runs used one SIGMA and the electronic free energy is wrong at other T (by the missing (kT)^2 part).
 2. How much of mu_H - G_H is electronic? Same construction as tdi/bias_proxy.py: f(x) = n (-T S_el per atom) / N_O vs x = H per O, slope f'(x) minus the pure-H
    value per atom = electronic contribution to mu_H - G_H (meV per added H), per T bin; its change with T is the electronic part of the transfer entropy ds.
Frames without energy_sigma0 are skipped (reported)."""
import argparse, re
import numpy as np
import pandas as pd
from ase.io import iread

ap = argparse.ArgumentParser(); ap.add_argument("xyz", nargs="+")
ap.add_argument("--pmin", type=float, default=None); ap.add_argument("--pmax", type=float, default=None); ap.add_argument("--xmax", type=float, default=3.0)
a = ap.parse_args()
KB = 8.617333262e-5


def h_per_o(label):
    c = str(label).split("_")[0].lower()
    if c == "h":
        return np.inf
    m = re.fullmatch(r"h([0-9.]+)o", c)
    return float(m.group(1)) if m else np.nan


rows, skipped = [], 0
for f in a.xyz:
    for at in iread(f, index=":"):
        e0 = at.info.get("energy_sigma0")
        if e0 is None:
            skipped += 1; continue
        try:
            F = at.get_potential_energy()
        except Exception:
            F = at.info.get("energy")
        if F is None:
            skipped += 1; continue
        z = at.numbers; nO, nH = int((z == 8).sum()), int((z == 1).sum())
        rows.append(dict(comp=at.info.get("config_type"), x=(nH / nO if nO else np.inf), T=at.info.get("T_K", np.nan), P=at.info.get("P_GPa", np.nan),
                         n=len(at), nO=nO, nH=nH, mTS=2 * (F - e0) * 1e3))   # meV per cell
df = pd.DataFrame(rows)
print(f"{len(df)} frames read, {skipped} without energy_sigma0 / energy (skipped)")
if df.empty:
    raise SystemExit("nothing to analyse")
if a.pmin is not None: df = df[df.P >= a.pmin]
if a.pmax is not None: df = df[df.P <= a.pmax]
df["Tb"] = (df["T"] / 1000).round() * 1000
df["mTS_atom"] = df.mTS / df.n
df["x"] = df.x.round(3)

print("\n-T S_el per atom (meV/atom), mean over frames, by composition (H per O; inf = pure H) and T bin:")
tab = df.pivot_table(index="x", columns="Tb", values="mTS_atom", aggfunc="mean")
print(tab.round(1).to_string())
if np.inf in tab.index:
    h = tab.loc[np.inf].dropna()
    if len(h) >= 2:
        T0 = h.index[0]
        print("pure H, (-T S_el) / (value at the lowest T x (T/T0)^2):  " + "  ".join(f"{int(t)} K {v / (h.iloc[0] * (t / T0) ** 2):.2f}" for t, v in h.items())
              + "   (~1 if SIGMA = kT of each run; ~ (T0/T)^2 if one SIGMA for all)")

print("\nelectronic contribution to mu_H - G_H (meV per added H) = f'(x) - (pure-H -T S_el per atom), f = n (-T S_el / atom) / N_O, per T bin:")
for Tb, g in df.groupby("Tb"):
    h = g[np.isinf(g.x)]; s = g[np.isfinite(g.x) & (g.x >= 2.0 - 1e-6) & (g.x <= a.xmax + 1e-6)]
    if not len(h) or s.x.nunique() < 2:
        print(f"  {int(Tb)} K: not enough compositions (pure H: {len(h)} frames, H2O..H{a.xmax:g}O: {s.x.nunique()} compositions)"); continue
    s = s.assign(f=s.mTS / s.nO); m = s.groupby("x").f.agg(["mean", "sem", "size"])
    b = np.polyfit(m.index.values - 2.0, m["mean"].values, 1)[0]
    dH = h.mTS_atom.mean()
    print(f"  {int(Tb)} K: f'(x) = {b:7.1f} meV per added H over x = {', '.join(f'{v:g}' for v in m.index)} ({int(m['size'].sum())} frames);  pure H {dH:7.1f} meV/atom ({len(h)} frames)"
          f"  ->  electronic part of mu_H - G_H = {b - dH:+7.1f} meV per H")
print("\nPositive = electronic free energy favours pure H (GRACE includes it at the training SIGMA). Compare its change between T bins with the fitted transfer entropy.")
