"""Collect every tdi_<tag>.json in a run directory into one table (and a CSV for the repo).
usage: python3 mlp_pipeline/tdi/summarize.py runs/tdi [-o docs/data/tdi_results.csv]
Pairs bcc/fcc entries with the same (P, T, H per O) and prints G_fcc - G_bcc (meV/atom) with the combined statistical error."""
import argparse, glob, json, os
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("-o", default=None); a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.dir, "tdi_*.json"))):
    d = json.load(open(f)); A = d.get("args", {}); h = d.get("hist", [])
    rows.append(dict(tag=d["tag"], phase=d["phase"], T_K=d["T"], P_target_GPa=d["P_target"], H_per_O=A.get("hper", 2.0) if d["nO"] else float("nan"), N=d["N"], nO=d["nO"], nH=d["nH"],
                     vpa_A3=d["vpa"], P_mlp_GPa=d.get("P_mlp"), k_Einstein=d.get("kE"), O_struct=d.get("O_struct"), nrep=A.get("nrep"), nodes=A.get("nodes"), power=A.get("power"), dt_fs=A.get("dt"), ps_per_node=h[-1]["t_ps"] if h else None,
                     G_meV_atom=d["G_meV_atom"], G_err_meV_atom=d["G_err_meV_atom"], model=A.get("model"), git=d.get("meta", {}).get("git", "")[:8], date=d.get("meta", {}).get("date", "")))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no tdi_*.json found (run tdi/collect.py first)")
pd.set_option("display.width", 220); print(df[["tag", "O_struct", "P_mlp_GPa", "ps_per_node", "G_meV_atom", "G_err_meV_atom", "git"]].to_string(index=False))
def pick(g, dt):
    """one row of g: the one with the same dt as `dt` if there is one (then the longest run), else the smallest dt (flagged by the caller)"""
    same = g[g.dt_fs == dt]
    return (same if len(same) else g[g.dt_fs == g.dt_fs.min()]).sort_values("ps_per_node").iloc[-1]


pairs = []
for (P, T, h, nr), g in df[df.nO > 0].groupby(["P_target_GPa", "T_K", "H_per_O", "nrep"]):   # same cell size only
    f, b = g[g.phase == "fcc"], g[g.phase == "bcc"]
    if len(f) and len(b):
        f = pick(f, b.dt_fs.min()); b = pick(b, f.dt_fs)
        pairs.append(dict(P_GPa=P, T_K=T, H_per_O=h, nrep=nr, dG_fcc_minus_bcc_meV_atom=f.G_meV_atom - b.G_meV_atom, err=(f.G_err_meV_atom ** 2 + b.G_err_meV_atom ** 2) ** .5, ps_fcc=f.ps_per_node, ps_bcc=b.ps_per_node,
                          dt_fcc=f.dt_fs, dt_bcc=b.dt_fs, note="" if f.dt_fs == b.dt_fs else "DIFFERENT dt"))
if pairs:
    print("\nG_fcc - G_bcc (meV/atom):"); print(pd.DataFrame(pairs).round(2).to_string(index=False))
# crystal mixing: H(2+n)O crystal (same phase, cell and T, P) -> H2O crystal + n pure-H fluid atoms, per added H; and the dilute limit from the insertion runs (mu_H - G_H)
solid = []
for _, m in df[(df.nO > 0) & df.phase.isin(["fcc", "bcc"]) & (df.H_per_O > 2.0)].iterrows():
    w = df[(df.phase == m.phase) & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K) & (df.H_per_O == 2.0) & (df.nrep == m.nrep)]
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(w) and len(hh):
        w, hh = pick(w, m.dt_fs), pick(hh, m.dt_fs); n = m.nH - 2 * m.nO
        dG = (m.N * m.G_meV_atom - w.N * w.G_meV_atom - n * hh.G_meV_atom) / n
        err = ((m.N * m.G_err_meV_atom) ** 2 + (w.N * w.G_err_meV_atom) ** 2 + (n * hh.G_err_meV_atom) ** 2) ** .5 / n
        solid.append(dict(phase=m.phase, P_GPa=m.P_target_GPa, T_K=m.T_K, H_per_O=m.H_per_O, n_added=int(n), dG_per_added_H_meV=dG, err=err, kT_meV=8.617333e-2 * m.T_K, dt_mix=m.dt_fs, dt_H2O=w.dt_fs, dt_H=hh.dt_fs, N_H=int(hh.N),
                          note="" if m.dt_fs == w.dt_fs == hh.dt_fs else "DIFFERENT dt"))
for _, m in df[df.phase == "insert"].iterrows():
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(hh):
        hh = pick(hh, m.dt_fs)
        solid.append(dict(phase="insert (fcc H2O, dilute)", P_GPa=m.P_target_GPa, T_K=m.T_K, H_per_O=2.0, n_added=int(m.N), dG_per_added_H_meV=m.G_meV_atom - hh.G_meV_atom, err=(m.G_err_meV_atom ** 2 + hh.G_err_meV_atom ** 2) ** .5,
                          kT_meV=8.617333e-2 * m.T_K, dt_mix=m.dt_fs, dt_H2O=float("nan"), dt_H=hh.dt_fs, N_H=int(hh.N), note="" if m.dt_fs == hh.dt_fs else "DIFFERENT dt"))
if solid:
    print("\nMixing free energy per added H, crystal H2O + n H(fluid) -> H(2+n/nO)O crystal (positive = H not soluble); insertion rows: mu_H(fcc H2O) - G_H(pure), meV:")
    print(pd.DataFrame(solid).round(2).to_string(index=False))
hull = []
for (P, T), g in df.groupby(["P_target_GPa", "T_K"]):
    w, hh = g[(g.phase == "liquid") & (g.H_per_O == 2.0)], g[g.phase == "H"]
    if len(w) and len(hh):
        w, hh = w.iloc[0], hh.iloc[0]
        for _, m in g[(g.phase == "liquid") & (g.H_per_O > 2.0)].iterrows():
            n = m.H_per_O - 2.0; fw, fh = 3 / (3 + n), n / (3 + n)   # atom fractions of the H2O part and of the added H in the mixture
            hull.append(dict(P_GPa=P, T_K=T, H_per_O=m.H_per_O, x_H=n / (1 + n), dG_mix_meV_atom=m.G_meV_atom - fw * w.G_meV_atom - fh * hh.G_meV_atom,
                             err=(m.G_err_meV_atom ** 2 + (fw * w.G_err_meV_atom) ** 2 + (fh * hh.G_err_meV_atom) ** 2) ** .5))
if hull:
    print("\nLiquid mixing free energy per atom, H2O + n H -> H(2+n)O (x_H = n/(1+n) as in the hull sheet; negative = mixing favourable), meV/atom:")
    print(pd.DataFrame(hull).round(2).to_string(index=False))
if a.o:
    df.to_csv(a.o, index=False); pd.DataFrame(pairs).to_csv(a.o.replace(".csv", "_dG.csv"), index=False); pd.DataFrame(solid).to_csv(a.o.replace(".csv", "_mix.csv"), index=False); pd.DataFrame(hull).to_csv(a.o.replace(".csv", "_hull.csv"), index=False); print("wrote", a.o, "and", a.o.replace(".csv", "_dG.csv"))
