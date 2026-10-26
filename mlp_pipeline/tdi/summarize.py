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
                     vpa_A3=d["vpa"], P_mlp_GPa=d.get("P_mlp"), k_Einstein=d.get("kE"), O_struct=d.get("O_struct"), nodes=A.get("nodes"), power=A.get("power"), dt_fs=A.get("dt"), ps_per_node=h[-1]["t_ps"] if h else None,
                     G_meV_atom=d["G_meV_atom"], G_err_meV_atom=d["G_err_meV_atom"], model=A.get("model"), git=d.get("meta", {}).get("git", "")[:8], date=d.get("meta", {}).get("date", "")))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no tdi_*.json found (run tdi/collect.py first)")
pd.set_option("display.width", 220); print(df[["tag", "O_struct", "P_mlp_GPa", "ps_per_node", "G_meV_atom", "G_err_meV_atom", "git"]].to_string(index=False))
pairs = []
for (P, T, h), g in df[df.nO > 0].groupby(["P_target_GPa", "T_K", "H_per_O"]):
    f, b = g[g.phase == "fcc"], g[g.phase == "bcc"]
    if len(f) and len(b):
        f, b = f.iloc[0], b.iloc[0]
        pairs.append(dict(P_GPa=P, T_K=T, H_per_O=h, dG_fcc_minus_bcc_meV_atom=f.G_meV_atom - b.G_meV_atom, err=(f.G_err_meV_atom ** 2 + b.G_err_meV_atom ** 2) ** .5, ps_fcc=f.ps_per_node, ps_bcc=b.ps_per_node))
if pairs:
    print("\nG_fcc - G_bcc (meV/atom):"); print(pd.DataFrame(pairs).round(2).to_string(index=False))
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
    df.to_csv(a.o, index=False); pd.DataFrame(pairs).to_csv(a.o.replace(".csv", "_dG.csv"), index=False); pd.DataFrame(hull).to_csv(a.o.replace(".csv", "_hull.csv"), index=False); print("wrote", a.o, "and", a.o.replace(".csv", "_dG.csv"))
