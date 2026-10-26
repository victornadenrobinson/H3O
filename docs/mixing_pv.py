"""P*dV_excess of H2O + n H -> H(2+n)O from the EOS columns of data-h3o-convex-hull.csv (DFT, liquid/mixed branch at 5000, 7000, 9000 K).
V_excess per added H = [V_O(mix) - V_O(H2O) - n V_H(pure)] / n at the same P,T (V_O: volume per O, V_H: per H atom); P V_excess in meV per added H and in units of kT.
usage: python3 docs/mixing_pv.py [data-h3o-convex-hull.csv]  -> prints a table, writes docs/data/mixing_pv.csv"""
import os, sys
import numpy as np, pandas as pd
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data-h3o-convex-hull.csv")
d = pd.read_csv(f)
N = {"h2.25": 0.25, "h3o": 1.0, "h5o": 3.0, "h7o": 5.0}
rows = []
for T in (5000, 7000, 9000):
    def curve(comp):
        p, v = d[f"P {T}K {comp}"], d[f"V {T}K {comp} norm"]
        m = p.notna() & v.notna(); p, v = p[m].values, v[m].values; o = np.argsort(p)
        return p[o], v[o]
    pw, vw = curve("h2o"); ph, vh = curve("h")
    for comp, n in N.items():
        pm, vm = curve(comp)
        for P in (200, 250, 300, 350, 400, 450, 500):
            if not (max(pw.min(), ph.min(), pm.min()) <= P <= min(pw.max(), ph.max(), pm.max())):
                continue
            Vw, Vh, Vm = np.interp(P, pw, vw), np.interp(P, ph, vh), np.interp(P, pm, vm)
            ex = (Vm - Vw - n * Vh) / n; pv = P / 160.2177 * ex * 1e3
            rows.append(dict(T_K=T, P_GPa=P, comp=comp, n_H_per_O_added=n, V_O_h2o=Vw, V_H_pure=Vh, V_O_mix=Vm, Vexcess_per_H_A3=ex, PVexcess_meV_per_H=pv, in_kT=pv / (8.617333e-2 * T)))
out = pd.DataFrame(rows); out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mixing_pv.csv"), index=False)
pd.set_option("display.width", 200)
for c in ("h2.25", "h3o"):
    print(f"\n{c}: P*Vexcess per added H (meV; in brackets kT)")
    print(out[out.comp == c].pivot(index="P_GPa", columns="T_K", values="PVexcess_meV_per_H").round(0).to_string())
