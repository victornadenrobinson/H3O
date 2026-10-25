"""Compare mined OUTCAR pressures with the original eos.dat tables (the source of the hull CSV).

usage: python 08_crosscheck_eos_dat.py mined_v2.csv ~/work/tdi/H3O [more roots...] [--root ~/work]
eos.dat line:  10.1  256H+128O N= 384 V[A^3]= 1030.3 rho[g/cc]= 3.7 T[K]= 5000 P[GPa]= 257.5 0.15 E[Ha]= ...
`label` (10.1) is the run sub-folder: mined runs matched where <folder-of-eos.dat>/<label>/ is in their path.
Reports dP = P_eos - P_external(mined) and tests dP against the ideal-gas term N kB T / V.
"""
import argparse, os, re
import numpy as np
import pandas as pd

KB_EV, EV_A3_GPA = 8.617333e-5, 160.21766
LINE = re.compile(r"^(\S+)\s+(\d+)H(?:\+(\d+)O)?\s+N=\s*(\d+)\s+V\[A\^3\]=\s*([\d.]+)\s+rho\[g/cc\]=\s*([\d.]+)\s+"
                  r"T\[K\]=\s*([\d.]+)\s+P\[GPa\]=\s*(-?[\d.]+)\s+([\d.]+)")


def parse_eos(roots, root):
    rows = []
    for top in roots:
        for d, _, fs in os.walk(top):
            for f in fs:
                if f != "eos.dat" and not f.endswith("eos.dat"):
                    continue
                for ln in open(os.path.join(d, f), errors="replace"):
                    m = LINE.match(ln.strip())
                    if m:
                        rows.append(dict(folder=os.path.relpath(d, root).lower(), label=m[1], N_H=int(m[2]), N_O=int(m[3] or 0),
                                         N=int(m[4]), V=float(m[5]), rho=float(m[6]), T=float(m[7]), P_eos=float(m[8]), P_err=float(m[9])))
    return pd.DataFrame(rows)


def main(a):
    root = os.path.abspath(os.path.expanduser(a.root))
    eos = parse_eos([os.path.expanduser(r) for r in a.roots], root)
    print(f"{len(eos)} eos.dat lines in {eos.folder.nunique()} folders")
    m = pd.read_csv(a.inv)
    m["dir"] = m.Path.str.lower().str.replace("\\", "/")
    m = m.dropna(subset=["True_P_GPa"])
    out = []
    for r in eos.itertuples():
        pre = f"{r.folder}/{r.label.lower()}/"
        sel = m[m.dir.str.contains(re.escape(pre), regex=True) & (m.N_H == r.N_H) & (m.N_O == r.N_O)]
        sel = sel[sel.N_steps >= a.min_steps]
        sel = sel[(sel.True_V_A3 - r.V).abs() < 0.01 * r.V]
        if sel.empty:
            continue
        w = sel.N_steps.values
        P_ext = float(np.average(sel.True_P_GPa, weights=w))
        P_tot = float(np.average(sel.P_total_GPa, weights=w)) if "P_total_GPa" in sel and sel.P_total_GPa.notna().all() else np.nan
        kin = r.N * KB_EV * r.T / r.V * EV_A3_GPA
        out.append(dict(folder=r.folder, label=r.label, N_H=r.N_H, N_O=r.N_O, T=r.T, V=r.V, n_runs=len(sel), P_eos=r.P_eos, P_ext=P_ext,
                        P_tot=P_tot, kin_est=kin, dP=r.P_eos - P_ext, dP_minus_kin=r.P_eos - P_ext - kin))
    x = pd.DataFrame(out)
    if x.empty:
        raise SystemExit("no eos.dat lines matched any mined runs (check --root and path prefixes)")
    x.to_csv("crosscheck_eos_dat.csv", index=False)
    print(f"{len(x)} of {len(eos)} eos.dat lines matched to mined runs\n")
    g = x.groupby("folder").agg(n=("dP", "size"), dP=("dP", "mean"), kin=("kin_est", "mean"), resid=("dP_minus_kin", "mean"),
                                resid_std=("dP_minus_kin", "std")).round(1)
    print(g.to_string())
    slope = np.polyfit(x.kin_est, x.dP, 1)
    print(f"\nfit  P_eos - P_ext = {slope[0]:.3f} * (N kB T/V) + {slope[1]:.1f} GPa   (1.0 and ~0 => eos.dat = external + ideal-gas term)")
    print(f"median dP = {x.dP.median():.1f} GPa;  median (dP - kinetic estimate) = {x.dP_minus_kin.median():.1f} GPa; std {x.dP_minus_kin.std():.1f}")
    if x.P_tot.notna().any():
        print(f"vs VASP 'total pressure': median P_eos - P_tot = {(x.P_eos - x.P_tot).median():.1f} GPa")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inv"); ap.add_argument("roots", nargs="+")
    ap.add_argument("--root", default="~/work", help="dir that mined paths are relative to")
    ap.add_argument("--min-steps", type=int, default=100)
    main(ap.parse_args())
