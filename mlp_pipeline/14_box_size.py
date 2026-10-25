"""Choose the minimum box size from the EOS and from the data, instead of guessing.

usage: python 14_box_size.py data-h3o-convex-hull.csv [mined_h3o.csv] [--T 9000]
Part A (hull CSV only): V per formula unit vs P -> cell edge of the production cells (N molecules/atoms) and atoms needed for a given edge.
Part B (needs the mined CSV): finite-size check. Pressure of each mined run minus hull pressure at the same density, binned by box edge,
per composition / T / lattice (bcc, fcc, other from the folder name). Where the median |dP| stays small for all larger boxes, the box is converged
*relative to the hull's own reference cells*.
"""
import argparse, re
import numpy as np
import pandas as pd
from common import HULL_COMPS, comp_label

AMU = 1.66053907
MASS = {"H": 1.00794, "O": 15.9994}


def hull_V(path, T):
    """The CSV's V column is the volume of the whole production cell (A^3); the atom count is inferred from rho*V/mean atomic mass."""
    h = pd.read_csv(path, low_memory=False); out = {}
    for suf, ratio in HULL_COMPS.items():
        c = [f"P {T}K {suf}", f"V {T}K {suf}", f"rho {T}K {suf}"]
        if all(x in h for x in c):
            d = h[c].apply(pd.to_numeric, errors="coerce").dropna().sort_values(c[0])
            if len(d) > 2:
                m_atom = MASS["H"] if np.isinf(ratio) else (MASS["O"] + ratio * MASS["H"]) / (1.0 + ratio)
                N = float(np.median(d[c[2]] * d[c[1]] / AMU / m_atom))
                out[suf] = dict(P=d[c[0]].values, V=d[c[1]].values, rho=d[c[2]].values, ratio=ratio, N=int(round(N)), Nraw=N)
    return out


def part_a(hull, T):
    Ps = (300, 400, 500, 700, 1000)
    print(f"=== A. production cells in the hull CSV at {T} K (V column = cell volume; N = atoms in the cell) ===")
    print(f"{'comp':8} {'N':>4}  " + "  ".join(f"edge@{p:4d}" for p in Ps) + "     v_atom (A^3) @ " + "/".join(str(p) for p in Ps))
    for suf, e in hull.items():
        L = [np.interp(p, e["P"], e["V"]) ** (1 / 3) for p in Ps]; v = [np.interp(p, e["P"], e["V"]) / e["N"] for p in Ps]
        print(f"{comp_label(e['ratio']):8} {e['N']:4d}  " + "  ".join(f"{x:9.2f}" for x in L) + "     " + " ".join(f"{x:.2f}" for x in v)
              + ("" if abs(e["Nraw"] - e["N"]) < 0.2 else f"   (N from rho*V = {e['Nraw']:.1f}: check)"))
    print("\natoms needed for a cubic cell of edge L, at the same P:   L = 8.0 / 8.5 / 9.0 / 10.0 A")
    for p in (300, 500, 700, 1000):
        print(f"  P = {p} GPa: " + "   ".join(f"{comp_label(e['ratio'])}: " + "/".join(f"{L ** 3 / (np.interp(p, e['P'], e['V']) / e['N']):4.0f}" for L in (8.0, 8.5, 9.0, 10.0))
                                              for e in hull.values()))


def part_b(hull_path, mined):
    import importlib; inv = importlib.import_module("01_bin_inventory")
    df = inv.load_inventory(mined)
    df = df[(df.N_steps >= 100) & (df.P_std_GPa.fillna(0) <= 20)].copy()
    df["rho"] = (df.N_H * MASS["H"] + df.N_O * MASS["O"]) * AMU / df.True_V_A3
    df["box"] = df.True_V_A3 ** (1 / 3)
    df["lat"] = df.Path.map(lambda p: "fcc" if "fcc" in p.lower() else ("bcc" if "bcc" in p.lower() else "other"))
    rows = []
    for T in inv.ISOTHERMS:
        hh = hull_V(hull_path, T)
        for r in df[(df.True_T_K - T).abs() <= inv.T_BIN].itertuples():
            e = hh.get(next((s for s, x in HULL_COMPS.items() if comp_label(x) == r.comp_snap), None))
            if e is None:
                continue
            o = np.argsort(e["rho"]); rho, P = e["rho"][o], e["P"][o]
            if rho[0] <= r.rho <= rho[-1]:
                rows.append(dict(comp=r.comp_snap, T=T, lat=r.lat, box=r.box, N=r.N_ions, dP=r.True_P_GPa - np.interp(r.rho, rho, P)))
    x = pd.DataFrame(rows)
    x["bin"] = pd.cut(x.box, [0, 6, 7, 8, 8.5, 9, 9.5, 10, 11, 20])
    print(f"\\n=== B. finite-size check: mined P - hull P(rho), by box edge ({len(x)} runs) ===")
    for (c, T), g in x.groupby(["comp", "T"]):
        t = g.groupby(["lat", "bin"], observed=True).dP.agg(n="size", med="median", std="std").round(1)
        t = t[t.n >= 3]
        if len(t):
            print(f"\\n{c} {T} K\\n{t.to_string()}")
    x.to_csv(out("box_size_dP.csv"), index=False)


if __name__ == "__main__":
    from common import out
    ap = argparse.ArgumentParser()
    ap.add_argument("hull"); ap.add_argument("mined", nargs="?"); ap.add_argument("--T", type=int, default=9000)
    args = ap.parse_args()
    part_a(hull_V(args.hull, args.T), args.T)
    if args.mined:
        part_b(args.hull, args.mined)
