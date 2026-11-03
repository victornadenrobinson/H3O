#!/usr/bin/env python3
"""dG and convex-hull summary of a Perlmutter pwidom run (perlmutter_pwidom.sh), from the same numbers the status view prints.
  python3 mlp_pipeline/pwidom_hull.py runs/pwidom_new_mnf3e3_500GPa_5000K [--err-scale 4] [--strict] [--mc 4000]
Definitions (n = extra H per O of the host: H2O 0, H2.083O 1/12, H2.25O 1/4; pure H is the reservoir):
  dg(n)      = mu_H(host n) - mu_H(pure H) per ONE more H, meV (pressure- and finite-size-corrected exactly as pwidom_scan.dg)
  dG_mix(n)  = integral_0^n dg dn'  [meV per O]  = formation free energy of H_(2+n)O from H2O + n H(reservoir); trapezoid between the scanned compositions
  per atom   = dG_mix / (3 + n);  x_H = (2 + n)/(3 + n).   Tie line H2O -- pure H is zero on this scale, so hull depth = -min(dG_mix per atom) on the lower convex hull.
  Minimum of dG_mix where dg crosses zero (dg < 0: H prefers the host). If dg < 0 over the whole scanned range the hull is NOT bracketed: the printed depth is a lower bound.
Errors: Monte Carlo over the dg errors x --err-scale (the audit found block errors 4-5x too small: try --err-scale 4). Melted hosts are excluded; --strict also drops UNCONVERGED / LOW KISH points.
Needs the repo's pwidom_scan (mlp_pipeline/tdi) next to this checkout, like the status view."""
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np


def hull(points, err_scale=1.0, mc=4000, seed=1):
    """points: list of (n, dg_meV, err_meV) sorted by n. Returns dict with the point table and the hull summary (pure function: testable without a run)."""
    P = sorted(points); n = np.array([p[0] for p in P], float); g = np.array([p[1] for p in P], float); e = np.array([p[2] for p in P], float) * err_scale
    def dgmix(gv):                                              # trapezoid from n = 0; if the first composition is > 0, dg is extended flat down to 0
        n0 = np.r_[0.0, n] if n[0] > 0 else n; g0 = np.r_[gv[0], gv] if n[0] > 0 else gv
        return np.r_[0.0, np.cumsum(0.5 * (g0[1:] + g0[:-1]) * np.diff(n0))] if n[0] > 0 else np.r_[0.0, np.cumsum(0.5 * (g0[1:] + g0[:-1]) * np.diff(n0))]
    def summarise(gv):
        G = dgmix(gv); nn = np.r_[0.0, n] if n[0] > 0 else n; gg = np.r_[gv[0], gv] if n[0] > 0 else gv
        f = G / (3 + nn)                                          # meV per atom
        cross = [i for i in range(len(gg) - 1) if gg[i] < 0 <= gg[i + 1]]
        if cross:
            i = cross[0]; ns = nn[i] + (0 - gg[i]) / (gg[i + 1] - gg[i]) * (nn[i + 1] - nn[i]); Gs = G[i] + 0.5 * (gg[i] + 0.0) * (ns - nn[i])
            return dict(bracketed=True, n_star=ns, depth_O=-Gs, depth_atom=-Gs / (3 + ns))
        if gg[0] >= 0: return dict(bracketed=False, n_star=0.0, depth_O=0.0, depth_atom=0.0, positive=True)
        j = int(np.argmin(G)); return dict(bracketed=False, n_star=nn[j], depth_O=-G[j], depth_atom=-f[j], positive=False)
    base = summarise(g); G = dgmix(g); nn = np.r_[0.0, n] if n[0] > 0 else n
    rng = np.random.default_rng(seed); runs = [summarise(rng.normal(g, e)) for _ in range(mc)] if mc and np.any(e > 0) else []
    out = dict(n=nn, G=G, per_atom=G / (3 + nn), x=(2 + nn) / (3 + nn), base=base)
    if runs:
        out["p_bracketed"] = float(np.mean([r["bracketed"] for r in runs])); d = np.array([r["depth_atom"] for r in runs]); out["depth_atom_ci"] = tuple(np.percentile(d, [16, 50, 84]))
        b = [r["n_star"] for r in runs if r["bracketed"]]; out["n_star_ci"] = tuple(np.percentile(b, [16, 50, 84])) if b else None
    return out


def load_run(run, scan=None):
    """(manifest, rows) of a Perlmutter pwidom run: rows = dicts(name, n, dg, err, flags) for every non-reference host, dg exactly as the status view computes it (pwidom_scan.dg)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tdi")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if scan is None:
        try: import pwidom_scan as scan
        except Exception as ex: sys.exit(f"cannot import pwidom_scan ({ex}); run from the checkout that has mlp_pipeline/tdi/pwidom_scan.py (the Perlmutter scan code)")
    out = Path(run); m = json.loads((out / "manifest.json").read_text()); res = {}
    for j in m["jobs"]:
        try: res[j["name"]] = json.loads(Path(j["result"]).read_text())
        except Exception: pass
    if m.get("reference"): ref = m["reference"]; res["pureH"] = ref.get("snapshot") or json.loads(Path(ref["result"]).read_text())
    h = res.get("pureH")
    if not h: sys.exit("no pure-H reference result in this run yet")
    rows = []
    for name, w in res.items():
        if name == "pureH": continue
        nO = w["natoms"] - w["N_H"]; n = w["N_H"] / nO - 2.0; v, err = scan.dg(w, h); flags = []
        if w.get("melted") or scan.o_growing(w): flags.append("MELTED")
        if abs(w["mu_ex_delete_meV"] - w["mu_ex_widom_meV"]) > max(200, 3 * err): flags.append("UNCONVERGED")
        if w["kish_frac_ins"] < 0.05: flags.append("LOW KISH")
        rows.append(dict(name=name, n=n, dg=v, err=err, flags=flags))
    rows.sort(key=lambda r: r["n"]); return m, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("run"); ap.add_argument("--err-scale", type=float, default=1.0); ap.add_argument("--strict", action="store_true"); ap.add_argument("--mc", type=int, default=4000); a = ap.parse_args()
    m, rows = load_run(a.run)
    rows.sort(key=lambda r: r["n"]); keep = [r for r in rows if "MELTED" not in r["flags"] and not (a.strict and r["flags"])]
    out = Path(a.run); print(f"{out}: {m['P']:g} GPa / {m['T']:g} K   (statistical errors only; shown x{a.err_scale:g})")
    print(f"{'host':8s}{'n':>7s}{'x_H':>7s}{'dg meV/H':>16s}  flags")
    for r in rows: print(f"{r['name']:8s}{r['n']:7.3f}{(2 + r['n']) / (3 + r['n']):7.3f}{r['dg']:9.0f} +-{r['err'] * a.err_scale:4.0f}  {' '.join(r['flags']) or '-'}" + ("   (excluded)" if r not in keep else ""))
    if len(keep) < 1: sys.exit("no usable host")
    hh = hull([(r["n"], r["dg"], r["err"]) for r in keep], a.err_scale, a.mc); b = hh["base"]
    print(f"\n{'n':>7s}{'dG_mix meV/O':>14s}{'meV/atom':>10s}")
    for ni, G, f in zip(hh["n"], hh["G"], hh["per_atom"]): print(f"{ni:7.3f}{G:14.1f}{f:10.2f}")
    if b["bracketed"]: ci = hh.get("n_star_ci"); print(f"\nhull minimum at n* = {b['n_star']:.3f} H/O (x_H = {(2 + b['n_star']) / (3 + b['n_star']):.3f}): depth {b['depth_O']:.1f} meV/O = {b['depth_atom']:.2f} meV/atom" + (f";  n* 16/50/84 % = {ci[0]:.3f}/{ci[1]:.3f}/{ci[2]:.3f}" if ci else ""))
    elif b.get("positive"): print(f"\nNO stable mixing in the scanned range: dg(0) >= 0 (H prefers the reservoir). hull depth 0.")
    else: print(f"\nhull NOT bracketed: dg < 0 up to n = {max(r['n'] for r in keep):.3f} (H still prefers the host); depth is a LOWER bound: {b['depth_O']:.1f} meV/O = {b['depth_atom']:.2f} meV/atom at n = {b['n_star']:.3f}. Scan larger n.")
    if "depth_atom_ci" in hh: c = hh["depth_atom_ci"]; print(f"depth per atom 16/50/84 % over the error draws: {c[0]:.2f} / {c[1]:.2f} / {c[2]:.2f} meV;  P(bracketed) = {hh['p_bracketed']:.2f}")
    print("Free-energy errors are statistical only; completion does not establish convergence. Flagged points are marked, melted ones excluded (--strict drops all flagged).")


if __name__ == "__main__":
    main()
