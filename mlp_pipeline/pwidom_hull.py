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


def pool_rows(runs_rows):
    """Pool replica runs of the same state: per host, mean dg over replicas; error = max(between-run sd / sqrt(k), mean stat err / sqrt(k)); a host melted in ANY replica is flagged MELTED."""
    by = {}
    for rows in runs_rows:
        for r in rows: by.setdefault(r["name"], []).append(r)
    out = []
    for name, lst in by.items():
        k = len(lst); dg = np.array([r["dg"] for r in lst]); st = np.array([r["err"] for r in lst]); sd = dg.std(ddof=1) if k > 1 else 0.0
        err = max(sd / np.sqrt(k), np.sqrt(np.mean(st ** 2) / k)) if k > 1 else st[0]
        fl = sorted({f for r in lst for f in r["flags"]}); out.append(dict(name=name, n=lst[0]["n"], dg=float(dg.mean()), err=float(err), flags=fl, k=k, spread=float(sd)))
    out.sort(key=lambda r: r["n"]); return out


def linear_extrapolation(points, err_scale=1.0, mc=4000, seed=2):
    """EXTRAPOLATION, not a measurement: weighted straight line dg(n) = a + b n through the usable compositions; if b > 0 the hull minimum is at n* = -a/b with depth a^2/(2b) per O. MC over the dg errors."""
    n = np.array([p[0] for p in points], float); g = np.array([p[1] for p in points], float); e = np.maximum(np.array([p[2] for p in points], float) * err_scale, 1e-6)
    def fit(gv):
        w = 1 / e ** 2; A = np.c_[np.ones_like(n), n]; M = A.T @ (A * w[:, None]); b = np.linalg.solve(M, A.T @ (w * gv)); return b[0], b[1]
    if len(n) < 2: return None
    a, b = fit(g); rng = np.random.default_rng(seed); res = []
    for _ in range(mc):
        aa, bb = fit(rng.normal(g, e)); res.append((aa, bb))
    res = np.array(res); ok = res[:, 1] > 0; nst = -res[ok, 0] / res[ok, 1]; ok2 = nst > 0
    out = dict(a=a, b=b, p_positive_slope=float(ok.mean()))
    if b > 0 and a < 0: out.update(n_star=-a / b, depth_O=a * a / (2 * b), depth_atom=a * a / (2 * b) / (3 - a / b))
    if ok2.any():
        ns = nst[ok2]; dep = (res[ok, 0][ok2] ** 2 / (2 * res[ok, 1][ok2])) / (3 + ns); out["n_star_ci"] = tuple(np.percentile(ns, [16, 50, 84])); out["depth_atom_ci"] = tuple(np.percentile(dep, [16, 50, 84])); out["p_valid"] = float(ok2.mean())
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("run", nargs="+", help="one run dir, or several replica runs of the same state (pooled with --pool)"); ap.add_argument("--pool", action="store_true", help="pool replica runs: per-host mean dg, error from the between-run scatter"); ap.add_argument("--err-scale", type=float, default=1.0); ap.add_argument("--strict", action="store_true"); ap.add_argument("--mc", type=int, default=4000); a = ap.parse_args()
    loaded = [load_run(r) for r in a.run]
    if len(loaded) > 1 and not a.pool:
        for r in a.run: print(f"=== {r}"); sys.argv = [sys.argv[0], r, "--err-scale", str(a.err_scale), "--mc", str(a.mc)] + (["--strict"] if a.strict else []); main()
        return
    m = loaded[0][0]; rows = pool_rows([x[1] for x in loaded]) if len(loaded) > 1 else loaded[0][1]
    rows.sort(key=lambda r: r["n"]); keep = [r for r in rows if "MELTED" not in r["flags"] and not (a.strict and r["flags"])]
    out = Path(a.run[0]); print(f"{out}{' (+%d replicas pooled)' % (len(a.run) - 1) if len(a.run) > 1 else ''}: {m['P']:g} GPa / {m['T']:g} K   (statistical errors only; shown x{a.err_scale:g})")
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
    if len(keep) >= 2:
        ex = linear_extrapolation([(r["n"], r["dg"], r["err"]) for r in keep], a.err_scale, a.mc)
        if ex and "n_star" in ex: print(f"EXTRAPOLATION (straight line dg(n) = {ex['a']:.0f} + {ex['b']:.0f} n): minimum at n* = {ex['n_star']:.3f}, depth {ex['depth_O']:.1f} meV/O = {ex['depth_atom']:.2f} meV/atom" + (f";  16/50/84 %: n* {ex['n_star_ci'][0]:.3f}/{ex['n_star_ci'][1]:.3f}/{ex['n_star_ci'][2]:.3f}, depth/atom {ex['depth_atom_ci'][0]:.2f}/{ex['depth_atom_ci'][1]:.2f}/{ex['depth_atom_ci'][2]:.2f};  P(slope > 0) = {ex['p_positive_slope']:.2f}" if "n_star_ci" in ex else ""))
        elif ex: print(f"EXTRAPOLATION: straight-line dg(n) has slope {ex['b']:.0f} meV/H per unit n (P(slope > 0) = {ex['p_positive_slope']:.2f}): no minimum implied")
    print("Free-energy errors are statistical only; completion does not establish convergence. Flagged points are marked, melted ones excluded (--strict drops all flagged).")


if __name__ == "__main__":
    main()
