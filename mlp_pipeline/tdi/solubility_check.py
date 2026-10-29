"""Validation of the GRACE answer for the solubility of H in superionic fcc H2O (docs/CURRENT_ISSUES.md section 0): how robust is n* = -mu0 / (2 s)?
Everything here is bookkeeping on the free energies that already exist (no MD, no GPU, numpy + pandas only).

Definitions (per O, meV): n = added H per O, a(n) = AVERAGE dG per added H of H2O + n H(fluid) -> H(2+n)O, G(n) = n a(n); the differential mu_H(n) - G_H = dG/dn.
Solubility limit = the composition where the differential chemical potential equals that of the pure H fluid: dG/dn = 0 (the minimum of G(n); beyond it H2O + H fluid is stable).
Regular solution (what tx_diagram.py / huang_hull.py use): a(n) = mu0 + s n, so dG/dn = mu0 + 2 s n and n* = -mu0 / (2 s); a(n) = 0 at n0 = -mu0 / s = 2 n*.

Mode A (default): docs/data/insertion_regsol_fit.csv (mu0, s per state point) ->
  1. n*, its Monte-Carlo error (mu0, s independent normals; a draw with s <= 0 has no minimum = 'unbounded'), minimum G(n*) per O and per atom,
  2. SHAPE test: G(n) = mu0 n + c n^p with the average pinned to the same value at the crystal composition n_a = 0.25 (where the direct H2.25O crystals sit), p = 1.5 ... 4:
     n* moves by how much if the curvature is not quadratic? (the data constrain mu0 at n -> 0 and a(0.25) only; the minimum lies in between),
  3. MODEL-ERROR test: n* when mu0 is shifted by delta (true = GRACE + delta; the bias proxy says delta ~ -40 +- 50 meV) and s is scaled by f_s,
  4. the same state points on the direct crystal TDI numbers (huang_hull.CRYSTAL): closure of the fit at n = 0.25,
  5. what a new run at n = 0.05 ... 0.25 must resolve: predicted a(n) under each shape, the error to expect and the ps per lambda needed to tell shapes apart.
Mode B (--solid <report.py out>_table_solid.csv, written by report.py): refits the solid rows of the new runs (insertion n = 8, 16, 27, 54 and crystals H2.02 ... H2.25O included) with
  (i) the linear average (as before), (ii) a(n) = mu0 + A n + B n^2 (free curvature), and (iii) the raw data: G(n) = n a(n) at each composition and the secant slope between neighbours;
  n* is only believed where the data bracket it (a measured point below and above n*).
usage: python3 mlp_pipeline/tdi/solubility_check.py [--fit docs/data/insertion_regsol_fit.csv] [--nmc 20000] [--bias -40 50] [--ps 5] [-o docs/data/solubility_check.csv]
       python3 mlp_pipeline/tdi/solubility_check.py --solid docs/figures/tdi_report_table_solid.csv      (report.py's table, with its QA exclusions)
       python3 mlp_pipeline/tdi/solubility_check.py --mix docs/data/tdi_results_mix.csv                  (summarize.py's table; numpy + pandas only, runs on the Young login node)
       python3 mlp_pipeline/tdi/solubility_check.py --selftest
"""
import argparse, csv, os, sys
import numpy as np

KB = 8.617333262e-2   # meV / K
NA = 0.25             # composition of the direct crystal points (27 added H in 108 O)
SHAPES = (1.5, 2.0, 3.0, 4.0)
HERE = os.path.dirname(os.path.abspath(__file__))


ALIAS = {"P": "P_GPa", "T": "T_K", "slope": "s", "slope_err": "s_err"}   # report.py's <out>_table_mu0.csv uses the short names


def rows(f):
    return [{ALIAS.get(k, k): float(v) for k, v in r.items()} for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]


def nstar_quad(mu0, s):
    """regular solution: n* = -mu0 / (2 s) (nan unless mu0 < 0 < s)"""
    mu0, s = np.asarray(mu0, float), np.asarray(s, float)
    return np.where((mu0 < 0) & (s > 0), -mu0 / (2 * np.where(s > 0, s, 1.0)), np.nan)


def nstar_shape(mu0, s, p, na=NA):
    """G(n) = mu0 n + c n^p with a(na) = mu0 + s na  ->  c = s na^(2-p);  dG/dn = mu0 + p c n^(p-1) = 0"""
    c = s * na ** (2 - p)
    return (-mu0 / (p * c)) ** (1 / (p - 1)) if (mu0 < 0 and c > 0) else np.nan


def gmin_per_atom(mu0, s):
    n = nstar_quad(mu0, s); return -mu0 ** 2 / (4 * s) / (3 + n)


def mc_nstar(mu0, e0, s, es, nmc, rng, cap=1.0):
    """quantiles of n* over normal draws of mu0, s; draws with s <= 0 (no minimum within the model) count as unbounded and enter the quantiles as `cap`"""
    m = rng.normal(mu0, e0, nmc); ss = rng.normal(s, es, nmc)
    bad = (ss <= 0) | (m >= 0); n = np.where(bad, cap, np.minimum(-m / (2 * np.where(ss > 0, ss, 1.0)), cap))
    return np.percentile(n, [16, 50, 84]), float(bad.mean())


def pr(*a):
    print(*a, flush=True)


def mode_a(a):
    fit = rows(a.fit); rng = np.random.default_rng(a.seed); out = []
    pr(f"== A1. regular solution n* = -mu0 / (2 s) from {a.fit} ({len(fit)} state points; errors are the weighted-fit errors of report.py, statistical only)")
    pr("   P/GPa   T/K  mu0/meV (+-)      s/meV per H/O (+-)   kT/meV  mu0/kT  n*=-mu0/2s   MC 16/50/84 %          unbounded  n0=-mu0/s  Gmin/O   Gmin/atom   H(2+n*)O")
    for r in fit:
        mu0, e0, s, es, T = r["mu0"], r["mu0_err"], r["s"], r["s_err"], r["T_K"]
        q, ub = mc_nstar(mu0, e0, s, es, a.nmc, rng); n = float(nstar_quad(mu0, s)); kt = KB * T
        pr(f"   {r['P_GPa']:5.0f} {T:6.0f}  {mu0:7.1f} ({e0:4.1f})   {s:7.0f} ({es:5.0f})      {kt:6.1f}  {mu0 / kt:6.2f}   {n:7.3f}     {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}   {100 * ub:5.1f} %     {-mu0 / s:6.3f}   {-mu0 ** 2 / (4 * s):6.1f}  {gmin_per_atom(mu0, s):7.2f}   H{2 + n:.3f}O")
        out.append(dict(P_GPa=r["P_GPa"], T_K=T, mu0=mu0, s=s, nstar=n, nstar_p16=q[0], nstar_p50=q[1], nstar_p84=q[2], unbounded=ub, Gmin_per_atom=float(gmin_per_atom(mu0, s))))
    pr("   note: s_err of 1200-1600 meV (400/6000, 400/6500, 500/6500, 500/7000) comes from fits with only 2 usable points; n* is then essentially undetermined at that (P, T). The three tightest states are 400/5000, 500/5000, 500/5500, 500/6000.")
    pr("   note: mu0 and s are anti-correlated in the fits (the n = 0.25 crystal pins a(0.25) = mu0 + 0.25 s); treating them as independent overstates the error of n* when the crystal point dominates.")

    pr(f"\n== A2. SHAPE test: G(n) = mu0 n + c n^p, a(n) pinned to mu0 + s*{NA} at n = {NA} (p = 2 is the regular solution). n* for p = " + ", ".join(f"{p:g}" for p in SHAPES))
    pr("   P/GPa   T/K   " + "   ".join(f"p={p:<4g}" for p in SHAPES) + "   spread (max/min)")
    for r in fit:
        v = [nstar_shape(r["mu0"], r["s"], p) for p in SHAPES]
        pr(f"   {r['P_GPa']:5.0f} {r['T_K']:6.0f}   " + "   ".join(f"{x:6.3f}" for x in v) + f"      {np.nanmax(v) / np.nanmin(v):.2f}")
    pr("   reading: a stiffer wall (p = 3, 4: the cost of H rises late) pushes n* UP; a softer one (p = 1.5) pushes it DOWN. The two end points (n -> 0 and n = 0.25) fix the scale, the shape is not measured:"
       " insertion at n = 8, 16 (0.074, 0.148) and crystals H2.05 / H2.10 / H2.20O measure it.")

    d0, ds = a.bias; sc = (0.7, 0.85, 1.0, 1.15, 1.3)
    pr(f"\n== A3. MODEL-ERROR test: true mu0 = GRACE mu0 + delta (delta < 0: GRACE mu_H too high, as the bias proxy says: {d0:+.0f} +- {ds:.0f} meV per H), true s = f_s * GRACE s. n* (regular solution):")
    pr("   P/GPa   T/K   f_s   " + "   ".join(f"d={d:+4.0f}" for d in (-200, -100, d0, 0, 100, 200)) + "   | delta that halves n* | delta that makes H 'insoluble' (n* < 0.01)")
    for r in fit:
        if r["s_err"] > 0.5 * r["s"]: continue   # undetermined states: not worth tabulating
        for f in sc:
            vals = [float(nstar_quad(r["mu0"] + d, f * r["s"])) for d in (-200, -100, d0, 0, 100, 200)]
            half = -(r["mu0"] / 2.0) ; ins = -r["mu0"] - 2 * f * r["s"] * 0.01   # n* = -(mu0+d)/(2 f s): n* halves at d = -mu0/2 ; n* < 0.01 at d > -mu0 - 0.02 f s
            pr(f"   {r['P_GPa']:5.0f} {r['T_K']:6.0f}  {f:4.2f}   " + "   ".join(f"{x:7.3f}" for x in vals) + f"   |  {half:+6.0f} meV             | {ins:+6.0f} meV")
    good = [r for r in fit if r["s_err"] <= 0.5 * r["s"]]
    shift = [float(nstar_quad(r["mu0"] + d0, r["s"]) / nstar_quad(r["mu0"], r["s"]) - 1) for r in good]; thr = [-r["mu0"] - 0.02 * r["s"] for r in good]
    pr(f"   reading: dn*/d(delta) = -1/(2 s) (~ -2e-4 per meV): the proxy shift delta = {d0:+.0f} meV changes n* by {100 * min(shift):+.0f} ... {100 * max(shift):+.0f} % over the determined state points, less than the shape spread of A2."
       f" H stays 'soluble' (n* > 0.01) for every delta below {min(thr):+.0f} (300-400 GPa) ... {max(thr):+.0f} meV (500 GPa); n* itself scales as 1/f_s, so a +-30 % error in s is a -+30 % error in n*.")

    per_h = 4.0 * a.static_offset   # H3O: 4 atoms per extra H
    pr(f"\n== A3b. the STATIC H3O offset of GRACE (+{a.static_offset:.0f} meV/atom above Huang's static DFT at 300-500 GPa, mlp_pipeline/h3o_static.py) = {per_h:.0f} meV per extra H. It is an ordered 0 K crystal at n = 1,"
       " the free energy of the dilute superionic solution may or may not inherit it. Three bracketing readings of the same size (true = GRACE minus the offset):")
    pr("   P/GPa   T/K    n*(GRACE)   A: all in mu0 (mu0 - off)   B: all in s (s - off, offset sits at n = 1)   C: half and half    (n*, ratio to GRACE)")
    for r in good:
        mu0, s_ = r["mu0"], r["s"]; n0 = float(nstar_quad(mu0, s_)); txt = []
        for m_, s2 in ((mu0 - per_h, s_), (mu0, s_ - per_h), (mu0 - per_h / 2, s_ - per_h / 2)):
            n2 = float(nstar_quad(m_, s2)); txt.append("no minimum (H3O-like phase favoured)" if not np.isfinite(n2) else f"{n2:6.3f} (x{n2 / n0:4.2f})")
        pr(f"   {r['P_GPa']:5.0f} {r['T_K']:6.0f}    {n0:6.3f}      " + "      ".join(f"{t:>28s}" for t in txt))
    pr("   reading: a constant per-H error (A) would almost double n*; an error that appears only where the lattice is full (B) raises it by 10-20 %; a bias of the size the DFT-frame proxy gave (A3: -40 meV) is a few %."
       " Which of the three is right is exactly what prep-static (VASP on the h3o_static structures: n = 1, 0.75, 0.5, 0.25 extra H per O) and prep-fep (hot frames) decide.")

    pr("\n== A4. closure: the fit against the direct crystal TDI points (huang_hull.CRYSTAL, n = 0.25 or 0.5; average dG per added H, meV). Residuals in sigma of the point (fit errors not included):")
    try:
        sys.path.insert(0, HERE)
        import huang_hull
        crystal = huang_hull.CRYSTAL
    except Exception as ex:
        crystal = []; pr(f"   (huang_hull.CRYSTAL not importable: {ex})")
    for r in fit:
        pts = [c for c in crystal if abs(c[0] - r["P_GPa"]) < 1 and abs(c[1] - r["T_K"]) < 1 and c[2] == NA]
        if not pts: continue
        mod = r["mu0"] + r["s"] * NA
        pr(f"   {r['P_GPa']:4.0f} GPa {r['T_K']:5.0f} K: fit a(0.25) = {mod:7.1f} meV;  crystals: " + ", ".join(f"{c[3]:6.1f}+-{c[4]:4.1f} ({(c[3] - mod) / c[4]:+.1f} sigma)" for c in pts)
           + f";  mean {np.mean([c[3] for c in pts]):.1f}, scatter between runs {np.std([c[3] for c in pts], ddof=1) if len(pts) > 1 else float('nan'):.1f} meV")
    pr("   note: the quoted errors of the crystal points are block errors of one run each. The scatter between repeated H2.25O runs (+-30 meV per H) and the Kronrod-Gauss quadrature estimate (up to ~85 meV per H"
       " at 500/6000, x13 amplification) are NOT in the fit weights: treat s_err as a lower limit.")

    pr(f"\n== A5. planning: predicted average a(n) = G/n (meV per added H) under each shape for the runs not yet in the fit, and the error to expect (insertion: ~40 meV at 2 ps, independent of the cell;"
       f" crystal in 108 O: ~1150 meV / n_added at 1.5 ps; both scale as 1/sqrt(ps per lambda); {a.ps:g} ps assumed)")
    for r in fit:
        if (r["P_GPa"], r["T_K"]) not in ((500, 6000), (500, 5000), (400, 5000)): continue
        pr(f"   -- {r['P_GPa']:.0f} GPa {r['T_K']:.0f} K (mu0 {r['mu0']:.0f}, s {r['s']:.0f}): n*(p=2) = {float(nstar_quad(r['mu0'], r['s'])):.3f}")
        pr("      n_add/108 O   n        " + "   ".join(f"p={p:<4g}" for p in SHAPES) + "   err_insertion  err_crystal   p=2 vs 3 (sigma)   p=2 vs 1.5 (sigma)")
        for nadd in (4, 8, 11, 16, 22, 27):
            n = nadd / 108; v = [r["mu0"] + r["s"] * NA ** (2 - p) * n ** (p - 1) for p in SHAPES]
            ei = 40.0 * np.sqrt(2.0 / a.ps); ec = 1150.0 / nadd * np.sqrt(1.5 / a.ps)
            pr(f"      {nadd:4d}        {n:5.3f}   " + "   ".join(f"{x:7.1f}" for x in v) + f"   {ei:7.0f}       {ec:7.0f}        {abs(v[1] - v[2]) / min(ei, ec):6.1f}             {abs(v[1] - v[0]) / min(ei, ec):6.1f}")
    sg = []
    for r in fit:
        if (r["P_GPa"], r["T_K"]) not in ((500, 6000), (500, 5000), (400, 5000)): continue
        for nadd in (8, 16):
            n = nadd / 108; v = [r["mu0"] + r["s"] * NA ** (2 - p) * n ** (p - 1) for p in SHAPES]; ei = 40.0 * np.sqrt(2.0 / a.ps)
            sg += [abs(v[1] - v[2]) / ei, abs(v[1] - v[0]) / ei]
    pr(f"   reading: NOMINAL separation of the shapes by insertion at n_add = 8 or 16 is {min(sg):.1f} ... {max(sg):.1f} sigma at {a.ps:g} ps per lambda (cost like n = 1; the per-H error of insertion is assumed independent of n)."
       " It is nominal because mu0 and s are refitted together with the new points; the crystal route only competes at n_add >= 22 and serves as the independent bookkeeping check.")
    if a.o:
        keys = list(out[0]); w = csv.DictWriter(open(a.o, "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(out); pr(f"\nwrote {a.o}")


# ---------------------------------------------------------------- mode B: refit of the solid rows of a new report
def wls(A, y, e):
    w = 1 / np.maximum(e, 1e-6) ** 2; cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * y)); chi2 = float(np.sum(w * (y - A @ b) ** 2))
    return b, cov, chi2


def nstar_poly(b):
    """a(n) = b0 + b1 n + b2 n^2 -> dG/dn = b0 + 2 b1 n + 3 b2 n^2 = 0: smallest positive root (nan if none)"""
    r = np.roots([3 * b[2], 2 * b[1], b[0]]) if len(b) > 2 and abs(b[2]) > 0 else np.array([-b[0] / (2 * b[1])]) if b[1] != 0 else np.array([])
    r = np.array([x.real for x in np.atleast_1d(r) if abs(np.imag(x)) < 1e-12 and x.real > 0])
    return float(r.min()) if len(r) else float("nan")


def solid_from_mix(path):
    """the solid-solution rows of summarize.py's <base>_mix.csv in the layout of report.py's table_solid.csv (no matplotlib needed). fcc-host insertion rows into H2O (dilute) and fcc crystals only;
    caveat = different dt / lambda grid, melted or marginal in free NVT, or no stage-0 pressure correction. NOT applied here (report.py does): QA exclusions (unfinished, |P_hi - P| > 12 %, H-H collapse, frozen liquids)."""
    import pandas as pd
    m = pd.read_csv(path); need = {"phase", "P_GPa", "T_K", "H_per_O", "n_added", "dG_per_added_H_meV", "err", "nO"}
    if not need <= set(m.columns):
        raise SystemExit(f"{path}: missing columns {sorted(need - set(m.columns))} (summarize.py of this version writes nO and tag into the mixing table: rerun it)")
    rows_ = []
    for _, r in m.iterrows():
        ph = str(r.phase); note = "" if pd.isna(r.get("note", "")) else str(r.get("note", ""))
        if ph == "fcc": kind, a_ = "fcc", r.H_per_O - 2.0
        elif ph.startswith("insert (fcc H2O, dilute)"): kind, a_ = "insert", r.n_added / r.nO
        else: continue
        rows_.append(dict(kind=kind, P=r.P_GPa, T=r.T_K, h=r.H_per_O, a=a_, n=r.n_added, dG=r.dG_per_added_H_meV, err=r.err, tags=str(r.get("tag", "")),
                          caveat=any(k in note for k in ("DIFFERENT", "MELTED", "marginal", "NO PRESSURE")), pcorr=np.nan if "NO PRESSURE CORRECTION" in note else 0.0))
    return pd.DataFrame(rows_, columns=["kind", "P", "T", "h", "a", "n", "dG", "err", "caveat", "tags", "pcorr"])


def mode_b(a):
    import pandas as pd
    src = a.solid or a.mix
    d = solid_from_mix(a.mix) if a.mix else pd.read_csv(a.solid); need = {"kind", "P", "T", "a", "n", "dG", "err", "caveat"}
    if not need <= set(d.columns):
        raise SystemExit(f"{src}: missing columns {sorted(need - set(d.columns))}")
    d = d[d.kind.isin(["insert", "fcc"])].copy(); pr(f"{len(d)} fcc-host rows (insertion 'insert' and fcc crystals) in {src}")
    if "pcorr" in d.columns:
        nc = d[(d.kind == "insert") & d.pcorr.isna()]
        if len(nc): pr(f"WARNING: {len(nc)} insertion rows have NO pressure correction (stage 0 older than the fix): their dG is at the cell's own pressure, not P_target -> excluded from the fits below: "
                       + "; ".join(f"{r.P:.0f}/{r['T']:.0f} n={r.n}" for _, r in nc.iterrows()))
        d = d[~((d.kind == "insert") & d.pcorr.isna())]
    out = []
    for (P, T), g in sorted(d.groupby(["P", "T"])):
        g = g[(~g.caveat.astype(bool)) & (g.a > 0)].sort_values("a"); kt = KB * T
        if g.a.nunique() < 2: continue
        x, y, e = g.a.values, g.dG.values, g.err.values
        pr(f"\n-- {P:.0f} GPa, {T:.0f} K: {len(g)} usable rows (kT {kt:.0f} meV)")
        pr("      kind    n_add     a=n/O      avg dG per added H (meV)      G(n)/O = n*avg (meV)    secant dG/dn to the previous point (meV per H/O -> per H)")
        prev = None
        for _, r in g.iterrows():
            if prev is None: sec = ""
            elif abs(r.a - prev.a) < 1e-9: sec = f"(same composition as the previous row: difference of the two routes {r.dG - prev.dG:+.1f} meV per H)"
            else: sec = f"{(r.a * r.dG - prev.a * prev.dG) / (r.a - prev.a):9.0f}  ({(r.a * r.dG - prev.a * prev.dG) / (r.a - prev.a) / 108:7.1f} per H in 108 O)"
            pr(f"      {r.kind:6s} {r.n:6.0f}   {r.a:7.4f}   {r.dG:8.1f} +- {r.err:5.1f}              {r.a * r.dG:9.1f}               {sec}"); prev = r
        res = dict(P_GPa=P, T_K=T, rows=len(g), a_min=x.min(), a_max=x.max())
        A1 = np.vstack([np.ones_like(x), x]).T; b1, c1, chi1 = wls(A1, y, e)
        n1 = float(nstar_quad(b1[0], b1[1])); res.update(mu0_lin=b1[0], s_lin=b1[1], chi2_lin=chi1, dof_lin=len(x) - 2, nstar_lin=n1)
        pr(f"   (i) linear average: mu0 {b1[0]:.1f} +- {np.sqrt(c1[0, 0]):.1f}, s {b1[1]:.0f} +- {np.sqrt(c1[1, 1]):.0f}  chi2 {chi1:.1f} / {len(x) - 2} dof  ->  n* = {n1:.3f}")
        if len(x) >= 4 and np.unique(x).size >= 3:
            A2 = np.vstack([np.ones_like(x), x, x ** 2]).T; b2, c2, chi2_ = wls(A2, y, e); n2 = nstar_poly(b2)
            res.update(mu0_quad=b2[0], A=b2[1], B=b2[2], chi2_quad=chi2_, dof_quad=len(x) - 3, nstar_quad=n2)
            pr(f"   (ii) a(n) = mu0 + A n + B n^2: mu0 {b2[0]:.1f}, A {b2[1]:.0f}, B {b2[2]:.0f}  chi2 {chi2_:.1f} / {len(x) - 3} dof  ->  n* = {n2:.3f}  "
               f"({'curvature significant: B/err = %.1f' % (b2[2] / np.sqrt(c2[2, 2])) if abs(b2[2] / np.sqrt(c2[2, 2])) > 2 else 'curvature NOT significant'})")
        lo = (x < (n1 if np.isfinite(n1) else 0.11)).any(); hi = (x > (n1 if np.isfinite(n1) else 0.11)).any()
        res["bracketed"] = bool(lo and hi)
        pr(f"   data below / above n*: {'yes / yes: n* is interpolated' if lo and hi else 'NO: n* is an extrapolation (need runs on both sides, e.g. n_add = 8 and 16 in 108 O)'}")
        out.append(res)
    if a.o and out:
        keys = list(dict.fromkeys(k for r in out for k in r)); w = csv.DictWriter(open(a.o, "w", newline=""), fieldnames=keys, restval=""); w.writeheader(); w.writerows(out); pr(f"\nwrote {a.o}")


# ---------------------------------------------------------------- self test: synthetic report with a known curve
def selftest():
    import tempfile, pandas as pd
    rng = np.random.default_rng(1); fails = 0
    pr("selftest: synthetic G(n) = mu0 n + c n^p with noise; the fits must recover n* (quadratic data: linear-average fit exact; cubic data: linear fit biased, free curvature close)")
    for p, tol_lin, tol_quad in ((2.0, 0.02, 0.03), (3.0, None, 0.04)):
        mu0, nstar_true = -500.0, 0.12; c = -mu0 / (p * nstar_true ** (p - 1))
        rowsl = []
        for kind, nadd in (("insert", 1), ("insert", 2), ("insert", 4), ("insert", 8), ("insert", 16), ("insert", 27), ("fcc", 27), ("fcc", 11)):
            n = nadd / 108; err = 25.0 if kind == "insert" else 30.0
            rowsl.append(dict(kind=kind, P=500.0, T=6000.0, h=2.0, a=n, n=nadd, dG=mu0 + c * n ** (p - 1) + rng.normal(0, err), err=err, caveat=False, tags="x", pcorr=-30.0))
        f = os.path.join(tempfile.mkdtemp(), "t.csv"); pd.DataFrame(rowsl).to_csv(f, index=False)
        a = argparse.Namespace(solid=f, mix=None, o=None); import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): mode_b(a)
        txt = buf.getvalue(); pr(f"-- true shape p = {p:g}, true n* = {nstar_true}"); pr("   " + txt.replace("\n", "\n   "))
        got_lin = float(txt.split("(i) linear average")[1].split("n* = ")[1].split()[0]); got_quad = float(txt.split("(ii)")[1].split("n* = ")[1].split()[0])
        ok_q = abs(got_quad - nstar_true) <= tol_quad
        ok_l = True if tol_lin is None else abs(got_lin - nstar_true) <= tol_lin
        chi_lin = float(txt.split("chi2 ")[1].split(" /")[0]); dof_lin = int(txt.split("chi2 ")[1].split("/ ")[1].split()[0])
        ok_c = (chi_lin > 3 * dof_lin) if p != 2.0 else (chi_lin < 3 * dof_lin)   # the linear fit must announce its own misfit on curved data (chi2 >> dof) and fit straight data
        pr(f"   linear n* {got_lin:.3f} ({'ok' if ok_l else 'FAIL'}), free-curvature n* {got_quad:.3f} ({'ok' if ok_q else 'FAIL'}), linear chi2 {chi_lin:.1f} / {dof_lin} dof ({'misfit flagged as it should' if ok_c and p != 2.0 else 'good fit' if ok_c else 'FAIL'})")
        fails += (not ok_q) + (not ok_l) + (not ok_c)
    pr("selftest " + ("PASSED" if not fails else f"FAILED ({fails})")); return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="docs/data/insertion_regsol_fit.csv"); ap.add_argument("--solid", default=None, help="mode B: report.py <out>_table_solid.csv (with the report's QA exclusions)")
    ap.add_argument("--mix", default=None, help="mode B without matplotlib: summarize.py <base>_mix.csv (e.g. docs/data/tdi_results_mix.csv; needs the nO column of this version)")
    ap.add_argument("--nmc", type=int, default=20000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bias", type=float, nargs=2, default=[-40.0, 50.0], metavar=("DELTA", "ERR"),
                    help="model-error shift of mu0 (meV) and its error: the bias proxy (valid_m + valid_l + train_m) gave GRACE mu_H - G_H too high by 40 +- 15 (50 allows for the systematics)")
    ap.add_argument("--static-offset", type=float, default=95.0, help="GRACE minus Huang static dH_f of H3O (meV/atom) used in A3b: +90 ... +100 at 300-500 GPa from h3o_static.py")
    ap.add_argument("--ps", type=float, default=5.0, help="ps per lambda assumed for the planning table"); ap.add_argument("-o", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(1 if selftest() else 0)
    (mode_b if (a.solid or a.mix) else mode_a)(a)


if __name__ == "__main__":
    main()
