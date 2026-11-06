#!/usr/bin/env python3
"""k-point correction of the Gamma-only labels: fit a smooth Delta(composition, density, T) to the 2x2x2 (MP) relabels, check it, apply it to whole tiers.
  fit:    $PY mlp_pipeline/kpt_correct.py fit runs/kpt222mp/labelled_mp222.extxyz --gamma runs/kpt222/frames.extxyz [-o runs/kpt222mp/kcorr]
  apply:  $PY mlp_pipeline/kpt_correct.py apply runs/kpt222mp/kcorr.json runs/train_mfin.extxyz runs/valid_mfin.extxyz [runs/train_mfinref.extxyz ...]
          -> <name>_kc.extxyz next to each input (then: bash mlp_pipeline/grace_prep_pkl.sh mfin_kc ... after renaming / tagging, see the end of the apply output)
Model (per atom, meV): de = sum_k c_k phi_k(x, n, t) with x = H atom fraction nH/N, n = N/V (atoms/A^3), t = T_el/10^4 K (T_el = the label's electronic temperature, SIGMA/kB),
  basis "quad" (default): 1, x, x^2, n, n x, n^2, t, t x, plus a pure-H indicator h and h n, h t (pure H is metallic and needs its own terms). Ridge least squares, weights 1/N (per-atom fit).
Because de depends only on (x, n, t) and not on positions, forces are unchanged and the consistent stress change is dP = n^2 d(de)/dn (from E = N de(n), V = N/n): the fit report
compares this predicted dP with the measured P_k - P_Gamma of the same frames (a consistency check of the energy model), and apply shifts E by N de and the stress by -dP (eV/A^3).
Checks printed by fit: residual sd per cell (eV) and per atom (meV) by data group and by composition class, leave-one-group-out cross-validation (each group predicted by a fit
without it), the range of (x, n, t) covered; apply counts frames outside that range (extrapolated) per group.
Inputs: the k-point labels (points_collect.py output: info kpt_src, E_gamma_eV, T_K) and the Gamma frames they came from (kpt_select.py output, with the Gamma stress)."""
import argparse, collections, json, math, os, sys
import numpy as np

GPA = 160.21766


def feats(nO, nH, V, T, mp=1.0):
    """basis; mp = 1 for a shifted Monkhorst-Pack mesh label, 0 for a Gamma-centred one: the mesh offset terms (mp, mp x, mp h) let both meshes be fitted together; apply uses mp = 1"""
    N = nO + nH; x = nH / N; n = N / V; t = T / 1e4; h = 1.0 if nO == 0 else 0.0
    return np.array([1, x, x * x, n, n * x, n * n, t, t * x, h, h * n, h * t, mp, mp * x, mp * h]), x, n, t, h


def dfeat_dn(x, n, t, h):
    return np.array([0, 0, 0, 1, x, 2 * n, 0, 0, 0, h, 0, 0, 0, 0])


NAMES = ["1", "x", "x^2", "n", "n x", "n^2", "t", "t x", "h", "h n", "h t", "mp", "mp x", "mp h"]


def mesh_of(info):
    """1.0 = Monkhorst-Pack, 0.0 = Gamma-centred, from the KPOINTS file of the label's point folder (points_collect stores point_dir); None if unknown"""
    d = info.get("point_dir")
    try: kp = open(os.path.join(str(d), "KPOINTS")).read().splitlines()[2].strip().lower()
    except (OSError, IndexError, TypeError): return None
    return 1.0 if kp.startswith("m") else 0.0 if kp.startswith("g") else None


def group(info):
    v = info.get("db_set", "old"); v = "F_toy" if v is False else "T" if v is True else str(v)
    return {"old": "AIMD m", "A": "ins A/B", "A_b1": "ins A/B", "A_part": "ins A/B", "B": "ins A/B", "C": "caches C", "C2": "caches C", "D_eos": "EOS D/E", "D_melt": "EOS D/E", "E": "EOS D/E",
            "F_toy": "toy F/G", "F": "toy F/G", "F_H": "toy F/G", "G": "toy F/G", "H": "H3O H", "PD": "pairs PD/PW", "PW": "pairs PD/PW", "LP": "low-P LP"}.get(v, v)


def comp_class(nO, nH):
    if nO == 0: return "pure H"
    r = nH / nO; return "H2O..H2.25O" if r <= 2.26 else "H2.25O..H3O" if r <= 3.01 else "H-rich > H3O"


def T_el(info):
    T = info.get("T_K", info.get("T_set"))
    try: T = float(T)
    except (TypeError, ValueError): return 5000.0
    return max(T, 1000.0) if T == T else 5000.0


def ridge(X, y, w, lam=1e-6):
    W = np.sqrt(w)[:, None]; A = X * W; b = y * W[:, 0]
    return np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ b)


def cmd_fit(a):
    from ase.io import read
    gam = {}
    for f in read(a.gamma, index=":"):
        gam[str(f.info.get("kpt_src"))] = f
    rows = []
    for f in read(a.klabels, index=":"):
        src = str(f.info.get("kpt_src")); g = gam.get(src)
        if g is None: continue
        nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum()); N = len(f); V = f.get_volume(); T = T_el(g.info)
        Eg = float(g.info.get("E_gamma_eV", g.get_potential_energy())); Ek = f.get_potential_energy()
        Pg = -np.trace(g.get_stress(voigt=False)) / 3 * GPA; Pk = -np.trace(f.get_stress(voigt=False)) / 3 * GPA
        mp = mesh_of(f.info); mp = 1.0 if mp is None else mp
        phi, x, n, t, h = feats(nO, nH, V, T, mp)
        rows.append(dict(src=src, grp=group(g.info), cls=comp_class(nO, nH), N=N, phi=phi, x=x, n=n, t=t, h=h, mp=mp, de=1000 * (Ek - Eg) / N, dP=Pk - Pg))
    if len(rows) < 15: sys.exit(f"only {len(rows)} matched frames (k labels with a Gamma partner by kpt_src): check the two files")
    X = np.array([r["phi"] for r in rows]); y = np.array([r["de"] for r in rows]); w = np.array([1.0 for r in rows])
    nmp = sum(r["mp"] for r in rows); print(f"meshes: {int(nmp)} Monkhorst-Pack, {len(rows) - int(nmp)} Gamma-centred labels" + ("" if 0 < nmp < len(rows) else " (one mesh only: the mesh terms are inert)"))
    if not 0 < nmp < len(rows): X[:, -3:] = 0
    c = ridge(X, y, w); pred = X @ c; res = y - pred
    dPp = np.array([r["n"] ** 2 * (dfeat_dn(r["x"], r["n"], r["t"], r["h"]) @ c) / 1000 * GPA for r in rows])   # meV/atom * A^3 -> eV/A^3 -> GPa
    dPm = np.array([r["dP"] for r in rows])
    print(f"{len(rows)} matched frames; de = E_k - E_Gamma per atom: mean {y.mean():+.2f}, sd {y.std():.2f}, range {y.min():+.2f}..{y.max():+.2f} meV/atom")
    print("coefficients (meV/atom): " + ", ".join(f"{nm} {v:+.3g}" for nm, v in zip(NAMES, c)))
    if 0 < nmp < len(rows): print(f"mesh offset (MP - Gamma-centred, from the fit): {c[-3] + c[-2] * 0.75:+.2f} meV/atom at x = 0.75 (water-like), {c[-3] + c[-2] + c[-1]:+.2f} for pure H  (k-point test: ~0.0 and +0.99)")
    print(f"fit residual: sd {res.std():.2f} meV/atom = {np.std(res * np.array([r['N'] for r in rows]) / 1000):.3f} eV/cell; R^2 {1 - res.var() / y.var():.3f}")
    def table(key):
        out = collections.defaultdict(list)
        for r, e, pm, pp in zip(rows, res, dPm, dPp): out[r[key]].append((e, e * r["N"] / 1000, r["de"], pm, pp))
        print(f"\n{'by ' + key:22s} {'n':>4s} {'<de>':>7s} {'res sd meV/at':>13s} {'res sd eV/cell':>14s} {'<dP_meas>':>10s} {'<dP_pred>':>10s}  GPa")
        for k_, v in sorted(out.items()):
            v = np.array(v); print(f"{k_:22s} {len(v):4d} {v[:, 2].mean():+7.2f} {v[:, 0].std():13.2f} {v[:, 1].std():14.3f} {v[:, 3].mean():+10.2f} {v[:, 4].mean():+10.2f}")
    table("cls"); table("grp")
    print("\nleave-one-group-out cross-validation (each group predicted by a fit without it; mean / sd of the error, meV/atom):")
    for gname in sorted({r["grp"] for r in rows}):
        tr = np.array([r["grp"] != gname for r in rows])
        if tr.sum() < 15 or (~tr).sum() < 2: continue
        cg = ridge(X[tr], y[tr], w[tr]); e = y[~tr] - X[~tr] @ cg
        print(f"  {gname:16s} n {int((~tr).sum()):3d}: mean {e.mean():+6.2f}  sd {e.std():5.2f}  max |e| {np.abs(e).max():5.2f}")
    rng = {k: [float(min(r[k] for r in rows)), float(max(r[k] for r in rows))] for k in ("x", "n", "t")}
    rngH = {k: [float(min(r[k] for r in rows if r["h"])), float(max(r[k] for r in rows if r["h"]))] for k in ("n", "t")} if any(r["h"] for r in rows) else None
    print(f"\ncovered: x {rng['x'][0]:.3f}-{rng['x'][1]:.3f}, n {rng['n'][0]:.3f}-{rng['n'][1]:.3f} atoms/A^3, T_el {1e4 * rng['t'][0]:.0f}-{1e4 * rng['t'][1]:.0f} K" + (f"; pure H n {rngH['n'][0]:.3f}-{rngH['n'][1]:.3f}" if rngH else ""))
    out = a.o or os.path.join(os.path.dirname(a.klabels), "kcorr")
    json.dump(dict(basis="quad", names=NAMES, coef=[float(v) for v in c], range=rng, range_pureH=rngH, n_frames=len(rows), res_sd_meV_atom=float(res.std()), source=a.klabels), open(out + ".json", "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True); grps = sorted({r["grp"] for r in rows}); cm = plt.get_cmap("tab10")
        for i, gname in enumerate(grps):
            m = np.array([r["grp"] == gname for r in rows])
            ax[0].scatter(y[m], pred[m], s=14, color=cm(i % 10), label=gname); ax[1].scatter(X[m, 1], res[m], s=14, color=cm(i % 10)); ax[2].scatter(dPm[m], dPp[m], s=14, color=cm(i % 10))
        lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]; ax[0].plot(lim, lim, "k-", lw=0.6); ax[0].set_xlabel("measured E_k - E_Gamma (meV/atom)"); ax[0].set_ylabel("fit"); ax[0].legend(fontsize=7, frameon=False)
        ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("H atom fraction x"); ax[1].set_ylabel("residual (meV/atom)")
        lp = [min(dPm.min(), dPp.min()), max(dPm.max(), dPp.max())]; ax[2].plot(lp, lp, "k-", lw=0.6); ax[2].set_xlabel("measured P_k - P_Gamma (GPa)"); ax[2].set_ylabel("dP = n^2 d(de)/dn from the energy fit (GPa)")
        fig.savefig(out + ".png", dpi=130); print("figure:", out + ".png")
    except Exception as e: print("(no figure:", e, ")")
    print("model:", out + ".json")


def cmd_apply(a):
    from ase.io import read, write
    from ase.calculators.singlepoint import SinglePointCalculator
    m = json.load(open(a.model)); c = np.array(m["coef"]); rng = m["range"]; rngH = m.get("range_pureH")
    for p in a.tiers:
        fr = read(p, index=":"); out = []; shift = collections.defaultdict(list); extrap = collections.Counter()
        for f in fr:
            nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum()); N = len(f); V = f.get_volume(); T = T_el(f.info)
            phi, x, n, t, h = feats(nO, nH, V, T, 1.0); de = float(phi @ c)   # corrected to the MP 2x2x2 level; dP = n ** 2 * float(dfeat_dn(x, n, t, h) @ c) / 1000 * GPA
            outside = not (rng["x"][0] - 0.02 <= x <= rng["x"][1] + 0.02 and rng["n"][0] * 0.97 <= n <= rng["n"][1] * 1.03 and rng["t"][0] - 0.05 <= t <= rng["t"][1] + 0.05)
            if h and rngH: outside = outside or not (rngH["n"][0] * 0.97 <= n <= rngH["n"][1] * 1.03)
            if outside: extrap[group(f.info)] += 1
            E = f.get_potential_energy() + N * de / 1000; st = f.get_stress(voigt=False) - np.eye(3) * dP / GPA
            g = f.copy(); g.info = dict(f.info); g.info.update(kcorr_meV_atom=round(de, 4), kcorr_dP_GPa=round(dP, 4), kcorr_extrap=bool(outside))
            if "energy_sigma0" in g.info: g.info["energy_sigma0"] = float(g.info["energy_sigma0"]) + N * de / 1000
            g.calc = SinglePointCalculator(g, energy=E, free_energy=E, forces=f.get_forces(), stress=st); out.append(g); shift[comp_class(nO, nH)].append((de, dP))
        o = p.replace(".extxyz", "_kc.extxyz"); write(o, out)
        print(f"{p}: {len(out)} frames -> {o}; extrapolated (outside the fitted range): {sum(extrap.values())} " + (str(dict(extrap)) if extrap else ""))
        for k_, v in sorted(shift.items()):
            v = np.array(v); print(f"   {k_:16s} {len(v):5d} frames: de {v[:, 0].mean():+6.2f} (range {v[:, 0].min():+.2f}..{v[:, 0].max():+.2f}) meV/atom, dP {v[:, 1].mean():+5.2f} GPa")
    print("next: give the corrected tiers a tag (e.g. cp runs/train_mfin_kc.extxyz runs/train_mfinkc.extxyz, same for valid) and bash mlp_pipeline/grace_prep_pkl.sh mfinkc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("klabels"); f.add_argument("--gamma", required=True); f.add_argument("-o", default=None)
    p = sp.add_parser("apply"); p.add_argument("model"); p.add_argument("tiers", nargs="+")
    a = ap.parse_args(); {"fit": cmd_fit, "apply": cmd_apply}[a.cmd](a)
