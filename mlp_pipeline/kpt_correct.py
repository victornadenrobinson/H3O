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


def force_change(g, f):
    """rms of F_k - F_Gamma over atoms (eV/A) and the rms Gamma force, atoms matched by species + minimum-image position (VASP reorders species); (nan, nan) if no match"""
    from ase.geometry import get_distances
    idx = np.full(len(f), -1)
    for z in set(f.numbers):
        a_, b_ = np.nonzero(f.numbers == z)[0], np.nonzero(g.numbers == z)[0]
        if len(a_) != len(b_): return float("nan"), float("nan")
        D = get_distances(f.positions[a_], g.positions[b_], cell=g.cell, pbc=True)[1]; j = D.argmin(axis=1)
        if D[np.arange(len(a_)), j].max() > 2e-3: return float("nan"), float("nan")
        idx[a_] = b_[j]
    try: Fg = g.get_forces()[idx]; Fk = f.get_forces()
    except Exception: return float("nan"), float("nan")
    return float(np.sqrt(((Fk - Fg) ** 2).mean())), float(np.sqrt((Fg ** 2).mean()))


def per_added_H(c, nO, nH, V, T):
    """shift of the energy of one added H implied by the fit (meV per H): N de at (nO, nH + 1, V) minus at (nO, nH, V), MP level"""
    e1 = (nO + nH + 1) * float(feats(nO, nH + 1, V, T, 1.0)[0] @ c); e0 = (nO + nH) * float(feats(nO, nH, V, T, 1.0)[0] @ c); return e1 - e0


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
        dF, Frms = force_change(g, f)
        # shift per added H implied by the fit is evaluated after the fit (needs the coefficients): keep the state
        rows.append(dict(src=src, grp=group(g.info), cls=comp_class(nO, nH), N=N, nO=nO, nH=nH, V=V, T=T, P=Pg, phi=phi, x=x, n=n, t=t, h=h, mp=mp, de=1000 * (Ek - Eg) / N, dP=Pk - Pg, dF=dF, Frms=Frms))
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
    for r in rows:
        r["cH"] = per_added_H(c, r["nO"], r["nH"], r["V"], r["T"])
        r["cHref"] = r["cH"] - float(feats(0, 729, 729 / r["n"], r["T"], 1.0)[0] @ c)     # minus the per-atom shift of pure H at the same density and T (the insertion reference; rough: pure H at the same P is denser)
    print("\nshift per added H implied by the fit (meV/H, sign E_k - E_Gamma), raw and relative to pure H at the same density and T (the k-point test: c_bias = -53 meV/H for H2O -> H2.25O at 500 GPa,"
          " 5000 K, defined Gamma - mesh relative to the pure-H reservoir, i.e. about +53 in this sign):")
    byc = collections.defaultdict(list)
    byr = collections.defaultdict(list)
    for r in rows:
        if r["nO"]: byc[r["cls"]].append(r["cH"]); byr[r["cls"]].append(r["cHref"])
    for k_, v in sorted(byc.items()): print(f"  {k_:16s} n {len(v):3d}: raw mean {np.mean(v):+7.1f} (range {min(v):+.1f}..{max(v):+.1f}); relative to pure H {np.mean(byr[k_]):+7.1f} (range {min(byr[k_]):+.1f}..{max(byr[k_]):+.1f})")
    fd = [r for r in rows if r["dF"] == r["dF"]]
    if fd:
        print("force change F_k - F_Gamma (rms over atoms; the correction leaves forces at Gamma level):")
        byf = collections.defaultdict(list)
        for r in fd: byf[r["cls"]].append((r["dF"], r["dF"] / max(r["Frms"], 1e-9)))
        for k_, v in sorted(byf.items()): v = np.array(v); print(f"  {k_:16s} n {len(v):3d}: {v[:, 0].mean():.3f} eV/A ({100 * v[:, 1].mean():.1f} % of the rms force), max {v[:, 0].max():.3f}")
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
        # maps: the correction in P, T and composition space, the fitted surface, the shift per added H, the force change
        fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        P_ = np.array([max(r["P"], 1.0) for r in rows]); T_ = np.array([r["T"] for r in rows]); x_ = np.array([r["x"] for r in rows])
        sc = ax[0, 0].scatter(P_, y, c=x_, cmap="viridis", s=18, edgecolor="k", lw=0.2); ax[0, 0].set_xscale("log"); ax[0, 0].set_xlabel("P (GPa, DFT Gamma label)"); ax[0, 0].set_ylabel("E_k - E_Gamma (meV/atom)"); fig.colorbar(sc, ax=ax[0, 0], label="H atom fraction")
        sc = ax[0, 1].scatter(P_, T_, c=y, cmap="RdBu_r", s=26, edgecolor="k", lw=0.2, vmin=-max(abs(y)), vmax=max(abs(y))); ax[0, 1].set_xscale("log"); ax[0, 1].set_xlabel("P (GPa)"); ax[0, 1].set_ylabel("T_el (K)"); fig.colorbar(sc, ax=ax[0, 1], label="E_k - E_Gamma (meV/atom)")
        sc = ax[0, 2].scatter(x_, y, c=np.log10(P_), cmap="plasma", s=18, edgecolor="k", lw=0.2); ax[0, 2].set_xlabel("H atom fraction (2/3 = H2O, 1 = pure H)"); ax[0, 2].set_ylabel("E_k - E_Gamma (meV/atom)"); fig.colorbar(sc, ax=ax[0, 2], label="log10 P")
        xs = np.linspace(2 / 3, 0.9, 60); ns = np.linspace(rng["n"][0], rng["n"][1], 60); XX, NN = np.meshgrid(xs, ns)
        ZZ = np.array([[float(np.array([1, xx, xx * xx, nn, nn * xx, nn * nn, 0.6, 0.6 * xx, 0, 0, 0, 1, xx, 0]) @ c) for xx in xs] for nn in ns])
        cs = ax[1, 0].contourf(XX, NN, ZZ, 20, cmap="RdBu_r"); fig.colorbar(cs, ax=ax[1, 0], label="fitted de at 6000 K (meV/atom)"); ax[1, 0].scatter([r["x"] for r in rows if not r["h"]], [r["n"] for r in rows if not r["h"]], s=8, color="k")
        ax[1, 0].set_xlabel("H atom fraction"); ax[1, 0].set_ylabel("n (atoms/A^3)"); ax[1, 0].set_title("fitted surface (mixtures, T_el 6000 K) + the 300 frames", fontsize=8, loc="left")
        for k_, col in zip(sorted(byc), ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"]):
            rr = [r for r in rows if r["cls"] == k_]; ax[1, 1].scatter([max(r["P"], 1) for r in rr], [r["cHref"] for r in rr], s=16, color=col, label=k_)
        ax[1, 1].axhline(53, color="gray", ls="--", lw=0.8); ax[1, 1].text(0.02, 0.04, "k-point test: c_bias 53 meV/H in this sign (500 GPa, 5000 K)", transform=ax[1, 1].transAxes, fontsize=7, color="gray")
        ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("P (GPa)"); ax[1, 1].set_ylabel("shift per added H rel. to pure H (meV/H)"); ax[1, 1].legend(fontsize=7, frameon=False)
        if fd:
            for k_, col in zip(sorted({r["cls"] for r in fd}), ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#8a5a2b"]):
                rr = [r for r in fd if r["cls"] == k_]; ax[1, 2].scatter([max(r["P"], 1) for r in rr], [100 * r["dF"] / max(r["Frms"], 1e-9) for r in rr], s=16, color=col, label=k_)
            ax[1, 2].set_xscale("log"); ax[1, 2].set_xlabel("P (GPa)"); ax[1, 2].set_ylabel("rms |F_k - F_Gamma| / rms F (%)"); ax[1, 2].legend(fontsize=7, frameon=False)
        fig.savefig(out + "_maps.png", dpi=130); print("figure:", out + "_maps.png")
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
            phi, x, n, t, h = feats(nO, nH, V, T, 1.0); de = float(phi @ c)   # corrected to the MP 2x2x2 level
            dP = n ** 2 * float(dfeat_dn(x, n, t, h) @ c) / 1000 * GPA
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
