#!/usr/bin/env python3
"""All EOS scans (eos_live.sh / 17_eos_md.py result tables) on one figure, against the DFT EOS of data-h3o-convex-hull.csv, in the style of the hull EOS plot:
V per formula unit vs P, one colour per composition (H2O, H2.25O, H3O, H5O, H7O, H), DFT = squares + line, model = circles (x = flagged: O froze / H-H collapse), dashed = linear mix
of the MODEL's own H2O and H EOS (V(HnO) = V(H2O) + (n - 2) V(H) at equal P). Lower panel: (P_model - P_DFT) / P_DFT at the model volumes.
  python3 mlp_pipeline/plot_eos_all.py [runs/eos_live] --T 7000 [--model ft_mfin_f_lr1e-2_e50] [--pmin 300 --pmax 1050] [-o runs/eos_live/eos_all_7000K.png]
Prints the table behind it (per composition: points, P range, mean / max |dP| vs DFT, flags).
--dft-frames runs/train_mfin.extxyz runs/valid_mfin.extxyz (needs ase: run with $PY): also compares with the DFT-MD frames of the same composition within --dT of T:
  energy  E_DFT - E_model per atom, E_model = the model's mean potential energy of the EOS run interpolated to the frame's volume (the labels are TOTEN, which the model was trained on);
  pressure  P_DFT(frame, virial + N kB T/V) - P_model at the frame's volume (reaches below the 300 GPa where the hull EOS starts); both in a third panel and a table per composition.
  Single frames scatter by a few meV/atom and GPa (thermal fluctuations); the per-state means (frames grouped by volume) are what matter. Keep --dT small (default 50 K):
  U changes by ~3 kB per atom per K (300 K ~ 40-80 meV/atom). A DFT frame in another phase than the model run (superionic solid vs liquid) differs by ~T*dS_melt ~ 200 meV/atom: see the
  per-data-set breakdown (D_melt / E sets contain solids)."""
import argparse, csv, glob, math, os
import numpy as np

COMPS = [("h2o", 2.0, "H$_2$O"), ("h2.25", 2.25, "H$_{2.25}$O"), ("h3o", 3.0, "H$_3$O"), ("h5o", 5.0, "H$_5$O"), ("h7o", 7.0, "H$_7$O"), ("h", None, "H")]
COL = ["#e34948", "#2a78d6", "#1baf7a", "#6250d6", "#eb6834", "#8a5a2b"]     # fixed per composition (red, blue, green, violet, orange, brown as in the hull figure)


def dft(comp, T, path="data-h3o-convex-hull.csv"):
    rows = list(csv.reader(open(path))); h = rows[0]
    try: iP, iV = h.index(f"P {T}K {comp}"), h.index(f"V {T}K {comp} norm")
    except ValueError: return np.empty((0, 2))
    out = []
    for r in rows[1:]:
        try: out.append((float(r[iP]), float(r[iV])))
        except (ValueError, IndexError): pass
    return np.array(sorted(out))


def comp_of(r):
    if r.get("phase") == "H": return "h"
    h = float(r.get("hper", 2.0)); return {2.0: "h2o", 2.25: "h2.25", 3.0: "h3o", 5.0: "h5o", 7.0: "h7o"}.get(h, f"h{h:g}o")


def interp_logP(V, P, v):
    """P at volume v from (V, P) points (log P linear in V), nan outside"""
    o = np.argsort(V); V, P = np.asarray(V)[o], np.asarray(P)[o]
    if len(V) < 2 or v < V[0] or v > V[-1]: return float("nan")
    return float(np.exp(np.interp(v, V, np.log(P))))


def frame_compare(a, data, cx):
    """DFT-MD frames of each composition within dT of T: E_DFT - E_model(V) per atom and P_DFT - P_model(V), per volume group; plotted against P in cx"""
    from ase.io import read
    KB_GPA = 8.617333e-5 * 160.21766
    fr = []
    for p in a.dft_frames:
        for f in read(p, index=":"):
            T = f.info.get("T_K", f.info.get("T_set"))
            if T is None or abs(float(T) - a.T) > a.dT: continue
            nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum()); N = len(f); V = f.get_volume()
            c = "h" if nO == 0 else {2.0: "h2o", 2.25: "h2.25", 3.0: "h3o", 5.0: "h5o", 7.0: "h7o"}.get(round(nH / nO, 2))
            if c is None: continue
            P = -float(np.trace(f.get_stress(voigt=False))) / 3 * 160.21766 + N * KB_GPA * float(T) / V
            fr.append(dict(c=c, vpa=V / N, E=f.get_potential_energy() / N, P=P, T=float(T), set=str(f.info.get("db_set", "old m tier")).replace("False", "F_toy")))
    print(f"\nDFT-MD frames within {a.dT:.0f} K of {a.T} K: {len(fr)}  (energies: TOTEN per atom; P = virial + N kB T/V)")
    print(f"{'comp':6s} {'frames':>6s} {'vpa range':>13s} {'P range GPa':>13s} {'<E_DFT - E_model> meV/at':>25s} {'sd':>6s} {'<P_DFT - P_model> GPa':>22s} {'sd':>6s}")
    for (c, h, lab), col in zip(COMPS, COL):
        pts = [q for q in data.get(c, []) if not q["bad"] and q["E"] == q["E"]]; fs = [q for q in fr if q["c"] == c]
        if len(pts) < 2 or not fs: continue
        o = np.argsort([q["vpa"] for q in pts]); vv = np.array([pts[i]["vpa"] for i in o]); EE = np.array([pts[i]["E"] for i in o]); PP = np.array([pts[i]["P"] for i in o])
        inside = [q for q in fs if vv[0] <= q["vpa"] <= vv[-1]]
        if not inside: print(f"{c:6s} {len(fs):6d}  (no frame inside the model's volume range {vv[0]:.2f}-{vv[-1]:.2f})"); continue
        dE = np.array([1000 * (q["E"] - np.interp(q["vpa"], vv, EE)) for q in inside]); dP = np.array([q["P"] - math.exp(np.interp(q["vpa"], vv, np.log(np.maximum(PP, 1e-3)))) for q in inside])
        print(f"{c:6s} {len(inside):6d} {min(q['vpa'] for q in inside):6.3f}-{max(q['vpa'] for q in inside):<6.3f} {min(q['P'] for q in inside):6.0f}-{max(q['P'] for q in inside):<6.0f} "
              f"{dE.mean():+25.1f} {dE.std():6.1f} {dP.mean():+22.1f} {dP.std():6.1f}")
        bys = {}
        for q, e, d in zip(inside, dE, dP): bys.setdefault(q["set"], []).append((e, d))
        print("        by data set: " + "; ".join(f"{k} {len(v)} fr: dE {np.mean([e for e, _ in v]):+.0f} (sd {np.std([e for e, _ in v]):.0f}) meV/at, dP {np.mean([d for _, d in v]):+.1f} GPa" for k, v in sorted(bys.items())))
        g = {}                                                             # per volume group (2 % bins): means, so thermal scatter of single frames averages out
        for q, e in zip(inside, dE): g.setdefault(round(math.log(q["vpa"]) / 0.02), []).append((q["P"], e))
        xs = [np.mean([p for p, _ in v]) for v in g.values()]; ys = [np.mean([e for _, e in v]) for v in g.values()]
        cx.plot(xs, ys, "o", color=col, ms=5, label=lab)
    cx.axhline(0, color="gray", lw=0.8); cx.set_ylabel("E$_{DFT}$ - E$_{model}$ (meV/atom)"); cx.grid(True, color="#eee"); cx.set_xlabel("P (GPa)")
    cx.set_title(f"DFT-MD frames within {a.dT:.0f} K: energy vs the model's mean energy at the same volume", loc="left", fontsize=8)


def french_isotherm(path, T):
    """(P GPa, V per H2O A^3) of French et al. 2009 at T: the isotherm itself if tabulated, else linear in T between the bracketing isotherms at their common densities
    (log P interpolated in density); fluid / superionic labels are not filtered (the table marks phases only at 1000-2000 K)"""
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#")) if r["table"] != "VIII"]
    iso = {}
    for r in rows: iso.setdefault(int(r["T_K"]), []).append((float(r["rho_gcc"]), float(r["P_GPa"])))
    Ts = sorted(iso)
    if T in iso: pts = sorted(iso[T])
    else:
        lo = [t for t in Ts if t < T]; hi = [t for t in Ts if t > T]
        if not lo or not hi: return np.empty((0, 2))
        a_, b_ = sorted(iso[lo[-1]]), sorted(iso[hi[0]]); w = (T - lo[-1]) / (hi[0] - lo[-1]); pts = []
        for rho, Pa in a_:
            if b_[0][0] <= rho <= b_[-1][0]:
                Pb = math.exp(np.interp(rho, [x for x, _ in b_], [math.log(y) for _, y in b_])); pts.append((rho, Pa + w * (Pb - Pa)))
    return np.array([(P, 18.015 / (rho * 0.60221)) for rho, P in pts])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="runs/eos_live"); ap.add_argument("--T", type=int, default=7000); ap.add_argument("--model", default="")
    ap.add_argument("--pmin", type=float, default=None); ap.add_argument("--pmax", type=float, default=None); ap.add_argument("-o", default=None)
    ap.add_argument("--lit", default="docs/data/french2009_h2o_eos.csv", help="H2O literature EOS (French et al. 2009, PBE DFT-MD); '' to leave out; interpolated linearly in T between isotherms")
    ap.add_argument("--dft-frames", nargs="*", default=[]); ap.add_argument("--dT", type=float, default=50.0)
    ap.add_argument("--all-runs", action="store_true", help="keep every run folder of a composition (default: only the run with the largest cell, e.g. not an old 162-atom test)"); a = ap.parse_args()
    data = {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "*.csv"))):
        if a.model and not os.path.basename(os.path.dirname(f)).startswith(a.model): continue
        for r in csv.DictReader(open(f)):
            try:
                if abs(float(r["T_set"]) - a.T) > 1: continue
                c = comp_of(r); fu = 1.0 if c == "h" else 1.0 + float(r["hper"])
                dO = float(r["D_O_cm2s"]) if r.get("D_O_cm2s") not in (None, "", "nan") else float("nan")
                bad = (c != "h" and (str(r.get("O_struct") or "") in ("bcc", "fcc", "hcp") or (dO == dO and dO < 2e-5))) or float(r.get("dmin_HH") or 9) < 0.25
                data.setdefault(c, []).append(dict(P=float(r["P_GPa"]), Pe=float(r.get("P_sem") or 0), V=fu * float(r["vpa"]), vpa=float(r["vpa"]), E=float(r.get("epot_per_atom") or "nan"),
                                                   bad=bad, src=os.path.basename(os.path.dirname(f))))
            except (KeyError, ValueError): continue
    if not data: raise SystemExit(f"no EOS rows at {a.T} K under {a.root}")
    if not a.all_runs:                                                       # one run per composition: the largest cell (atom count from the folder tag, else the row count)
        for c, pts in data.items():
            srcs = sorted({q["src"] for q in pts}, key=lambda s_: (int(m.group(1)) if (m := __import__("re").search(r"_(\d+)at$", s_)) else 0, sum(q["src"] == s_ for q in pts)))
            if len(srcs) > 1: print(f"{c}: using {srcs[-1]}, ignoring {srcs[:-1]} (--all-runs keeps all)")
            data[c] = [q for q in pts if q["src"] == srcs[-1]]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nrow = 3 if a.dft_frames else 2
    fig, axs = plt.subplots(nrow, 1, figsize=(6.4, 8 + 2.6 * (nrow - 2)), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3] + [1.3] * (nrow - 2)), constrained_layout=True)
    ax, bx = axs[0], axs[1]
    model_eos = {}
    print(f"{'comp':6s} {'pts':>4s} {'P range GPa':>14s} {'mean dP%':>9s} {'max|dP|%':>9s} {'flagged':>8s}  source")
    for (c, h, lab), col in zip(COMPS, COL):
        d = dft(c, a.T)
        if len(d): ax.plot(d[:, 0], d[:, 1], "-", color=col, lw=1.2, zorder=2); ax.plot(d[:, 0], d[:, 1], "s", color=col, mec="k", mew=0.6, ms=6, zorder=3, label=f"{lab} DFT")
        pts = sorted(data.get(c, []), key=lambda q: q["P"])
        if not pts: continue
        good = [q for q in pts if not q["bad"]]; flag = [q for q in pts if q["bad"]]
        ax.errorbar([q["P"] for q in good], [q["V"] for q in good], xerr=[q["Pe"] for q in good], fmt="o", mfc="white", mec=col, ecolor=col, mew=1.4, ms=6, zorder=4, label=f"{lab} model")
        if flag: ax.plot([q["P"] for q in flag], [q["V"] for q in flag], "x", color=col, ms=8, mew=1.6, zorder=4)
        model_eos[c] = (np.array([q["V"] for q in good]), np.array([q["P"] for q in good]))
        dp = []
        if len(d):
            for q in good:
                pd_ = interp_logP(d[:, 1], d[:, 0], q["V"])
                if pd_ == pd_: dp.append((q["P"], 100 * (q["P"] - pd_) / pd_))
        if dp: bx.plot([x for x, _ in dp], [y for _, y in dp], "o-", color=col, ms=4, lw=1)
        print(f"{c:6s} {len(pts):4d} {min(q['P'] for q in pts):6.0f}-{max(q['P'] for q in pts):<7.0f} "
              f"{(np.mean([y for _, y in dp]) if dp else float('nan')):+9.2f} {(max(abs(y) for _, y in dp) if dp else float('nan')):9.2f} {len(flag):8d}  {sorted({q['src'] for q in pts})}")
    lit = french_isotherm(a.lit, a.T) if a.lit and os.path.exists(a.lit) else None
    if lit is not None and len(lit):
        ax.plot(lit[:, 0], lit[:, 1], "D", color="k", mfc="none", ms=6, mew=1.2, zorder=5, label=f"H$_2$O French 2009 (PBE) {a.T} K" + ("" if lit.dtype else ""))
        if "h2o" in model_eos and len(model_eos["h2o"][0]) > 1:
            Vm, Pm = model_eos["h2o"]; dl = [(P, 100 * (interp_logP(Vm, Pm, V) - P) / P) for P, V in lit]; dl = [x for x in dl if x[1] == x[1]]
            if dl: bx.plot([x for x, _ in dl], [y for _, y in dl], "D--", color="k", mfc="none", ms=5, lw=0.8, label="H$_2$O model vs French 2009"); bx.legend(fontsize=7, frameon=False)
            print(f"H2O model vs French 2009 at {a.T} K: " + ", ".join(f"{P:.0f} GPa {y:+.1f}%" for P, y in dl))
    if "h2o" in model_eos and "h" in model_eos:   # linear mix of the model's own end members at equal P
        (Vw, Pw), (Vh, Ph) = model_eos["h2o"], model_eos["h"]
        Pg = np.geomspace(max(Pw.min(), Ph.min()), min(Pw.max(), Ph.max()), 40)
        vw = np.interp(np.log(Pg), np.log(Pw[np.argsort(Pw)]), Vw[np.argsort(Pw)]); vh = np.interp(np.log(Pg), np.log(Ph[np.argsort(Ph)]), Vh[np.argsort(Ph)])
        for (c, h, lab), col in zip(COMPS, COL):
            if h and h > 2 and c in data: ax.plot(Pg, vw + (h - 2) * vh, "--", color=col, lw=1, alpha=0.8)
        ax.plot([], [], "--", color="gray", label="linear mix (model H$_2$O + H)")
    if a.pmin is not None or a.pmax is not None: ax.set_xlim(a.pmin, a.pmax)
    ax.set_ylim(bottom=0); ax.set_ylabel("V (A$^3$ per formula unit)"); ax.grid(True, color="#eee"); ax.legend(fontsize=7, ncol=2, frameon=False); ax.set_title(f"{a.T} K: model (circles) vs DFT (squares)", loc="left")
    bx.axhline(0, color="gray", lw=0.8); bx.set_ylabel("P$_{model}$ - P$_{DFT}$ (%)"); bx.set_xlabel("P (GPa)"); bx.grid(True, color="#eee")
    if a.dft_frames: frame_compare(a, data, axs[2])
    out = a.o or os.path.join(a.root, f"eos_all_{a.T}K{('_' + a.model) if a.model else ''}.png")
    fig.savefig(out, dpi=150); print("figure:", out)


if __name__ == "__main__":
    main()
