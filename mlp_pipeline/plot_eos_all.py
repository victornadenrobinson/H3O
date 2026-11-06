#!/usr/bin/env python3
"""All EOS scans (eos_live.sh / 17_eos_md.py result tables) on one figure, against the DFT EOS of data-h3o-convex-hull.csv, in the style of the hull EOS plot:
V per formula unit vs P, one colour per composition (H2O, H2.25O, H3O, H5O, H7O, H), DFT = squares + line, model = circles (x = flagged: O froze / H-H collapse), dashed = linear mix
of the MODEL's own H2O and H EOS (V(HnO) = V(H2O) + (n - 2) V(H) at equal P). Lower panel: (P_model - P_DFT) / P_DFT at the model volumes.
  python3 mlp_pipeline/plot_eos_all.py [runs/eos_live] --T 7000 [--model ft_mfin_f_lr1e-2_e50] [--pmin 300 --pmax 1050] [-o runs/eos_live/eos_all_7000K.png]
Prints the table behind it (per composition: points, P range, mean / max |dP| vs DFT, flags)."""
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="runs/eos_live"); ap.add_argument("--T", type=int, default=7000); ap.add_argument("--model", default="")
    ap.add_argument("--pmin", type=float, default=None); ap.add_argument("--pmax", type=float, default=None); ap.add_argument("-o", default=None); a = ap.parse_args()
    data = {}
    for f in sorted(glob.glob(os.path.join(a.root, "*", "*.csv"))):
        if a.model and not os.path.basename(os.path.dirname(f)).startswith(a.model): continue
        for r in csv.DictReader(open(f)):
            try:
                if abs(float(r["T_set"]) - a.T) > 1: continue
                c = comp_of(r); fu = 1.0 if c == "h" else 1.0 + float(r["hper"])
                bad = (c != "h" and float(r.get("D_O_cm2s") or 0) < 2e-6) or float(r.get("dmin_HH") or 9) < 0.25
                data.setdefault(c, []).append(dict(P=float(r["P_GPa"]), Pe=float(r.get("P_sem") or 0), V=fu * float(r["vpa"]), bad=bad, src=os.path.basename(os.path.dirname(f))))
            except (KeyError, ValueError): continue
    if not data: raise SystemExit(f"no EOS rows at {a.T} K under {a.root}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(6.4, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]), constrained_layout=True)
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
    out = a.o or os.path.join(a.root, f"eos_all_{a.T}K{('_' + a.model) if a.model else ''}.png")
    fig.savefig(out, dpi=150); print("figure:", out)


if __name__ == "__main__":
    main()
