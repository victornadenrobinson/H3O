#!/usr/bin/env python3
"""P-T maps of everything the EOS scans record, one row per composition, from all runs/eos_live/*/*.csv (all temperatures; one run per composition and T, the largest cell):
  columns: D_O, D_H (cm^2/s, log colour), n_OH (H per O within 1.2 A), f_H2 (fraction of H in mutual-nearest H-H pairs < 0.85 A), E_DFT - E_model (meV/atom, with --dft-frames),
  and the smoothed RDFs g_OH / g_HH of the lowest-, middle- and highest-pressure liquid run at the lowest T (rdf.csv of each volume). Frozen points (O lattice / D_O < 2e-5) are drawn as x.
  $PY mlp_pipeline/plot_eos_maps.py [runs/eos_live] [--model ft_mfin_f_lr1e-2_e50] [--dft-frames runs/train_mfin.extxyz runs/valid_mfin.extxyz] [-o runs/eos_live/eos_maps.png]
E_DFT - E_model: DFT-MD frames (energy = TOTEN per atom, the training label) against the model's mean potential energy of the scan at the same composition, the same T (within --dT)
and the frame's volume (interpolated); averaged per state (composition, T, 2 % volume bin) and shown at the frame pressure. Needs ase only with --dft-frames.
With a single temperature each panel is one row of points; it becomes a map as scans at more T are added (T=5000 / 9000 in eos_live.sh)."""
import argparse, collections, csv, glob, math, os, re
import numpy as np

COMPS = [("h2o", 2.0, "H2O"), ("h2.25", 2.25, "H2.25O"), ("h3o", 3.0, "H3O"), ("h5o", 5.0, "H5O"), ("h7o", 7.0, "H7O"), ("h", None, "H")]


def fnum(r, k):
    try: v = float(r.get(k, "nan")); return v
    except (TypeError, ValueError): return float("nan")


def comp_of(r):
    if r.get("phase") == "H": return "h"
    return {2.0: "h2o", 2.25: "h2.25", 3.0: "h3o", 5.0: "h5o", 7.0: "h7o"}.get(round(fnum(r, "hper"), 2))


def load(root, model):
    runs = collections.defaultdict(lambda: collections.defaultdict(list))   # comp -> T -> rows (one folder: the largest cell)
    for f in sorted(glob.glob(os.path.join(root, "*", "*.csv"))):
        src = os.path.basename(os.path.dirname(f))
        if model and not src.startswith(model): continue
        nat = int(m.group(1)) if (m := re.search(r"_(\d+)at$", src)) else 0
        for r in csv.DictReader(open(f)):
            c = comp_of(r)
            if c is None: continue
            r["_src"], r["_nat"], r["_dir"] = src, nat, os.path.dirname(f); runs[c][int(round(fnum(r, "T_set")))].append(r)
    for c in runs:
        for T in runs[c]:
            best = max({(r["_nat"], r["_src"]) for r in runs[c][T]})[1]; runs[c][T] = [r for r in runs[c][T] if r["_src"] == best]
    return runs


def frozen(r, c):
    dO = fnum(r, "D_O_cm2s")
    return c != "h" and (str(r.get("O_struct") or "") in ("bcc", "fcc", "hcp") or (dO == dO and dO < 2e-5))


def dft_states(files, runs, dT):
    """per (comp, T of the model scan, volume bin): mean P_DFT and mean E_DFT - E_model (meV/atom)"""
    from ase.io import read
    KB_GPA = 8.617333e-5 * 160.21766; acc = collections.defaultdict(list)
    for p in files:
        for f in read(p, index=":"):
            T = f.info.get("T_K", f.info.get("T_set"))
            if T is None: continue
            nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum()); N = len(f)
            c = "h" if nO == 0 else {2.0: "h2o", 2.25: "h2.25", 3.0: "h3o", 5.0: "h5o", 7.0: "h7o"}.get(round(nH / nO, 2))
            if c is None or c not in runs: continue
            Tm = min(runs[c], key=lambda t: abs(t - float(T)))
            if abs(Tm - float(T)) > dT: continue
            rows = [r for r in runs[c][Tm] if not frozen(r, c) and fnum(r, "epot_per_atom") == fnum(r, "epot_per_atom")]
            if len(rows) < 2: continue
            vv = np.array(sorted(fnum(r, "vpa") for r in rows)); EE = np.array([fnum(r, "epot_per_atom") for r in sorted(rows, key=lambda r: fnum(r, "vpa"))])
            v = f.get_volume() / N
            if not vv[0] <= v <= vv[-1]: continue
            P = -float(np.trace(f.get_stress(voigt=False))) / 3 * 160.21766 + N * KB_GPA * float(T) / f.get_volume()
            acc[(c, Tm, round(math.log(v) / 0.02))].append((P, 1000 * (f.get_potential_energy() / N - np.interp(v, vv, EE))))
    return {k: (np.mean([p for p, _ in v]), np.mean([e for _, e in v]), np.std([e for _, e in v]), len(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="runs/eos_live"); ap.add_argument("--model", default=""); ap.add_argument("--dft-frames", nargs="*", default=[])
    ap.add_argument("--dT", type=float, default=50.0); ap.add_argument("-o", default=None); a = ap.parse_args()
    runs = load(a.root, a.model)
    comps = [c for c, _, _ in COMPS if c in runs]
    if not comps: raise SystemExit(f"no EOS tables under {a.root}")
    dft = dft_states(a.dft_frames, runs, a.dT) if a.dft_frames else {}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    cols = [("D_O_cm2s", "D_O (cm$^2$/s)", "log"), ("D_H_cm2s", "D_H (cm$^2$/s)", "log"), ("n_OH_1p2", "H per O (< 1.2 A)", "lin"), ("f_H2", "H in H$_2$-like pairs", "lin")] + ([("dE", "E$_{DFT}$ - E$_{model}$ (meV/at)", "div")] if dft else []) + [("rdf", "g(r): OH solid, HH dashed", "rdf")]
    fig, axs = plt.subplots(len(comps), len(cols), figsize=(3.3 * len(cols), 2.6 * len(comps)), squeeze=False, constrained_layout=True)
    name = {c: n for c, _, n in COMPS}
    for i, c in enumerate(comps):
        Ts = sorted(runs[c])
        for j, (key, lab, kind) in enumerate(cols):
            ax = axs[i, j]
            if kind == "rdf":
                rows = sorted([r for r in runs[c][Ts[0]] if not frozen(r, c)], key=lambda r: fnum(r, "P_GPa"))
                pick = [rows[k] for k in sorted({0, len(rows) // 2, len(rows) - 1})] if rows else []
                for r, col in zip(pick, ["#2a78d6", "#1baf7a", "#e34948"]):
                    fn = os.path.join(r["_dir"], f"{r['tag']}_v{fnum(r, 'vpa'):.4f}", "rdf.csv")
                    if not os.path.exists(fn): continue
                    d = list(csv.DictReader(open(fn))); rr = [float(x["r_A"]) for x in d]
                    if "g_OH" in d[0]: ax.plot(rr, [float(x["g_OH"]) for x in d], "-", color=col, lw=1.2, label=f"{fnum(r, 'P_GPa'):.0f} GPa")
                    if "g_HH" in d[0]: ax.plot(rr, [float(x["g_HH"]) for x in d], "--", color=col, lw=1)
                ax.set_xlim(0.4, 3.5); ax.set_xlabel("r (A)"); ax.legend(fontsize=6, frameon=False); ax.set_title(f"{name[c]} {Ts[0]} K: {lab}", fontsize=8, loc="left")
                if not pick: ax.text(0.5, 0.5, "no rdf.csv (older runs)", transform=ax.transAxes, ha="center", fontsize=7)
                continue
            if kind == "div":
                pts = [(P, Tm, e, n) for (cc, Tm, _), (P, e, sd, n) in dft.items() if cc == c]
                if pts:
                    lim = max(20, max(abs(p[2]) for p in pts)); sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=[p[2] for p in pts], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), s=[12 + 4 * min(p[3], 10) for p in pts], edgecolor="k", lw=0.3)
                    fig.colorbar(sc, ax=ax, shrink=0.8)
                    print(f"{name[c]:7s} E_DFT - E_model per state (meV/atom): " + ", ".join(f"{P:.0f} GPa {Tm} K {e:+.0f} (n {n})" for P, Tm, e, n in sorted(pts)))
            else:
                good = [(fnum(r, "P_GPa"), T, fnum(r, key)) for T in Ts for r in runs[c][T] if not frozen(r, c) and fnum(r, key) == fnum(r, key)]
                bad = [(fnum(r, "P_GPa"), T) for T in Ts for r in runs[c][T] if frozen(r, c)]
                if good:
                    vals = [g[2] for g in good]; norm = LogNorm(max(min(v for v in vals if v > 0), 1e-7), max(vals)) if kind == "log" and max(vals) > 0 else None
                    sc = ax.scatter([g[0] for g in good], [g[1] for g in good], c=[max(v, 1e-7) for v in vals] if kind == "log" else vals, cmap="viridis", norm=norm, s=40, edgecolor="k", lw=0.3)
                    fig.colorbar(sc, ax=ax, shrink=0.8)
                if bad: ax.scatter([b[0] for b in bad], [b[1] for b in bad], marker="x", color="#e34948", s=30, label="frozen"); ax.legend(fontsize=6, frameon=False)
            ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("P (GPa)"); ax.set_ylabel("T (K)"); ax.set_title(f"{name[c]}: {lab}", fontsize=8, loc="left")
            if len(Ts) == 1: ax.set_ylim(Ts[0] - 1500, Ts[0] + 1500)
    out = a.o or os.path.join(a.root, "eos_maps" + (f"_{a.model}" if a.model else "") + ".png")
    fig.savefig(out, dpi=130); print("figure:", out, "| temperatures per composition:", {name[c]: sorted(runs[c]) for c in comps})


if __name__ == "__main__":
    main()
