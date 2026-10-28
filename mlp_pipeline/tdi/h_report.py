"""Pure-hydrogen report: is our G_H (the reference of every dg = mu_H - G_H and of the hull) a smooth, atomic-fluid free energy, or does it carry
molecular H2 / the molecular -> atomic liquid-liquid transition (LLPT)? Two steps:
  analyse (Young, ~/grace-venv/bin/python: needs ASE):  structure from the snapshots of the highest-lambda node of every pure-H run (essentially the real fluid):
      H-H g(r); molecular indicators: f_short = fraction of H whose nearest neighbour is closer than RMOL (0.85 A; H2 bond 0.74 A),
      f_pair = fraction of H in mutual-nearest-neighbour pairs closer than RMOL, persist_k = fraction of those pairs still paired k snapshots later
      (snapshots every 20 recorded samples, 20 fs at dt 0.5); molecules persist for many vibrations, chance contacts in an atomic fluid do not.
      Plus G_H, error, V/atom, MLP pressure from summarize.py's table.   -> <out>_runs.csv, <out>_gr.csv
  plot (any python with matplotlib):  G_H(T) per P, S = -dG/dT, the dG/dP = V consistency between pressures, g(r), molecular indicators vs T and P   -> <out>.pdf + pngs
usage (repo root):
  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_prep -o runs/checks/summary.csv
  ~/grace-venv/bin/python mlp_pipeline/tdi/h_report.py analyse --summary runs/checks/summary.csv runs/tdi [--toy runs/toyscan] [-o docs/data/pure_h]
  python3 mlp_pipeline/tdi/h_report.py plot [-o docs/data/pure_h] [--pdf docs/report_pure_h]
--toy DIR...: pure-H runs of the toy pair potential (purely repulsive H-H: cannot bind, so its indicators are the 'atomic fluid' baseline at similar density);
their G comes from tdi_<tag>.json (written by collect.py at the end of every launch).
"""
import argparse, csv, glob, json, os, sys
import numpy as np

RMOL = 0.85


def frames(path):
    from ase.io import read
    fr = read(path, index=":")
    return [f.get_positions() for f in fr], fr[0].cell.array if fr else None


def mic_d(pos, cell):
    inv = np.linalg.inv(cell)
    d = pos[:, None, :] - pos[None, :, :]; s = d @ inv; s -= np.round(s); d = s @ cell
    r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf); return r


def structure(path, dt_snap_fs):
    P, cell = frames(path)
    if not P or len(P) < 4: return None
    P = P[len(P) // 5:]   # skip the first 20% (equilibration of that node)
    n = len(P[0]); V = abs(np.linalg.det(cell)); rho = n / V
    edges = np.arange(0, 3.02, 0.02); h = np.zeros(len(edges) - 1); fs, fp, pairs = [], [], []
    for p in P:
        r = mic_d(p, cell); h += np.histogram(r[np.triu_indices(n, 1)], edges)[0]
        nn = r.argmin(1); dn = r[np.arange(n), nn]
        fs.append(np.mean(dn < RMOL))
        mutual = (nn[nn] == np.arange(n)) & (dn < RMOL)
        fp.append(np.mean(mutual)); pairs.append({(min(i, nn[i]), max(i, nn[i])) for i in np.where(mutual)[0]})
    shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = h / (len(P) * n / 2 * rho * shell)
    rmax = 0.5 * V / max(np.linalg.norm(np.cross(cell[i], cell[(i + 1) % 3])) for i in range(3))   # half the smallest cell width: g(r) is incomplete beyond it
    g = np.where(edges[1:] <= rmax, g, np.nan)
    pers = {}
    for k in (1, 2, 5):
        num = sum(len(pairs[i] & pairs[i + k]) for i in range(len(pairs) - k)); den = sum(len(pairs[i]) for i in range(len(pairs) - k))
        pers[k] = num / den if den else float("nan")
    r = 0.5 * (edges[1:] + edges[:-1]); first = r[np.nanargmax(g)]
    return dict(n_frames=len(P), rho=rho, f_short=float(np.mean(fs)), f_pair=float(np.mean(fp)), persist_1=pers[1], persist_2=pers[2], persist_5=pers[5],
                dt_snap_fs=dt_snap_fs, r_peak=float(first), g_peak=float(np.nanmax(g)), r_min=float(np.min([mic_d(p, cell).min() for p in P[::max(1, len(P) // 20)]]))), (r, g)


def analyse(a):
    rows, grs = [], []
    summ = {}
    if a.summary:
        for r in csv.DictReader(open(a.summary)):
            if r["phase"] == "H": summ[(r["dir"], r["tag"])] = r
    todo = [(d, False) for d in a.dirs] + [(d, True) for d in (a.toy or [])]
    for d, toy in todo:
        for f in sorted(glob.glob(os.path.join(d, "**", "H_*_stage0.json"), recursive=True)):
            dd = os.path.dirname(f); tag = os.path.basename(f)[:-len("_stage0.json")]
            st = json.load(open(f)); A = st["args"]
            traj = os.path.join(dd, f"{tag}_node{A['nodes'] - 1}_traj.extxyz")
            if not os.path.exists(traj):
                print(f"{dd}/{tag}: no snapshots of the highest lambda value"); continue
            res = structure(traj, A.get("every", 2) * 20 * A["dt"])
            if res is None: print(f"{dd}/{tag}: too few snapshots"); continue
            s, (r, g) = res
            G = Ge = Phi = float("nan")
            row = summ.get((os.path.basename(dd), tag))   # summarize.py keys rows by the directory's base name: check the volume too (two dirs can share a base name)
            if row and abs(float(row["vpa_A3"]) - st["vpa"]) > 1e-6 * st["vpa"]: row = None
            if row:
                G, Ge = float(row["G_meV_atom"]), float(row["G_err_meV_atom"]); Phi = float(row["P_hi_GPa"] or "nan")
            elif os.path.exists(os.path.join(dd, f"tdi_{tag}.json")):
                j = json.load(open(os.path.join(dd, f"tdi_{tag}.json"))); G, Ge, Phi = j["G_meV_atom"], j["G_err_meV_atom"], j.get("P_mlp", float("nan"))
            rows.append(dict(model="toy" if toy else "GRACE", dir=dd, tag=tag, P_GPa=st["P_target"], T_K=st["T"], N=st["N"], vpa_A3=st["vpa"], lam_hi=round(sorted(st.get("lam", [1]))[-1], 4) if st.get("lam") else "",
                             G_meV_atom=G, G_err_meV_atom=Ge, P_hi_GPa=Phi, **s))
            for ri, gi in zip(r, g):
                if not np.isfinite(gi): continue
                grs.append(dict(model=rows[-1]["model"], tag=tag, P_GPa=st["P_target"], T_K=st["T"], r_A=round(float(ri), 3), g=round(float(gi), 4)))
            print(f"{rows[-1]['model']:5s} {tag:28s} P {st['P_target']:5.0f} T {st['T']:5.0f}: f_short {s['f_short']:.3f} f_pair {s['f_pair']:.3f} persist(20/40/100 fs) "
                  f"{s['persist_1']:.2f}/{s['persist_2']:.2f}/{s['persist_5']:.2f}  g(r) peak {s['r_peak']:.2f} A  min d {s['r_min']:.2f} A  G {G:.1f} +- {Ge:.1f}", flush=True)
    for name, data in ((a.o + "_runs.csv", rows), (a.o + "_gr.csv", grs)):
        if data:
            with open(name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
            print("wrote", name)


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    INK2, MUTED, GRID = "#52514d", "#8a897f", "#e4e3dd"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
    R = list(csv.DictReader(open(a.o + "_runs.csv"))); GR = list(csv.DictReader(open(a.o + "_gr.csv")))
    for r in R:
        for k, v in r.items():
            try: r[k] = float(v)
            except (TypeError, ValueError): pass
    Ps = sorted({r["P_GPa"] for r in R if r["model"] == "GRACE"}) or sorted({r["P_GPa"] for r in R})
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
    col = {P: ramp[min(len(ramp) - 1, int(i * (len(ramp) - 1) / max(1, len(Ps) - 1)))] for i, P in enumerate(Ps)}
    figs = {}
    best = {}
    for r in R:   # one row per (model, P, T): the longest-run / smallest error
        k = (r["model"], r["P_GPa"], r["T_K"])
        if np.isfinite(r["G_meV_atom"]) and (k not in best or r["G_err_meV_atom"] < best[k]["G_err_meV_atom"]): best[k] = r
    G = [v for v in best.values() if v["model"] == "GRACE"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for P in Ps:
        s = sorted((r["T_K"], r["G_meV_atom"], r["G_err_meV_atom"]) for r in G if r["P_GPa"] == P)
        if not s: continue
        T, g, e = map(np.array, zip(*s)); ax[0].errorbar(T, g, e, fmt="o-", color=col[P], ms=5, capsize=2, label=f"{P:.0f} GPa")
        if len(T) >= 2:
            S = -np.diff(g) / np.diff(T) * 1e3 / 8.617333262e-2 / 1e3; Tm = 0.5 * (T[1:] + T[:-1])   # S/k_B per atom
            Se = np.hypot(e[1:], e[:-1]) / np.diff(T) / 8.617333262e-2
            ax[1].errorbar(Tm, S, Se, fmt="o-", color=col[P], ms=5, capsize=2, label=f"{P:.0f} GPa")
    ax[0].set(xlabel="T (K)", ylabel="G_H (meV/atom)", title="pure-H free energy (GRACE TDI)"); ax[0].legend()
    ax[1].set(xlabel="T (K)", ylabel="S / k_B per atom  (= -dG/dT)", title="entropy from neighbouring T: dissociation would show as a bump"); fig.tight_layout(); figs["G"] = fig
    # consistency: G(P2) - G(P1) = integral of V dP at fixed T (trapezoid with the run volumes)
    fig, ax = plt.subplots(figsize=(8, 4.4)); EV = 160.21766
    for T in sorted({r["T_K"] for r in G}):
        s = sorted((r["P_GPa"], r["G_meV_atom"], r["G_err_meV_atom"], r["vpa_A3"]) for r in G if r["T_K"] == T)
        for (p1, g1, e1, v1), (p2, g2, e2, v2) in zip(s, s[1:]):
            pred = 0.5 * (v1 + v2) * (p2 - p1) / EV * 1e3
            ax.errorbar(0.5 * (p1 + p2), (g2 - g1) - pred, np.hypot(e1, e2), fmt="o", ms=6, capsize=2, color="#2a78d6")
            ax.annotate(f"{T:.0f} K", (0.5 * (p1 + p2), (g2 - g1) - pred), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(0, color="#1f1f1e", lw=1)
    ax.set(xlabel="P midpoint (GPa)", ylabel="[G(P2) - G(P1)] - V dP  (meV/atom)", title="thermodynamic consistency between pressures (0 = consistent; trapezoid in V)"); fig.tight_layout(); figs["consistency"] = fig
    # g(r)
    tags = sorted({(r["model"], float(r["P_GPa"]), float(r["T_K"]), r["tag"]) for r in GR})
    Pg = sorted({t[1] for t in tags if t[0] == "GRACE"})
    if Pg:
        fig, axs = plt.subplots(1, len(Pg), figsize=(max(7.5, 3.6 * len(Pg)), 3.8), sharey=True, squeeze=False)
        Tall = sorted({t[2] for t in tags}); cmap = plt.get_cmap("plasma")
        for ax, P in zip(axs[0], Pg):
            for m, _, T, tag in [t for t in tags if t[0] == "GRACE" and t[1] == P]:
                rr = [(float(x["r_A"]), float(x["g"])) for x in GR if x["tag"] == tag and x["model"] == m]
                r_, g_ = zip(*rr); ax.plot(r_, g_, color=cmap((T - min(Tall)) / max(1, max(Tall) - min(Tall)) * 0.85), lw=1.5, label=f"{T:.0f} K")
            ax.axvline(0.74, color=MUTED, ls=":", lw=1); ax.text(0.76, 0.1, "H2 bond", fontsize=7, color=MUTED)
            ax.set(xlabel="r (A)", xlim=(0.4, 3), title=f"{P:.0f} GPa"); ax.legend(fontsize=6.5)
        axs[0][0].set_ylabel("g_HH(r), highest lambda value"); fig.suptitle("H-H pair distribution (GRACE): a molecular fluid has a sharp peak at 0.74 A", fontsize=9); fig.tight_layout(); figs["gr"] = fig
    # molecular indicators
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for m, mk in (("GRACE", "o"), ("toy", "D")):
        for P in sorted({r["P_GPa"] for r in R if r["model"] == m}):
            s = sorted((r["T_K"], r["f_short"], r["f_pair"], r["persist_2"]) for r in R if r["model"] == m and r["P_GPa"] == P)
            T, fs, fp, p2 = map(np.array, zip(*s)); c = col.get(P, MUTED) if m == "GRACE" else MUTED
            kw = dict(fmt=mk + ("-" if m == "GRACE" else "--"), color=c, ms=5, label=f"{m} {P:.0f} GPa")
            ax[0].errorbar(T, fs, **kw); ax[1].errorbar(T, fp, **kw); ax[2].errorbar(T, p2, **kw)
    ax[0].set(xlabel="T (K)", ylabel=f"fraction of H with a neighbour < {RMOL} A", title="short contacts")
    ax[1].set(xlabel="T (K)", ylabel="fraction in mutual-nearest pairs", title="pairs (molecule candidates)")
    ax[2].set(xlabel="T (K)", ylabel="fraction of pairs still paired 40 fs later", title="pair persistence: molecules ~1, chance contacts ~0", ylim=(0, 1))
    ax[2].legend(fontsize=6.5); fig.suptitle("Molecular character of the pure-H fluid (grey dashed: toy pair potential, purely repulsive H-H = atomic-fluid baseline)", fontsize=9); fig.tight_layout(); figs["molecular"] = fig
    with PdfPages(a.pdf + ".pdf") as pdf:
        fig = plt.figure(figsize=(11, 8.5)); fig.text(0.06, 0.93, "Pure hydrogen: is G_H an atomic-fluid free energy?", fontsize=15, weight="bold", color="#1f1f1e")
        lines = ["Every dg = mu_H(ice) - G_H and every hull point uses G_H of these runs: an error in G_H shifts all of them by the same amount.",
                 "Questions: (1) is the fluid atomic at our (P, T), i.e. beyond the molecular -> atomic transition (DFT-PBE: ~150 GPa near 2000 K, below ~1000 K by 300 GPa)?",
                 "(2) is G_H(P, T) smooth and thermodynamically consistent (dG/dP = V)? A crossing of the transition would show as molecules (g(r) peak at 0.74 A,",
                 "persistent pairs) and as a bump in S = -dG/dT.", "", "Runs (model, P, T, N, G_H meV/atom, molecular pair fraction, persistence over 40 fs):"]
        for r in sorted(R, key=lambda r: (r["model"], r["P_GPa"], r["T_K"])):
            lines.append(f"   {r['model']:5s} {r['P_GPa']:5.0f} GPa {r['T_K']:5.0f} K  N {r['N']:4.0f}  G {r['G_meV_atom']:9.1f} +- {r['G_err_meV_atom']:5.1f}  "
                         f"f_pair {r['f_pair']:.3f}  persist {r['persist_2']:.2f}  g(r) peak {r['r_peak']:.2f} A")
        for i, l in enumerate(lines[:60]): fig.text(0.06, 0.87 - 0.0135 * i, l, fontsize=7.5, family="monospace" if l.startswith("   ") else None, color=INK2)
        pdf.savefig(fig); plt.close(fig)
        for n, f in figs.items():
            pdf.savefig(f); f.savefig(os.path.join(a.png_dir, f"pure_h_{n}.png"), dpi=130); plt.close(f)
    print("wrote", a.pdf + ".pdf")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["analyse", "plot"]); ap.add_argument("dirs", nargs="*", default=["runs/tdi"])
    ap.add_argument("--summary", default=None); ap.add_argument("--toy", nargs="*", default=None); ap.add_argument("-o", default="docs/data/pure_h")
    ap.add_argument("--pdf", default="docs/report_pure_h"); ap.add_argument("--png-dir", default="docs/figures")
    a = ap.parse_args(); {"analyse": analyse, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
