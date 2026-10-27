"""One-command report of the TDI results: figures (pdf with all pages, plus png/svg per figure) from the CSVs written by summarize.py.
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_prep -o docs/data/tdi_results.csv      (makes tdi_results.csv, _dG, _mix, _hull)
                    python mlp_pipeline/tdi/report.py [--refresh runs/tdi runs/tdi_prep ...] [--out docs/figures/tdi_report] [--fmt pdf png svg]
Needs pandas + matplotlib (the Young venv may not have matplotlib: the CSVs are small, so render on any machine that has it; --refresh needs the TDI runs and ASE-free summarize.py only).
Figures: hull (liquid mixing free energy vs x_H, per pressure, with the DFT hull sheet), fcc-bcc (H2O vs Wilson 2013 and Cheng 2021), solid mixing per added H, insertion mu_H vs n, liquid-solid
crossing G_liquid - G_fcc vs T, start-configuration protocol (tdi_prep vs cold start), P-T coverage, QA (drift, quadrature, pressure mismatch, excluded runs).
QA rule: a run is EXCLUDED from every derived quantity (and listed on the QA page) if it is unfinished or if its MLP pressure at the highest lambda differs from the target by more than --pmax-rel (default 12%):
that is how a corrupted run shows up (e.g. a stage0.json overwritten with another lambda grid gives P_hi far from the target). Hollow markers = the numbers come from runs with a caveat (different dt / lambda grid, crystal melted in free NVT).
Derived quantities are recomputed here from the retained rows with the same formulas as summarize.py (equal cell-size class, same dt preferred, then the longest run), so they can differ from the summarize tables where a run was excluded."""
import argparse, csv, os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--base", default="docs/data/tdi_results.csv", help="summarize.py -o file (its _dG/_mix/_hull siblings are read for notes)")
ap.add_argument("--refresh", nargs="*", default=None, metavar="DIR", help="run summarize.py on these runs/ directories first (default dirs: runs/tdi runs/tdi_prep if none given)")
ap.add_argument("--out", default="docs/figures/tdi_report"); ap.add_argument("--fmt", nargs="*", default=["png", "svg"], help="per-figure formats; the pdf with every page is always written")
ap.add_argument("--lit", default="docs/data/dg_fcc_bcc.csv"); ap.add_argument("--cheng", default="docs/data/cheng2021_dmu_bcc_fcc.csv"); ap.add_argument("--dft-hull", default="data-h3o-convex-hull.csv")
ap.add_argument("--pmax-rel", type=float, default=0.12, help="exclude runs with |P_hi - P_target| / P_target above this")
a = ap.parse_args()

if a.refresh is not None:
    dirs = a.refresh or ["runs/tdi", "runs/tdi_prep"]
    os.makedirs(os.path.dirname(a.base) or ".", exist_ok=True)
    with open(a.base.replace(".csv", "_summary.txt"), "w") as fh:
        r = subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), *dirs, "-o", a.base], stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode: raise SystemExit(f"summarize.py failed (exit {r.returncode}); see {a.base.replace('.csv', '_summary.txt')}")
if not os.path.exists(a.base): raise SystemExit(f"{a.base} not found: run summarize.py ... -o {a.base} first (or give --refresh)")
raw = pd.read_csv(a.base)
mixnote = pd.read_csv(a.base.replace(".csv", "_mix.csv")) if os.path.exists(a.base.replace(".csv", "_mix.csv")) else pd.DataFrame()

# ---------------------------------------------------------------- QA
kT = lambda T: 8.617333e-2 * T   # meV
raw["P_dev"] = (raw.P_hi_GPa - raw.P_target_GPa) / raw.P_target_GPa
why = []
for _, r in raw.iterrows():
    w = []
    if not str(r.status).startswith("done"): w.append(f"unfinished ({r.status})")
    if np.isfinite(r.P_dev) and abs(r.P_dev) > a.pmax_rel: w.append(f"P_hi {r.P_hi_GPa:.0f} vs target {r.P_target_GPa:.0f} GPa")
    why.append("; ".join(w))
raw["excluded"] = why
df = raw[raw.excluded == ""].copy()
print(f"{len(raw)} runs, {len(raw) - len(df)} excluded by QA:"); print(raw[raw.excluded != ""][["dir", "tag", "excluded"]].to_string(index=False))


def best(g, dt=None, nodes=None):
    """one row of g: dt as given if present else the smallest dt; same number of lambda values if possible; then the longest run"""
    s = g[g.dt_fs == dt] if dt is not None else g
    g = s if len(s) else g[g.dt_fs == g.dt_fs.min()]
    if nodes is not None and (g.nodes == nodes).any(): g = g[g.nodes == nodes]
    return g.sort_values("ps_per_node").iloc[-1]


def sel(d, **kw):
    for k, v in kw.items(): d = d[(d[k] - v).abs() < 1e-6] if isinstance(v, float) else d[d[k] == v]
    return d


def lower_hull(x, y):
    """lower convex hull (monotone chain) of points (x, y), returned as arrays"""
    p = sorted(zip(x, y)); h = []
    for q in p:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0: h.pop()
        h.append(q)
    return np.array([q[0] for q in h]), np.array([q[1] for q in h])


# ---------------------------------------------------------------- derived quantities (retained rows only)
hull = []
for (P, T), g in df.groupby(["P_target_GPa", "T_K"]):
    wg, hg = g[(g.phase == "liquid") & (g.H_per_O == 2.0)], g[g.phase == "H"]
    if len(wg) and len(hg):
        for _, m in g[(g.phase == "liquid") & (g.H_per_O > 2.0)].iterrows():
            w, hh = best(wg, m.dt_fs, m.nodes), best(hg, m.dt_fs, m.nodes); n = m.H_per_O - 2.0; fw, fh = 3 / (3 + n), n / (3 + n)
            hull.append(dict(P=P, T=T, x=n / (1 + n), dG=m.G_meV_atom - fw * w.G_meV_atom - fh * hh.G_meV_atom, err=(m.G_err_meV_atom ** 2 + (fw * w.G_err_meV_atom) ** 2 + (fh * hh.G_err_meV_atom) ** 2) ** .5,
                             caveat=not (m.dt_fs == w.dt_fs == hh.dt_fs and m.nodes == w.nodes == hh.nodes)))
hull = pd.DataFrame(hull, columns=["P", "T", "x", "dG", "err", "caveat"])

pairs = []
d0 = df[df.nO > 0].copy(); d0["cell"] = np.where(d0.nO < 75, "small", "big")
for (P, T, h, cell), g in d0.groupby(["P_target_GPa", "T_K", "H_per_O", "cell"]):
    f, b = g[g.phase == "fcc"], g[g.phase == "bcc"]
    if len(f) and len(b):
        f = best(f, b.dt_fs.min()); b = best(b, f.dt_fs, f.nodes)
        pairs.append(dict(P=P, T=T, h=h, cell=cell, dG=f.G_meV_atom - b.G_meV_atom, err=(f.G_err_meV_atom ** 2 + b.G_err_meV_atom ** 2) ** .5, caveat=(f.dt_fs != b.dt_fs) or (f.nodes != b.nodes)))
pairs = pd.DataFrame(pairs, columns=["P", "T", "h", "cell", "dG", "err", "caveat"])

solid = []
for _, m in df[(df.nO > 0) & df.phase.isin(["fcc", "bcc"]) & (df.H_per_O > 2.0)].iterrows():
    w = df[(df.phase == m.phase) & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K) & (df.H_per_O == 2.0) & (df.nO == m.nO)]
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(w) and len(hh):
        w, hh = best(w, m.dt_fs, m.nodes), best(hh, m.dt_fs, m.nodes); n = m.nH - 2 * m.nO
        note = ""
        if len(mixnote):
            q = mixnote[(mixnote.phase == m.phase) & (mixnote.P_GPa == m.P_target_GPa) & (mixnote.T_K == m.T_K) & (mixnote.n_added == n)]
            note = " ".join(str(x) for x in q.note.fillna("")) if len(q) else ""
        solid.append(dict(kind=m.phase, P=m.P_target_GPa, T=m.T_K, h=m.H_per_O, n=int(n), dG=(m.N * m.G_meV_atom - w.N * w.G_meV_atom - n * hh.G_meV_atom) / n,
                          err=((m.N * m.G_err_meV_atom) ** 2 + (w.N * w.G_err_meV_atom) ** 2 + (n * hh.G_err_meV_atom) ** 2) ** .5 / n,
                          caveat=("MELTED" in note) or not (m.dt_fs == w.dt_fs == hh.dt_fs and m.nodes == w.nodes == hh.nodes)))
for _, m in df[df.phase == "insert"].iterrows():
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(hh):
        hh = best(hh, m.dt_fs)
        solid.append(dict(kind="insert", P=m.P_target_GPa, T=m.T_K, h=2.0, n=int(m.N), dG=m.G_meV_atom - hh.G_meV_atom, err=(m.G_err_meV_atom ** 2 + hh.G_err_meV_atom ** 2) ** .5, caveat=m.dt_fs != hh.dt_fs))
solid = pd.DataFrame(solid, columns=["kind", "P", "T", "h", "n", "dG", "err", "caveat"])

cross = []   # G_liquid - G_fcc at equal P, T, composition (per atom): negative = liquid more stable (melted)
for (P, T, h), g in df[df.nO > 0].groupby(["P_target_GPa", "T_K", "H_per_O"]):
    l, f = g[g.phase == "liquid"], g[(g.phase == "fcc") & (g.nO > 75)]
    if len(l) and len(f):
        l = best(l); f = best(f, l.dt_fs, l.nodes)
        cross.append(dict(P=P, T=T, h=h, dG=l.G_meV_atom - f.G_meV_atom, err=(l.G_err_meV_atom ** 2 + f.G_err_meV_atom ** 2) ** .5, caveat=(l.dt_fs != f.dt_fs) or (l.nodes != f.nodes)))
cross = pd.DataFrame(cross, columns=["P", "T", "h", "dG", "err", "caveat"])

prep = []   # protocol (dir tdi_prep, O held then released) minus the cold start of the same system, and the H2O control (both without protocol)
if "tdi_prep" in set(raw.dir):
    for _, p in df[(df.dir == "tdi_prep") & (df.phase == "fcc")].iterrows():
        c = df[(df.dir != "tdi_prep") & (df.phase == "fcc") & (df.P_target_GPa == p.P_target_GPa) & (df.T_K == p.T_K) & (df.H_per_O == p.H_per_O) & (df.nO == p.nO)]
        if len(c):
            c = best(c, p.dt_fs, p.nodes)
            prep.append(dict(P=p.P_target_GPa, T=p.T_K, h=p.H_per_O, dG=p.G_meV_atom - c.G_meV_atom, err=(p.G_err_meV_atom ** 2 + c.G_err_meV_atom ** 2) ** .5, ps_prep=p.ps_per_node, ps_cold=c.ps_per_node,
                             control=p.H_per_O == 2.0, caveat=(p.dt_fs != c.dt_fs) or (c.ps_per_node < 0.5 * p.ps_per_node)))
prep = pd.DataFrame(prep, columns=["P", "T", "h", "dG", "err", "ps_prep", "ps_cold", "control", "caveat"])

# ---------------------------------------------------------------- literature
lit = pd.read_csv(a.lit, comment="#") if os.path.exists(a.lit) else pd.DataFrame()
if len(lit):
    lit = lit[lit.quantity == "G"].copy(); lit["dG"] = np.where(lit.sign.str.strip() == "bcc-fcc", -lit.dG_meV_atom, lit.dG_meV_atom)
cheng = pd.read_csv(a.cheng, comment="#") if os.path.exists(a.cheng) else pd.DataFrame()
if len(cheng):
    cheng["pbe"] = -cheng.dmu_pbe_eV_mol * 1000 / 3; cheng["mlp"] = -cheng.dmu_mlp_eV_mol * 1000 / 3; cheng["err"] = cheng.err_fep_eV_mol * 1000 / 3   # G_fcc - G_bcc = -(mu_bcc - mu_fcc), per atom
dft = {}
if os.path.exists(a.dft_hull):
    rows = list(csv.reader(open(a.dft_hull)))[1:6]
    f = lambda s: float(s) if s not in ("", None) else np.nan
    xs = [f(r[1]) for r in rows]
    def col(c): return np.array([f(r[c]) if len(r) > c else np.nan for r in rows])
    dft = {("unlabelled (400 GPa assumed)", 9000): (col(2), col(3)), ("unlabelled (400 GPa assumed)", 7000): (col(4), col(5)), ("unlabelled (400 GPa assumed)", 5000): (col(6), col(7)),
           ("350 GPa", 9000): (col(10), col(11)), ("350 GPa", 7000): (col(12), col(13)), ("350 GPa", 5000): (col(14), col(15))}
    dft_x = np.array(xs)   # the sheet's second x column is mislabelled; both blocks use x = 0, 0.2, 0.5, 0.75, 1

# ---------------------------------------------------------------- figures
figs = {}
col_T = plt.cm.plasma(np.linspace(0.05, 0.85, 8))
def Tcol(T, Ts): return col_T[list(Ts).index(T) % 8]
def hollow(ax, x, y, e, cav, c, m, label=None):
    for k, (xx, yy, ee, cv) in enumerate(zip(x, y, e, cav)):
        ax.errorbar(xx, yy, ee, fmt=m, color=c, mfc="none" if cv else c, ms=5, capsize=2, label=label if k == 0 else None)

# 1 liquid hull
if len(hull):
    Ps = sorted(hull.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.4 * len(Ps), 4.2), sharey=True, squeeze=False)
    for ax, P in zip(axs[0], Ps):
        h = hull[hull.P == P]; Ts = sorted(h["T"].unique())
        for T in Ts:
            g = h[h["T"] == T].sort_values("x"); c = Tcol(T, Ts); x = np.r_[0, g.x, 1]; y = np.r_[0, g.dG, 0]
            hollow(ax, g.x, g.dG, g.err, g.caveat, c, "o", f"{T:.0f} K"); ax.plot(*lower_hull(x, y), "-", color=c, lw=.9, alpha=.8)
        lab = "unlabelled (400 GPa assumed)" if P == 400 else "350 GPa" if P == 350 else None
        if lab:
            for (l2, T), (y, e) in dft.items():
                if l2 == lab and T in Ts: ax.errorbar(dft_x + 0.012, y, e, fmt="s", color=Tcol(T, Ts), mfc="none", ms=7, alpha=.5, capsize=1, label=f"DFT sheet {T} K")
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{P:.0f} GPa"); ax.set_xlabel("x_H = n/(1+n)  (H$_{2+n}$O)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
    axs[0][0].set_ylabel("liquid mixing free energy (meV/atom)"); fig.suptitle("Liquid hull: G(H2O + n H -> H2+nO) per atom. Lines: lower convex hull incl. x=0,1. Hollow = caveat (dt/lambda grid). Squares: DFT hull sheet (header says eV/atom, values read as meV/atom)", fontsize=8)
    figs["hull"] = fig

# 2 fcc - bcc vs literature
Tset = [3000, 5000]; fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, T in zip(axs, Tset):
    if len(lit):
        for s, g in lit[(lit["T_K"] - T).abs() < 250].groupby("source"):
            g = g.sort_values("P_GPa"); ax.errorbar(g.P_GPa, g.dG, g.err_meV_atom.fillna(0), fmt="s--", color="tab:green", mfc="none", ms=5, capsize=2, label=s)
    if len(cheng):
        g = cheng[cheng["T_K"] == T].sort_values("P_GPa")
        if len(g):
            ax.plot(g.P_GPa, g.pbe, "-", color="tab:blue", label="Cheng 2021 (PBE-corrected)"); ax.fill_between(g.P_GPa, g.pbe - g.err, g.pbe + g.err, color="tab:blue", alpha=.15); ax.plot(g.P_GPa, g.mlp, ":", color="tab:blue", label="Cheng 2021 (their MLP)")
    for h, c, m in ((2.0, "k", "o"), (2.25, "tab:red", "^")):
        g = pairs[(pairs["T"] == T) & (pairs.h == h)].sort_values("P")
        if len(g):
            for cell, mm, sub in (("big", m, g[g.cell == "big"]), ("small", "v", g[g.cell == "small"])):
                if len(sub): hollow(ax, sub.P, sub.dG, sub.err, sub.caveat | (cell == "small"), c, mm, f"this work H{h:g}O fcc-bcc ({cell} cell)")
    ax.axhline(0, color="k", lw=.6); ax.set_xscale("log"); ax.set_title(f"T = {T} K"); ax.set_xlabel("P (GPa)"); ax.grid(alpha=.25, which="both"); ax.legend(fontsize=6)
axs[0].set_ylabel("G_fcc - G_bcc (meV/atom)"); fig.suptitle("fcc - bcc superionic H2O vs literature (negative = fcc stable). Hollow = small cell or different dt/lambda grid. Our values are MLP (ft_m2), classical nuclei", fontsize=8)
figs["fcc_bcc"] = fig

# 3 solid mixing per added H
sm = solid[solid.kind.isin(["fcc", "bcc"])]
if len(sm):
    Ps = sorted(sm.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.6 * len(Ps), 4.2), sharey=True, squeeze=False)
    for ax, P in zip(axs[0], Ps):
        for kind, c, m in (("fcc", "tab:blue", "o"), ("bcc", "tab:orange", "^"), ("insert", "tab:green", "s")):
            g = solid[(solid.P == P) & (solid.kind == kind)].sort_values("T")
            for n, gg in g.groupby("n"):
                hollow(ax, gg["T"] + (15 * n if kind == "insert" else 0), gg.dG, gg.err, gg.caveat, c, m, f"{kind} n={n}" if kind == "insert" else f"{kind} H2.xO ({n} added)")
        Tl = np.linspace(solid[solid.P == P]["T"].min(), solid[solid.P == P]["T"].max(), 20) if len(solid[solid.P == P]) else []
        if len(Tl): ax.fill_between(Tl, -kT(Tl), kT(Tl), color="gray", alpha=.12, label="+-kT")
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{P:.0f} GPa"); ax.set_xlabel("T (K)"); ax.grid(alpha=.25); ax.legend(fontsize=5, ncol=2)
    axs[0][0].set_ylabel("dG per added H (meV)"); fig.suptitle("Crystal H2O + n H(fluid) -> H2+n/nO crystal, per added H (positive = H not soluble). Insertion: fixed-V mu_H(host) - G_H; hollow = melted in free NVT / different dt or lambda grid", fontsize=8)
    figs["mix_solid"] = fig

# 4 insertion: n dependence
ins = solid[solid.kind == "insert"]
if len(ins):
    fig, ax = plt.subplots(figsize=(6, 4.2)); keys = sorted(set(zip(ins.P, ins["T"]))); out = []
    for k, (P, T) in enumerate(keys):
        g = ins[(ins.P == P) & (ins["T"] == T)].sort_values("n"); c = plt.cm.tab20(k % 20)
        ax.errorbar(g.n, g.dG, g.err, fmt="o-", color=c, ms=4, capsize=2, lw=.8, label=f"{P:.0f} GPa {T:.0f} K")
        if len(g) >= 2:
            w = 1 / np.maximum(g.err.values, 1e-6) ** 2; A = np.vstack([np.ones(len(g)), g.n.values]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * g.dG.values))
            out.append(f"{P:.0f} GPa {T:.0f} K: n->0 {b[0]:7.0f} +- {cov[0, 0] ** .5:4.0f} meV, slope {b[1]:5.0f} meV/H")
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("number of inserted H n (fixed V)"); ax.set_ylabel("mu_H(insertion) - G_H(pure) (meV)"); ax.grid(alpha=.25); ax.legend(fontsize=5, ncol=2)
    ax.set_title("Insertion vs n: the slope is the fixed-V pressure rise of the cell; weighted line fits give n->0", fontsize=8); figs["insert_n"] = fig
    print("insertion n->0 extrapolation (weighted linear fit, error from the fit only):"); print("\n".join(out))

# 5 liquid-solid crossing
if len(cross):
    Ps = sorted(cross.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.4 * len(Ps), 4.2), sharey=True, squeeze=False)
    for ax, P in zip(axs[0], Ps):
        for h, c in ((2.0, "k"), (2.25, "tab:red"), (3.0, "tab:blue")):
            g = cross[(cross.P == P) & (cross.h == h)].sort_values("T")
            if len(g): hollow(ax, g["T"], g.dG, g.err, g.caveat, c, "o", f"H{h:g}O"); ax.plot(g["T"], g.dG, "-", color=c, lw=.6)
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{P:.0f} GPa"); ax.set_xlabel("T (K)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
    axs[0][0].set_ylabel("G_liquid - G_fcc (meV/atom)"); fig.suptitle("Melting: negative = liquid more stable than the fcc crystal at the same (P, T, composition). Zero crossing = TDI melting temperature (statistical errors only)", fontsize=8)
    figs["melt"] = fig

# 6 start-configuration protocol
if len(prep):
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(prep) + 2), 4.2)); g = prep.sort_values(["control", "h", "P", "T"]).reset_index(drop=True)
    for k, r in g.iterrows():
        ax.errorbar(k, r.dG, r.err, fmt="o" if not r.control else "s", color="tab:red" if not r.control else "gray", mfc="none" if r.caveat else None, capsize=2)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{r.P:.0f}/{r.T:.0f}\nH{r.h:g}O" for _, r in g.iterrows()], fontsize=6, rotation=90); ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
    ax.set_ylabel("G(prep run) - G(cold start) (meV/atom)"); ax.set_title("Start-configuration protocol (hold O, thermalise H, release O) vs cold start, P(GPa)/T(K). Grey squares: H2O control (no protocol in either: tests longer run / grid)", fontsize=7); figs["prep"] = fig

# 7 coverage
fig, ax = plt.subplots(figsize=(7, 4.6)); mk = {"fcc": "s", "bcc": "^", "liquid": "o", "H": "D", "insert": "P"}
for ph, g in raw.groupby("phase"):
    bad = g.excluded != ""
    ax.scatter(g.P_target_GPa[~bad] + 4 * (g.H_per_O.fillna(2)[~bad] - 2), g.T_K[~bad], marker=mk.get(ph, "x"), s=26, alpha=.7, label=ph)
    if bad.any(): ax.scatter(g.P_target_GPa[bad], g.T_K[bad], marker="x", color="red", s=40, label=f"{ph} excluded")
ax.set_xlabel("P (GPa)  (composition offset 4 GPa per extra H per O)"); ax.set_ylabel("T (K)"); ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2); ax.set_title("Coverage of the TDI runs", fontsize=9); figs["coverage"] = fig

# 8 QA
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2)); ok = raw[raw.excluded == ""]
axs[0].scatter(ok.ps_per_node, ok.drift_meV, s=12, c=ok.G_err_meV_atom, cmap="viridis"); axs[0].axhline(0, color="k", lw=.5); axs[0].set_xlabel("ps per lambda"); axs[0].set_ylabel("drift = G(all) - G(first half) (meV/atom)"); axs[0].set_title("convergence (colour: G error)", fontsize=8)
q = raw.quad_diff_meV.dropna()
if len(q): axs[1].hist(q, bins=20); axs[1].set_xlabel("Kronrod - Gauss (meV/atom)"); axs[1].set_title("quadrature error estimate", fontsize=8)
axs[2].scatter(raw.P_target_GPa, 100 * raw.P_dev, s=12, c=np.where(raw.excluded != "", "red", "tab:blue")); axs[2].axhline(0, color="k", lw=.5); axs[2].set_xlabel("P target (GPa)"); axs[2].set_ylabel("P_hi - P_target (%)"); axs[2].set_title("MLP pressure at the highest lambda (red = excluded)", fontsize=8)
for x in axs: x.grid(alpha=.25)
exc = raw[raw.excluded != ""]
fig.text(0.01, 0.0, "Excluded: " + ("; ".join(f"{r.dir}/{r.tag} ({r.excluded})" for _, r in exc.iterrows())[:900] if len(exc) else "none"), fontsize=5, va="bottom", wrap=True); figs["qa"] = fig

# ---------------------------------------------------------------- write
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with PdfPages(a.out + ".pdf") as pdf:
    for name, fig in figs.items():
        fig.tight_layout(rect=(0, 0.03, 1, 0.95)); pdf.savefig(fig)
        for fm in a.fmt: fig.savefig(f"{a.out}_{name}.{fm}", dpi=150)
print("wrote", a.out + ".pdf", "and", ", ".join(f"{a.out}_{n}.{fm}" for n in figs for fm in a.fmt))
