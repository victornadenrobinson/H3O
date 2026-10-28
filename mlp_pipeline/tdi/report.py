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
ap.add_argument("--state-check", default="runs/checks/liquid_check.csv", help="liquid_check.py --csv output: liquid runs with a frozen high-lambda node and crystal / insertion runs with a MELTED? one are excluded")
ap.add_argument("--no-gt", action="store_true", help="do not also run plot_all.py (G vs simulation time, separate svg next to the report, not in the pdf)")
ap.add_argument("--gt-dirs", nargs="*", default=[], help="directories for plot_all.py (default: its own default, runs/tdi* without tdi_test)")
ap.add_argument("--qmax", type=float, default=50.0, help="exclude runs with |Kronrod - Gauss| above this (meV/atom)")
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
lc = pd.read_csv(a.state_check) if a.state_check and os.path.exists(a.state_check) else pd.DataFrame(columns=["dir", "tag", "verdict"])
badstate = {(d, t): v for (d, t), v in lc[lc.verdict.isin(["frozen?", "MELTED?"])].groupby(["dir", "tag"]).verdict.first().items()}
if not len(lc): print(f"(no {a.state_check}: liquid / solid state of the high-lambda nodes not checked; run liquid_check.py --csv)")
for _, r in raw.iterrows():
    w = []
    if not str(r.status).startswith("done"): w.append(f"unfinished ({r.status})")
    if np.isfinite(r.P_dev) and abs(r.P_dev) > a.pmax_rel: w.append(f"P_hi {r.P_hi_GPa:.0f} vs target {r.P_target_GPa:.0f} GPa")
    if np.isfinite(r.quad_diff_meV) and abs(r.quad_diff_meV) > a.qmax: w.append(f"Kronrod-Gauss {r.quad_diff_meV:.0f} meV/atom (lambda integral not converged, or a melted crystal held by springs)")
    if np.isfinite(r.get("min_dHH_A", np.nan)) and r.min_dHH_A < 0.25: w.append(f"H-H COLLAPSE (min d(H-H) {r.min_dHH_A:.2f} A: MLP hole, G meaningless)")
    if (r.dir, r.tag) in badstate: w.append("liquid FROZEN at a high-lambda node (liquid_check)" if badstate[(r.dir, r.tag)] == "frozen?" else "O sublattice MELTED at a high-lambda node (liquid_check)")
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
        solid.append(dict(kind=m.phase, P=m.P_target_GPa, T=m.T_K, h=m.H_per_O, a=m.H_per_O - 2.0, n=int(n), dG=(m.N * m.G_meV_atom - w.N * w.G_meV_atom - n * hh.G_meV_atom) / n,
                          err=((m.N * m.G_err_meV_atom) ** 2 + (w.N * w.G_err_meV_atom) ** 2 + (n * hh.G_err_meV_atom) ** 2) ** .5 / n,
                          caveat=("MELTED" in note) or not (m.dt_fs == w.dt_fs == hh.dt_fs and m.nodes == w.nodes == hh.nodes), tags=f"{m.dir}/{m.tag} | {w.dir}/{w.tag} | {hh.dir}/{hh.tag}"))
for _, m in df[(df.phase == "insert") & (~df.tag.str.contains(r"_h\d", regex=True)) & (~df.tag.str.contains("_Hhost"))].iterrows():   # H2O hosts only (H(2+a)O hosts: HPER; pure-H host = method check)
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(hh):
        hh = best(hh, m.dt_fs)
        solid.append(dict(kind="insert_bcc" if "_bcc" in m.tag else "insert", P=m.P_target_GPa, T=m.T_K, h=2.0, a=m.N / m.nO, n=int(m.N), dG=m.G_meV_atom - hh.G_meV_atom,
                          pcorr=float(m.get("P_corr_meV", np.nan)), err=(m.G_err_meV_atom ** 2 + hh.G_err_meV_atom ** 2) ** .5, caveat=m.dt_fs != hh.dt_fs, tags=f"{m.dir}/{m.tag} | {hh.dir}/{hh.tag}"))
solid = pd.DataFrame(solid, columns=["kind", "P", "T", "h", "a", "n", "dG", "err", "caveat", "tags", "pcorr"])
if len(solid) and (solid.dG.abs() > 1500).any():
    print("mixing/insertion rows with |dG| > 1500 meV per H (off scale in the figure; check their input runs):"); print(solid[solid.dG.abs() > 1500][["kind", "P", "T", "n", "dG", "tags"]].round(0).to_string(index=False))

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

# ---------------------------------------------------------------- derived tables (printed and saved next to the figures)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
for name, t in (("hull", hull), ("fcc_bcc", pairs), ("mixing", solid), ("melting", cross), ("prep", prep)):
    if len(t):
        t.to_csv(f"{a.out}_table_{name}.csv", index=False); print(f"\n== {name} (meV/atom; mixing and insertion per added H; caveat = dt / lambda grid / melted differ):")
        print(t.drop(columns=[c for c in ("tags",) if c in t]).round(2).to_string(index=False))

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
    dft_x = np.array(xs); _n = np.array([v / (1 - v) if v < 1 else np.inf for v in dft_x]); dft_xa = np.array([n / (3 + n) if np.isfinite(n) else 1.0 for n in _n])   # sheet x = n/(1+n) -> atom fraction n/(3+n)
    # the sheet's second x column is mislabelled; both blocks use x = 0, 0.2, 0.5, 0.75, 1

# ---------------------------------------------------------------- figures
figs = {}
from matplotlib.colors import LinearSegmentedColormap
HEAT = LinearSegmentedColormap.from_list("heat", ["#1f3fbf", "#1fa3d6", "#f2c230", "#e8590c", "#b30000"])   # saturated all the way (no pale middle): blue = cold ... red = hot
TN = plt.Normalize(3000, 9000)   # temperature -> colour by heat in every figure
def Tcol(T, Ts=None): return HEAT(TN(T))
def Tbar(fig, axs, label="T (K)"):
    sm = plt.cm.ScalarMappable(norm=TN, cmap=HEAT); sm.set_array([]); cax = fig.add_axes([0.935, 0.18, 0.012, 0.62]); fig.colorbar(sm, cax=cax, label=label); fig._cb = True   # own axes right of the panels (tight_layout below leaves room)
def hollow(ax, x, y, e, cav, c, m, label=None):
    for k, (xx, yy, ee, cv) in enumerate(zip(x, y, e, cav)):
        ax.errorbar(xx, yy, ee, fmt=m, color=c, mfc="none" if cv else c, ms=5, capsize=2, label=label if k == 0 else None)

# 1 liquid hull
if len(hull):
    Ps = sorted(hull.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.4 * len(Ps), 4.2), sharey=True, squeeze=False)
    for ax, P in zip(axs[0], Ps):
        h = hull[hull.P == P]; Ts = sorted(h["T"].unique())
        for T in Ts:
            g = h[h["T"] == T].sort_values("x"); c = Tcol(T); xa = g.x / (1 - g.x) / (3 + g.x / (1 - g.x)); x = np.r_[0, xa, 1]; y = np.r_[0, g.dG, 0]   # n = x/(1-x); atom fraction of the added H = n/(3+n)
            hollow(ax, xa, g.dG, g.err, g.caveat, c, "o", f"{T:.0f} K"); ax.plot(*lower_hull(x, y), "-", color=c, lw=.9, alpha=.8)
        lab = "unlabelled (400 GPa assumed)" if P == 400 else "350 GPa" if P == 350 else None
        if lab:
            for (l2, T), (y, e) in dft.items():
                if l2 == lab and T in Ts: ax.errorbar(dft_xa + 0.01, y, e, fmt="s", color=Tcol(T), mfc="none", ms=7, alpha=.6, capsize=1, label=f"DFT sheet {T} K")
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{P:.0f} GPa"); ax.set_xlabel("added-H atom fraction n/(3+n)  (mixture H$_2$O + n H)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
    axs[0][0].set_ylabel("liquid mixing free energy (meV/atom)"); Tbar(fig, list(axs[0])); fig.suptitle("Liquid hull: G(H2O + n H -> H2+nO) per atom vs n/(3+n) (n = H per O - 2). Lines: lower convex hull incl. 0,1. Hollow = caveat (dt/lambda grid). Squares: DFT hull sheet (header says eV/atom, values read as meV/atom)", fontsize=8)
    figs["hull"] = fig

# 2 fcc - bcc vs literature
Tset = [3000, 5000]; fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, T in zip(axs, Tset):
    if len(lit):
        for s, g in lit[((lit["T_K"] - T).abs() < 250) & lit.source.str.startswith("Wilson")].groupby("source"):   # Cheng comes from the full table below; old 'this work' rows of the csv are replaced by the current TDI results
            g = g.sort_values("P_GPa"); ax.errorbar(g.P_GPa, g.dG, g.err_meV_atom.fillna(0), fmt="s--", color="tab:green", mfc="none", ms=6, capsize=2, label="Wilson 2013 (DFT-PBE TI)")
    if len(cheng):
        g = cheng[cheng["T_K"] == T].sort_values("P_GPa")
        if len(g):
            ax.plot(g.P_GPa, g.pbe, "-", color="tab:blue", label="Cheng 2021 (PBE-corrected)"); ax.fill_between(g.P_GPa, g.pbe - g.err, g.pbe + g.err, color="tab:blue", alpha=.15); ax.plot(g.P_GPa, g.mlp, ":", color="tab:blue", label="Cheng 2021 (their MLP)")
    for h, c, m in ((2.0, "k", "o"), (2.25, "tab:red", "D")):
        g = pairs[(pairs["T"] == T) & (pairs.h == h)].sort_values("P")
        if len(g):
            for cell, mm, sub in (("big", m, g[g.cell == "big"]), ("small", "v", g[g.cell == "small"])):
                if len(sub): hollow(ax, sub.P, sub.dG, sub.err, sub.caveat | (cell == "small"), c, mm, f"this work H{h:g}O fcc-bcc ({cell} cell)")
    ax.axhline(0, color="k", lw=.6); ax.set_xscale("log"); ax.set_xticks([100, 200, 300, 500, 1000, 2000, 4000]); ax.set_xticklabels(["100", "200", "300", "500", "1000", "2000", "4000"]); ax.minorticks_off(); ax.set_xlim(85, 4500); ax.set_title(f"T = {T} K"); ax.set_xlabel("P (GPa)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
axs[0].set_ylabel("G_fcc - G_bcc (meV/atom)"); fig.suptitle("fcc - bcc superionic H2O vs literature (negative = fcc stable). Hollow = small cell or different dt/lambda grid. Our values are MLP (ft_m2), classical nuclei", fontsize=8)
figs["fcc_bcc"] = fig

# 2b fcc (ABC) - hcp (ABAB) in the same orthohexagonal cell, from the stage tables
ph = []
for (P, T, h), g in df[df.phase.isin(["fccl", "hcp"])].groupby(["P_target_GPa", "T_K", "H_per_O"]):
    f, b = g[g.phase == "fccl"], g[g.phase == "hcp"]
    if len(f) and len(b):
        f = best(f, b.dt_fs.min()); b = best(b, f.dt_fs, f.nodes)
        ph.append(dict(P=P, T=T, h=h, dG=f.G_meV_atom - b.G_meV_atom, err=(f.G_err_meV_atom ** 2 + b.G_err_meV_atom ** 2) ** .5, caveat=(f.dt_fs != b.dt_fs) or (f.nodes != b.nodes)))
ph = pd.DataFrame(ph, columns=["P", "T", "h", "dG", "err", "caveat"])
if len(ph):
    ph.to_csv(f"{a.out}_table_fcc_hcp.csv", index=False); fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for T, g in ph.groupby("T"):
        g = g.sort_values("P"); hollow(ax, g.P, g.dG, g.err, g.caveat, Tcol(T), "o", f"{T:.0f} K"); ax.plot(g.P, g.dG, "-", color=Tcol(T), lw=.7)
    ax.axhline(0, color="k", lw=.6); ax.set_xlabel("P (GPa)"); ax.set_ylabel("G_fcc(ABC) - G_hcp (meV/atom)"); ax.grid(alpha=.25); ax.legend(fontsize=6)
    ax.set_title("fcc (ABCABC) vs hcp (ABABAB) H2O in the same 96-O cell; negative = fcc stable. The MLP was not trained on hcp: a control, not a prediction", fontsize=7); figs["fcc_hcp"] = fig

# 3 solid mixing per added H
sm = solid[solid.kind.isin(["fcc", "bcc"])]
if len(sm):
    Ps = sorted(sm.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.8 * len(Ps), 4.4), sharey=True, squeeze=False); YL = (-1000, 800); nout = 0
    for ax, P in zip(axs[0], Ps):
        d = solid[solid.P == P]; jit = 0.012 * max(float(d["T"].max() - d["T"].min()), 500.0)
        for kind, m in (("fcc", "o"), ("bcc", "^"), ("insert", "s"), ("insert_bcc", "D")):
            for n, gg in d[d.kind == kind].sort_values("T").groupby("n"):
                gg = gg[(gg.dG > YL[0]) & (gg.dG < YL[1])] if len(gg) else gg; nout += int(((d.kind == kind) & (d.n == n)).sum() - len(gg))
                for _, r in gg.iterrows():
                    ax.errorbar(r["T"] + (jit * {1: -1, 2: 0, 4: 1}.get(n, 0) if kind.startswith("insert") else 0), r.dG, r.err, fmt=m, color=Tcol(r["T"]), mfc="none" if r.caveat else Tcol(r["T"]), mec="k" if not r.caveat else Tcol(r["T"]), ms=6, capsize=2)
        for kind, m, t in (("fcc", "o", "fcc H2.xO crystal"), ("bcc", "^", "bcc H2.xO crystal"), ("insert", "s", "insertion into fcc (n = 1, 2, 4: left, centre, right)"), ("insert_bcc", "D", "insertion into bcc")): ax.plot([], [], m, color="gray", label=t)
        Tl = np.linspace(d["T"].min() - 100, d["T"].max() + 100, 20); ax.fill_between(Tl, -kT(Tl), kT(Tl), color="gray", alpha=.12, label="+-kT")
        ax.set_ylim(*YL); ax.axhline(0, color="k", lw=.6); ax.set_title(f"{P:.0f} GPa"); ax.set_xlabel("T (K)"); ax.grid(alpha=.25); ax.legend(fontsize=6, loc="lower right")
    axs[0][0].set_ylabel("dG per added H (meV)"); Tbar(fig, list(axs[0])); fig.suptitle("Crystal H2O + n H(fluid) -> H2+n/nO crystal, per added H (positive = H not soluble). Hollow = melted in free NVT / different dt or lambda grid" + (f"; {nout} off-scale rows not drawn (printed in the log)" if nout else ""), fontsize=8)
    figs["mix_solid"] = fig

# 4 insertion: n dependence, one panel per pressure, colour = T
ins = solid[solid.kind == "insert"]
if len(ins):
    Ps = sorted(ins.P.unique()); fig, axs = plt.subplots(1, len(Ps), figsize=(4.6 * len(Ps), 4.4), sharey=True, squeeze=False); out = []
    for ax, P in zip(axs[0], Ps):
        for T in sorted(ins[ins.P == P]["T"].unique()):
            g = ins[(ins.P == P) & (ins["T"] == T)].sort_values("n").copy(); c = Tcol(T); pc = g["pcorr"].values if "pcorr" in g else np.full(len(g), np.nan)
            mixed = np.isfinite(pc).any() and not np.isfinite(pc).all()
            if mixed:   # never fit pressure-corrected and uncorrected rows together: put all on the uncorrected basis
                g["dG"] = g.dG.values - np.nan_to_num(pc)
            ax.errorbar(g.n, g.dG, g.err, fmt="o-", color=c, ms=5, capsize=2, lw=1.0, label=f"{T:.0f} K")
            if len(g) >= 2:
                w = 1 / np.maximum(g.err.values, 1e-6) ** 2; A = np.vstack([np.ones(len(g)), g.n.values]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * g.dG.values))
                ax.plot([0, g.n.max()], [b[0], b[0] + b[1] * g.n.max()], "--", color=c, lw=.7); out.append(f"{P:.0f} GPa {T:.0f} K: n->0 {b[0]:7.0f} +- {cov[0, 0] ** .5:4.0f} meV, slope {b[1]:5.0f} meV/H"
                                                                                             + ("   (mixed P correction: all rows on the uncorrected basis)" if mixed else "   (P-corrected)" if np.isfinite(pc).all() else ""))
        ax.axhline(0, color="k", lw=.6); ax.set_xlabel("inserted H n (fixed V, 108 O host)"); ax.set_title(f"{P:.0f} GPa", fontsize=9); ax.grid(alpha=.25); ax.legend(fontsize=6); ax.set_xlim(0, None)
    axs[0][0].set_ylabel("mu_H(insertion) - G_H(pure) (meV per H)"); Tbar(fig, list(axs[0]))
    fig.suptitle("Insertion vs n per pressure: dashed = weighted linear fit to n->0; the slope contains the fixed-V pressure rise of the cell (before the stage-0 correction)", fontsize=8); figs["insert_n"] = fig
    print("insertion n->0 extrapolation (weighted linear fit, error from the fit only):"); print("\n".join(out))

# 4b insertion into fcc vs bcc host at the same (P, T): mu_H - G_H(pure) vs n; table of the difference per n
insb = solid[solid.kind == "insert_bcc"]
if len(insb):
    keys = sorted(set(zip(insb.P, insb["T"]))); nc = min(3, len(keys)); nr = -(-len(keys) // nc); fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.6 * nr), squeeze=False); rowsb = []
    for ax, (P, T) in zip(axs.ravel(), keys):
        for kind, m, c in (("insert", "s", "tab:blue"), ("insert_bcc", "D", "tab:orange")):
            g = solid[(solid.kind == kind) & (solid.P == P) & (solid["T"] == T)].sort_values("n")
            if len(g): ax.errorbar(g.n, g.dG, g.err, fmt=m + "-", color=c, ms=5, capsize=2, lw=.9, label="fcc host" if kind == "insert" else "bcc host")
        gf, gb = solid[(solid.kind == "insert") & (solid.P == P) & (solid["T"] == T)].set_index("n"), solid[(solid.kind == "insert_bcc") & (solid.P == P) & (solid["T"] == T)].set_index("n")
        for n in sorted(set(gf.index) & set(gb.index)): rowsb.append(dict(P=P, T=T, n=n, mu_fcc_minus_bcc=gf.dG[n] - gb.dG[n], err=float(np.hypot(gf.err[n], gb.err[n]))))
        ax.axhline(0, color="k", lw=.5); ax.set_title(f"{P:.0f} GPa, {T:.0f} K", fontsize=8, color=Tcol(T)); ax.set_xlabel("inserted H n"); ax.set_ylabel("mu_H - G_H(pure) (meV)", fontsize=7); ax.grid(alpha=.25); ax.legend(fontsize=6)
    for ax in axs.ravel()[len(keys):]: ax.axis("off")
    fig.suptitle("Insertion into fcc vs bcc H2O host (fixed V, before the pressure correction)", fontsize=8); figs["insert_hosts"] = fig
    if rowsb: print("\n== mu_H(fcc host) - mu_H(bcc host) per n (meV; negative = H prefers fcc):"); print(pd.DataFrame(rowsb).round(1).to_string(index=False))

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
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{r.P:.0f}/{r['T']:.0f}\nH{r.h:g}O" for _, r in g.iterrows()], fontsize=6, rotation=90); ax.axhline(0, color="k", lw=.6); ax.grid(alpha=.25)
    ax.set_ylabel("G(prep run) - G(cold start) (meV/atom)"); ax.set_title("Start-configuration protocol (hold O, thermalise H, release O) vs cold start, P(GPa)/T(K). Grey squares: H2O control (no protocol in either: tests longer run / grid)", fontsize=7); figs["prep"] = fig

# 6b melting line in P-T space: TDI G_liquid - G_fcc sign changes (+ free-NVT verdicts of H2O crystals where runs/melt* exist)
fig, ax = plt.subplots(figsize=(7.5, 5.2)); anyd = False
sys.path.insert(0, os.path.join(HERE, "..")); from tdi.verdict import verdict
nv = []
for dd in ("runs/melt_tdi", "runs/melt", "runs/melt_comp", "runs/eos_scan"):
    import glob
    for f in sorted(glob.glob(os.path.join(dd, "*.csv"))):
        try: t = pd.read_csv(f)
        except Exception: continue
        for _, r in t.iterrows():
            if r.get("branch", "solid") != "solid" or r.get("phase") not in ("fcc", "bcc") or "T_set" not in r or "P_GPa" not in r: continue
            h = r.get("hper", 2.0); h = 2.0 if not np.isfinite(h) else float(h)
            if abs(h - 2.0) > 0.02: continue
            pt = r.get("P_target", np.nan); nv.append(dict(P=float(pt) if np.isfinite(pt) else float(r["P_GPa"]), T=float(r["T_set"]), v=verdict(r)))
nv = pd.DataFrame(nv, columns=["P", "T", "v"])
if len(nv):
    anyd = True
    for v, m, c, lab in (("melted", "x", "tab:red", "free NVT: melted"), ("marginal", "^", "tab:orange", "free NVT: marginal"), ("solid", "s", "tab:green", "free NVT: solid (6 ps; superheating possible)")):
        g = nv[nv.v == v]
        if len(g): ax.scatter(g.P, g["T"], marker=m, s=30, **({"facecolors": "none", "edgecolors": c} if m == "s" else {"color": c}), label=lab, linewidths=1.2)
if len(cross):
    anyd = True
    for h, m, off in ((2.0, "o", 0), (2.25, "D", 6)):
        g = cross[cross.h == h]
        if len(g): ax.scatter(g.P + off, g["T"], c=np.where(g.dG < 0, "tab:red", "tab:blue"), marker=m, s=40, edgecolors="k", linewidths=.6, label=f"TDI H{h:g}O: red liquid lower G, blue crystal lower G")
    for h, ls in ((2.0, "-"), (2.25, "--")):
        pts = []
        for P, g in cross[cross.h == h].groupby("P"):
            g = g.sort_values("T").reset_index(drop=True)
            for i in range(len(g) - 1):
                if g.dG[i] * g.dG[i + 1] < 0: pts.append((P, g["T"][i] + (g["T"][i + 1] - g["T"][i]) * g.dG[i] / (g.dG[i] - g.dG[i + 1])))
        if len(pts) >= 1: ax.plot(*zip(*sorted(pts)), "k" + ls, marker="*", ms=11, label=f"TDI melting estimate, H{h:g}O (linear interpolation of the sign change)")
ax.set_xlabel("P (GPa)"); ax.set_ylabel("T (K)"); ax.grid(alpha=.25); ax.set_title("Melting of the fcc crystal in P-T space (statistical errors only; the TDI crystal is spring-held, so a molten crystal gives a wrong G)", fontsize=7)
if anyd: ax.legend(fontsize=5.5, loc="upper left"); figs["meltline"] = fig

# 6c finite size: G of the same state point and dt at different system sizes
d1 = df[df.phase != "insert"].assign(h=df.H_per_O.fillna(-1)); rowsfs = []
for key, g in d1.groupby(["phase", "P_target_GPa", "T_K", "h", "dt_fs"]):
    g = g.sort_values("ps_per_node").groupby("N").tail(1)   # the longest run of each size
    if g.N.nunique() >= 2: rowsfs.append((key, g.sort_values("N")))
if rowsfs:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, g in rowsfs:
        ref = g.iloc[-1]; lab = f"{key[0]} {key[1]:.0f} GPa {key[2]:.0f} K" + ("" if key[3] < 0 else f" H{key[3]:g}O")
        axs[0].errorbar(1 / g.N, g.G_meV_atom - ref.G_meV_atom, np.hypot(g.G_err_meV_atom, ref.G_err_meV_atom), fmt="o-", ms=4, lw=.8, capsize=2, label=lab)
        axs[1].scatter(g.N, g.G_err_meV_atom, s=14 + 22 * g.ps_per_node, alpha=.7, label=lab)
    axs[0].axhline(0, color="k", lw=.6); axs[0].set_xlabel("1/N (atoms)"); axs[0].set_ylabel("G(N) - G(largest N) (meV/atom)"); axs[0].legend(fontsize=5); axs[0].grid(alpha=.25)
    axs[0].set_title("Finite size (extensive part cancelled per atom; runs differ in ps per lambda, see right)", fontsize=7)
    axs[1].set_xlabel("N (atoms)"); axs[1].set_ylabel("statistical error of G (meV/atom)"); axs[1].set_title("Error vs size; marker area grows with ps per lambda", fontsize=7); axs[1].grid(alpha=.25); axs[1].set_xscale("log")
    figs["finite_size"] = fig

# 6d chemical potential of H: mu_H - G_H(pure) (meV per added H) vs composition a = added H per O, with the ideal-solution slope and a linear (regular-solution-like) fit; vs P and T
liqH = hull.assign(n=hull.x / (1 - hull.x)); liqH["a"] = liqH.n; liqH["perH"] = liqH.dG * (3 + liqH.n) / liqH.n; liqH["perH_err"] = liqH.err * (3 + liqH.n) / liqH.n
mu0 = []   # (P, T, mu0, err, slope, slope_err) from weighted linear fits of insertion + crystal rows with caveat False and a <= 0.3
grp = [(k, g) for k, g in solid[solid.kind.isin(["insert", "fcc", "bcc"])].groupby(["P", "T"]) if len(g) >= 2]
if grp:
    nc = min(4, len(grp)); nr = -(-len(grp) // nc); fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.4 * nr), squeeze=False, sharey=False)
    for ax, ((P, T), g) in zip(axs.ravel(), sorted(grp)):
        for kind, m, c in (("insert", "s", "tab:green"), ("fcc", "o", "tab:blue"), ("bcc", "^", "tab:orange")):
            q = g[g.kind == kind]
            if len(q): hollow(ax, q.a, q.dG, q.err, q.caveat, c, m, {"insert": "insertion (fixed V)", "fcc": "fcc crystal (direct)", "bcc": "bcc crystal (direct)"}[kind])
        l = liqH[(liqH.P == P) & (liqH["T"] == T) & (liqH.a <= 1.2)]
        if len(l): ax.errorbar(l.a, l.perH, l.perH_err, fmt="d", color="tab:red", ms=4, capsize=2, label="liquid (per added H)")
        f = g[(~g.caveat) & (g.a <= 0.3)]
        if len(f) >= 2 and f.a.nunique() >= 2:
            w = 1 / np.maximum(f.err.values, 1e-6) ** 2; A = np.vstack([np.ones(len(f)), f.a.values]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * f.dG.values))
            xx = np.linspace(0, max(0.3, f.a.max()), 20); ax.plot(xx, b[0] + b[1] * xx, "k-", lw=.9, label="linear fit (regular-solution-like)"); ax.plot(xx, b[0] + kT(T) * xx / 4, "k:", lw=.9, label="ideal fluid-like H (M sites/O -> inf): per-added-H average, slope kT/4 (differential kT/2)")
            mu0.append((P, T, b[0], cov[0, 0] ** .5, b[1], cov[1, 1] ** .5))
        ax.axhline(0, color="k", lw=.5); ax.set_title(f"{P:.0f} GPa, {T:.0f} K", fontsize=8, color=Tcol(T)); ax.grid(alpha=.25); ax.set_xlabel("added H per O", fontsize=7); ax.set_ylabel("mu_H - G_H(pure) (meV)", fontsize=7); ax.legend(fontsize=4.5)
    for ax in axs.ravel()[len(grp):]: ax.axis("off")
    fig.suptitle("Chemical potential of H relative to pure H vs composition. Fixed-V insertion values still contain the pressure rise of the cell (apply the stage-0 pressure correction first)", fontsize=7); figs["mu_x"] = fig
mu0 = pd.DataFrame(mu0, columns=["P", "T", "mu0", "mu0_err", "slope", "slope_err"])
if len(mu0):
    mu0.to_csv(f"{a.out}_table_mu0.csv", index=False); print("\n== mu_H(n->0) - G_H(pure) and slope per added H-per-O (meV), weighted linear fit:"); print(mu0.round(1).to_string(index=False))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    for _, r in mu0.iterrows():
        axs[0].errorbar(r.P, r.mu0, r.mu0_err, fmt="o", color=Tcol(r["T"]), capsize=2); axs[1].errorbar(r["T"], r.mu0, r.mu0_err, fmt="o", color=Tcol(r["T"]), capsize=2)
        axs[2].errorbar(1000 / r["T"], r.mu0 / kT(r["T"]), r.mu0_err / kT(r["T"]), fmt="o", color=plt.cm.viridis((r.P - 250) / 400), capsize=2)
    for ax in axs: ax.axhline(0, color="k", lw=.5); ax.grid(alpha=.25)
    axs[0].set_xlabel("P (GPa)"); axs[0].set_ylabel("mu_H(n->0) - G_H(pure) (meV)"); axs[0].set_title("vs pressure (colour: T)", fontsize=8)
    axs[1].set_xlabel("T (K)"); axs[1].set_title("vs temperature (colour: T)", fontsize=8); axs[2].set_xlabel("1000 / T (1/kK)"); axs[2].set_ylabel("mu0 / kT"); axs[2].set_title("van 't Hoff style: slope = enthalpy/k (colour: P, dark 250 ... yellow 650 GPa)", fontsize=8)
    figs["mu_PT"] = fig

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
if (q.abs() > 20).any(): print("quadrature estimate |Kronrod - Gauss| > 20 meV/atom (run likely not converged in lambda):"); print(raw[raw.quad_diff_meV.abs() > 20][["dir", "tag", "quad_diff_meV"]].round(1).to_string(index=False))
if len(q): axs[1].hist(q, bins=20); axs[1].set_xlabel("Kronrod - Gauss (meV/atom)"); axs[1].set_title("quadrature error estimate", fontsize=8)
axs[2].scatter(raw.P_target_GPa, 100 * raw.P_dev, s=12, c=np.where(raw.excluded != "", "red", "tab:blue")); axs[2].axhline(0, color="k", lw=.5); axs[2].set_xlabel("P target (GPa)"); axs[2].set_ylabel("P_hi - P_target (%)"); axs[2].set_title("MLP pressure at the highest lambda (red = excluded)", fontsize=8)
for x in axs: x.grid(alpha=.25)
exc = raw[raw.excluded != ""]
fig.text(0.01, 0.0, "Excluded: " + ("; ".join(f"{r.dir}/{r.tag} ({r.excluded})" for _, r in exc.iterrows())[:900] if len(exc) else "none"), fontsize=5, va="bottom", wrap=True); figs["qa"] = fig

# ---------------------------------------------------------------- write
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with PdfPages(a.out + ".pdf") as pdf:
    for name, fig in figs.items():
        fig.tight_layout(rect=(0, 0.03, 0.92 if getattr(fig, "_cb", False) else 1, 0.95)); pdf.savefig(fig)
        for fm in a.fmt: fig.savefig(f"{a.out}_{name}.{fm}", dpi=150)
print("wrote", a.out + ".pdf", "and", ", ".join(f"{a.out}_{n}.{fm}" for n in figs for fm in a.fmt))

if not a.no_gt and os.path.isdir("runs"):   # G vs simulation time of every run (separate file, not part of the pdf)
    gt = os.path.join(os.path.dirname(a.out) or ".", "tdi_G_vs_time.svg"); r = subprocess.run([sys.executable, os.path.join(HERE, "plot_all.py"), *a.gt_dirs, "-o", gt], capture_output=True, text=True)
    print(f"wrote {gt}" if r.returncode == 0 else f"plot_all.py failed (exit {r.returncode}): {r.stderr.strip()[-400:]}")
