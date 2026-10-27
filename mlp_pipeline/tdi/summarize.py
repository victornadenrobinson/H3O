"""Summary of every TDI run found, recomputed from the saved lambda series (no need to run collect.py first, partial runs included).
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py [dir ...] [-o docs/data/tdi_results.csv] [--plots]
Without directories it scans runs/tdi* (not runs/tdi_test). One row per tag and directory: G (meV/atom; for insertion runs mu_H per added H), the Gauss-Legendre statistical error, the drift (G over all samples minus G over the first
half, a convergence hint), the share of the target time done, and the MLP pressure at the highest lambda. Then: G_fcc - G_bcc for equal cell sizes, mixing free energy per added H (crystals and insertion) and the liquid hull."""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.gcalc import G_from_nodes
from tdi.verdict import verdict
EV_A3_GPA = 160.21766

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("-o", default=None)
ap.add_argument("--discard", type=float, default=0.0, help="discard this fraction of the start of every lambda series before integrating (e.g. 0.5 = use the second half only): a direct test of equilibration / convergence of every table below"); a = ap.parse_args()
dirs = a.dirs or [d for d in sorted(glob.glob("runs/tdi*")) if os.path.isdir(d) and os.path.basename(d) != "tdi_test"]
if not dirs:
    raise SystemExit("no runs/tdi* directory here: run from the repo root or give the directory")
rows, skipped, stale = [], [], []
for d in dirs:
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; st = json.load(open(f0)); A = st.get("args", {})
        lam, w, gsub = analyze.rule_nodes(A); per_ps = 1000 / A["dt"] / A["every"]
        def load(k):   # a file being rewritten by a running chunk can be unreadable for a moment: treat as empty
            f = os.path.join(d, f"{tag}_node{k}.npy")
            try:
                return np.load(f) if os.path.exists(f) else np.array([])
            except Exception:
                return np.array([])
        ser = [load(k) for k in range(A["nodes"])]
        if os.path.exists(os.path.join(d, f"{tag}_node{A['nodes']}.npy")):   # node files beyond what the json declares: the json was probably overwritten by a later launch with other settings
            stale.append(f"{os.path.basename(d)}/{tag}: node{A['nodes']}.npy exists but stage0.json declares {A['nodes']} lambda values (stale json?)")
        copied = []
        for k in range(A["nodes"]):   # a failed lambda value with a negligible weight (weight < 1e-4, e.g. lambda 0 after a GPU memory error) takes the series of the next lambda value
            if len(ser[k]) == 0 and w[k] < 1e-4:
                j = next((j for j in range(k + 1, A["nodes"]) if len(ser[j]) > 0), None)
                if j is not None:
                    ser[k] = ser[j].copy(); copied.append(k)
        partial_gk = False
        if gsub is not None and any(len(x) == 0 for x in ser[len(gsub[0]):]) and all(len(x) > 0 for x in ser[:len(gsub[0])]):   # extension to Gauss-Kronrod not finished: use the Gauss values alone
            lam, w, ser = gsub[0], gsub[1], ser[:len(gsub[0])]; gsub = None; partial_gk = True
        n = min(len(x) for x in ser)
        if n < 25:
            skipped.append(f"{os.path.basename(d)}/{tag}: " + ("no samples" if max(len(x) for x in ser) == 0 else f"only {sum(len(x) > 0 for x in ser)}/{A['nodes']} lambda values have samples"))
            continue
        lo = int(a.discard * n); ser = [x[lo:n] for x in ser]; n = n - lo
        if n < 25:
            skipped.append(f"{os.path.basename(d)}/{tag}: fewer than 25 samples left after --discard {a.discard}"); continue
        G, Ge, dF, m, sd = G_from_nodes(st, lam, w, [x[:n] for x in ser], st["N"]); Gh = G_from_nodes(st, lam, w, [x[:n // 2] for x in ser], st["N"])[0]
        qd = G - G_from_nodes(st, gsub[0], gsub[1], [x[:n] for x in ser[:len(gsub[0])]], st["N"])[0] if gsub is not None else float("nan")   # Kronrod minus Gauss: quadrature error estimate
        ps = n / per_ps; P_hi = float("nan"); ft = os.path.join(d, f"{tag}_node{A['nodes'] - 1}_thermo.csv")
        if os.path.exists(ft):
            try:
                P_hi = float(pd.read_csv(ft).P_MLP_GPa.mean())
            except Exception:
                pass
        rows.append(dict(dir=os.path.basename(d), tag=tag, phase=st["phase"], T_K=st["T"], P_target_GPa=st["P_target"], H_per_O=A.get("hper", 2.0) if st["nO"] else float("nan"), N=st["N"], nO=st["nO"], nH=st["nH"], nrep=A.get("nrep"),
                         nodes=A["nodes"], power=A["power"], dt_fs=A["dt"], ps_per_node=ps, ps_target=A["ps"], status=("done" if ps >= A["ps"] - 1e-6 else f"{100 * ps / A['ps']:.0f}%") + (f" (lambda {','.join(map(str, copied))} copied)" if copied else "") + (f" (Gauss {A['ngauss']} only: Kronrod extension pending)" if partial_gk else ""), quad_diff_meV=qd, rule=A.get("rule") or "gl",
                         G_meV_atom=G, G_err_meV_atom=Ge, drift_meV=G - Gh, P_hi_GPa=P_hi, vpa_A3=st["vpa"], P_corr_meV=st.get("dG_P_eV", np.nan) / st["N"] * 1e3,
                         dG_P2_meV_atom=(np.nan if st["phase"] == "insert" else -st["V"] / st["N"] * (P_hi - st["P_target"]) ** 2 / (2 * st.get("B_GPa", np.nan)) / EV_A3_GPA * 1e3), git=st.get("meta", {}).get("git", "")[:8], date=st.get("meta", {}).get("date", "")))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no tag with samples found in " + ", ".join(dirs) + ("\nskipped: " + "; ".join(skipped) if skipped else ""))
df = df.sort_values(["phase", "P_target_GPa", "T_K", "H_per_O", "nrep", "dt_fs", "dir"]).reset_index(drop=True)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"All runs{' (first ' + str(a.discard) + ' of every series discarded)' if a.discard else ''} (G in meV/atom, insertion rows: mu_H per added H; drift = G(all) - G(first half); quad_diff = Gauss-Kronrod minus Gauss-only (quadrature error estimate, only for rule gk runs); P_hi = MLP pressure at the highest lambda, target in P_target; dG_P2 = -V (P_hi - P_target)^2 / (2 B N): second-order correction G(P_target) - (F + P_target V) when the MLP pressure misses the target, NOT applied; nan = no bulk modulus in the stage0 json (older runs); insertion rows: P_corr = pressure correction to P_target included in G):")
print(df[["dir", "tag", "N", "dt_fs", "nodes", "ps_per_node", "status", "G_meV_atom", "G_err_meV_atom", "drift_meV", "quad_diff_meV", "P_target_GPa", "P_hi_GPa", "dG_P2_meV_atom", "P_corr_meV"]].round(2).to_string(index=False))
if stale:
    print("\nWARNING, more node files than the json declares: " + "; ".join(stale))
if skipped:
    print("\nnot summarised (no or incomplete samples): " + "; ".join(skipped))
def pick(g, dt, nodes=None):
    """one row of g: same dt as `dt` if there is one, else the smallest dt; among those the same number of lambda values as `nodes` if there is one; then the longest run (the caller flags differences)"""
    same = g[g.dt_fs == dt]; g = same if len(same) else g[g.dt_fs == g.dt_fs.min()]
    if nodes is not None and (g.nodes == nodes).any():
        g = g[g.nodes == nodes]
    return g.sort_values("ps_per_node").iloc[-1]


pairs = []
df["cell"] = np.where(df.nO < 75, "small", "big")
for (P, T, h, cell), g in df[df.nO > 0].groupby(["P_target_GPa", "T_K", "H_per_O", "cell"]):   # equal cell-size class only: fcc 32 O with bcc 54 O, fcc 108 O with bcc 128 O
    f, b = g[g.phase == "fcc"], g[g.phase == "bcc"]
    if len(f) and len(b):
        f = pick(f, b.dt_fs.min()); b = pick(b, f.dt_fs, f.nodes)
        note = ("DIFFERENT dt " if f.dt_fs != b.dt_fs else "") + ("DIFFERENT lambda grid" if f.nodes != b.nodes else "") + (" not done" if not (f.status.startswith("done") and b.status.startswith("done")) else "")
        pairs.append(dict(P_GPa=P, T_K=T, H_per_O=h, cell=cell, nO_fcc=int(f.nO), nO_bcc=int(b.nO), dG_fcc_minus_bcc_meV_atom=f.G_meV_atom - b.G_meV_atom, err=(f.G_err_meV_atom ** 2 + b.G_err_meV_atom ** 2) ** .5, ps_fcc=f.ps_per_node, ps_bcc=b.ps_per_node,
                          dt_fcc=f.dt_fs, dt_bcc=b.dt_fs, drift_fcc=f.drift_meV, drift_bcc=b.drift_meV, note=note.strip()))
if pairs:
    print("\nG_fcc - G_bcc (meV/atom), equal cell-size class:"); print(pd.DataFrame(pairs).round(2).to_string(index=False))

# free-NVT stability checks (17_eos_md.py rows): did the crystal at this composition, P and T keep its O lattice? same verdict rule as plot_pt.py (solid / marginal / melted)
def _nvt_table():
    out = []
    for d in ("runs/melt_tdi", "runs/melt", "runs/melt_comp", "runs/eos_scan"):
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            try: t = pd.read_csv(f)
            except Exception: continue
            for _, r in t.iterrows():
                if r.get("branch", "solid") != "solid" or r.get("phase") not in ("fcc", "bcc") or "T_set" not in r or "P_GPa" not in r: continue
                pt = r.get("P_target", np.nan)
                out.append(dict(hold=float(r["hold_O_ps"]) if np.isfinite(r.get("hold_O_ps", np.nan)) else 0.0, phase=r["phase"], hper=float(r["hper"]) if np.isfinite(r.get("hper", np.nan)) else 2.0, T=float(r["T_set"]), P=float(pt) if np.isfinite(pt) else float(r["P_GPa"]), exact=bool(np.isfinite(pt)),
                                v=verdict(r)))
    return pd.DataFrame(out, columns=["hold", "phase", "hper", "T", "P", "exact", "v"])
NVT = _nvt_table()
def nvt_flag(phase, h, P, T):
    """text for the notes column: free NVT verdict of this crystal at (P, T), or 'not NVT-checked'"""
    g = NVT[(NVT.phase == phase) & ((NVT.hper - h).abs() < 0.02) & (NVT["T"] == T) & (((NVT.P - P).abs() <= 2) | (~NVT.exact & ((NVT.P - P).abs() <= 0.1 * P)))] if len(NVT) else NVT
    lab = f"H{h:g}O"; g = g[g.v != "unknown"] if len(g) else g   # too short to judge: as if not checked
    if not len(g): return f"{lab} not NVT-checked"
    nm = int((g.v == "melted").sum()); ng = int((g.v == "marginal").sum())
    if not (nm or ng): return ""
    f, h = g[g.hold == 0], g[g.hold > 0]   # free start vs O held first
    return f"{lab} {'MELTED' if nm else 'marginal'} in free NVT: {nm} melted / {ng} marginal of {len(g)} runs (free start {int((f.v == 'melted').sum())}/{len(f)}, O-hold {int((h.v == 'melted').sum())}/{len(h)} melted)"
# crystal mixing: H(2+n)O crystal (same phase, cell and T, P) -> H2O crystal + n pure-H fluid atoms, per added H; and the dilute limit from the insertion runs (mu_H - G_H)
solid = []
for _, m in df[(df.nO > 0) & df.phase.isin(["fcc", "bcc"]) & (df.H_per_O > 2.0)].iterrows():
    w = df[(df.phase == m.phase) & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K) & (df.H_per_O == 2.0) & (df.nO == m.nO)]
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(w) and len(hh):
        w, hh = pick(w, m.dt_fs, m.nodes), pick(hh, m.dt_fs, m.nodes); n = m.nH - 2 * m.nO
        dG = (m.N * m.G_meV_atom - w.N * w.G_meV_atom - n * hh.G_meV_atom) / n
        err = ((m.N * m.G_err_meV_atom) ** 2 + (w.N * w.G_err_meV_atom) ** 2 + (n * hh.G_err_meV_atom) ** 2) ** .5 / n
        solid.append(dict(phase=m.phase, P_GPa=m.P_target_GPa, T_K=m.T_K, H_per_O=m.H_per_O, n_added=int(n), dG_per_added_H_meV=dG, err=err,
                          P2_corr_per_H_meV=(m.N * m.dG_P2_meV_atom - w.N * w.dG_P2_meV_atom - n * hh.dG_P2_meV_atom) / n, kT_meV=8.617333e-2 * m.T_K, dt_mix=m.dt_fs, dt_H2O=w.dt_fs, dt_H=hh.dt_fs, ps_mix=m.ps_per_node, ps_H2O=w.ps_per_node, ps_H=hh.ps_per_node, nodes_mix=m.nodes, nodes_H2O=w.nodes, nodes_H=hh.nodes, N_H=int(hh.N),
                          note=("DIFFERENT dt " if not (m.dt_fs == w.dt_fs == hh.dt_fs) else "") + ("DIFFERENT lambda grid " if not (m.nodes == w.nodes == hh.nodes) else "") + "; ".join(x for x in (nvt_flag(m.phase, m.H_per_O, m.P_target_GPa, m.T_K), nvt_flag(m.phase, 2.0, m.P_target_GPa, m.T_K)) if x)))
for _, m in df[df.phase == "insert"].iterrows():
    hh = df[(df.phase == "H") & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K)]
    if len(hh):
        hh = pick(hh, m.dt_fs)
        solid.append(dict(phase="insert (fcc H2O, dilute)", P_GPa=m.P_target_GPa, T_K=m.T_K, H_per_O=2.0, n_added=int(m.N), dG_per_added_H_meV=m.G_meV_atom - hh.G_meV_atom, err=(m.G_err_meV_atom ** 2 + hh.G_err_meV_atom ** 2) ** .5,
                          kT_meV=8.617333e-2 * m.T_K, dt_mix=m.dt_fs, dt_H2O=float("nan"), dt_H=hh.dt_fs, N_H=int(hh.N), note=("" if m.dt_fs == hh.dt_fs else "DIFFERENT dt ")
                          + ("" if np.isfinite(m.P_corr_meV) else "NO PRESSURE CORRECTION (stage 0 older than the fix: rerun stage 0 of this run) ") + nvt_flag("fcc", 2.0, m.P_target_GPa, m.T_K)))
if solid:
    print("\nMixing free energy per added H, crystal H2O + n H(fluid) -> H(2+n/nO)O crystal (positive = H not soluble); insertion rows: mu_H(fcc H2O) - G_H(pure), meV; P2_corr_per_H = the dG_P2 corrections combined like G (not applied):")
    print(pd.DataFrame(solid).round(2).to_string(index=False))
hull = []
for (P, T), g in df.groupby(["P_target_GPa", "T_K"]):
    wg, hg = g[(g.phase == "liquid") & (g.H_per_O == 2.0)], g[g.phase == "H"]
    if len(wg) and len(hg):
        for _, m in g[(g.phase == "liquid") & (g.H_per_O > 2.0)].iterrows():
            w, hh = pick(wg, m.dt_fs, m.nodes), pick(hg, m.dt_fs, m.nodes)
            n = m.H_per_O - 2.0; fw, fh = 3 / (3 + n), n / (3 + n)   # atom fractions of the H2O part and of the added H in the mixture
            note = ("DIFFERENT dt " if not (m.dt_fs == w.dt_fs == hh.dt_fs) else "") + ("DIFFERENT lambda grid " if not (m.nodes == w.nodes == hh.nodes) else "") + ("not done" if not all(str(x.status).startswith("done") for x in (m, w, hh)) else "")
            hull.append(dict(P_GPa=P, T_K=T, H_per_O=m.H_per_O, x_H=n / (1 + n), x_H_atom=fh, dG_mix_meV_atom=m.G_meV_atom - fw * w.G_meV_atom - fh * hh.G_meV_atom,
                             err=(m.G_err_meV_atom ** 2 + (fw * w.G_err_meV_atom) ** 2 + (fh * hh.G_err_meV_atom) ** 2) ** .5, ps_mix=m.ps_per_node, ps_H2O=w.ps_per_node, ps_H=hh.ps_per_node,
                             nodes_mix=m.nodes, nodes_H2O=w.nodes, nodes_H=hh.nodes, N_mix=int(m.N), N_H2O=int(w.N), N_H=int(hh.N), note=note.strip()))
if hull:
    print("\nLiquid mixing free energy per atom, H2O + n H -> H(2+n)O (x_H = n/(1+n) as in the hull sheet; negative = mixing favourable), meV/atom. A hull of these per-atom values must be drawn against x_H_atom = n/(3+n) (fraction of atoms that are added H): against x_H its tie lines are not straight:")
    print(pd.DataFrame(hull).round(2).to_string(index=False))
if a.o:
    df.to_csv(a.o, index=False); pd.DataFrame(pairs).to_csv(a.o.replace(".csv", "_dG.csv"), index=False); pd.DataFrame(solid).to_csv(a.o.replace(".csv", "_mix.csv"), index=False); pd.DataFrame(hull).to_csv(a.o.replace(".csv", "_hull.csv"), index=False); print("wrote", a.o, "and", a.o.replace(".csv", "_dG.csv"))
