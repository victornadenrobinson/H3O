"""Simple thermodynamic model of dilute H insertion in superionic H2O across (P, T), from the insertion runs and the pure-H runs.
usage (repo root):  python3 mlp_pipeline/tdi/summarize.py -o runs/checks/summary.csv        (writes the table this script reads)
                    python3 mlp_pipeline/tdi/insertion_model.py runs/checks/summary.csv [-o runs/checks/insertion_model] [--host fcc] [--approx-corr --bulk 2000]
Model.  dg(P, T) = mu_H(in the H2O crystal, n -> 0) - G_H(pure H, per atom) at the same (P, T). Thermodynamics: d dg/dT|_P = -ds (entropy of transfer of one H from pure H
into the crystal), d dg/dP|_T = dv (partial volume of H in the crystal minus the volume per atom of pure H). Fitted to first order around (P0, T0):
        dg = dg0 - ds (T - T0) + dv (P - P0)
Steps: (1) per (P, T): weighted line through dg(n) of the n = 1, 2, 4 runs -> dg at n -> 0 and the slope s (per added H; the solid-solution non-ideality, H2O 108 O cell);
(2) weighted least squares of the model over (P, T); (3) independent check of dv: v_H = V (P_full - P_host) / (n B) from the insertion stage 0 (runs whose stage0 has the pressure
correction) minus V/N of the pure-H run at the same (P, T); (4) prediction: the dg = 0 line (dilute insertion unfavourable beyond it) and, from the slope, the solubility limit
x_sol ~ -dg0 / (2 s) / n_O H per O where the differential mu_H reaches G_H.
Caveats: insertion rows without the stage-0 pressure correction are biased (first order in the cell pressure offset): they are flagged and, with --approx-corr, corrected using
P_hi(n) as P_full, its n -> 0 extrapolation as P_host and B = --bulk (or the json's B_GPa). Only state points where the host stays solid mean anything (check the free-NVT notes)."""
import argparse, json, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("summary", help="CSV written by summarize.py -o"); ap.add_argument("-o", default=None); ap.add_argument("--host", default="fcc", choices=["fcc", "bcc"])
ap.add_argument("--approx-corr", action="store_true"); ap.add_argument("--bulk", type=float, default=None, help="bulk modulus (GPa) for --approx-corr when the json has none")
ap.add_argument("--P0", type=float, default=450.0); ap.add_argument("--T0", type=float, default=6000.0); a = ap.parse_args()
EV_A3_GPA, KB = 160.21766, 8.617333262e-2   # GPa A^3 per eV; meV per K
df = pd.read_csv(a.summary); runs_root = os.path.dirname(os.path.dirname(os.path.abspath(a.summary)))   # runs/checks/x.csv -> runs/


def stage0(r):
    for base in (os.path.join(runs_root, r.dir), os.path.join("runs", r.dir)):
        f = os.path.join(base, f"{r.tag}_stage0.json")
        if os.path.exists(f):
            return json.load(open(f))
    return {}


ins = df[df.phase == "insert"].copy()
ins = ins[ins.tag.str.contains("_bcc") == (a.host == "bcc")]; ins = ins[~ins.tag.str.contains("_Hhost")]
H = df[df.phase == "H"]
rows = []
for (P, T), g in ins.groupby(["P_target_GPa", "T_K"]):
    hh = H[(H.P_target_GPa == P) & (H.T_K == T)]
    if not len(hh):
        print(f"{P:.0f} GPa {T:.0f} K: no pure-H run, skipped"); continue
    h = hh.sort_values("ps_per_node").iloc[-1]; sth = stage0(h)
    g = g.sort_values("N"); n = g.N.values.astype(float); mu = g.G_meV_atom.values.copy(); err = g.G_err_meV_atom.values.copy()
    corr = g.P_corr_meV.values if "P_corr_meV" in g else np.full(len(g), np.nan); flag = ""
    if np.any(~np.isfinite(corr)):
        if a.approx_corr and len(g) >= 2 and np.all(np.isfinite(g.P_hi_GPa.values)):
            ph = g.P_hi_GPa.values; s_ = np.polyfit(n, ph, 1); P_host = s_[1]
            for i, r in enumerate(g.itertuples()):
                if np.isfinite(corr[i]): continue
                st = stage0(r); B = st.get("B_GPa") or a.bulk
                if not st:
                    raise SystemExit(f"--approx-corr: no stage0 json for {r.dir}/{r.tag} (run from the repo root, summary under runs/checks/)")
                if not B: raise SystemExit("--approx-corr needs --bulk (no B_GPa in the stage0 json)")
                mu[i] += -st["V"] / (2 * B) * ((ph[i] - P) ** 2 - (P_host - P) ** 2) / EV_A3_GPA / n[i] * 1e3
            flag = "approx P corr"
        else:
            flag = "NO P corr"
    dg = mu - h.G_meV_atom; e = np.hypot(err, h.G_err_meV_atom)
    if len(n) >= 2:
        w = 1 / e ** 2; X = np.vstack([np.ones_like(n), n]).T; C = np.linalg.inv(X.T @ (X * w[:, None])); b = C @ X.T @ (w * dg)
        dg0, dg0e, s, se = b[0], np.sqrt(C[0, 0]), b[1], np.sqrt(C[1, 1])
    else:
        dg0, dg0e, s, se = dg[0], e[0], np.nan, np.nan
    # independent partial volume of H in the crystal (needs the stage-0 pressure runs) and volume per atom of pure H
    vH = [st["V"] * (st["P_full_GPa"] - st["P_host_GPa"]) / (st["N"] * st["B_GPa"]) for st in (stage0(r) for r in g.itertuples()) if st.get("P_full_GPa") and st.get("B_GPa")]
    vpure = sth["V"] / sth["N"] * (1 + (h.P_hi_GPa - P) / sth["B_GPa"]) if sth.get("B_GPa") and np.isfinite(h.P_hi_GPa) else (sth["V"] / sth["N"] if sth else np.nan)
    rows.append(dict(P=P, T=T, n=",".join(str(int(x)) for x in n), dg0=dg0, dg0_err=dg0e, slope=s, slope_err=se, nO=int(g.nO.iloc[0]),
                     v_H=np.mean(vH) if vH else np.nan, v_pureH=vpure, dv_direct=(np.mean(vH) - vpure) if vH else np.nan, kT=KB * T, note=flag))
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit("no insertion rows with a pure-H partner")
print(f"dilute insertion into {a.host} H2O: dg = mu_H(n -> 0) - G_H(pure), meV per added H (slope = d(dg per added H)/dn in the {R.nO.iloc[0]}-O cell)")
print(R.round(3).to_string(index=False))
ok = R[np.isfinite(R.dg0_err) & (R.dg0_err > 0)]
if len(ok) >= 3 and ok.P.nunique() >= 2 and ok["T"].nunique() >= 2:
    X = np.vstack([np.ones(len(ok)), ok["T"] - a.T0, ok.P - a.P0]).T; w = 1 / ok.dg0_err.values ** 2
    C = np.linalg.inv(X.T @ (X * w[:, None])); b = C @ X.T @ (w * ok.dg0.values); be = np.sqrt(np.diag(C))
    res = ok.dg0.values - X @ b; chi2 = float(np.sum(w * res ** 2)); dof = max(1, len(ok) - 3); scale = np.sqrt(max(1.0, chi2 / dof))
    ds, dv = -b[1], b[2] * EV_A3_GPA / 1e3   # meV/K; meV/GPa -> A^3 (1 GPa A^3 = 6.24 meV)
    print(f"\nfit over {len(ok)} state points (P0 {a.P0:g} GPa, T0 {a.T0:g} K): dg0 = {b[0]:.0f} +- {be[0] * scale:.0f} meV;  ds = {ds:+.3f} +- {be[1] * scale:.3f} meV/K "
          f"({ds / KB:+.2f} k_B);  dv = {dv:+.3f} +- {be[2] * scale * EV_A3_GPA / 1e3:.3f} A^3  (chi2/dof {chi2 / dof:.1f}; errors scaled by {scale:.1f})")
    print(f"   dh = dg0 + T0 ds = {b[0] + a.T0 * ds:.0f} meV (enthalpy of transfer at P0, T0): {'enthalpy' if abs(b[0] + a.T0 * ds) > abs(a.T0 * ds) else 'entropy'}-dominated")
    dd = R.dv_direct.dropna()
    print(f"   independent dv from volumes (v_H from the stage-0 pressure runs - v of pure H): " + (f"{dd.mean():+.3f} A^3 (n {len(dd)})" if len(dd) else "not available yet (needs stage-0 pressure corrections: young_tdi_insert_p0.sbatch)"))
    if abs(b[1]) > 2 * be[1] * scale:
        Ps = np.linspace(R.P.min() - 100, R.P.max() + 100, 5)
        print("   dg0 = 0 (dilute insertion becomes unfavourable) at T = " + ", ".join(f"{a.T0 + (-(b[0] + b[2] * (p - a.P0))) / b[1]:.0f} K ({p:.0f} GPa)" for p in Ps) + "  (linear extrapolation)")
    else:
        print("   no significant T dependence: no dg0 = 0 temperature can be extrapolated from these points")
    if R.slope.notna().any():
        xs = (-R.dg0 / (2 * R.slope)) / R.nO
        print("   solubility limit x_sol ~ -dg0 / (2 slope) / n_O (H per O, regular-solution extrapolation): " + ", ".join(f"{x:.3f} ({p:.0f}/{t:.0f})" for x, p, t in zip(xs, R.P, R["T"]) if np.isfinite(x) and x > 0))
else:
    print("\nnot enough state points for the (P, T) fit: need >= 3 points with >= 2 pressures and >= 2 temperatures")
    b = None
if a.o:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.8)); lim = max(200, np.nanmax(np.abs(R.dg0)))
    sc = ax.scatter(R.P, R["T"], c=R.dg0, cmap="RdBu_r", vmin=-lim, vmax=lim, s=90, edgecolors="#1f1f1e", linewidths=0.8, zorder=3)
    for r in R.itertuples():
        ax.annotate(f"{r.dg0:.0f}", (r.P, r.T), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#1f1f1e")
    if b is not None:
        PP, TT = np.meshgrid(np.linspace(R.P.min() - 100, R.P.max() + 100, 60), np.linspace(R["T"].min() - 1500, R["T"].max() + 1500, 60))
        Z = b[0] + b[1] * (TT - a.T0) + b[2] * (PP - a.P0); cs = ax.contour(PP, TT, Z, levels=[-400, -200, 0, 200], colors="#6b6b66", linewidths=1)
        ax.clabel(cs, fmt="%d meV", fontsize=7)
    fig.colorbar(sc, label="dg0 = mu_H(dilute) - G_H (meV); blue = insertion favourable")
    ax.set_xlabel("P (GPa)"); ax.set_ylabel("T (K)"); ax.set_title(f"dilute H insertion into {a.host} H2O (linear model contours; solid host only)", loc="left", fontsize=9)
    os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.o}.{ext}", dpi=150, bbox_inches="tight")
    R.to_csv(f"{a.o}.csv", index=False); print(f"wrote {a.o}.pdf / .png / .csv")
