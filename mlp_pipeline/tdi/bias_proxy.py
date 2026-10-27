"""Cheap model-bias proxy for mu_H - G_H: how the MLP energy error depends on composition, from the DFT frames already benchmarked by 04_benchmark.py.
usage (repo root):  python3 mlp_pipeline/tdi/bias_proxy.py runs/bench_<model>_valid_m.csv [more bench CSVs, e.g. the train set] [--pmin 300 --pmax 600 --tmin 4000 --tmax 7000]
                    [--xmax 3] [--by T] [-o runs/checks/bias_proxy]
Idea. To first order (free-energy perturbation), the model error in a free energy is the mean energy error <E_MLP - E_DFT>. Per frame, Delta = n * e_signed
(04_benchmark's e_signed: model - DFT per atom after a per-element shift). Write the error per O as f(x) = Delta / N_O with x = H per O. Then the error of the
insertion chemical potential at fixed N_O is d Delta / d N_H = f'(x), and the error of the pure-H reference is delta_H = Delta / N_H of pure-H frames, so
        error(mu_H - G_H) ~ f'(x) - delta_H.
The per-element shift (or any constant per-element offset of the model) drops out exactly: it adds s_H to both terms. Same rule of thumb as CURRENT_ISSUES:
per-atom bias delta(x) gives f'(x) = delta + (1 + x) delta'(x), e.g. 4 meV/atom between H2O and H2.25O -> ~13 meV/atom x 3.25 / 0.25 ~ 50 meV per added H.
Output: per-composition mean signed error (meV/atom and meV/O, SEM over frames), finite differences between neighbouring compositions, a weighted linear fit
of f(x) over 2 <= x <= --xmax (slope = f'), and the implied error of mu_H - G_H. Caveats: frames from one AIMD run are correlated (the SEM is the larger of
the frame SEM and the SEM over run means when the CSV has a source column with >= 3 runs per composition); DFT frames sample the DFT ensemble, not the MLP one (first-order proxy only); compositions
are the AIMD ones (H2O, H2.25O, H2.5O, H3O, ...), so f' at x -> 2 is a slope over 0.25-1 H per O, not the dilute derivative."""
import argparse, os, re
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("csv", nargs="+", help="bench_*.csv from 04_benchmark.py (needs e_signed)")
ap.add_argument("--pmin", type=float, default=None); ap.add_argument("--pmax", type=float, default=None)
ap.add_argument("--tmin", type=float, default=None); ap.add_argument("--tmax", type=float, default=None)
ap.add_argument("--phase", default=None, help="keep frames whose phase_tag contains this (pure-H frames are always kept)")
ap.add_argument("--xmax", type=float, default=3.0, help="largest H per O in the linear fit of f(x) (default 3: H2O..H3O)")
ap.add_argument("--by", default=None, choices=["T", "source_csv"], help="repeat the analysis per T or per input file")
ap.add_argument("-o", default=None, help="output prefix: writes <o>_by_comp.csv")
a = ap.parse_args()


def h_per_o(label):
    c = str(label).split("_")[0].lower()
    if c == "h":
        return np.inf
    m = re.fullmatch(r"h([0-9.]+)o", c)
    return float(m.group(1)) if m else np.nan


dfs = []
for f in a.csv:
    d = pd.read_csv(f)
    if "e_signed" not in d:
        raise SystemExit(f"{f}: no e_signed column (04_benchmark.py older than 2026-10-26: rerun it)")
    d["source_csv"] = os.path.basename(f); dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df["x"] = df.comp.map(h_per_o)
bad = df.x.isna()
if bad.any():
    print(f"skipped {bad.sum()} frames with unknown composition: {sorted(df.comp[bad].astype(str).unique())[:8]}")
df = df[~bad].copy()
pure = np.isinf(df.x)
df["NO"] = np.where(pure, 0.0, df.n / (1 + df.x.where(~pure, 0)))   # from the label; the atom counts (04_benchmark from 2026-10-27) win
if "N_O" in df:
    df["NO"] = df.N_O.astype(float).fillna(df.NO)
df["NH"] = df.n - df.NO
if not np.allclose(df.NO[~pure], df.NO[~pure].round(), atol=0.01):
    print("WARNING: n / (1 + x) is not an integer for some frames: composition label and atom count disagree")
for col, lo, hi in (("P_dft", a.pmin, a.pmax), ("T", a.tmin, a.tmax)):
    if lo is not None:
        df = df[df[col] >= lo]
    if hi is not None:
        df = df[df[col] <= hi]
if a.phase and "phase_tag" in df:
    df = df[np.isinf(df.x) | df.phase_tag.astype(str).str.contains(a.phase)]
df["D"] = df.n * df.e_signed * 1e3                     # meV per frame (model - DFT, shifted)
df["f"] = np.where(np.isinf(df.x), np.nan, df.D / df.NO.replace(0, np.nan))   # meV per O
df["dH"] = np.where(np.isinf(df.x), df.D / df.NH, np.nan)                     # meV per H atom, pure H only


def sem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def analyse(d, title):
    print(f"\n=== {title}: {len(d)} frames, P {d.P_dft.min():.0f}-{d.P_dft.max():.0f} GPa, T {d['T'].min():.0f}-{d['T'].max():.0f} K")
    rows = []
    for x, g in sorted(d.groupby("x"), key=lambda t: t[0]):
        nrun = g.source.nunique() if "source" in g and g.source.notna().any() else np.nan
        # SEM over AIMD runs (frames of one run are correlated); used instead of the frame SEM when larger and >= 3 runs
        fr = g.groupby("source").f.mean() if nrun == nrun and nrun >= 3 and np.isfinite(x) else None
        er = g.groupby("source").e_signed.mean() if nrun == nrun and nrun >= 3 else None
        rows.append(dict(x=x, comp=g.comp.astype(str).str.split("_").str[0].iloc[0], frames=len(g), runs=nrun,
                         e_meV_atom=1e3 * g.e_signed.mean(), e_sem=1e3 * np.nanmax([sem(g.e_signed), sem(er) if er is not None else np.nan]),
                         f_meV_O=g.f.mean() if np.isfinite(x) else np.nan,
                         f_sem=np.nanmax([sem(g.f), sem(fr) if fr is not None else np.nan]) if np.isfinite(x) else np.nan,
                         P=g.P_dft.mean(), T=g["T"].mean()))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    h = t[np.isinf(t.x)]
    dH, dH_e = (h.e_meV_atom.iloc[0], h.e_sem.iloc[0]) if len(h) else (np.nan, np.nan)
    if not len(h):
        print("no pure-H frames: only f'(x) is available (the pure-H reference error is unknown)")
    s = t[np.isfinite(t.x) & (t.x >= 2.0 - 1e-9)].reset_index(drop=True)
    print("\nfinite differences f'(x) = d(error)/d(N_H) at fixed N_O, and f' - delta_H = error of mu_H - G_H (meV per added H):")
    for i in range(len(s) - 1):
        dx = s.x[i + 1] - s.x[i]; fp = (s.f_meV_O[i + 1] - s.f_meV_O[i]) / dx
        fe = np.hypot(s.f_sem[i + 1], s.f_sem[i]) / dx
        print(f"  {s.comp[i]:>7} -> {s.comp[i + 1]:<7} x {s.x[i]:.3g}-{s.x[i + 1]:.3g}:  f' = {fp:8.1f} +- {fe:5.1f}   f' - delta_H = {fp - dH:8.1f} +- {np.hypot(fe, dH_e):5.1f}")
    fit = s[(s.x <= a.xmax + 1e-9) & s.f_sem.notna() & (s.f_sem > 0)]
    if len(fit) >= 2:
        w = 1 / fit.f_sem ** 2; X = np.vstack([np.ones(len(fit)), fit.x - 2.0]).T
        cov = np.linalg.inv(X.T @ (X * w.values[:, None])); c = cov @ X.T @ (w * fit.f_meV_O)
        chi2 = float(np.sum(w * (fit.f_meV_O - X @ c) ** 2)); dof = len(fit) - 2
        scale = np.sqrt(chi2 / dof) if dof > 0 and chi2 > dof else 1.0            # inflate by sqrt(chi2/dof) when the line does not fit
        b, be = c[1], np.sqrt(cov[1, 1]) * scale
        print(f"linear fit of f(x) over x = {', '.join(f'{v:g}' for v in fit.x)}: f' = {b:.1f} +- {be:.1f} meV per added H (chi2/dof {chi2:.1f}/{dof})")
        print(f"==> implied error of mu_H - G_H (model - DFT): {b - dH:.0f} +- {np.hypot(be, dH_e):.0f} meV per H"
              f"   [pure-H delta_H = {dH:.1f} +- {dH_e:.1f} meV/atom]")
        print("    sign: positive = the model puts mu_H in ice too high relative to pure H (true dilute mu_H - G_H more negative than GRACE's)")
    return t


res = [analyse(df, "all frames")]
if a.by:
    for k, g in df.groupby(a.by):
        if g.x.nunique() >= 2:
            res.append(analyse(g, f"{a.by} = {k}").assign(**{a.by: k}))
if a.o:
    os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True)
    pd.concat(res, ignore_index=True).to_csv(a.o + "_by_comp.csv", index=False); print("wrote", a.o + "_by_comp.csv")
