"""Solid 'convex hull' of H in superionic H2O: formation free energy per atom of H2O(crystal) + n H(pure fluid) -> H(2+a)O(crystal), from the insertion runs and the
direct crystal runs, with the solubility limit from the tangent to pure H.
usage (repo root):  python3 mlp_pipeline/tdi/summarize.py -o runs/checks/summary.csv
                    python3 mlp_pipeline/tdi/solid_hull.py runs/checks/summary.csv [-o runs/checks/solid_hull] [--host fcc] [--xmax 0.1]
                    python3 mlp_pipeline/tdi/solid_hull.py --fits runs/checks/insertion_model.csv ...   (curves only, from insertion_model.py's table)
Axes. End members H2O (crystal, 3 atoms) and H (pure fluid, 1 atom). x = n / (3 nO + n) = fraction of the atoms that belong to the H end member (a = n / nO added H per O,
x = a / (3 + a)); dG(x) per atom = [G(cell) - 3 nO G(H2O)/atom - n G_H] / (3 nO + n). In these coordinates the lever rule holds, the slope at x -> 0 is the dilute
mu_H - G_H, and the tangent from pure H (x = 1, dG = 0) touches the curve where the differential mu_H = G_H: the solubility limit.
Insertion. Each run gives the average over its n added H, avg(n) = [G(N + n) - G(N)] / n - G_H (summary rows, P-corrected where stage 0 has it). Weighted line
avg(n) = dg0 + s n (same fit as insertion_model.py) -> n avg(n) = n dg0 + s n^2 -> dG(x) with n = 3 nO x / (1 - x); band = 16-84% of the fit covariance.
The differential mu_H - G_H = dg0 + 2 s n, so a_sol = -dg0 / (2 s nO) (same as insertion_model's x_sol). The line is regular-solution-like: an extrapolation beyond n = 4.
Crystal points: H(2+a)O crystal vs H2O crystal (same phase, cell, P, T) + n pure H, as in summarize.py; hollow = caveat (different dt / lambda grid, or the crystal melted
in free NVT). Insertion points: hollow if the run has no stage-0 pressure correction (first-order bias, typically -26 to -80 meV per added H after correction)."""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C_INS, C_FCC, C_BCC, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#222222", "#8a8a85"


def pick(g, dt, nodes=None):
    """same rule as summarize.pick: same dt if there is one (else the smallest), then the same lambda grid if there is one, then the longest run"""
    same = g[g.dt_fs == dt]; g = same if len(same) else g[g.dt_fs == g.dt_fs.min()]
    if nodes is not None and (g.nodes == nodes).any():
        g = g[g.nodes == nodes]
    return g.sort_values("ps_per_node").iloc[-1]


def nvt_melted():
    """(phase, H per O, P, T) of crystals that melted in a free-NVT run (same sources and verdict rule as summarize.py); empty if the runs are not here"""
    try:
        from tdi.verdict import verdict
    except Exception:
        return set()
    out = set()
    for d in ("runs/melt_tdi", "runs/melt", "runs/melt_comp", "runs/eos_scan"):
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            try: t = pd.read_csv(f)
            except Exception: continue
            for _, r in t.iterrows():
                if r.get("branch", "solid") != "solid" or r.get("phase") not in ("fcc", "bcc") or "T_set" not in r or "P_GPa" not in r: continue
                if verdict(r) == "melted":
                    pt = r.get("P_target", np.nan); P = float(pt) if np.isfinite(pt) else float(r["P_GPa"])
                    h = float(r["hper"]) if np.isfinite(r.get("hper", np.nan)) else 2.0
                    out.add((r["phase"], round(h, 2), round(P / 25) * 25, float(r["T_set"])))
    return out


def states_from_summary(path, host):
    df = pd.read_csv(path); melted = nvt_melted(); S = {}
    H = df[df.phase == "H"]
    ins = df[(df.phase == "insert") & (~df.tag.str.contains("_Hhost"))]
    ins = ins[ins.tag.str.contains("_bcc") == (host == "bcc")]
    for (P, T), g in ins.groupby(["P_target_GPa", "T_K"]):
        hh = H[(H.P_target_GPa == P) & (H.T_K == T)]
        if not len(hh):
            print(f"{P:.0f} GPa {T:.0f} K: insertion runs but no pure-H run: skipped"); continue
        g = g[g.dt_fs == g.dt_fs.min()]; h = pick(hh, g.dt_fs.iloc[0])
        pts = [dict(n=int(r.N), nO=int(r.nO), avg=r.G_meV_atom - h.G_meV_atom, err=float(np.hypot(r.G_err_meV_atom, h.G_err_meV_atom)),
                    pcorr=bool(np.isfinite(getattr(r, "P_corr_meV", np.nan))), caveat=r.dt_fs != h.dt_fs) for r in g.itertuples()]
        S[(P, T)] = dict(P=P, T=T, nO=pts[0]["nO"], ins=pts, cryst=[])
    for _, m in df[(df.nO > 0) & df.phase.isin(["fcc", "bcc"]) & (df.H_per_O > 2.0)].iterrows():
        w = df[(df.phase == m.phase) & (df.P_target_GPa == m.P_target_GPa) & (df.T_K == m.T_K) & (df.H_per_O == 2.0) & (df.nO == m.nO)]
        hh = H[(H.P_target_GPa == m.P_target_GPa) & (H.T_K == m.T_K)]
        if not (len(w) and len(hh)):
            continue
        w, h = pick(w, m.dt_fs, m.nodes), pick(hh, m.dt_fs, m.nodes); n = int(m.nH - 2 * m.nO)
        dG = (m.N * m.G_meV_atom - w.N * w.G_meV_atom - n * h.G_meV_atom) / n
        err = ((m.N * m.G_err_meV_atom) ** 2 + (w.N * w.G_err_meV_atom) ** 2 + (n * h.G_err_meV_atom) ** 2) ** .5 / n
        mel = any((m.phase, round(hp, 2), round(m.P_target_GPa / 25) * 25, float(m.T_K)) in melted for hp in (m.H_per_O, 2.0))
        cav = (not (m.dt_fs == w.dt_fs == h.dt_fs and m.nodes == w.nodes == h.nodes)) or mel
        st = S.setdefault((m.P_target_GPa, m.T_K), dict(P=m.P_target_GPa, T=m.T_K, nO=int(m.nO), ins=[], cryst=[]))
        st["cryst"].append(dict(phase=m.phase, n=n, nO=int(m.nO), avg=dG, err=err, caveat=cav, why=("MELTED in free NVT" if mel else "") + ("" if m.dt_fs == w.dt_fs == h.dt_fs and m.nodes == w.nodes == h.nodes else " dt/grid differ")))
    for st in S.values():
        add_fits(st, host)
    return list(S.values())


def wfit(n, y, e):
    n, y, e = (np.asarray(v, float) for v in (n, y, e))
    w = 1 / e ** 2; X = np.c_[np.ones_like(n), n]; cov = np.linalg.inv(X.T @ (X * w[:, None])); b = cov @ X.T @ (w * y)
    chi = float(np.sum(w * (y - X @ b) ** 2)); dof = len(n) - 2
    if dof > 0 and chi > dof:
        cov = cov * chi / dof   # inflate when the line does not fit the points
    return b, cov


def add_fits(st, host):
    """'ins': line through the insertion averages (n <= 4: the dilute end, extrapolated); 'all': the same line also through the direct crystal points of this host
    without caveat (the only constraint at larger n)"""
    p = st["ins"]; c = [q for q in st["cryst"] if q["phase"] == host and not q["caveat"]]
    st["fits"] = {}
    if len({q["n"] for q in p}) >= 2:
        st["fits"]["ins"] = wfit([q["n"] for q in p], [q["avg"] for q in p], [q["err"] for q in p])
    if c and len({q["n"] for q in p + c}) >= 2:
        st["fits"]["all"] = wfit([q["n"] for q in p + c], [q["avg"] for q in p + c], [q["err"] for q in p + c])


def states_from_fits(path):
    R = pd.read_csv(path); out = []
    for r in R.itertuples():
        if np.isfinite(r.slope):
            out.append(dict(P=r.P, T=r.T, nO=int(r.nO), ins=[], cryst=[], fits={"ins": (np.array([r.dg0, r.slope]), np.diag([r.dg0_err ** 2, r.slope_err ** 2]))}))
    return out


def curve(b, nO, x):
    """dG formation per atom (meV) at atom fractions x for avg(n) = b0 + b1 n"""
    n = 3 * nO * x / (1 - x)
    return n * (b[0] + b[1] * n) / (3 * nO + n)


def analyse(b, cov, nO, xg, nsamp=4000, seed=0):
    """16-84% band of the curve, and the tangent point to pure H (x = 1, dG = 0): central value and 16-84% range, as added H per O"""
    cg = curve(b, nO, xg); i_t = int(np.argmin(cg / (1 - xg)))
    rng = np.random.default_rng(seed); B = rng.multivariate_normal(b, cov, nsamp)
    n = 3 * nO * xg / (1 - xg); Y = n[None, :] * (B[:, :1] + B[:, 1:] * n[None, :]) / (3 * nO + n[None, :])
    xs = xg[np.argmin(Y / (1 - xg[None, :]), axis=1)]
    return dict(lo=np.percentile(Y, 16, 0), hi=np.percentile(Y, 84, 0), x_t=xg[i_t], g_t=cg[i_t], bounded=i_t < len(xg) - 1,
                a_sol=3 * xg[i_t] / (1 - xg[i_t]), a_lo=np.percentile(3 * xs / (1 - xs), 16), a_hi=np.percentile(3 * xs / (1 - xs), 84), dGmin=float(cg.min()))


def point_xy(q):
    N = 3 * q["nO"] + q["n"]
    return q["n"] / N, q["n"] * q["avg"] / N, q["n"] * q["err"] / N


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("summary", nargs="?", default=None); ap.add_argument("--fits", default=None, help="insertion_model.py CSV instead of summary.csv (curves only)")
    ap.add_argument("-o", default="runs/checks/solid_hull"); ap.add_argument("--host", default="fcc", choices=["fcc", "bcc"]); ap.add_argument("--xmax", type=float, default=0.1)
    a = ap.parse_args()
    if not (a.summary or a.fits):
        ap.error("give summary.csv or --fits")
    states = states_from_fits(a.fits) if a.fits else states_from_summary(a.summary, a.host)
    plot(states, a.o, a.xmax, a.host)


STYLE = {"ins": dict(color=C_INS, lw=2, ls="-", label="fit to insertion n = 1, 2, 4 (extrapolated)"),
         "all": dict(color=INK, lw=1.4, ls="-.", label="fit to insertion + direct crystal")}


def plot(states, outp, xmax=0.1, host="fcc", note=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    states = sorted(states, key=lambda s: (s["P"], s["T"]))
    if not states:
        raise SystemExit("no state point with insertion or crystal data")
    nc = min(4, len(states)); nr = -(-len(states) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.9 * nc, 3.4 * nr + 0.6), squeeze=False)
    xg = np.linspace(1e-5, 0.6, 6000); xs = np.linspace(0, xmax, 400); rows = []
    for ax, st in zip(axs.ravel(), states):
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1); row = dict(P=st["P"], T=st["T"]); ys = [0.0]; tt = []
        for key in ("ins", "all"):
            if key not in st.get("fits", {}):
                continue
            b, cov = st["fits"][key]; A = analyse(b, cov, st["nO"], xg); m = xg <= xmax
            if key == "ins":
                ax.fill_between(xg[m], A["lo"][m], A["hi"][m], color=C_INS, alpha=0.15, lw=0, zorder=1)
            ax.plot(xs, curve(b, st["nO"], xs), zorder=2, **STYLE[key])
            if A["bounded"] and A["x_t"] <= xmax:
                ax.plot([A["x_t"], xmax], [A["g_t"], A["g_t"] * (1 - xmax) / (1 - A["x_t"])], color=STYLE[key]["color"], lw=0.8, ls=":", zorder=2)
                ax.plot([A["x_t"]], [A["g_t"]], "o", ms=5, mfc="white", mec=STYLE[key]["color"], zorder=4)
            ys.append(A["dGmin"])
            tag = "ins" if key == "ins" else "ins+cryst"
            tt.append(f"{tag} {A['a_sol']:.2f} ({A['a_lo']:.2f}-{A['a_hi']:.2f})" if A["bounded"] else f"{tag} > {3 * xg[-1] / (1 - xg[-1]):.1f}")
            row.update({f"{key}_dg0": b[0], f"{key}_dg0_err": cov[0, 0] ** .5, f"{key}_slope": b[1], f"{key}_slope_err": cov[1, 1] ** .5,
                        f"{key}_a_sol": A["a_sol"] if A["bounded"] else np.nan, f"{key}_a_sol_lo": A["a_lo"], f"{key}_a_sol_hi": A["a_hi"], f"{key}_dG_min_meV_atom": A["dGmin"]})
        for q in st["ins"]:
            x, y, e = point_xy(q); hol = not q["pcorr"]; ys += [y - e, y + e]
            ax.errorbar([x], [y], [e], fmt="s", ms=6, color=C_INS, mfc="white" if hol else C_INS, mec=C_INS, capsize=2, lw=1, zorder=3)
        for q in st["cryst"]:
            x, y, e = point_xy(q); c, mk = (C_FCC, "o") if q["phase"] == "fcc" else (C_BCC, "^"); ys += [y - e, y + e]
            ax.errorbar([x], [y], [e], fmt=mk, ms=7, color=c, mfc="white" if q["caveat"] else c, mec=c, capsize=2, lw=1, zorder=3)
            row[f"{q['phase']}_x"] = x; row[f"{q['phase']}_dG_meV_atom"] = y; row[f"{q['phase']}_perH_meV"] = q["avg"]; row[f"{q['phase']}_note"] = q.get("why", "")
        lo, hi = min(ys), max(ys); pad = 0.25 * max(hi - lo, 4.0)
        ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, xmax)
        ax.set_title(f"{st['P']:.0f} GPa, {st['T']:.0f} K", fontsize=9, color=INK, loc="left")
        if tt:
            ax.text(0.98, 0.04, "a_sol (H/O): " + "\n".join(tt), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK)
        ax.grid(alpha=0.2, lw=0.5)
        sec = ax.secondary_xaxis("top", functions=(lambda x: 3 * x / (1 - x), lambda a: a / (3 + a)))
        sec.set_xlabel("added H per O", fontsize=7, color=MUTED); sec.tick_params(labelsize=6.5, colors=MUTED)
        ax.set_xlabel("x = n / (3 nO + n)", fontsize=7); ax.set_ylabel("formation dG (meV/atom)", fontsize=7); ax.tick_params(labelsize=7)
        rows.append(row)
    for ax in axs.ravel()[len(states):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Line2D([], [], **STYLE["ins"]), Patch(color=C_INS, alpha=0.15, label="16-84% of the insertion fit"), Line2D([], [], **STYLE["all"]),
          Line2D([], [], marker="s", ls="", color=C_INS, mfc="white", label="insertion run (hollow: no P correction)"),
          Line2D([], [], marker="o", ls="", color=C_FCC, label="fcc crystal, direct (hollow: caveat, not fitted)"), Line2D([], [], marker="^", ls="", color=C_BCC, label="bcc crystal, direct"),
          Line2D([], [], marker="o", ls=":", color=INK, mfc="white", lw=0.8, label="tangent to pure H (x = 1): solubility limit")]
    fig.legend(handles=hd, loc="lower center", ncol=4 if nc >= 3 else 2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Solid hull: H2O({host}) + n H(fluid) -> H(2+a)O({host}), formation free energy per atom (slope at x -> 0 = dilute mu_H - G_H)" + (f"\n{note}" if note else ""),
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, (0.09 if nc >= 3 else 0.2) / nr ** 0.5, 1, 0.95))
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    fig.savefig(outp + ".png", dpi=150); fig.savefig(outp + ".pdf")
    R = pd.DataFrame(rows); R.to_csv(outp + ".csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False)); print("wrote", outp + ".png / .pdf / .csv")
    return R


if __name__ == "__main__":
    main()
