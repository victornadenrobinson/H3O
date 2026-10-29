"""T-x phase diagram of H2O + H at fixed P from our free energies (GRACE TDI), per atom vs X = n/(3+n) (atom fraction of added H; X = 0 H2O, X = 1 pure H):
  solid solution (superionic fcc H(2+n)O): g_s = (mu0 n + s n^2) / (3 + n): mu0 + s n is the AVERAGE dG per added H (insertion of n H and the direct crystals both
      measure averages; docs/data/insertion_regsol_fit.csv), so the next-H chemical potential is mu0 + 2 s n; mu0, s linear in T;
      only n <= NMAX (0.6) is used (beyond: extrapolation of the regular-solution form);
  liquid: g_l = (1 - X) dG_melt(T) + dG_mix(X, T), dG_melt = G_liq - G_fcc of H2O (docs/data/melt_dG_grace.csv, linear in T),
      dG_mix = ideal atomic O-H mixing (g_ideal) + X (1 - X) (a + b (1 - 2X)) fitted to the liquid mixing points (docs/data/liquid_mix_grace.csv) at each T, a, b linear in T between the T with data;
  all relative to H2O solid + fluid H (pure H fluid: 0). At each T the lower convex hull gives: solid solution up to the SOLVUS x_s, two phases (solid + liquid),
  liquid from x_l (LIQUIDUS composition) on; if the hull starts on the liquid, the H2O solid is gone (melted). Monte Carlo over all input errors -> 16-84 % bands.
Check printed: direct TDI G_liq - G_fcc of H2.25O vs the model's g_l - g_s at n = 0.25.
Limits: temperatures outside the data are extrapolated (marked); the liquid fit has 3-5 compositions; finite-size and model (GRACE) errors are not in the bands.
Second solid (eutectics): --compound HPERO:DG[:DS] adds a line compound H(HPERO)O with g_c(T) = DG - T DS (meV/atom, DS in meV/atom/K) on the same reference
  (H2O superionic solid + fluid H at the same P, T; NOT Huang's 0 K reference: theirs is vs H2O Pbcm + solid H, include your own entropy / reference shift).
  With it the hull can show solid + compound and compound + liquid fields; the temperatures where the hull's phase sequence changes (eutectic, peritectic,
  melting of a phase) are printed. Without a compound there is one solid only (pure H is fluid here) and no eutectic at these T.
Planets: Uranus / Neptune bands from docs/data/ice_giant_profiles.csv (approximate T(P), see its header): at each panel's P a horizontal bar at the planet's T (range = slope range)
  over X = --planet-x (default 0.1-0.4): H mass fraction w in the water-rich layer -> n = 18.015 w / (1.008 (1 - w)) added H per O, X = n / (3 + n); w = 2 % -> X 0.11, 5 % -> 0.24,
  10 % -> 0.40 (three-layer models put roughly 5-15 wt% H/He into the inner envelope; He is not in our system). --no-planets to leave them out.
usage: python3 mlp_pipeline/tdi/tx_diagram.py [--P 400 500 300] [--T 3500 8000] [--xmax 1] [--compound 3:-20:0.002] [-o docs/figures/tx_diagram]   -> <o>.png, <o>.csv
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NMAX = 0.6


def rows(f):
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]


def linfit(T, y, e):
    T, y, e = map(np.asarray, (T, y, e))
    if len(T) == 1: return np.array([0.0, y[0]]), np.array([0.0, e[0]])
    w = 1 / np.maximum(e, 1e-6) ** 2; A = np.vstack([T - 6000, np.ones_like(T)]).T * np.sqrt(w)[:, None]
    c, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None); cov = np.linalg.inv(A.T @ A); return c, np.sqrt(np.diag(cov))


def g_ideal(X, T):
    """ideal ATOMIC mixing of O and H per atom (meV) relative to the end members (H2O liquid, x_O = 1/3; pure H): kT [x_O ln x_O + x_H ln x_H] - (1 - X) kT s(1/3), x_O = (1 - X) / 3.
    The liquid is atomic (dissociated) and added H is the same species as the host H: no log term at the H2O end (the H density stays finite), a log term at the pure-H end
    (O becomes dilute). It keeps g_l convex next to pure H; the fitted excess part absorbs the rest (a plain Redlich-Kister fit made a spurious two-liquid gap at X > 0.96)"""
    X = np.clip(np.asarray(X, float), 0, 1 - 1e-12); xo = np.clip((1 - X) / 3, 1e-12, 1); xh = 1 - xo
    return 0.08617333 * T * ((xo * np.log(xo) + xh * np.log(xh)) - (1 - X) * (np.log(1 / 3) / 3 + 2 / 3 * np.log(2 / 3)))


def rk_fit(X, d, e, T=0.0):
    X, d, e = map(np.asarray, (X, d, e)); w = 1 / e ** 2; d = d - g_ideal(X, T)
    B = np.vstack([X * (1 - X), X * (1 - X) * (1 - 2 * X)]).T
    if len(X) < 3: B = B[:, :1]
    c, *_ = np.linalg.lstsq(B * np.sqrt(w)[:, None], d * np.sqrt(w), rcond=None)
    return np.r_[c, 0.0][:2]


def lower_hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0: h.pop()
        h.append(p)
    return h


class Model:
    def __init__(self, P, rng=None):
        fit = [r for r in rows("docs/data/insertion_regsol_fit.csv") if r["P_GPa"] == P]
        melt = [r for r in rows("docs/data/melt_dG_grace.csv") if r["P_GPa"] == P and r["H_per_O"] == 2]
        liq = [r for r in rows("docs/data/liquid_mix_grace.csv") if r["P_GPa"] == P]
        if not fit or not melt or not liq: raise ValueError(f"{P} GPa: missing insertion / melting / liquid data")
        z = (lambda e: rng.normal(0, e)) if rng is not None else (lambda e: 0.0)
        self.Tr = dict(solid=(min(r["T_K"] for r in fit), max(r["T_K"] for r in fit)), melt=(min(r["T_K"] for r in melt), max(r["T_K"] for r in melt)),
                       liquid=(min(r["T_K"] for r in liq), max(r["T_K"] for r in liq)))
        self.mu0 = linfit([r["T_K"] for r in fit], [r["mu0"] + z(r["mu0_err"]) for r in fit], [r["mu0_err"] for r in fit])[0]
        self.s = linfit([r["T_K"] for r in fit], [max(200.0, r["s"] + z(min(r["s_err"], r["s"] * 0.5))) for r in fit], [r["s_err"] for r in fit])[0]
        self.dm = linfit([r["T_K"] for r in melt], [r["dG_meV_atom"] + z(r["err"]) for r in melt], [r["err"] for r in melt])[0]
        self.lT, self.lab = [], []
        for T in sorted({r["T_K"] for r in liq}):
            g = [r for r in liq if r["T_K"] == T]; X = [r["n"] / (3 + r["n"]) for r in g]
            self.lT.append(T); self.lab.append(rk_fit(X, [r["dG_meV_atom"] + z(r["err"]) for r in g], [r["err"] for r in g], T))
        self.lab = np.array(self.lab)

    def gs(self, X, T):
        n = 3 * X / (1 - X); mu0 = self.mu0[0] * (T - 6000) + self.mu0[1]; s = max(self.s[0] * (T - 6000) + self.s[1], 100.0)
        return (mu0 * n + s * n ** 2) / (3 + n)

    def gl(self, X, T):
        a = np.interp(T, self.lT, self.lab[:, 0]); b = np.interp(T, self.lT, self.lab[:, 1]); dm = self.dm[0] * (T - 6000) + self.dm[1]
        return (1 - X) * dm + g_ideal(X, T) + X * (1 - X) * (a + b * (1 - 2 * X))

    def hull(self, T, comp=None):
        """lower convex hull at T: list of (X, g, phase) with phase s (solid solution), l (liquid), c (compound)"""
        Xs = np.linspace(0, NMAX / (3 + NMAX), 60); Xl = np.linspace(0, 1, 201)
        pts = list(zip(Xs, self.gs(Xs, T), ["s"] * len(Xs))) + list(zip(Xl, self.gl(Xl, T), ["l"] * len(Xl)))   # whole arrays at once: the per-point calls made the Monte Carlo slow
        if comp: pts.append((comp[0], comp[1] - T * comp[2], "c"))
        return lower_hull(pts)

    def phases(self, T, comp=None):
        """(x_s solvus or nan if no solid, x_l composition of the liquid nearest the H2O side, solid stable at X = 0)"""
        h = self.hull(T, comp)
        if h[0][2] != "s": xs = np.nan
        else: xs = max(p[0] for p in h if p[2] == "s")
        xl = min((p[0] for p in h if p[2] == "l" and not (p[0] < (xs if np.isfinite(xs) else -1))), default=1.0)
        return xs, xl, h[0][2] == "s"

    def extrap(self, T):
        return any(T < lo - 1 or T > hi + 1 for lo, hi in self.Tr.values())


def regions(h, xg, tol=0.012):
    """phase-field label at each composition of xg from a hull: single phase (s, l) where the hull runs over sampled points of one phase, else 'a+b' of the tie line"""
    xs = np.array([p[0] for p in h]); out = []
    for x in xg:
        k = min(max(np.searchsorted(xs, x, side="right") - 1, 0), len(h) - 2)
        a, b = h[k], h[k + 1]
        if a[2] == b[2] and a[2] != "c" and b[0] - a[0] <= tol: out.append(a[2])
        elif abs(x - a[0]) < 1e-9 and a[2] == "c": out.append("c")
        else: out.append("+".join(sorted({a[2], b[2]}, key="scl".index)) if a[2] != b[2] else a[2] + "+" + b[2])
    return out


def sequence(h, tol=0.012):
    """phase sequence along X of a hull, e.g. 's | s+l | l' or 's | s+c | c | c+l | l'"""
    seq = []
    for a, b in zip(h[:-1], h[1:]):
        lab = [a[2]] if a[2] == b[2] and a[2] != "c" and b[0] - a[0] <= tol else ([a[2], "+".join(sorted({a[2], b[2]}, key="scl".index)) if a[2] != b[2] else a[2] + "+" + b[2], b[2]])
        for l in lab:
            if not seq or seq[-1] != l: seq.append(l)
    return " | ".join(seq)


REG = {"s": ("#cde2fb", "superionic H(2+n)O solid solution"), "s+l": ("#f3f2ee", "solid solution + liquid"), "l": ("#fbe3ec", "liquid (H2O-H miscible)"),
       "l+l": ("#ece0f5", "two liquids (miscibility gap)"), "s+c": ("#dfeadf", "solid solution + compound"), "c+l": ("#f6ead8", "compound + liquid"), "c": ("#3f8f4f", "compound")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--P", type=float, nargs="+", default=[400, 500, 300]); ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("--T", type=float, nargs=2, default=None, metavar=("TMIN", "TMAX"), help="temperature range (default: data range - 500 K ... + 700 K)")
    ap.add_argument("--xmax", type=float, default=0.6, help="largest X shown (1 = pure H; X > 0.5 has no liquid data: extrapolated)")
    ap.add_argument("--compound", default=None, help="HPERO:DG[:DS]  line compound H(HPERO)O, g = DG - T DS meV/atom (e.g. 3:-20:0.002); same DG, DS at every P")
    ap.add_argument("--planet-x", type=float, nargs=2, default=[0.1, 0.4], help="X range of the Uranus / Neptune bars"); ap.add_argument("--no-planets", action="store_true")
    ap.add_argument("-o", default="docs/figures/tx_diagram"); a = ap.parse_args()
    planets = [] if a.no_planets else [r for r in csv.DictReader(l for l in open("docs/data/ice_giant_profiles.csv") if not l.startswith("#"))]
    comp = None
    if a.compound:
        f = [float(v) for v in a.compound.split(":")] + [0.0]; nc = f[0] - 2; comp = (nc / (3 + nc), f[1], f[2])
        print(f"compound H{f[0]:g}O at X = {comp[0]:.3f}: g_c(T) = {f[1]:g} - T x {f[2]:g} meV/atom (vs H2O superionic solid + fluid H at P, T)")
    fig, axs = plt.subplots(1, len(a.P), figsize=(6.2 * len(a.P), 5.6), squeeze=False); out = []
    rng = np.random.default_rng(1); xg = np.linspace(0, a.xmax, 241)
    for ax, P in zip(axs[0], a.P):
        try:
            m0 = Model(P)
        except ValueError as e:
            ax.set_title(f"{P:.0f} GPa: {e}", fontsize=8); continue
        lo = min(v[0] for v in m0.Tr.values()) - 500; hi = max(m0.Tr["melt"][1], m0.Tr["solid"][1]) + 700
        if a.T: lo, hi = a.T
        Ts = np.arange(lo, hi + 1, 50.0)
        mcs = [Model(P, rng) for _ in range(a.mc)]
        res = np.array([[mm.phases(T, comp)[:2] for T in Ts] for mm in mcs])   # (mc, T, 2)
        hulls = [m0.hull(T, comp) for T in Ts]; best = np.array([m0.phases(T, comp)[:2] for T in Ts])
        q = lambda k, p: np.nanpercentile(res[:, :, k], p, axis=0)
        psolid = np.mean(np.isfinite(res[:, :, 0]), axis=0)
        xs, xl = best[:, 0], best[:, 1]; ok = np.isfinite(xs)
        lab = np.array([regions(h, xg) for h in hulls]); keys = [k for k in REG if (lab == k).any()]
        for k in sorted(set(lab.ravel()) - set(REG)): REG[k] = ("#dddddd", k)
        keys += [k for k in sorted(set(lab.ravel())) if k not in keys]
        code = np.vectorize(lambda v: keys.index(v))(lab)
        from matplotlib.colors import ListedColormap
        ax.pcolormesh(xg, Ts, code, cmap=ListedColormap([REG[k][0] for k in keys]), vmin=-0.5, vmax=len(keys) - 0.5, shading="nearest")
        for k in keys:
            if k != "c": ax.fill_between([], [], color=REG[k][0], label=REG[k][1])
        ax.plot(xs, Ts, color="#1c5cab", lw=2, label="solvus (max H in the solid)"); ax.plot(np.where(ok, xl, np.nan), Ts, color="#d55181", lw=2, label="liquidus (liquid in equilibrium)")
        if comp:
            cs = np.array([any(p[2] == "c" for p in h) for h in hulls])
            ax.plot(np.where(cs, comp[0], np.nan), Ts, color=REG["c"][0], lw=3, label="compound (stable)")
            xr = np.array([min((p[0] for p in h if p[2] == "l" and p[0] > comp[0]), default=np.nan) if c else np.nan for h, c in zip(hulls, cs)])
            ax.plot(xr, Ts, color="#d55181", lw=2, ls="--", label="liquidus on the H side of the compound")
        with np.errstate(all="ignore"):
            ax.fill_betweenx(Ts, q(0, 16), q(0, 84), color="#1c5cab", alpha=0.18, lw=0); ax.fill_betweenx(Ts, q(1, 16), q(1, 84), color="#d55181", alpha=0.18, lw=0)
        Tm = Ts[np.argmax(~ok)] if (~ok).any() else np.nan
        for r in planets:
            Ta = float(r["T_anchor_K"]); Pa = float(r["P_anchor_GPa"]); t = [Ta * (P / Pa) ** float(r[k]) for k in ("a_min", "a_max")]; T1, T2 = min(t), max(t)
            cold = "cold" in r["profile"]; c = "#2a9d8f" if r["planet"] == "Uranus" else "#3a4fb8"
            ax.add_patch(plt.Rectangle((a.planet_x[0], T1 - 30), a.planet_x[1] - a.planet_x[0], T2 - T1 + 60, fc=c, alpha=0.25 if cold else 0.5, ec=c, lw=1.2, ls="--" if cold else "-",
                                       label=f"{r['planet']}, {r['profile']} (approx.)"))
            ax.text(a.planet_x[1] + 0.005, (T1 + T2) / 2, f"{r['planet'][0]} {'cold' if cold else 'R11'}", fontsize=7, color=c, va="center")
            print(f"   {r['planet']:8s} {r['profile']:22s} at {P:.0f} GPa: T ~ {T1:.0f}-{T2:.0f} K (scaled from {Ta:.0f} K at {Pa:.0f} GPa)")
        ax.axhline(Tm, color="k", lw=0.8, ls=":"); ax.text(0.02 * a.xmax, Tm + 40, f"pure H2O melts ~{Tm:.0f} K (model)", fontsize=7.5)
        for T in Ts:
            if m0.extrap(T): ax.axhspan(T - 25, T + 25, color="w", alpha=0.35, lw=0)
        if a.xmax > 0.5: ax.axvspan(0.5, a.xmax, color="w", alpha=0.25, lw=0); ax.text(0.51, Ts[1], "no liquid data\n(X > 0.5)", fontsize=7)
        sec = ax.secondary_xaxis("top", functions=(lambda x: 2 + 3 * x / np.maximum(1 - x, 1e-9), lambda h: (h - 2) / (h + 1)))
        sec.set_xticks([t for t in [2, 2.25, 2.5, 3, 4, 6, 10, 20] if (t - 2) / (t + 1) <= a.xmax]); sec.set_xlabel("H per O")
        ax.set(xlim=(0, a.xmax), ylim=(Ts[0], Ts[-1]), xlabel="X = added-H atom fraction n/(3+n)   (0: H2O, 1: pure H)", ylabel="T (K)", title=f"{P:.0f} GPa")
        ax.legend(fontsize=6.5, loc="upper right"); ax.grid(alpha=0.25)
        print(f"\n{P:.0f} GPa (data T: solid {m0.Tr['solid']}, melting {m0.Tr['melt']}, liquid {m0.Tr['liquid']}; outside = extrapolated, light rows)")
        print(f"   pure H2O melting (model) ~{Tm:.0f} K;  P(solid exists) from MC at that T: {psolid[np.argmin(abs(Ts - Tm))]:.2f}")
        seqs = [sequence(h) for h in hulls]
        print("   phase sequence along X (H2O -> pure H) and where it changes (eutectic / peritectic / melting temperatures):")
        print(f"      {Ts[0]:6.0f} K: {seqs[0]}")
        for T0, T1, s0, s1 in zip(Ts[:-1], Ts[1:], seqs[:-1], seqs[1:]):
            if s0 != s1: print(f"      {(T0 + T1) / 2:6.0f} K (+-25): {s0}  ->  {s1}")
        for T, h, (s_, l_), lo16, hi84 in zip(Ts, hulls, best, q(0, 16), q(0, 84)):
            if T % 500 == 0:
                pa = next((p for p in h if p[0] > s_ + 1e-9), None) if np.isfinite(s_) else None
                with_ = "" if pa is None else (f", coexisting with the compound (X {pa[0]:.3f})" if pa[2] == "c" else "")
                print(f"   {T:6.0f} K: " + (f"solvus H{2 + 3 * s_ / (1 - s_):.2f}O (X {s_:.3f}, 16-84 % {lo16:.3f}-{hi84:.3f}){with_}, liquidus X {l_:.2f} (H{2 + 3 * l_ / (1 - l_):.1f}O)" if np.isfinite(s_) else "no solid solution")
                      + ("   [extrapolated]" if m0.extrap(T) else ""))
                out.append(dict(P_GPa=P, T_K=T, solvus_X=s_, solvus_X16=lo16, solvus_X84=hi84, liquidus_X=l_, extrapolated=int(m0.extrap(T)), sequence=seqs[list(Ts).index(T)]))
        for r in [r for r in rows("docs/data/melt_dG_grace.csv") if r["P_GPa"] == P and r["H_per_O"] == 2.25]:
            X = 0.25 / 3.25; print(f"   check H2.25O at {r['T_K']:.0f} K: direct TDI G_liq - G_fcc {r['dG_meV_atom']:6.1f} +- {r['err']:.1f}, model {m0.gl(X, r['T_K']) - m0.gs(X, r['T_K']):6.1f} meV/atom")
    fig.suptitle("T-x diagram of H2O + H from GRACE free energies (classical nuclei); bands = 16-84 % (statistical only); pale rows = extrapolated in T"
                 + (f"; compound {a.compound} is an ASSUMED input" if comp else ""), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(a.o + ".png", dpi=130)
    if out:
        with open(a.o + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nwrote", a.o + ".png", a.o + ".csv")


if __name__ == "__main__":
    main()
