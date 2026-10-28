"""T-x phase diagram of H2O + H at fixed P from our free energies (GRACE TDI), per atom vs X = n/(3+n) (atom fraction of added H; X = 0 H2O, X = 1 pure H):
  solid solution (superionic fcc H(2+n)O): g_s = (mu0 n + s n^2) / (3 + n): mu0 + s n is the AVERAGE dG per added H (insertion of n H and the direct crystals both
      measure averages; docs/data/insertion_regsol_fit.csv), so the next-H chemical potential is mu0 + 2 s n; mu0, s linear in T;
      only n <= NMAX (0.6) is used (beyond: extrapolation of the regular-solution form);
  liquid: g_l = (1 - X) dG_melt(T) + dG_mix(X, T), dG_melt = G_liq - G_fcc of H2O (docs/data/melt_dG_grace.csv, linear in T),
      dG_mix = X (1 - X) (a + b (1 - 2X)) fitted to the liquid mixing points (docs/data/liquid_mix_grace.csv) at each T, a, b linear in T between the T with data;
  all relative to H2O solid + fluid H (pure H fluid: 0). At each T the lower convex hull gives: solid solution up to the SOLVUS x_s, two phases (solid + liquid),
  liquid from x_l (LIQUIDUS composition) on; if the hull starts on the liquid, the H2O solid is gone (melted). Monte Carlo over all input errors -> 16-84 % bands.
Check printed: direct TDI G_liq - G_fcc of H2.25O vs the model's g_l - g_s at n = 0.25.
Limits: temperatures outside the data are extrapolated (marked); the liquid fit has 3-5 compositions; finite-size and model (GRACE) errors are not in the bands.
usage: python3 mlp_pipeline/tdi/tx_diagram.py [--P 400 500 300] [-o docs/figures/tx_diagram]   -> <o>.png, <o>.csv
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


def rk_fit(X, d, e):
    X, d, e = map(np.asarray, (X, d, e)); w = 1 / e ** 2
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
            self.lT.append(T); self.lab.append(rk_fit(X, [r["dG_meV_atom"] + z(r["err"]) for r in g], [r["err"] for r in g]))
        self.lab = np.array(self.lab)

    def gs(self, X, T):
        n = 3 * X / (1 - X); mu0 = self.mu0[0] * (T - 6000) + self.mu0[1]; s = max(self.s[0] * (T - 6000) + self.s[1], 100.0)
        return (mu0 * n + s * n ** 2) / (3 + n)

    def gl(self, X, T):
        a = np.interp(T, self.lT, self.lab[:, 0]); b = np.interp(T, self.lT, self.lab[:, 1]); dm = self.dm[0] * (T - 6000) + self.dm[1]
        return (1 - X) * dm + X * (1 - X) * (a + b * (1 - 2 * X))

    def phases(self, T):
        """(x_s solvus or nan if no solid, x_l liquidus composition, solid stable at X = 0)"""
        Xs = np.linspace(0, NMAX / (3 + NMAX), 60); Xl = np.linspace(0, 1, 201)
        pts = [(x, self.gs(x, T), "s") for x in Xs] + [(x, self.gl(x, T), "l") for x in Xl]
        h = lower_hull([(p[0], p[1], p[2]) for p in pts])
        if h[0][2] == "l": return np.nan, 0.0, False
        last_s = max(i for i, p in enumerate(h) if p[2] == "s")
        xs = h[last_s][0]; xl = h[last_s + 1][0] if last_s + 1 < len(h) else 1.0
        return xs, xl, True

    def extrap(self, T):
        return any(T < lo - 1 or T > hi + 1 for lo, hi in self.Tr.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--P", type=float, nargs="+", default=[400, 500, 300]); ap.add_argument("--mc", type=int, default=300)
    ap.add_argument("-o", default="docs/figures/tx_diagram"); a = ap.parse_args()
    fig, axs = plt.subplots(1, len(a.P), figsize=(6.2 * len(a.P), 5.4), squeeze=False); out = []
    rng = np.random.default_rng(1)
    for ax, P in zip(axs[0], a.P):
        try:
            m0 = Model(P)
        except ValueError as e:
            ax.set_title(f"{P:.0f} GPa: {e}", fontsize=8); continue
        lo = min(v[0] for v in m0.Tr.values()) - 500; hi = max(m0.Tr["melt"][1], m0.Tr["solid"][1]) + 700
        Ts = np.arange(lo, hi + 1, 50.0)
        mcs = [Model(P, rng) for _ in range(a.mc)]
        res = np.array([[mm.phases(T)[:2] for T in Ts] for mm in mcs])   # (mc, T, 2)
        best = np.array([m0.phases(T)[:2] for T in Ts])
        q = lambda k, p: np.nanpercentile(res[:, :, k], p, axis=0)
        psolid = np.mean(np.isfinite(res[:, :, 0]), axis=0)
        xs, xl = best[:, 0], best[:, 1]; ok = np.isfinite(xs)
        ax.fill_betweenx(Ts[ok], 0, xs[ok], color="#cde2fb", label="superionic H(2+n)O solid solution")
        ax.fill_betweenx(Ts[ok], xs[ok], xl[ok], color="#f3f2ee", label="solid solution + H-rich liquid")
        ax.fill_betweenx(Ts, np.where(ok, xl, 0), 1, color="#fbe3ec", label="liquid (H2O-H miscible)")
        ax.plot(xs, Ts, color="#1c5cab", lw=2, label="solvus (max H in the solid)"); ax.plot(np.where(ok, xl, np.nan), Ts, color="#d55181", lw=2, label="liquidus (liquid in equilibrium)")
        with np.errstate(all="ignore"):
            ax.fill_betweenx(Ts, q(0, 16), q(0, 84), color="#1c5cab", alpha=0.18, lw=0); ax.fill_betweenx(Ts, q(1, 16), q(1, 84), color="#d55181", alpha=0.18, lw=0)
        Tm = Ts[np.argmax(~ok)] if (~ok).any() else np.nan
        ax.axhline(Tm, color="k", lw=0.8, ls=":"); ax.text(0.55, Tm + 40, f"pure H2O melts ~{Tm:.0f} K (model)", fontsize=7.5)
        for T in Ts:
            if m0.extrap(T): ax.axhspan(T - 25, T + 25, color="w", alpha=0.35, lw=0)
        sec = ax.secondary_xaxis("top", functions=(lambda x: 2 + 3 * x / np.maximum(1 - x, 1e-9), lambda h: (h - 2) / (h + 1)))
        sec.set_xticks([2, 2.25, 2.5, 3, 4, 6]); sec.set_xlabel("H per O")
        ax.set(xscale="linear", xlim=(0, 0.6), ylim=(Ts[0], Ts[-1]), xlabel="X = added-H atom fraction n/(3+n)   (0: H2O, 1: pure H)", ylabel="T (K)", title=f"{P:.0f} GPa")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
        print(f"\n{P:.0f} GPa (data T: solid {m0.Tr['solid']}, melting {m0.Tr['melt']}, liquid {m0.Tr['liquid']}; outside = extrapolated, light rows)")
        print(f"   pure H2O melting (model) ~{Tm:.0f} K;  P(solid exists) from MC at that T: {psolid[np.argmin(abs(Ts - Tm))]:.2f}")
        for T, (s_, l_), lo16, hi84 in zip(Ts, best, q(0, 16), q(0, 84)):
            if T % 500 == 0:
                print(f"   {T:6.0f} K: " + (f"solvus H{2 + 3 * s_ / (1 - s_):.2f}O (X {s_:.3f}, 16-84 % {lo16:.3f}-{hi84:.3f}), coexisting liquid X {l_:.2f} (H{2 + 3 * l_ / (1 - l_):.1f}O)" if np.isfinite(s_) else "no solid (all liquid)")
                      + ("   [extrapolated]" if m0.extrap(T) else ""))
                out.append(dict(P_GPa=P, T_K=T, solvus_X=s_, solvus_X16=lo16, solvus_X84=hi84, liquidus_X=l_, extrapolated=int(m0.extrap(T))))
        for r in [r for r in rows("docs/data/melt_dG_grace.csv") if r["P_GPa"] == P and r["H_per_O"] == 2.25]:
            X = 0.25 / 3.25; print(f"   check H2.25O at {r['T_K']:.0f} K: direct TDI G_liq - G_fcc {r['dG_meV_atom']:6.1f} +- {r['err']:.1f}, model {m0.gl(X, r['T_K']) - m0.gs(X, r['T_K']):6.1f} meV/atom")
    fig.suptitle("T-x diagram of H2O + H from GRACE free energies (classical nuclei): superionic solid solution, two-phase field, miscible liquid; bands = 16-84 % (statistical only); pale rows = extrapolated in T", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(a.o + ".png", dpi=130)
    if out:
        with open(a.o + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("\nwrote", a.o + ".png", a.o + ".csv")


if __name__ == "__main__":
    main()
