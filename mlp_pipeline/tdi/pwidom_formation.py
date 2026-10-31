"""Plot reported transfer costs and their provisional composition integral.

This is not an equilibrium convex hull: interpolation and unconverged input can
dominate. Statistical propagation bounds omit sampling and model bias; no
validated equilibrium solubility is inferred.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def integrate(n, mu):
    if len(n) < 2 or n[0] != 0 or np.any(np.diff(n) <= 0):
        raise ValueError("Need at least two distinct sorted compositions including n=0")
    values = np.r_[0, np.cumsum(np.diff(n) * (mu[1:] + mu[:-1]) / 2)]
    grid = np.linspace(0, n[-1], 500)
    indices = np.minimum(np.searchsorted(n, grid, side="right") - 1, len(n)-2)
    delta = grid - n[indices]
    integral = values[indices] + mu[indices]*delta + .5 * (
        (mu[indices+1]-mu[indices])/(n[indices+1]-n[indices])) * delta**2
    return values, grid, integral


def statistical_error_bound(n, err):
    """Maximal-positive-correlation bound for the trapezoid's statistical sigma.

    All integration weights are nonnegative. Since |cov_ij| <= sigma_i sigma_j,
    sum_i weight_i sigma_i bounds the propagated sigma for any covariance with
    the supplied diagonal. It is not a bound on unconverged-sampling bias.
    """
    weights = np.zeros(len(n))
    bounds = [0.0]
    for i, delta in enumerate(np.diff(n), start=1):
        weights[i-1] += delta/2; weights[i] += delta/2
        bounds.append(float(weights @ err))
    return np.array(bounds)/(3+n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="docs/data/pwidom_pm_500GPa_3000K_reported.csv")
    ap.add_argument("--P", type=float, default=500)
    ap.add_argument("--T", type=float, default=3000)
    ap.add_argument("--out", required=True, help="fresh filename prefix for PNG/PDF/integral CSV")
    ap.add_argument("--compare", help="earlier reported CSV, shown for comparison only; not pooled")
    ap.add_argument("--compare-label", default="Original short scan (shared parent)")
    ap.add_argument("--label", default="Current scan", help="legend label for the primary dataset")
    a = ap.parse_args()
    with open(a.data) as file:
        rows = [r for r in csv.DictReader(line for line in file if not line.startswith("#"))
                if float(r["P_GPa"]) == a.P and float(r["T_K"]) == a.T]
    rows.sort(key=lambda r: float(r["n"]))
    n, mu, err = (np.array([float(r[k]) for r in rows]) for k in ("n", "dg_meV", "err_meV"))
    values, grid, integral = integrate(n, mu)
    error_bound = statistical_error_bound(n, err)
    out = Path(a.out)
    for suffix in (".png", ".pdf", ".csv"):
        if out.with_suffix(suffix).exists():
            raise ValueError(f"Output already exists: {out.with_suffix(suffix)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    color = "#ce477f"
    if a.compare:
        with open(a.compare) as file:
            previous = [r for r in csv.DictReader(line for line in file if not line.startswith("#"))
                        if float(r["P_GPa"]) == a.P and float(r["T_K"]) == a.T]
        previous.sort(key=lambda r: float(r["n"]))
        pn, pm, pe = (np.array([float(r[k]) for r in previous]) for k in ("n", "dg_meV", "err_meV"))
        pv, pg, pi = integrate(pn, pm)
        axes[0].errorbar(pn, pm, yerr=pe, fmt="o--", color="0.6", mfc="white",
                         capsize=3, alpha=.7, label=a.compare_label)
        axes[1].plot(pg, pi/(3+pg), "--", color="0.6", alpha=.7,
                     label=a.compare_label)
        axes[1].errorbar(pn, pv/(3+pn), yerr=statistical_error_bound(pn, pe),
                         fmt="o", color="0.6", mfc="white", capsize=3, alpha=.7)
    for i, r in enumerate(rows):
        axes[0].errorbar(n[i], mu[i], yerr=err[i], fmt="s", color=color,
                        mfc=color if r["flag"] == "ok" else "white", capsize=4)
    axes[0].plot(n, mu, "--", color=color, alpha=.7, label=f"{a.label}; convergence not established")
    axes[0].set_title(f"{a.P:g} GPa, {a.T:g} K — GRACE ft_m2, fitted proxies")
    axes[0].set_ylabel(r"$\Delta\mu_H = \mu_H^{\mathrm{mixture}} - \mu_H^{\mathrm{pure\ H}}$ (meV / H)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(grid, integral/(3+grid), "--", color=color,
                 label="Integral of interpolated transfer costs (provisional)")
    axes[1].errorbar(n, values/(3+n), yerr=error_bound, fmt="s", mfc="white",
                    color=color, capsize=4, label="Statistical error bound (maximal positive correlation)")
    annotation_indices = range(1,len(n)) if len(n) <= 4 else sorted({int(np.argmin(values/(3+n))),len(n)-1})
    for index in annotation_indices:
        nn, value = n[index], (values/(3+n))[index]
        axes[1].annotate(f"{value:.1f} meV/atom", (nn, value), xytext=(-5, 12),
                         textcoords="offset points", ha="right", fontsize=9)
    axes[1].set_ylabel("Provisional formation free energy (meV / atom)")
    axes[1].set_xlabel("Extra H per O, n  [H(2+n)O]")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25); ax.axhline(0, color="k", lw=.7)
        ax.set_xlim(-.01, n[-1]+.03)
    sec = axes[0].secondary_xaxis("top", functions=(lambda x: x*108, lambda k:k/108))
    sec.set_xlabel("Extra H per 108-O cell"); sec.set_xticks([0, 9, 18, 27])
    fig.text(.5, .015, "Exploratory curve; convergence is not established. Bars omit sampling/model bias.\n"
             "Lower bars bound propagated statistical sigma; shared-reference covariance is unavailable.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=180)
    plt.close(fig)
    np.savetxt(out.with_suffix(".csv"), np.c_[n, mu, err, values, values/(3+n), error_bound],
        delimiter=",", header="n,dg_meV,stat_err_meV,integral_meV_per_O,formation_meV_atom,stat_sigma_upper_bound_meV_atom", comments="")
    print(f"Wrote {out}.png/.pdf/.csv; provisional, no equilibrium solubility inferred")


if __name__ == "__main__":
    main()
