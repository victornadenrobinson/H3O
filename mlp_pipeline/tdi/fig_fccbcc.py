"""fcc - bcc figure with every isotherm (report.py page 'fcc_bcc_all'): G_fcc - G_bcc vs P for each T (colour = T) with Cheng et al. 2021 (classical PBE) at their
2000-5000 K in the same colours, and vs T for each P (isobars: the bcc -> fcc crossing on heating and the near-melting convergence).
pairs: DataFrame with P, T, h, cell ('big' 108/128 O, 'small' 32/54 O), dG, err, caveat; cheng: DataFrame with T_K, P_GPa, pbe, err (meV/atom, G_fcc - G_bcc) or empty.
Standalone test: python3 mlp_pipeline/tdi/fig_fccbcc.py docs/data/grace_pt_points.csv docs/data/cheng2021_dmu_bcc_fcc.csv out.png
"""
import numpy as np


def fig_fccbcc_all(pairs, cheng, plt):
    p = pairs[(pairs.h == 2.0)].copy()
    Ts = sorted(p["T"].unique()); Ps = sorted(p["P"].unique())
    cmapT = plt.get_cmap("plasma"); colT = lambda T: cmapT(0.05 + 0.8 * (T - min(Ts)) / max(1, max(Ts) - min(Ts)))
    cmapP = plt.get_cmap("viridis"); colP = lambda P: cmapP(0.05 + 0.85 * (P - min(Ps)) / max(1, max(Ps) - min(Ps)))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6.2))
    for T in Ts:
        g = p[(p["T"] == T) & (p.cell == "big")].sort_values("P")
        if len(g):
            ax[0].errorbar(g.P, g.dG, g.err, fmt="o-", color=colT(T), ms=5, lw=1.4, capsize=2, label=f"{T:.0f} K")
            for _, r in g[g.caveat].iterrows(): ax[0].plot(r.P, r.dG, "o", mfc="white", mec=colT(T), ms=5)
        s = p[(p["T"] == T) & (p.cell == "small")]
        if len(s): ax[0].errorbar(s.P, s.dG, s.err, fmt="v", color=colT(T), mfc="white", ms=7, capsize=2, label=f"{T:.0f} K, 32/54-O cell")
        if len(cheng):
            c = cheng[cheng["T_K"] == T].sort_values("P_GPa")
            if len(c): ax[0].plot(c.P_GPa, c.pbe, "--", color=colT(T), lw=1.2, alpha=0.9)
    ax[0].plot([], [], "k--", lw=1.2, label="Cheng 2021 (PBE, classical), same colour = same T")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set(xlabel="P (GPa)", ylabel="G_fcc - G_bcc (meV/atom)", xlim=(80, 620),
                                                    title="isotherms: below 0 = fcc stable (open = different dt / lambda grid)")
    ax[0].legend(fontsize=6.5, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for P in Ps:
        g = p[(p.P == P) & (p.cell == "big")].sort_values("T")
        if len(g): ax[1].errorbar(g["T"], g.dG, g.err, fmt="o-", color=colP(P), ms=4, lw=1.4, capsize=2, label=f"{P:.0f} GPa")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set(xlabel="T (K)", ylabel="G_fcc - G_bcc (meV/atom)", title="isobars: bcc -> fcc on heating; the two converge near melting")
    ax[1].legend(fontsize=6.5, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    for a in ax: a.grid(alpha=0.25)
    fig.suptitle("fcc vs bcc superionic H2O, all temperatures (this work: GRACE ft_m2, classical nuclei, 108-O fcc / 128-O bcc)", fontsize=9)
    fig.tight_layout(); return fig


if __name__ == "__main__":   # test on the data-file copy of the summary
    import sys, pandas as pd, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = pd.read_csv(sys.argv[1], comment="#"); g = g[g.kind == "fccbcc"]
    pairs = pd.DataFrame(dict(P=g.P_GPa.astype(float), T=g.T_K.astype(float), h=g.H_per_O.astype(float), cell="big", dG=g.value.astype(float), err=g.err.astype(float),
                              caveat=g.note.fillna("").str.contains("grid|dt")))
    pairs = pd.concat([pairs, pd.DataFrame([dict(P=400.0, T=3000.0, h=2.0, cell="small", dG=-35.22, err=3.48, caveat=False)])])
    c = pd.read_csv(sys.argv[2], comment="#"); c["pbe"] = -c.dmu_pbe_eV_mol * 1000 / 3; c["err"] = c.err_fep_eV_mol * 1000 / 3
    fig_fccbcc_all(pairs, c, plt).savefig(sys.argv[3], dpi=120); print("wrote", sys.argv[3])
