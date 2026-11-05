#!/usr/bin/env python3
"""Scatter plots of a training tier: where the DFT labels sit in pressure, temperature and composition, coloured by data group (8 groups of db_set), train filled / valid hollow.
   $PY mlp_pipeline/plot_training_data.py --train runs/train_mfin.extxyz --valid runs/valid_mfin.extxyz -o runs/plots/training_mfin
Writes <o>.png (4 panels) and <o>.csv (one row per frame: group, set, split, nO, nH, natoms, x_H, T_K, P_GPa, E/atom, max|F|) and prints frames per group x split.
P = DFT virial pressure from the stress label (no ionic N kT/V term). T = info T_K, else T_set (low-P MD), else blank (static / relaxed / toy frames: plotted in the 'no T' strip)."""
import argparse, collections, csv, os
import numpy as np

GROUPS = [  # (label, db_set names); frames without db_set = the old mined AIMD tier (m)
    ("AIMD tier m", {"old"}),
    ("GRACE-MD insertions A/B", {"A", "A_b1", "A_part", "B"}),
    ("PWidom host caches C", {"C", "C2"}),
    ("EOS / melt / low-P EOS D,E", {"D_eos", "D_melt", "E"}),
    ("toy defects + pure H F,G", {"F", "F_toy", "F_H", "G", "T"}),
    ("H3O supercells H", {"H"}),
    ("DFT pairs + weighted trials PD,PW", {"PD", "PW"}),
    ("low-P / cold / hydrates LP", {"LP"}),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]   # validated categorical order (dataviz reference palette, light)


def setname(f):
    v = f.info.get("db_set", "old")
    return "F_toy" if v is False else "T" if v is True else str(v)


def group_of(s):
    for i, (_, ss) in enumerate(GROUPS):
        if s in ss: return i
    return None


def rows_from(path, split):
    from ase.io import read
    out = []
    for f in read(path, index=":"):
        s = setname(f); g = group_of(s)
        st = f.get_stress(voigt=False); fo = f.get_forces(); nO = int((f.numbers == 8).sum()); nH = int((f.numbers == 1).sum())
        T = f.info.get("T_K", f.info.get("T_set"))
        out.append(dict(group=GROUPS[g][0] if g is not None else "other", gi=len(GROUPS) if g is None else g, set=s, split=split, nO=nO, nH=nH, natoms=len(f),
                        x_H=nH / max(1, nH + nO), T_K=float(T) if T not in (None, "", "nan") else float("nan"), P_GPa=-float(np.trace(st)) / 3 * 160.21766,
                        E_atom=f.get_potential_energy() / len(f), Fmax=float(np.abs(fo).max())))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--valid"); ap.add_argument("-o", default="runs/plots/training_data"); a = ap.parse_args()
    R = rows_from(a.train, "train") + (rows_from(a.valid, "valid") if a.valid else [])
    os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
    keys = ["group", "set", "split", "nO", "nH", "natoms", "x_H", "T_K", "P_GPa", "E_atom", "Fmax"]
    with open(a.o + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys, extrasaction="ignore"); w.writeheader(); w.writerows(R)
    c = collections.Counter((r["gi"], r["split"]) for r in R)
    labels = [g[0] for g in GROUPS] + ["other"]
    print(f"{'group':36s} {'train':>6s} {'valid':>6s}   sets")
    for i, lab in enumerate(labels):
        if c[(i, "train")] + c[(i, "valid")] == 0: continue
        ss = collections.Counter(r["set"] for r in R if r["gi"] == i)
        print(f"{lab:36s} {c[(i, 'train')]:6d} {c[(i, 'valid')]:6d}   " + " ".join(f"{k}:{v}" for k, v in sorted(ss.items())))
    print(f"{'total':36s} {sum(v for (i, s), v in c.items() if s == 'train'):6d} {sum(v for (i, s), v in c.items() if s == 'valid'):6d}")
    others = sorted({r["set"] for r in R if r["gi"] == len(GROUPS)})
    if others: print("  sets not in any group (grey):", others)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    col = COLORS + ["#9a9a9a"]
    Tvals = np.array([r["T_K"] for r in R]); Tnone = -800.0                       # frames without T drawn in a strip below 0 K
    def sc(axx, xk, yk, ytrans=None):
        for i in range(len(labels)):
            for split, filled in (("train", True), ("valid", False)):
                rr = [r for r in R if r["gi"] == i and r["split"] == split]
                if not rr: continue
                x = np.array([r[xk] for r in rr]); y = np.array([r[yk] for r in rr]); y = ytrans(y) if ytrans else y
                axx.scatter(x, y, s=14, marker="o", facecolors=col[i] if filled else "none", edgecolors=col[i], linewidths=0.9 if not filled else 0.4, alpha=0.75 if filled else 0.9, zorder=3 if not filled else 2)
        axx.grid(True, color="#e6e6e6", lw=0.6, zorder=0)
    fixT = lambda y: np.where(np.isfinite(y), y, Tnone)
    sc(ax[0, 0], "P_GPa", "T_K", fixT); ax[0, 0].set_xscale("symlog", linthresh=10); ax[0, 0].set_xlabel("DFT pressure (GPa, virial, symlog)"); ax[0, 0].set_ylabel("T (K)")
    ax[0, 0].axhline(Tnone / 2, color="#bbb", lw=0.6); ax[0, 0].text(0.01, 0.015, "no T (static / relaxed / toy)", transform=ax[0, 0].transAxes, color="#777", fontsize=8)
    ax[0, 0].set_title("pressure vs temperature", loc="left", color="#222")
    sc(ax[0, 1], "P_GPa", "x_H"); ax[0, 1].set_xscale("symlog", linthresh=10); ax[0, 1].set_xlabel("DFT pressure (GPa, virial, symlog)"); ax[0, 1].set_ylabel("H fraction  nH / (nH + nO)")
    for hpo, nm in ((2, "H2O"), (3, "H3O"), (4, "H4O"), (6, "H6O"), (None, "pure H")):
        xh = 1.0 if hpo is None else hpo / (hpo + 1); ax[0, 1].axhline(xh, color="#ccc", lw=0.6, zorder=1); ax[0, 1].text(1.0, xh, f" {nm}", transform=ax[0, 1].get_yaxis_transform(), va="center", fontsize=8, color="#777")
    ax[0, 1].set_title("pressure vs composition", loc="left", color="#222")
    sc(ax[1, 0], "P_GPa", "Fmax"); ax[1, 0].set_xscale("symlog", linthresh=10); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("DFT pressure (GPa, virial, symlog)"); ax[1, 0].set_ylabel("max |F| per frame (eV/A)")
    ax[1, 0].set_title("largest force per frame", loc="left", color="#222")
    used = [i for i in range(len(labels)) if c[(i, "train")] + c[(i, "valid")]]
    y = np.arange(len(used))[::-1]; tr = np.array([c[(i, "train")] for i in used]); va = np.array([c[(i, "valid")] for i in used])
    ax[1, 1].barh(y, tr, color=[col[i] for i in used], height=0.62, edgecolor="white", linewidth=2)
    ax[1, 1].barh(y, va, left=tr, color="none", edgecolor=[col[i] for i in used], height=0.62, linewidth=1.2, hatch="////")
    for yy, t, v in zip(y, tr, va): ax[1, 1].text(t + v, yy, f"  {t} + {v}", va="center", fontsize=8, color="#333")
    ax[1, 1].set_yticks(y); ax[1, 1].set_yticklabels([labels[i] for i in used]); ax[1, 1].set_xlabel("frames (train filled + valid hatched)")
    ax[1, 1].set_xlim(0, max(tr + va) * 1.25); ax[1, 1].grid(True, axis="x", color="#e6e6e6", lw=0.6); ax[1, 1].set_axisbelow(True)
    ax[1, 1].set_title(f"frames per group: {tr.sum()} train, {va.sum()} valid", loc="left", color="#222")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", markerfacecolor=col[i], markeredgecolor=col[i], markersize=6, label=labels[i]) for i in used]
    h += [Line2D([], [], marker="o", ls="", markerfacecolor="#666", markeredgecolor="#666", markersize=6, label="train (filled)"),
          Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#666", markersize=6, label="valid (hollow)")]
    fig.legend(handles=h, loc="outside upper center", ncol=5, frameon=False, fontsize=8.5)
    fig.savefig(a.o + ".png", dpi=150); print("wrote", a.o + ".png", "and", a.o + ".csv")


if __name__ == "__main__":
    main()
