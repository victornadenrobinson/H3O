"""Plot + table of the PLANNED DFT labelling (docs/DFT_PLAN.md): points per set and per (composition, T). Allocations are plan targets, edit PLAN below when the plan changes.
usage: python mlp_pipeline/plot_data_plan.py [--out docs/data_plan.png]"""
import argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TS = [3000, 4000, 5000, 6000, 7000]
ROWS = ["pure H", "H2O", "H2O+dilute H (2.0-2.1)", "H2.25O", "H2.5O", "H2.75O", "H3O"]
# set -> (label, group, {row: {T: points}})
PLAN = {
 "A  GRACE-MD + n H":        ("generated here", {2: {3000: 90, 5000: 330, 7000: 90}}),
 "B+ n->n+1 chains":         ("generated here", {2: {5000: 180}}),
 "C  pwidom caches":         ("Young GRACE", {2: {4000: 30, 5000: 90, 6000: 30}, 3: {5000: 30}, 4: {5000: 20}}),
 "C2 pwidom reverse test":   ("Young GRACE", {0: {4000: 30, 5000: 80, 6000: 30}, 1: {4000: 30, 5000: 80, 6000: 30}, 3: {4000: 30, 5000: 80, 6000: 30}}),
 "C3 TDI-state D stats":     ("Young GRACE", {0: {5000: 40, 6000: 40}, 1: {5000: 40, 6000: 40}, 3: {5000: 40, 6000: 40}}),
 "D  EOS / melt_comp":       ("Young GRACE", {r: {5000: 20, 7000: 17} for r in (3, 4, 5, 6)}),
 "D2 4000/6000 K fill":      ("Young GRACE", {r: {4000: 12, 6000: 12} for r in (2, 3, 4, 5, 6)}),
 "E  low P 50/100/200":      ("GRACE NVT, new", {r: {3000: 12, 5000: 12, 7000: 12} for r in (1, 3, 4, 5, 6)}),
 "F  toy defect cells":      ("toy potential", {1: {5000: 47}, 3: {5000: 47}, 4: {5000: 47}, 6: {5000: 47}, 0: {5000: 32}}),
 "G  pure H 729 GRACE-MD":   ("GRACE-MD", {0: {5000: 30}}),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/data_plan.png"); a = ap.parse_args()
    tot = {k: sum(v for d in p.values() for v in d.values()) for k, (g, p) in PLAN.items()}
    M = np.zeros((len(ROWS), len(TS)))
    for k, (g, p) in PLAN.items():
        for r, d in p.items():
            for T, n in d.items(): M[r, TS.index(T)] += n
    print(f"{'set':26s} {'source':16s} {'points':>6s}"); [print(f"{k:26s} {PLAN[k][0]:16s} {n:6d}") for k, n in tot.items()]
    print(f"{'total':26s} {'':16s} {sum(tot.values()):6d}   (~{sum(tot.values()) / 450:.1f} four-node hours at 450 points/h, 729-atom points extra)")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    names = list(tot)[::-1]; cols = {"generated here": "#4c78a8", "Young GRACE": "#59a14f", "GRACE NVT, new": "#f28e2b", "toy potential": "#b07aa1", "GRACE-MD": "#76b7b2"}
    ax[0].barh(names, [tot[n] for n in names], color=[cols[PLAN[n][0]] for n in names])
    for i, n in enumerate(names): ax[0].text(tot[n] + 5, i, str(tot[n]), va="center", fontsize=9)
    ax[0].set_xlabel("planned DFT points (target)"); ax[0].set_title(f"DFT labelling plan: {sum(tot.values())} points")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols.values()], labels=list(cols), loc="lower right", fontsize=8, frameon=False)
    im = ax[1].imshow(M, cmap="viridis", aspect="auto"); ax[1].set_xticks(range(len(TS)), [f"{t} K" for t in TS]); ax[1].set_yticks(range(len(ROWS)), ROWS)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax[1].text(j, i, int(M[i, j]) if M[i, j] else "", ha="center", va="center", color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=9)
    ax[1].set_title("planned new points per composition x T"); plt.colorbar(im, ax=ax[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(a.out, dpi=130); print("wrote", a.out)


if __name__ == "__main__":
    main()
