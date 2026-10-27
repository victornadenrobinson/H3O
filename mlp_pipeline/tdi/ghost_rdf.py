"""Where does the inserted proton sit, lambda by lambda? Radial distribution of the ghost(s) around host O and host H from the saved node trajectories (no MD).
usage (repo root, python with ase + numpy, matplotlib for the figure):
    python mlp_pipeline/tdi/ghost_rdf.py runs/tdi_long/insert_500GPa_6000K_n1 [-o runs/checks/ghost_rdf] [--rmax 3.0]
Per lambda value: g_ghost-O(r) and g_ghost-H(r), the nearest host-O and host-H distances (median, 1st percentile), the fraction of frames with the ghost closer than 0.7 A to
any host atom (overlap), and the O coordination of the ghost within the first minimum of the host H-O g(r).
Sanity checks: at the largest lambda the ghost is an ordinary proton, so its g(r) must match the host protons' g_H-O / g_H-H (reference, dashed); a large difference means the
lambda ~ 1 sampling is not equilibrated (ghost trapped) or the ghost sits in a different site. At small lambda the ghost is nearly ideal: g(r) ~ 1 with an overlap fraction near
the volume fraction inside 0.7 A spheres."""
import argparse, json, os, sys
import numpy as np
from ase.io import read
from ase.geometry import get_distances
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze

ap = argparse.ArgumentParser(); ap.add_argument("run", help="<dir>/<tag> of an insertion run"); ap.add_argument("-o", default=None); ap.add_argument("--rmax", type=float, default=3.0)
ap.add_argument("--dr", type=float, default=0.05); ap.add_argument("--max-frames", type=int, default=400); a = ap.parse_args()
st = json.load(open(a.run + "_stage0.json")); A = st["args"]; lam = analyze.rule_nodes(A)[0]; ng = int(st["N"]) if st["phase"] == "insert" else 0
if not ng:
    raise SystemExit("not an insertion run")
edges = np.arange(0.0, a.rmax + 1e-9, a.dr); rc = 0.5 * (edges[1:] + edges[:-1]); shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)


def rdf(d, n_ref, n_frames, V):
    """g(r) of distances d (all frames, all centres) to n_ref atoms of a species, normalised by the ideal count n_ref / V * shell per centre and frame"""
    h = np.histogram(d, edges)[0]
    return h / (n_frames * (n_ref / V) * shell)


res = []
for k in np.argsort(lam):
    f = f"{a.run}_node{k}_traj.extxyz"
    if not os.path.exists(f):
        continue
    frames = read(f, index=":"); frames = frames[::max(1, len(frames) // a.max_frames)]
    nat = len(frames[0]); host = np.arange(nat - ng); gh = np.arange(nat - ng, nat); Z = frames[0].numbers
    iO = host[Z[host] == 8]; iH = host[Z[host] == 1]; V = frames[0].get_volume()
    dO, dH, nnO, nnH, ref_HO, ref_HH = [], [], [], [], [], []
    for fr in frames:
        p = fr.positions
        for g in gh:
            if len(iO):
                d = get_distances(p[g][None], p[iO], cell=fr.cell, pbc=True)[1][0]; dO.append(d); nnO.append(d.min())
            d = get_distances(p[g][None], p[iH], cell=fr.cell, pbc=True)[1][0]; dH.append(d); nnH.append(d.min())
        sel = iH[:: max(1, len(iH) // 16)]   # reference: 16 host protons per frame
        if len(iO):
            ref_HO.append(get_distances(p[sel], p[iO], cell=fr.cell, pbc=True)[1].ravel())
        D = get_distances(p[sel], p[iH], cell=fr.cell, pbc=True)[1]; D[D < 1e-8] = np.inf; ref_HH.append(D.ravel())
    nf = len(frames) * ng; nO, nH = len(iO), len(iH)
    r = dict(k=int(k), lam=float(lam[k]), frames=len(frames), V=V,
             gO=rdf(np.concatenate(dO), nO, nf, V) if nO else None, gH=rdf(np.concatenate(dH), nH, nf, V),
             gO_ref=rdf(np.concatenate(ref_HO), nO, len(frames) * len(sel), V) if nO else None, gH_ref=rdf(np.concatenate(ref_HH), nH - 1, len(frames) * len(sel), V),
             nnO=np.array(nnO), nnH=np.array(nnH))
    near = np.minimum(r["nnO"], r["nnH"]) if nO else r["nnH"]; r["overlap"] = float(np.mean(near < 0.7))
    res.append(r)
if not res:
    raise SystemExit("no node trajectories found")
# first minimum of the host H-O g(r) after its first peak: coordination shell of a bonded proton
top = res[-1]
if top["gO_ref"] is not None:
    i1 = int(np.argmax(top["gO_ref"])); j = i1 + int(np.argmin(top["gO_ref"][i1:i1 + int(1.0 / a.dr)])); rmin = rc[j]
else:
    rmin = 1.2
print(f"{a.run}: {ng} ghost(s), {len(res)} lambda values; overlap = ghost closer than 0.7 A to a host atom; CN_O = host O within {rmin:.2f} A (first minimum of the host H-O g(r))")
print(f"{'k':>3} {'lambda':>10} {'frames':>6} {'nnO med':>8} {'nnO 1%':>7} {'nnH med':>8} {'nnH 1%':>7} {'overlap':>8} {'CN_O':>6} {'|g-g_host| O':>12} {'H':>6}")
for r in res:
    nO_density = (st.get("nO") or 0) / r["V"]
    cnO = float(np.sum(r["gO"][rc < rmin] * shell[rc < rmin]) * nO_density) if r["gO"] is not None else float("nan")
    m = (rc > 0.5) & (rc < a.rmax)
    dgo = float(np.sum(np.abs(r["gO"] - r["gO_ref"])[m]) * a.dr / np.sum(r["gO_ref"][m] * a.dr)) if r["gO"] is not None else float("nan")
    dgh = float(np.sum(np.abs(r["gH"] - r["gH_ref"])[m]) * a.dr / np.sum(r["gH_ref"][m] * a.dr))
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")
    print(f"{r['k']:>3} {r['lam']:10.3e} {r['frames']:6d} {q(r['nnO'], 50):8.3f} {q(r['nnO'], 1):7.3f} {q(r['nnH'], 50):8.3f} {q(r['nnH'], 1):7.3f} {r['overlap']:8.4f} {cnO:6.2f} {dgo:12.3f} {dgh:6.3f}")
print("last two columns: relative L1 difference between the ghost's g(r) and the host protons' g(r) over 0.5-rmax A: at the largest lambda it should be small (~ the noise, <~0.1-0.2 with few frames)")
if a.o:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    nrm = colors.LogNorm(vmin=min(r["lam"] for r in res), vmax=1.0); cmap = cm.viridis_r
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for r in res:
        c = cmap(nrm(r["lam"]))
        if r["gO"] is not None:
            ax[0].plot(rc, r["gO"], color=c, lw=1.5)
        ax[1].plot(rc, r["gH"], color=c, lw=1.5)
    if top["gO_ref"] is not None:
        ax[0].plot(rc, top["gO_ref"], color="#1f1f1e", lw=2, ls="--", label="host protons (lambda max)")
    ax[1].plot(rc, top["gH_ref"], color="#1f1f1e", lw=2, ls="--", label="host protons (lambda max)")
    for x_, t in zip(ax, ("ghost - host O", "ghost - host H")):
        x_.set_title(t, loc="left"); x_.set_xlabel("r (A)"); x_.set_ylabel("g(r)"); x_.axhline(1, color="#9a9a94", lw=0.8); x_.legend(frameon=False, loc="upper right")
    sm = cm.ScalarMappable(norm=nrm, cmap=cmap); fig.colorbar(sm, ax=ax, label="lambda")
    os.makedirs(os.path.dirname(a.o) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.o}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {a.o}.pdf / .png")
