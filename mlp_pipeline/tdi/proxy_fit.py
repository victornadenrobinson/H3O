"""Refit the insertion proxy of pwidom.py (OH and HH spline pairs) to the REAL model's insertion energies at probe points, using the pwidom frames that already exist.

Why. pwidom.py is unbiased for any proxy that does not depend on the frames of the estimate; the proxy only sets the variance (std of dU - phi, Kish fraction). toy_fit.py
force-matches DFT frames; what pwidom needs is dU(x) = U(frame + H at x) - U(frame) of GRACE where the Rosenbluth step looks (low phi). For a pair proxy dU(x) = c + sum_j phi_Hj(|x - r_j|)
is LINEAR in the spline coefficients: one weighted least-squares solve, no forces, no MD. The O-O pair, the core and the cutoff of the base fit are kept.

No MD: `label` reads the cached host frames (pwidom *_frames.npz; fcc / bcc / pure-H hosts have cubic cells, the cell is V^(1/3) from the file) of as many states as you give it
(compositions, P, T), a few frames each, and asks the real model for dU at probe points (Rosenbluth-selected on the starting proxy at --temper x T, plus uniform points).
One probes file per source (resumable; existing ones are skipped). extxyz sources (e.g. insertion trajectories, --drop-last 1 removes the ghost) bring their own cell and need --T.

usage (repo root; label needs the real model and a GPU, fit is numpy only):
  python mlp_pipeline/tdi/proxy_fit.py label runs/pwidom*/pw_*_frames.npz --device cuda --model grace-ft:... --proxy docs/data/toyfit_t4.json --per-source 10 -o runs/pwidom_fit
  python mlp_pipeline/tdi/proxy_fit.py fit runs/pwidom_fit/*_probes.npz --base docs/data/toyfit_t4.json --skip pureH --hold pwidom__pw_fcc_500GPa_6000K -o runs/pwidom_fit/proxy_ice --plot
  python mlp_pipeline/tdi/pwidom.py run --like <stage0.json> --proxy runs/pwidom_fit/proxy_ice.json ...        (production frames of the held-out state: all of them)
--model defaults to toyice; use the model of the runs (the `model` field of pwidom's output json / stage0.json args).
fit: --only / --skip select sources by substring of the probes file name (e.g. ice vs pure H: separate proxies), --hold keeps matching sources OUT of the fit entirely and reports them as
held-out states (use them for the production state: then the proxy has never seen those frames); inside the other sources every --fold-th frame is held out. Weights exp(-(dU - ref)/(wT kT))
per source (ref = 1st percentile of that source's dU), so the fit is accurate where the sampler goes. A pair proxy fitted to a pool of states is an average over them: the per-source table
shows which states it describes well; fit ice and pure H separately if they differ.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
KB = 8.617333262e-5


def lse(x):
    m = np.max(x); return m + np.log(np.sum(np.exp(x - m)))


def source_name(f):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(os.path.dirname(os.path.abspath(f))) + "__" + os.path.splitext(os.path.basename(f))[0])


def source_frames(f, a):
    """-> pos (F, N, 3), cell (F, 3, 3), numbers (N,), T (K) of one source"""
    if f.endswith(".npz"):
        z = np.load(f); pos = z["host"]; nums = z["numbers"]; assert pos.shape[1] == len(nums), f"{f}: frames and numbers disagree"
        L = float(z["V"]) ** (1 / 3); cell = np.repeat((np.eye(3) * L)[None], len(pos), 0); T = float(z["T"])
    else:
        from ase.io import iread
        fr = list(iread(f, index=":")); n = len(fr[0]) - a.drop_last; assert a.T, f"{f}: extxyz sources need --T"
        pos = np.array([r.positions[:n] for r in fr]); cell = np.array([np.asarray(r.cell) for r in fr]); nums = fr[0].numbers[:n]; T = a.T
    return pos, cell, nums, T


def cmd_label(a):
    from ase import Atoms
    from pwidom import Model, proxy_params, proxy_energy
    srcs = [f for f in a.src if not a.select or any(x in f for x in a.select)]   # --select: keep sources whose path contains any of these (e.g. 500GPa_6000K)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/")); srcs = srcs[i::n]   # --shard i/n: this process takes every n-th source (independent processes, no coordination)
    print(f"{len(srcs)} of {len(a.src)} sources" + (f" (shard {a.shard})" if a.shard else ""), flush=True)
    p = proxy_params(a.proxy); model = Model(a.model, a.device, a.batch); os.makedirs(a.o, exist_ok=True); t0 = time.time(); ndone = 0
    for si, f in enumerate(srcs):
        out = os.path.join(a.o, source_name(f) + "_probes.npz")
        if os.path.exists(out) and not a.force:
            print(f"[{si + 1}/{len(srcs)}] {out} exists: skipped", flush=True); continue
        pos, cell, nums, T = source_frames(f, a); isH = nums == 1; kT = KB * T; rng = np.random.default_rng(a.seed + si)
        keep = np.unique(np.linspace(0, len(pos) - 1, min(a.per_source, len(pos))).astype(int)); pos, cell = pos[keep], cell[keep]; F = len(pos)
        pts, fid, kind, phi0 = [], [], [], []
        for k in range(F):
            c = rng.random((a.K, 3)) @ cell[k]; phi = proxy_energy(p, c, pos[k], isH, cell[k])
            if a.nsel:
                w = -phi / (a.temper * kT); s = rng.choice(a.K, a.nsel, p=np.exp(w - lse(w))); pts += list(c[s]); phi0 += list(phi[s]); kind += [1] * a.nsel; fid += [k] * a.nsel
            if a.nuni:
                u = rng.random((a.nuni, 3)) @ cell[k]; pts += list(u); phi0 += list(proxy_energy(p, u, pos[k], isH, cell[k])); kind += [0] * a.nuni; fid += [k] * a.nuni
        fid = np.array(fid); pts = np.array(pts)
        structs = [Atoms(numbers=nums, positions=pos[k], cell=cell[k], pbc=True) for k in range(F)]
        structs += [Atoms(numbers=np.r_[nums, 1], positions=np.vstack([pos[k], x]), cell=cell[k], pbc=True) for k, x in zip(fid, pts)]
        E = model.energies(structs); dU = E[F:] - E[:F][fid]; ndone += len(structs)
        np.savez(out, pos=pos, cell=cell, numbers=nums, T=T, pts=pts, fid=fid, kind=np.array(kind), phi_start=np.array(phi0), dU=dU, E0=E[:F], proxy=str(a.proxy), temper=a.temper, source=os.path.abspath(f))
        sel = np.array(kind) == 1
        print(f"[{si + 1}/{len(srcs)}] {os.path.basename(out)}: {F} frames of {len(nums)} atoms, {len(pts)} probes, T {T:g} K; selected-probe gap of the starting proxy "
              f"{np.std((dU - np.array(phi0))[sel]) / kT:.2f} kT; {ndone} single points, {(time.time() - t0) / 60:.1f} min", flush=True)


def pair_rows(pts, pos, isH, cell, rc, chunk=256):
    """all (probe, distance, host atom is H) pairs within rc, periodic images included (same image rule as toy_ice.ghost_energy)"""
    inv = np.linalg.inv(cell); V = abs(np.linalg.det(cell)); width = V / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    nimg = [int(np.floor(rc / w + 0.5)) for w in width]; I, R, H = [], [], []
    for i0 in range(0, len(pts), chunk):
        ds = (pos @ inv)[None] - (pts[i0:i0 + chunk] @ inv)[:, None]; ds -= np.round(ds)
        for sx in range(-nimg[0], nimg[0] + 1):
            for sy in range(-nimg[1], nimg[1] + 1):
                for sz in range(-nimg[2], nimg[2] + 1):
                    r = np.linalg.norm((ds + np.array([sx, sy, sz])) @ cell, axis=-1); m = r < rc; i, j = np.nonzero(m)
                    I.append(i + i0); R.append(r[m]); H.append(isH[j])
    return np.concatenate(I), np.concatenate(R), np.concatenate(H)


def design(bases, K, rows, M):
    """X (M x (n_OH + n_HH + 1)) of spline values (linear extrapolation below r_lo as in toy_ice._spline_pair), plus the fixed K dl^2 core energy per probe"""
    I, R, H = rows; off = np.r_[0, np.cumsum([b.n for b in bases])]; X = np.zeros((M, off[-1] + 1)); core = np.zeros(M)
    for k, b in enumerate(bases):
        m = H == bool(k); r = R[m]; i = I[m]; v, dv = b(r); dl = np.where(r < b.r_lo, b.r_lo - r, 0.0); val = v - dl[:, None] * dv
        for n in range(b.n):
            X[:, off[k] + n] += np.bincount(i, weights=val[:, n], minlength=M)
        core += K * np.bincount(i, weights=dl ** 2, minlength=M)
    X[:, -1] = 1.0
    return X, core


def kish(gap_kT):
    q = np.exp(-(gap_kT - gap_kT.min())); return float(q.sum() ** 2 / (q ** 2).sum() / len(q))


def cmd_fit(a):
    from toy_fit import Basis
    from pwidom import proxy_params, proxy_energy
    from tdi import toy_ice
    base = json.load(open(a.base)); rc = a.rc or base["rc"]; K = base["K"]
    files = [f for f in a.data if (not a.only or any(s in os.path.basename(f) for s in a.only)) and not any(s in os.path.basename(f) for s in a.skip)]
    assert files, "no probes files selected (--only / --skip)"
    names = [os.path.basename(f).replace("_probes.npz", "") for f in files]; hold_src = np.array([any(h in n for h in a.hold) for n in names])
    fpos, fcell, fnum = [], [], []; P, D, KD, SRC, FID, KT, FLOC = [], [], [], [], [], [], []
    for s, f in enumerate(files):
        z = np.load(f); n0 = len(fpos); F = len(z["pos"])
        fpos += list(z["pos"]); fcell += list(z["cell"]); fnum += [z["numbers"]] * F
        P.append(z["pts"]); D.append(z["dU"]); KD.append(z["kind"]); SRC.append(np.full(len(z["dU"]), s)); FID.append(z["fid"] + n0); KT.append(np.full(len(z["dU"]), KB * float(z["T"]))); FLOC.append(z["fid"])
    pts, dU, kind, src, fid, kT, floc = (np.concatenate(x) for x in (P, D, KD, SRC, FID, KT, FLOC)); M = len(pts)
    hold = hold_src[src]; test_frame = (~hold) & (floc % a.fold == a.fold - 1); test = hold | test_frame; trn = ~test
    print(f"{len(files)} sources ({int(hold_src.sum())} held out entirely), {M} probes: {int(trn.sum())} training, {int(test_frame.sum())} in held-out frames of training sources, {int(hold.sum())} in held-out sources", flush=True)
    I, R, H = [], [], []
    for f in np.unique(fid):
        idx = np.nonzero(fid == f)[0]; i, r, h = pair_rows(pts[idx], fpos[f], fnum[f] == 1, fcell[f], rc); I.append(idx[i]); R.append(r); H.append(h)
    I, R, H = np.concatenate(I), np.concatenate(R), np.concatenate(H)
    ref = np.zeros(M)   # per source: 1st percentile of dU (the sampler's region sits near the bottom of each state's dU)
    for s in range(len(files)):
        m = src == s; ref[m] = np.percentile(dU[m & ~hold], 1) if (m & ~hold).any() else np.percentile(dU[m], 1)
    w = np.clip(np.exp(-(dU - ref) / (a.wT * kT)), 1e-4, 1.0); tr = trn & (dU < ref + a.cap * kT)
    r_lo = []
    for k in (0, 1):   # shortest O-H / H-H distance among the probes that carry weight
        m = (H == bool(k)) & tr[I] & (w[I] > 1e-2); r_lo.append(max(0.3, float(R[m].min()) - a.core_pad) if m.any() else 1.0)
    bases = [Basis(r_lo[k], rc, a.dr) for k in (0, 1)]; X, core = design(bases, K, (I, R, H), M); y = dU - core
    sw = np.sqrt(w * tr)[:, None]; A = (X * sw).T @ (X * sw); b = (X * sw).T @ (y[:, None] * sw); o = np.r_[0, np.cumsum([bb.n for bb in bases])]
    for k in (0, 1):
        D2 = np.diff(np.eye(bases[k].n), 2, axis=0); A[o[k]:o[k + 1], o[k]:o[k + 1]] += a.smooth * (w * tr).sum() * D2.T @ D2
    c = np.linalg.solve(A, b[:, 0]); phi_new = X @ c - c[-1]
    p0 = proxy_params(a.base); phi_old = np.zeros(M)
    for f in np.unique(fid):
        idx = np.nonzero(fid == f)[0]; phi_old[idx] = proxy_energy(p0, pts[idx], fpos[f], fnum[f] == 1, fcell[f])
    lowE = dU < ref + 10 * kT
    print(f"fit: {int(tr.sum())} training probes (dU < ref + {a.cap:g} kT), weight scale {a.wT:g} kT, r_lo OH {r_lo[0]:.3f} HH {r_lo[1]:.3f} A, {bases[0].n + bases[1].n} coefficients\n")
    print(f"std(dU - phi) in kT:                         {'all probes':>11} {'selected':>9} {'dU<ref+10kT':>12}   Kish (selected)   overlap probes with phi too low by >5 kT")
    res = {}

    def sd(g, m): return float(np.std(g[m])) if m.sum() > 2 else float("nan")
    for name, msk in (("training probes", trn), ("held-out frames (training sources)", test_frame), ("held-out sources", hold)):
        if msk.sum() < 3: continue
        for lab, phi in (("old (base fit)", phi_old), ("new", phi_new)):
            g = (dU - phi) / kT; s_sel = msk & (kind == 1)
            kk = kish(g[s_sel]) if s_sel.sum() > 2 else float("nan")
            hi = msk & (dU >= ref + a.cap * kT); off = np.median(g[s_sel]) if s_sel.sum() else 0.0; tl = f"{np.mean(g[hi] - off > 5):.3f} of {int(hi.sum())}" if hi.sum() else "-"
            print(f"  {name:36s} {lab:15s} {sd(g, msk):6.2f} {sd(g, s_sel):9.2f} {sd(g, msk & lowE):12.2f}   {kk:.3f}             {tl}"); res[f"{name}|{lab.split()[0]}"] = dict(all=sd(g, msk), selected=sd(g, s_sel), low=sd(g, msk & lowE), kish=kk)
    print(f"\nper source (held-out probes only: held-out sources and held-out frames), selected probes, std(dU - phi) in kT:\n  {'source':56s} {'T (K)':>6} {'n':>4} {'old':>6} {'new':>6} {'Kish new':>9}")
    for s in range(len(files)):
        m = test & (src == s) & (kind == 1)
        if m.sum() < 3: continue
        g0, g1 = (dU - phi_old)[m] / kT[m], (dU - phi_new)[m] / kT[m]; print(f"  {names[s][:56]:56s} {kT[m][0] / KB:6.0f} {int(m.sum()):4d} {np.std(g0):6.2f} {np.std(g1):6.2f} {kish(g1):9.3f}" + ("  [held out]" if hold_src[s] else ""))
    fit = json.loads(json.dumps(base)); fit["rc"] = rc
    for k, nm in ((0, "OH"), (1, "HH")):
        fit["pairs"][nm] = dict(t=bases[k].t.tolist(), c=c[o[k]:o[k + 1]].tolist(), r_lo=r_lo[k], n=bases[k].n)
    fit["data"] = dict(base=os.path.basename(a.base), insertion_fit=dict(sources=names, held_out_sources=[n for n, h in zip(names, hold_src) if h], n_probes=int(M), n_train=int(tr.sum()), T_K=sorted(set(float(x / KB) for x in kT)),
                       wT=a.wT, cap_kT=a.cap, dr=a.dr, smooth=a.smooth, fold=a.fold, held_out=res, note="OH and HH refitted to insertion energies dU of the real model; OO, K, mu_O, mu_H are the base fit's (unused by the proxy)"))
    os.makedirs(os.path.dirname(os.path.abspath(a.o)) or ".", exist_ok=True); json.dump(fit, open(a.o + ".json", "w"), indent=1); print("\nwrote", a.o + ".json")
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); m = test & (kind == 1)
        if m.sum() > 3:
            for phi, lab, col in ((phi_old, "old proxy", "#9aa5b1"), (phi_new, "new proxy", "#d9480f")):
                g = (dU - phi)[m] / kT[m]; ax[0].hist(g - g.mean(), bins=40, alpha=.7, color=col, label=f"{lab}: std {g.std():.2f} kT")
            ax[0].set_xlabel("(dU - phi) / kT on held-out selected probes (mean removed)"); ax[0].set_ylabel("probes"); ax[0].legend(frameon=False, fontsize=8)
        r = np.linspace(0.3, rc, 300)
        for nm, col in (("OH", "#2b6cb0"), ("HH", "#1baf7a")):
            kd = 1 if nm == "OH" else 2; ax[1].plot(r, toy_ice.pair_energy(kd, r, fit=fit)[0], color=col, label=nm + " new"); ax[1].plot(r, toy_ice.pair_energy(kd, r, fit=base)[0], color=col, ls=":", label=nm + " old")
        ax[1].set_ylim(-3, 15); ax[1].set_xlabel("r (A)"); ax[1].set_ylabel("phi (eV)"); ax[1].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(a.o + ".png", dpi=130); print("wrote", a.o + ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label"); l.add_argument("src", nargs="+", help="pwidom *_frames.npz (cubic host cells) and / or extxyz host frames; one probes file per source in -o")
    l.add_argument("--select", nargs="*", default=[], help="only sources whose path contains any of these substrings"); l.add_argument("--shard", default=None, help="i/n: this process labels every n-th source (run n processes with i = 0 .. n-1, e.g. one per few CPU cores)")
    l.add_argument("--T", type=float, default=0.0, help="temperature of extxyz sources (npz files carry it)"); l.add_argument("--drop-last", type=int, default=0, help="extxyz: remove this many trailing atoms (e.g. an inserted ghost)")
    l.add_argument("--per-source", type=int, default=10, help="frames per source, evenly spaced in time (cost ~ per-source x (nsel + nuni + 1) single points per source)"); l.add_argument("--force", action="store_true", help="relabel existing probes files")
    l.add_argument("--seed", type=int, default=5); l.add_argument("--model", default="toyice"); l.add_argument("--device", default="cuda"); l.add_argument("--batch", type=int, default=16)
    l.add_argument("--proxy", default="none", help="starting proxy for the selection (json / toyice / none = uniform points only)"); l.add_argument("--K", type=int, default=2000)
    l.add_argument("--nsel", type=int, default=16); l.add_argument("--nuni", type=int, default=8); l.add_argument("--temper", type=float, default=2.0, help="selection at temper x T"); l.add_argument("-o", required=True, help="output directory")
    f = sub.add_parser("fit"); f.add_argument("data", nargs="+", help="probes files from label"); f.add_argument("--base", required=True, help="toy_fit json: OO, K, rc and the starting proxy"); f.add_argument("-o", required=True)
    f.add_argument("--only", nargs="*", default=[], help="keep sources whose probes file name contains any of these"); f.add_argument("--skip", nargs="*", default=[], help="drop sources whose name contains any of these")
    f.add_argument("--hold", nargs="*", default=[], help="sources (name substring) kept out of the fit and reported as held-out states: use the state of the production run")
    f.add_argument("--rc", type=float, default=0.0); f.add_argument("--dr", type=float, default=0.25); f.add_argument("--wT", type=float, default=3.0, help="weight exp(-(dU - ref) / (wT kT))"); f.add_argument("--cap", type=float, default=20.0, help="drop probes above ref + cap kT")
    f.add_argument("--smooth", type=float, default=1e-3); f.add_argument("--core-pad", type=float, default=0.05); f.add_argument("--fold", type=int, default=4, help="inside each training source every fold-th frame is held out"); f.add_argument("--plot", action="store_true")
    a = ap.parse_args(); {"label": cmd_label, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
