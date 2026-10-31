"""Refit the insertion proxy of pwidom.py (OH and HH spline pairs) to the REAL model's insertion energies at probe points, instead of force-matching DFT frames.

Why. pwidom.py is unbiased for any proxy phi; the proxy only sets the variance (std of dU - phi, Kish fraction). toy_fit.py fits forces / energies / stress of DFT
frames; what pwidom needs is dU(x) = U(frame + H at x) - U(frame) of GRACE where the Rosenbluth step looks (low phi). For a pair proxy dU(x) = c + sum_j phi_Hj(|x - r_j|)
is LINEAR in the spline coefficients: one weighted least-squares solve, no forces. The O-O pair and the core / cutoff of the base fit are kept.

usage (repo root; label needs the real model and a GPU, fit is numpy only):
  1. host frames that are NOT the ones of the production estimate (another seed or state point):
       python mlp_pipeline/tdi/pwidom.py run --like runs/tdi/<tag>_stage0.json --model grace-ft:... --frames-only --seed 11 --tag fit_ice --out runs/pwidom_fit
  2. label probe points with the real model (the host is rebuilt from the same options as pwidom.py, for the cell):
       python mlp_pipeline/tdi/proxy_fit.py label runs/pwidom_fit/fit_ice_frames.npz --like runs/tdi/<tag>_stage0.json --proxy docs/data/toyfit_t4.json \\
           --nsel 16 --nuni 8 --temper 2 -o runs/pwidom_fit/fit_ice_probes.npz
     (several sources of the same composition: list them; extxyz sources bring their own cell, --drop-last N removes N extra atoms, e.g. the ghost of an insertion trajectory)
  3. fit and compare on held-out frames:
       python mlp_pipeline/tdi/proxy_fit.py fit runs/pwidom_fit/fit_ice_probes.npz --base docs/data/toyfit_t4.json -o runs/pwidom_fit/proxy_ice
  4. use it:  pwidom.py run ... --proxy runs/pwidom_fit/proxy_ice.json   (reusing the production frames; fix the proxy before that run and do not refit on those frames)

label draws, per frame: --nsel points by a Rosenbluth step on the starting proxy at temperature --temper x T (temper > 1 broadens it so the new proxy is also trained where
the old one did not look) and --nuni uniform points. fit weights each probe by exp(-(dU - dU_ref) / (--wT kT)) (dU_ref: 1st percentile), so the fit is accurate where the
sampler goes; the held-out report (frame blocks, never used in the fit) gives std(dU - phi) in kT for the old and the new proxy.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
KB = 8.617333262e-5


def lse(x):
    m = np.max(x); return m + np.log(np.sum(np.exp(x - m)))


def load_sources(a):
    """-> pos (F, N, 3), cell (F, 3, 3), numbers (N,) from pwidom frames npz (host key; the cell from the rebuilt host) and / or extxyz files (own cells)"""
    pos, cell, nums = [], [], None
    for f in a.src:
        if f.endswith(".npz"):
            from pwidom import apply_like, build_host
            if a.like:
                apply_like(a)
            host = build_host(a); z = np.load(f); assert abs(float(z["V"]) - host.get_volume()) < 1e-6 * host.get_volume() and int(z["natoms"]) == len(host), f"{f}: host options do not match the frames"
            x = z["host"][::a.stride]; pos += list(x); cell += [np.asarray(host.cell)] * len(x); nu = host.numbers
        else:
            from ase.io import iread
            fr = list(iread(f, index=f"::{a.stride}")); n = len(fr[0]) - a.drop_last
            pos += [r.positions[:n] for r in fr]; cell += [np.asarray(r.cell) for r in fr]; nu = fr[0].numbers[:n]
        assert nums is None or np.array_equal(nums, nu), f"{f}: other composition / order than the earlier sources: label them separately"
        nums = nu
    if a.max_frames and len(pos) > a.max_frames:
        k = np.sort(np.random.default_rng(0).choice(len(pos), a.max_frames, replace=False)); pos = [pos[i] for i in k]; cell = [cell[i] for i in k]
    return np.array(pos), np.array(cell), nums


def cmd_label(a):
    from ase import Atoms
    from pwidom import Model, proxy_params, proxy_energy
    pos, cell, nums = load_sources(a); isH = nums == 1; F, N = pos.shape[:2]; kT = KB * a.T; rng = np.random.default_rng(a.seed)
    p = proxy_params(a.proxy); model = Model(a.model, a.device, a.batch); t0 = time.time(); pts, fid, kind, phi0, S = [], [], [], [], []
    for f in range(F):
        V = abs(np.linalg.det(cell[f])); c = rng.random((a.K, 3)) @ cell[f]; phi = proxy_energy(p, c, pos[f], isH, cell[f])
        if a.nsel:
            w = -phi / (a.temper * kT); s = rng.choice(a.K, a.nsel, p=np.exp(w - lse(w))); pts += list(c[s]); phi0 += list(phi[s]); kind += [1] * a.nsel; fid += [f] * a.nsel
        if a.nuni:
            u = rng.random((a.nuni, 3)) @ cell[f]; pts += list(u); phi0 += list(proxy_energy(p, u, pos[f], isH, cell[f])); kind += [0] * a.nuni; fid += [f] * a.nuni
    fid = np.array(fid); pts = np.array(pts)
    structs = [Atoms(numbers=nums, positions=pos[f], cell=cell[f], pbc=True) for f in range(F)]
    structs += [Atoms(numbers=np.r_[nums, 1], positions=np.vstack([pos[f], x]), cell=cell[f], pbc=True) for f, x in zip(fid, pts)]
    print(f"{F} frames of {N} atoms, {len(pts)} probes ({a.nsel} selected + {a.nuni} uniform per frame): {len(structs)} real-model single points", flush=True)
    E = model.energies(structs); dU = E[F:] - E[:F][fid]
    np.savez(a.o, pos=pos, cell=cell, numbers=nums, T=a.T, pts=pts, fid=fid, kind=np.array(kind), phi_start=np.array(phi0), dU=dU, E0=E[:F], proxy=str(a.proxy), temper=a.temper)
    print(f"wrote {a.o} ({(time.time() - t0) / 60:.1f} min, {model.ncalls} model calls); dU range {dU.min():.2f} .. {np.median(dU):.2f} (median) eV, "
          f"selected probes: std(dU - phi) with the starting proxy {np.std((dU - np.array(phi0))[np.array(kind) == 1]) / kT:.2f} kT", flush=True)


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
    I, R, H = rows; ncoef = [b.n for b in bases]; off = np.r_[0, np.cumsum(ncoef)]; X = np.zeros((M, off[-1] + 1)); core = np.zeros(M)
    for k, b in enumerate(bases):
        m = H == bool(k); r = R[m]; i = I[m]; v, dv = b(r); dl = np.where(r < b.r_lo, b.r_lo - r, 0.0); val = v - dl[:, None] * dv
        for n in range(b.n):
            X[:, off[k] + n] += np.bincount(i, weights=val[:, n], minlength=M)
        core += K * np.bincount(i, weights=dl ** 2, minlength=M)
    X[:, -1] = 1.0
    return X, core


def kish(gap, kT):
    q = np.exp(-(gap - gap.min()) / kT); return float(q.sum() ** 2 / (q ** 2).sum() / len(q))


def cmd_fit(a):
    from toy_fit import Basis
    from pwidom import proxy_params, proxy_energy
    from tdi import toy_ice
    base = json.load(open(a.base)); rc = a.rc or base["rc"]; K = base["K"]
    parts = [np.load(f) for f in a.data]; kT = KB * float(parts[0]["T"])
    # all datasets are stacked; frame ids are made unique per dataset
    pos = [z["pos"] for z in parts]; cell = [z["cell"] for z in parts]; numbers = [z["numbers"] for z in parts]; off_f = np.r_[0, np.cumsum([len(x) for x in pos])]
    fid = np.concatenate([z["fid"] + o for z, o in zip(parts, off_f)]); pts = np.concatenate([z["pts"] for z in parts]); kind = np.concatenate([z["kind"] for z in parts]); dU = np.concatenate([z["dU"] for z in parts])
    ds_of_frame = np.concatenate([np.full(len(x), i) for i, x in enumerate(pos)]); fpos = np.concatenate(pos); fcell = np.concatenate(cell); M = len(pts)
    test = (fid // a.block) % a.fold == a.fold - 1
    print(f"{M} probes in {len(fpos)} frames; {int(test.sum())} probes in held-out frame blocks (block {a.block} frames, every {a.fold}th block)", flush=True)
    # distances of every probe to the O / H atoms of its frame
    I, R, H = [], [], []
    for f in np.unique(fid):
        idx = np.nonzero(fid == f)[0]; isH = numbers[ds_of_frame[f]] == 1
        i, r, h = pair_rows(pts[idx], fpos[f], isH, fcell[f], rc); I.append(idx[i]); R.append(r); H.append(h)
    I, R, H = np.concatenate(I), np.concatenate(R), np.concatenate(H)
    ref = np.percentile(dU[~test], 1); w = np.clip(np.exp(-(dU - ref) / (a.wT * kT)), 1e-4, 1.0); tr = (~test) & (dU < ref + a.cap * kT)
    rows_tr = (I, R, H)
    r_lo = []
    for k in (0, 1):   # shortest O-H / H-H distance among the probes that carry weight
        m = (H == bool(k)) & tr[I] & (w[I] > 1e-2); r_lo.append(max(0.3, float(R[m].min()) - a.core_pad if m.any() else 1.0))
    bases = [Basis(r_lo[k], rc, a.dr) for k in (0, 1)]
    X, core = design(bases, K, rows_tr, M); y = dU - core; nc = X.shape[1]
    sw = np.sqrt(w * tr)[:, None]; A = (X * sw).T @ (X * sw); b = (X * sw).T @ (y[:, None] * sw)
    o = np.r_[0, np.cumsum([bb.n for bb in bases])]
    for k in (0, 1):
        D2 = np.diff(np.eye(bases[k].n), 2, axis=0); A[o[k]:o[k + 1], o[k]:o[k + 1]] += a.smooth * (w * tr).sum() * D2.T @ D2
    c = np.linalg.solve(A, b[:, 0]); phi_new = X @ c - c[-1] + 0.0
    # starting proxy on the same probes
    p0 = proxy_params(a.base); phi_old = np.zeros(M)
    for f in np.unique(fid):
        idx = np.nonzero(fid == f)[0]; phi_old[idx] = proxy_energy(p0, pts[idx], fpos[f], numbers[ds_of_frame[f]] == 1, fcell[f])
    lowE = dU < ref + 10 * kT
    print(f"fit: {int(tr.sum())} training probes (dU < ref + {a.cap:g} kT), weight scale {a.wT:g} kT, r_lo OH {r_lo[0]:.3f} HH {r_lo[1]:.3f} A, {bases[0].n + bases[1].n} coefficients\n")
    print(f"std(dU - phi) in kT:            {'all probes':>12} {'selected':>10} {'dU<ref+10kT':>12}   Kish (selected)")
    res = {}
    for name, sel_t in (("train", ~test), ("held-out", test)):
        for lab, phi in (("old (base fit)", phi_old), ("new", phi_new)):
            s_all = sel_t; s_sel = sel_t & (kind == 1); s_low = sel_t & lowE
            def sd(m): return float(np.std((dU - phi)[m]) / kT) if m.sum() > 2 else float("nan")
            kk = kish((dU - phi)[s_sel], kT) if s_sel.sum() > 2 else float("nan")
            print(f"  {name:9s} {lab:15s}        {sd(s_all):12.2f} {sd(s_sel):10.2f} {sd(s_low):12.2f}   {kk:.3f}"); res[f"{name}_{lab.split()[0]}"] = dict(all=sd(s_all), selected=sd(s_sel), low=sd(s_low), kish=kk)
    fit = json.loads(json.dumps(base)); fit["rc"] = rc
    for k, nm in ((0, "OH"), (1, "HH")):
        fit["pairs"][nm] = dict(t=bases[k].t.tolist(), c=c[o[k]:o[k + 1]].tolist(), r_lo=r_lo[k], n=bases[k].n)
    fit["data"] = dict(base=os.path.basename(a.base), insertion_fit=dict(sources=[os.path.basename(f) for f in a.data], n_probes=int(M), n_train=int(tr.sum()), T_K=float(parts[0]["T"]), wT=a.wT, cap_kT=a.cap,
                       dr=a.dr, smooth=a.smooth, held_out=res, note="OH and HH refitted to insertion energies dU of the real model; OO, K, mu_O, mu_H are the base fit's (unused by the proxy)"))
    os.makedirs(os.path.dirname(os.path.abspath(a.o)) or ".", exist_ok=True); json.dump(fit, open(a.o + ".json", "w"), indent=1); print("\nwrote", a.o + ".json")
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for axx, phi, ttl in ((ax[0], phi_old, "old proxy"), (ax[1], phi_new, "new proxy")):
            m = test; axx.scatter(dU[m & ~lowE], (phi + np.mean((dU - phi)[test & lowE]))[m & ~lowE], s=4, color="#c3c8cf"); axx.scatter(dU[m & lowE], (phi + np.mean((dU - phi)[test & lowE]))[m & lowE], s=6, color="#2b6cb0")
            lim = [dU[m].min(), ref + 15 * kT]; axx.plot(lim, lim, color="#d9480f", lw=1); axx.set_xlim(lim); axx.set_ylim(lim); axx.set_xlabel("dU real model (eV)"); axx.set_ylabel("phi (eV, offset)"); axx.set_title(ttl + " (held-out)", fontsize=10)
        r = np.linspace(0.3, rc, 300)
        for nm, col in (("OH", "#2b6cb0"), ("HH", "#1baf7a")):
            ax[2].plot(r, toy_ice.pair_energy(1 if nm == "OH" else 2, r, fit=fit)[0], color=col, label=nm + " new"); ax[2].plot(r, toy_ice.pair_energy(1 if nm == "OH" else 2, r, fit=base)[0], color=col, ls=":", label=nm + " old")
        ax[2].set_ylim(-3, 15); ax[2].set_xlabel("r (A)"); ax[2].set_ylabel("phi (eV)"); ax[2].legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(a.o + ".png", dpi=130); print("wrote", a.o + ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label"); l.add_argument("src", nargs="+", help="pwidom *_frames.npz and / or extxyz host frames (same atoms, same order)")
    l.add_argument("--like", default=None, help="host options of the npz frames from an insertion stage0.json (as pwidom.py)")
    l.add_argument("--phase", default="fcc"); l.add_argument("--vpa", type=float, default=0.0); l.add_argument("--nrep", type=int, default=2); l.add_argument("--hper", type=float, default=2.0); l.add_argument("--natoms", type=int, default=0)
    l.add_argument("--seed", type=int, default=5); l.add_argument("--T", type=float, default=0.0); l.add_argument("--P", type=float, default=0.0); l.add_argument("--bulk-gpa", type=float, default=0.0)
    l.add_argument("--model", default="toyice"); l.add_argument("--device", default="cuda"); l.add_argument("--batch", type=int, default=16)
    l.add_argument("--proxy", default="none", help="starting proxy for the selection (json / toyice / none = uniform points only)"); l.add_argument("--K", type=int, default=2000)
    l.add_argument("--nsel", type=int, default=16); l.add_argument("--nuni", type=int, default=8); l.add_argument("--temper", type=float, default=2.0, help="selection at temper x T")
    l.add_argument("--stride", type=int, default=1); l.add_argument("--max-frames", type=int, default=0); l.add_argument("--drop-last", type=int, default=0); l.add_argument("-o", required=True)
    f = sub.add_parser("fit"); f.add_argument("data", nargs="+", help="probe npz files from label (same T)"); f.add_argument("--base", required=True, help="toy_fit json: OO, K, rc and the starting proxy"); f.add_argument("-o", required=True)
    f.add_argument("--rc", type=float, default=0.0); f.add_argument("--dr", type=float, default=0.25); f.add_argument("--wT", type=float, default=3.0, help="weight exp(-(dU - ref) / (wT kT))"); f.add_argument("--cap", type=float, default=20.0, help="drop probes above ref + cap kT")
    f.add_argument("--smooth", type=float, default=1e-3); f.add_argument("--core-pad", type=float, default=0.05); f.add_argument("--block", type=int, default=10, help="frames per held-out block"); f.add_argument("--fold", type=int, default=4, help="every fold-th block is held out")
    f.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.cmd == "label":
        if a.like is None and not (a.vpa and a.T):
            ap.error("label: --like, or --vpa and --T (npz sources); --T for extxyz")
        cmd_label(a)
    else:
        cmd_fit(a)


if __name__ == "__main__":
    main()
