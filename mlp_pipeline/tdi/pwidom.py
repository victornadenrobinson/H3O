"""Proxy-guided Widom insertion / deletion (Rosenbluth-weighted test particle with a cheap pair-potential proxy) + Bennett acceptance ratio: the excess chemical
potential of one H in a host at fixed V WITHOUT a lambda path (no soft-core intermediate states, no quadrature, no lambda MD). docs/MU_METHODS.md, route R1.

Idea. Plain Widom fails at 400-500 GPa because almost every uniform trial point overlaps an atom: the average <exp(-beta dU)> is carried by rare points that are
never drawn. Here each trial draws K candidate points uniformly (K ~ 10^3-10^4, cheap), evaluates a PROXY insertion energy phi (pair potential, e.g. the toy_fit.py
spline pairs fitted to DFT) at all of them, picks one with probability exp(-beta phi_k) / sum, and calls the real model (GRACE) only there. Exactly unbiased for ANY
proxy (the proxy only sets the variance):
  insertion (frames of the host, N H):     exp(-beta mu_ex) = < W exp(-beta (dU_s - phi_s)) >,     W = (1/K) sum_k exp(-beta phi_k)
  deletion  (frames of host + 1 H, N + 1): exp(+beta mu_ex) = < exp(+beta (dU_t - phi_t)) / W_t >,  W_t = (1/K) [exp(-beta phi_t) + sum_{k=2..K} exp(-beta phi_k)]
    (t = a real H of the frame, phi_t its proxy energy with the rest, K - 1 fresh uniform candidates in the frame without t). Both are the same "work"
    w = dU - phi - kT ln W in an extended space (configuration + candidates + selection), so the forward and reverse w satisfy Crooks and BAR combines them.
  dU = U(frame + H at x) - U(frame) with the real model; proxy 'none' (phi = 0, K = 1) is plain Widom / deletion.
mu_H at fixed V = mu_ex + kT ln((N_H + 1) / V) + 3 kT ln Lambda_H (same counting as run_real.py's insertion reference); then the same stage-0 pressure correction to
P_target as the insertion TDI (-V/(2B) [(P_full - P)^2 - (P_host - P)^2], P_host and P_full from this script's own two MD runs). The real-model calls are independent
single points: batched on the GPU (tdi/batched.py GraceBatch), --batch structures per call.

usage (repo root):
  python mlp_pipeline/tdi/pwidom.py run --model toyice --phase fcc --vpa 2.08 --T 6000 --P 450 --bulk-gpa 1400 --proxy docs/data/toyfit_t4.json --out runs/pwidom
  python mlp_pipeline/tdi/pwidom.py run --model toyice --phase H --natoms 128 --vpa 1.30 ...                     (pure-H reservoir: mu_H(fluid) = G_H + finite-size term)
  python mlp_pipeline/tdi/pwidom.py dg runs/pwidom/pw_fcc_450GPa_6000K.json runs/pwidom/pw_H_450GPa_6000K.json  dg = mu_H(ice) - G_H, same convention as toy_scan.sh
  --proxy: a toy_fit.py json (spline pairs), 'toyice' (the analytic toy pair forms, a deliberately poor proxy), or 'none' (plain Widom, K = 1)
  MD frames are cached in <out>/<tag>_frames.npz (host and host + 1 H, plus their MLP pressures): a rerun with another --proxy / --K reuses them (no MD).
  --frames-only: just the MD (e.g. run it once, then analyse with several proxies).
Output: <out>/<tag>[_<label>].json with mu_ex (Widom, deletion, BAR; 5-block errors over frames), the proxy diagnostics (std of dU - phi in kT; Kish effective
fraction of the insertion weights) and mu_H at fixed V and at P_target.
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
EV_A3_GPA = 160.21766
KB = 8.617333262e-5


def lse(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True); return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def model_fingerprint(spec):
    """short hash of the resolved Hamiltonian: the model kind plus, for grace-ft: / mace-ft: checkpoints, the file names and first 16 MB of every file (the same weights hash the same on another machine);
    toyice: the TOYICE_FIT file. Cached MD frames and pressures belong to ONE Hamiltonian: another model's frames are not an equilibrium ensemble of this one."""
    import hashlib
    kind, _, rest = spec.partition(":"); h = hashlib.sha256(kind.encode())
    if kind.endswith("-ft") and rest and os.path.exists(rest):
        root = rest if os.path.isdir(rest) else os.path.dirname(rest)
        for f in ([rest] if os.path.isfile(rest) else sorted(os.path.join(d, x) for d, _, fs in os.walk(rest) for x in fs))[:200]:
            h.update(os.path.relpath(f, root).encode()); h.update(open(f, "rb").read(1 << 24))
    elif kind == "toyice" and os.environ.get("TOYICE_FIT") and os.path.exists(os.environ["TOYICE_FIT"]):
        h.update(open(os.environ["TOYICE_FIT"], "rb").read())
    else:
        h.update(rest.encode())
    return h.hexdigest()[:16]


def build_host(a):
    from importlib import import_module
    if a.phase == "H":
        rr = import_module("tdi.run_real")
        return rr.build_h_n(a.vpa, a.natoms) if a.natoms else rr.build_h(a.vpa, a.nrep)
    return import_module("17_eos_md").build(a.phase, a.vpa, a.nrep, a.seed, a.hper)


def proxy_params(spec):
    """pair-potential parameter dict for toy_ice.ghost_energy, or None for 'none'"""
    from tdi import toy_ice
    if spec == "none":
        return None
    p = dict(toy_ice.DEFAULTS)
    if spec == "toyice":
        return p
    fit = json.load(open(spec)); p.update(fit=fit, rc=fit["rc"]); return p


def proxy_energy(p, pts, pos, isH, cell, chunk=512):
    """proxy insertion energy (eV) of each point in pts (m x 3) into the frame pos (isH: bool per atom); 0 for proxy 'none'"""
    if p is None:
        return np.zeros(len(pts))
    from tdi.toy_ice import ghost_energy
    return np.concatenate([ghost_energy(pts[i:i + chunk], pos, isH, cell, p) for i in range(0, len(pts), chunk)])


class Model:
    """real-model energies of many structures: GraceBatch (one model call per --batch structures) for a tensorpotential calculator, else one call each"""

    def __init__(self, spec, device, batch):
        from importlib import import_module
        self.calc = import_module("04_benchmark").get_calc_spec(spec, device); self.batch = batch; self.gb = None; self.ncalls = 0
        if type(self.calc).__name__ == "TPCalculator" and batch > 1:
            from tdi.batched import GraceBatch
            self.gb = GraceBatch(self.calc)

    def energies(self, structs):
        out = []
        for i in range(0, len(structs), self.batch):
            part = structs[i:i + self.batch]; self.ncalls += 1
            if self.gb is not None:
                if not getattr(self, "checked", False):   # batched vs single calls once, on the first structures (raises above tolerance: rerun with --batch 1)
                    print(self.gb.selfcheck([x.copy() for x in part[:4]]), flush=True); self.checked = True
                out += [r[0] for r in self.gb([0], [(part, False)])[0]]
            else:
                for at in part:
                    at.calc = self.calc; out.append(float(at.get_potential_energy())); at.calc = None
        return np.array(out)


def md_frames(model, at0, a, seed, label):
    """free NVT Langevin MD with the real model (no springs): positions every --every-fs after --equil-ps, and the MLP pressure (incl. kinetic) at each frame"""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    at = at0.copy(); at.calc = model.calc; rng = np.random.default_rng(seed); t0 = time.time()
    MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=rng)
    dyn = Langevin(at, a.dt * units.fs, temperature_K=a.T, friction=0.01, rng=rng)
    dyn.run(int(a.equil_ps * 1000 / a.dt)); every = max(1, int(round(a.every_fs / a.dt))); nfr = int(a.ps * 1000 / a.every_fs)
    pos, P = [], []
    for k in range(nfr):
        dyn.run(every); pos.append(at.positions.copy())
        P.append(-at.get_stress()[:3].mean() * EV_A3_GPA + len(at) * units.kB * a.T / at.get_volume() * EV_A3_GPA)
        if (k + 1) % max(1, nfr // 5) == 0:
            print(f"  MD {label}: {k + 1}/{nfr} frames, P so far {np.mean(P):.1f} GPa, {(time.time() - t0) / 60:.1f} min", flush=True)
    return np.array(pos), np.array(P)


def o_msd(pos, numbers):
    """O mean-square displacement (A^2) from the first frame, relative to the O centre of mass (the flowing H drags the whole O lattice), in 10 time slices; nan without O"""
    o = numbers == 8
    if not o.any():
        return []
    d = pos[:, o] - pos[0, o]; d -= d.mean(1, keepdims=True); m = (d ** 2).sum(-1).mean(1)
    return [float(x) for x in m[np.linspace(0, len(m) - 1, 10).astype(int)]]


def blocks(x, nb=5):
    return np.array_split(np.asarray(x), nb)


def berr(vals):
    v = np.asarray(vals, float); return float(np.std(v, ddof=1) / np.sqrt(len(v)))


def bar(wf, wr, kT):
    """BAR for dF = F(N+1) - F(N): wf forward works (insertion), wr = minus the reverse works (deletion), as toy_validate.bar"""
    from tdi.toy_validate import bar as _bar
    return _bar(np.asarray(wf), np.asarray(wr), kT)


def frame_mean_P(P, nb=5):
    return float(np.mean(P)), berr([b.mean() for b in blocks(P, nb)])


def apply_like(a):
    """--like <insertion stage0.json>: host (phase, size, H per O), volume, T, P, bulk modulus and model of an existing run_real.py insertion run (same host as the TDI)"""
    st = json.load(open(a.like)); g = st["args"]; hp = g.get("host_phase", "fcc")
    assert st.get("phase") == "insert", f"{a.like}: not an insertion stage0 json"
    a.phase = "H" if hp == "H" else hp; a.vpa = st["vpa"]; a.T = st["T"]; a.P = st["P_target"]; a.bulk_gpa = a.bulk_gpa or st.get("B_GPa", 0.0); a.hper = g.get("hper", 2.0)
    a.natoms = g.get("natoms") or 0; a.nrep = g.get("nrep") or {"bcc": 4, "H": 5}.get(hp, 3)
    if a.model == "toyice" and g.get("model"): a.model = g["model"]
    if not a.bulk_gpa:
        raise SystemExit(f"{a.like} has no B_GPa (run made before the pressure correction): give --bulk-gpa")
    print(f"--like {a.like}: {a.phase} host, nrep {a.nrep}, natoms {a.natoms}, H per O {a.hper:g}, vpa {a.vpa:.5f}, {a.T:g} K, {a.P:g} GPa, B {a.bulk_gpa:.0f} GPa, model {a.model}", flush=True)


def cmd_run(a):
    if a.like:
        apply_like(a)
    assert a.vpa and a.T and a.P, "--vpa, --T and --P (or --like)"
    from ase import Atoms
    from tdi import refs
    kT = KB * a.T; tag = a.tag or f"pw_{a.phase}{'' if a.hper == 2.0 or a.phase == 'H' else f'_h{a.hper:g}o'}_{a.P:g}GPa_{a.T:.0f}K"
    os.makedirs(a.out, exist_ok=True); ff = os.path.join(a.out, f"{tag}_frames.npz")
    host = build_host(a); cell = np.asarray(host.cell); V = host.get_volume(); isH = host.numbers == 1; NH = int(isH.sum())
    model = Model(a.model, a.device, a.batch); rng = np.random.default_rng(a.seed + 17); mfp = model_fingerprint(a.model)
    if os.path.exists(ff):
        z = np.load(ff); assert abs(float(z["V"]) - V) < 1e-6 * V and int(z["natoms"]) == len(host) and float(z["T"]) == a.T, f"{ff}: other V / size / T: use another --out or --tag"
        if "model_fp" in z.files:
            if str(z["model_fp"]) != mfp:
                raise SystemExit(f"{ff}: frames and pressures were made with another Hamiltonian (fingerprint {z['model_fp']} vs {mfp} for {a.model}): they are not an equilibrium ensemble of this model; use another --out or --tag")
        else:
            print(f"WARNING: {ff} has no model fingerprint (made before the check): the model is NOT verified", flush=True)
        fh, Ph, fp, Pf = z["host"], z["P_host"], z["full"], z["P_full"]; print(f"frames from {ff}: host {len(fh)}, host + 1 H {len(fp)}")
    else:
        from tdi.run_real import add_ghosts
        full0 = add_ghosts(host, 1, a.seed + 7)
        fh, Ph = md_frames(model, host, a, a.seed, "host"); fp, Pf = md_frames(model, full0, a, a.seed + 1, "host + 1 H")
        np.savez(ff, host=fh, P_host=Ph, full=fp, P_full=Pf, V=V, natoms=len(host), T=a.T, numbers=host.numbers, model_fp=mfp)
        print("wrote", ff, flush=True)
    msd = {k: o_msd(x, nu) for k, x, nu in (("host", fh, host.numbers), ("full", fp, np.r_[host.numbers, 1]))}
    melted = any(m and m[-1] > 1.0 for m in msd.values())   # solid O: < ~0.3 A^2 at 3000-6000 K; a melted crystal diffuses several A^2 per ps
    if msd["host"]:
        print(f"O MSD (A^2, relative to the O centre of mass) over the host run: {' '.join(f'{x:.2f}' for x in msd['host'])}; host + 1 H: {' '.join(f'{x:.2f}' for x in msd['full'])}"
              + ("  MELTED: the O lattice diffuses, this is not the crystal (results are written but flagged)" if melted else "  (solid)"), flush=True)
    if a.frames_only:
        return
    p = proxy_params(a.proxy); K = 1 if p is None else a.K; t0 = time.time()
    isH1 = np.r_[isH, True]; numbers1 = np.r_[host.numbers, 1]
    # ---- insertion: per host frame, a.sel selections from one set of K uniform candidates (each selection is an unbiased trial). The proxy work is done frame
    # by frame; the real-model structures of all frames are collected and evaluated together (batched calls)
    sfr = list(range(0, len(fh), a.stride)); S0, S1, ins = [], [], []
    for f in sfr:
        x = fh[f]; c = rng.random((K, 3)) @ cell; phi = proxy_energy(p, c, x, isH, cell)
        lnW = lse(-phi / kT) - np.log(K); pr = np.exp(-phi / kT - lse(-phi / kT)); s = rng.choice(K, a.sel, p=pr)
        S0.append(Atoms(numbers=host.numbers, positions=x, cell=cell, pbc=True))
        S1 += [Atoms(numbers=numbers1, positions=np.vstack([x, c[j]]), cell=cell, pbc=True) for j in s]; ins.append((phi[s], lnW))
    # ---- deletion: per full frame, a.ndel randomly chosen H (incl. the added one: all H are identical)
    iH1 = np.where(isH1)[0]; dfr = list(range(0, len(fp), a.stride)); D0, D1, dele = [], [], []
    for f in dfr:
        x = fp[f]; ts = rng.choice(iH1, min(a.ndel, len(iH1)), replace=False); D0.append(Atoms(numbers=numbers1, positions=x, cell=cell, pbc=True)); ws = []
        for t in ts:
            keep = np.ones(len(x), bool); keep[t] = False; xr = x[keep]; hr = isH1[keep]
            D1.append(Atoms(numbers=numbers1[keep], positions=xr, cell=cell, pbc=True))
            phi_t = proxy_energy(p, x[t:t + 1], xr, hr, cell)[0]
            if K > 1:
                c = rng.random((K - 1, 3)) @ cell; phik = np.r_[phi_t, proxy_energy(p, c, xr, hr, cell)]
            else:
                phik = np.array([phi_t])
            ws.append((phi_t, lse(-phik / kT) - np.log(K)))
        dele.append(ws)
    print(f"proxy {a.proxy} K {K}: {len(S1)} insertion and {len(D1)} deletion trials prepared ({(time.time() - t0) / 60:.1f} min); real-model single points: {len(S0) + len(S1) + len(D0) + len(D1)}", flush=True)
    E = model.energies(S0 + S1 + D0 + D1); n0, n1, m0 = len(S0), len(S1), len(D0)
    e0, e1, f0, f1 = E[:n0], E[n0:n0 + n1].reshape(n0, a.sel), E[n0 + n1:n0 + n1 + m0], E[n0 + n1 + m0:]
    wf_frame, gap_ins = [], []
    for k, (phis, lnW) in enumerate(ins):
        dU = e1[k] - e0[k]; wf_frame.append(dU - phis - kT * lnW); gap_ins += list(dU - phis)
    wr_frame, gap_del = [], []; j = 0
    for k, ws in enumerate(dele):
        er = f1[j:j + len(ws)]; j += len(ws)
        wr_frame.append(np.array([f0[k] - e - ph - kT * lw for e, (ph, lw) in zip(er, ws)])); gap_del += [f0[k] - e - ph for e, (ph, _) in zip(er, ws)]
    wf, wr = np.concatenate(wf_frame), np.concatenate(wr_frame)
    mu_w = -kT * (lse(-wf / kT) - np.log(len(wf))); mu_d = kT * (lse(wr / kT) - np.log(len(wr))); mu_b = bar(wf, -wr, kT)
    bf, br = [np.concatenate(b) for b in blocks(wf_frame)], [np.concatenate(b) for b in blocks(wr_frame)]
    ew = berr([-kT * (lse(-b / kT) - np.log(len(b))) for b in bf]); ed = berr([kT * (lse(b / kT) - np.log(len(b))) for b in br])
    eb = berr([bar(x, -y, kT) for x, y in zip(bf, br)])
    q = np.exp(-(wf - wf.min()) / kT); kish = float(q.sum() ** 2 / (q ** 2).sum() / len(q))   # effective fraction of the insertion samples
    ideal = kT * np.log((NH + 1) / V) + 3 * kT * np.log(refs.thermal_wavelength(float(host.get_masses()[isH][0]), a.T))
    (ph, eph), (pf, epf) = frame_mean_P(Ph), frame_mean_P(Pf)
    B = a.bulk_gpa; dGP = -V / (2 * B) * ((pf - a.P) ** 2 - (ph - a.P) ** 2) / EV_A3_GPA if B else 0.0
    dGPe = V / B * float(np.hypot((pf - a.P) * epf, (ph - a.P) * eph)) / EV_A3_GPA if B else 0.0
    r = dict(tag=tag, model=a.model, proxy=a.proxy, K=K, sel=a.sel, ndel=a.ndel, phase=a.phase, hper=a.hper, T=a.T, P_target=a.P, vpa=a.vpa, V=V, N_H=NH, natoms=len(host),
             n_frames_host=len(sfr), n_frames_full=len(wr_frame), n_ins=len(wf), n_del=len(wr), model_calls=model.ncalls,
             mu_ex_widom_meV=mu_w * 1e3, mu_ex_widom_err_meV=ew * 1e3, mu_ex_delete_meV=mu_d * 1e3, mu_ex_delete_err_meV=ed * 1e3, mu_ex_bar_meV=mu_b * 1e3, mu_ex_bar_err_meV=eb * 1e3,
             proxy_gap_std_kT_ins=float(np.std(gap_ins) / kT), proxy_gap_std_kT_del=float(np.std(gap_del) / kT), kish_frac_ins=kish,
             ideal_meV=ideal * 1e3, P_host_GPa=ph, P_host_err_GPa=eph, P_full_GPa=pf, P_full_err_GPa=epf, B_GPa=B, dG_P_meV=dGP * 1e3, dG_P_err_meV=dGPe * 1e3,
             msd_O_host=msd["host"], msd_O_full=msd["full"], melted=melted,
             model_fp=mfp, pressure_corrected=bool(B),
             mu_fixedV_meV=(mu_b + ideal) * 1e3, mu_P_meV=(mu_b + ideal + dGP) * 1e3, mu_P_err_meV=float(np.hypot(eb, dGPe)) * 1e3)
    if not B:
        print("WARNING: no --bulk-gpa: mu_P_meV is the FIXED-VOLUME value (no pressure correction to P_target); the json says pressure_corrected = false", flush=True)
    lab = a.label or (os.path.basename(a.proxy).split(".")[0] if a.proxy not in ("none", "toyice") else a.proxy)
    fo = os.path.join(a.out, f"{tag}_{lab}_K{K}.json"); json.dump(r, open(fo, "w"), indent=1)
    print(f"{tag} proxy {lab} K {K}: {len(wf)} insertions ({len(sfr)} frames), {len(wr)} deletions, {model.ncalls} model calls, {(time.time() - t0) / 60:.1f} min")
    print(f"  mu_ex: Widom {mu_w * 1e3:8.1f} +- {ew * 1e3:5.1f}   deletion {mu_d * 1e3:8.1f} +- {ed * 1e3:5.1f}   BAR {mu_b * 1e3:8.1f} +- {eb * 1e3:5.1f} meV")
    print(f"  proxy quality: std(dU - phi) {r['proxy_gap_std_kT_ins']:.2f} kT (insertion) {r['proxy_gap_std_kT_del']:.2f} kT (deletion); Kish effective fraction of insertions {kish:.3f}")
    print(f"  P host {ph:.1f} +- {eph:.1f}, host + 1 H {pf:.1f} +- {epf:.1f} GPa; mu_H fixed V {r['mu_fixedV_meV']:.1f}, at {a.P:g} GPa {r['mu_P_meV']:.1f} +- {r['mu_P_err_meV']:.1f} meV  -> {fo}", flush=True)


def cmd_dg(a):
    """dg = mu_H(ice) - G_H(pure H) from two run jsons (ice host, pure-H host), both at P_target; G_H = mu_H(pure H, N -> N+1) - fs with the ideal-gas finite-size
    term fs = kT [ln(N+1) - ln(N!)/N - 1] (toy_validate T3 / toy_scan.sh convention)"""
    from math import lgamma, log
    w, h = (json.load(open(f)) for f in (a.ice, a.hfluid)); kT = KB * w["T"] * 1e3; N = h["N_H"]
    fs = kT * (log(N + 1) - lgamma(N + 1) / N - 1)
    for k in ("T", "P_target"):
        if abs(w[k] - h[k]) > 1e-6 * max(1.0, abs(w[k])):
            raise SystemExit(f"dg: the two runs have a different {k} ({w[k]:g} vs {h[k]:g}): not the same state")
    if w.get("model_fp") and h.get("model_fp"):
        if w["model_fp"] != h["model_fp"]:
            raise SystemExit(f"dg: the two runs used different Hamiltonians (fingerprints {w['model_fp']} vs {h['model_fp']})")
    else:
        print("WARNING: dg: a run has no model fingerprint (made before the check): the model is not verified", flush=True)
    if not (w.get("pressure_corrected", bool(w.get("B_GPa"))) and h.get("pressure_corrected", bool(h.get("B_GPa")))):
        print("WARNING: dg: a run has NO pressure correction (no bulk modulus): that mu_H is the fixed-volume value, dg is not at the target pressure", flush=True)
    dg = w["mu_P_meV"] - (h["mu_P_meV"] - fs); e = float(np.hypot(w["mu_P_err_meV"], h["mu_P_err_meV"]))
    if w.get("melted"):
        print(f"WARNING: {a.ice}: the crystal MELTED in the free MD (O MSD {w['msd_O_host'][-1]:.2f} / {w['msd_O_full'][-1]:.2f} A^2): dg is not that of the solid")
    print(f"dg = mu_H(ice) - G_H = {dg:.1f} +- {e:.1f} meV at {w['P_target']:g} GPa {w['T']:.0f} K (ice {w['mu_P_meV']:.1f}, pure H {h['mu_P_meV']:.1f}, fs {fs:.1f}; proxies {w['proxy']} / {h['proxy']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="toyice"); r.add_argument("--device", default="cuda"); r.add_argument("--phase", default="fcc", help="fcc / bcc (O lattice, 17_eos_md.build) or H")
    r.add_argument("--like", default=None, help="take host, volume, T, P, B and model from an insertion run's <tag>_stage0.json"); r.add_argument("--vpa", type=float, default=0.0); r.add_argument("--nrep", type=int, default=2); r.add_argument("--hper", type=float, default=2.0); r.add_argument("--natoms", type=int, default=0, help="pure H: atoms (n^3, 2n^3 or 4n^3; 0 = bcc 2 nrep^3)")
    r.add_argument("--T", type=float, default=0.0); r.add_argument("--P", type=float, default=0.0, help="target pressure (GPa) for the pressure correction"); r.add_argument("--bulk-gpa", type=float, default=0.0)
    r.add_argument("--ps", type=float, default=4.0); r.add_argument("--equil-ps", type=float, default=0.5); r.add_argument("--dt", type=float, default=0.5); r.add_argument("--every-fs", type=float, default=20.0)
    r.add_argument("--proxy", default="none"); r.add_argument("--K", type=int, default=2000, help="uniform candidates per trial (proxy only)")
    r.add_argument("--sel", type=int, default=4, help="real-model insertions per host frame"); r.add_argument("--ndel", type=int, default=4, help="real-model deletions per host + 1 H frame")
    r.add_argument("--stride", type=int, default=1, help="use every stride-th cached frame"); r.add_argument("--batch", type=int, default=16, help="structures per batched model call (GRACE)")
    r.add_argument("--seed", type=int, default=3); r.add_argument("--out", default="runs/pwidom"); r.add_argument("--tag", default=None); r.add_argument("--label", default=None)
    r.add_argument("--frames-only", action="store_true")
    d = sub.add_parser("dg"); d.add_argument("ice"); d.add_argument("hfluid")
    a = ap.parse_args(); {"run": cmd_run, "dg": cmd_dg}[a.cmd](a)


if __name__ == "__main__":
    main()
