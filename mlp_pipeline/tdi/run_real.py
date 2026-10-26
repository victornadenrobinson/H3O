"""Thermodynamic integration for crystalline H2O (O on bcc/fcc sites, mobile protons) with a GRACE model: one phase, one volume per job.

Path ('direct', Wilson et al. 2013 type):  U(lam) = (1-lam) E_Einstein(O) + lam U_MLP,   reference = Einstein crystal for O + ideal gas for H.
  F = F_ref + int_0^1 <U_MLP - E_Einstein>_lam dlam      (+ kinetic terms; classical nuclei)       G = F + P_target V
Modes:  (default) all in one process: lambda=1 run, then all nodes in chunks, G printed vs simulation time;
        --stage0   only the lambda=1 run (Einstein constant, MLP pressure, structure, reference free energy) -> <outdir>/<tag>_stage0.json
        --node K   only node K (reads stage0 json): many of these can share one GPU (see young_tdi_par.sh); tdi/collect.py integrates them and prints G(t).
Steps: (1) lam=1 run: equilibration check, Einstein constant k = 3 kT / <|u|^2> of the O atoms, MLP pressure and O-structure label at this volume;
(2) the lambda nodes (Gauss-Legendre in lam^power) advance in chunks of --chunk-ps; every chunk prints G (meV/atom) and writes <outdir>/tdi_<tag>.json.
G uses P_target*V (not the MLP pressure at V): this is the Gibbs energy at P_target to second order in (P_MLP - P_target); the MLP pressure is printed as a check.
usage (GPU, repo root):  GRACE_UQ=1 python mlp_pipeline/tdi/run_real.py grace-ft:runs/grace/ft_m2 --phase bcc --P 400 --T 3000 --from-dft inventory_binned.csv
"""
import argparse, json, os, sys, time
from importlib import import_module
from math import lgamma
import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tdi import refs, engine, analyze

EV_A3_GPA = 160.21766


class LambdaRun:
    def __init__(self, atoms0, components, lam, T, idxO, dt, tau, seed, every, equil_steps):
        self.lam, self.T, self.every, self.equil, self.series, self.tk, self.snaps = lam, T, every, equil_steps, [], [], []
        self.eful, self.espr = [], []
        self.calc = engine.TDICalculator(components, {"E": 1 - lam, "full": lam}, needed=("full", "E"))
        self.at = atoms0.copy(); self.at.calc = self.calc; self.idxO = idxO; self.sites0 = atoms0.positions[idxO].copy()
        self.at.set_constraint(engine.FixGroupCom(idxO))
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(self.at, temperature_K=T, rng=rng)
        self.at.set_momenta(self.at.get_momenta(), apply_constraint=True)
        self.dyn = Langevin(self.at, dt * units.fs, temperature_K=T, friction=1.0 / (tau * units.fs), fixcm=False, rng=rng)
        self.ndof = self.at.get_number_of_degrees_of_freedom(); self.started = False; self.steps = 0; self.t0 = time.time()

    def _rec(self):
        self.at.get_forces()
        p = self.calc.parts
        self.series.append(p["full"] - p["E"]); self.eful.append(p["full"]); self.espr.append(p["E"])
        self.tk.append(2 * self.at.get_kinetic_energy() / (self.ndof * units.kB))
        if len(self.series) % 20 == 0:
            self.snaps.append(self.at.get_positions().copy())

    def start(self):
        """equilibration (nothing recorded), then attach the recorder"""
        self.dyn.run(self.equil); self.dyn.attach(self._rec, interval=self.every); self.started = True

    def advance(self, n):
        if not self.started:
            self.start()
        self.dyn.run(n); self.steps += n

    def diag(self, last=None):
        """one-line health report over the last `last` samples (default all): temperatures, energies per atom, closest H-H / H-O contacts, O displacement"""
        sl = slice(-last, None) if last else slice(None)
        at = self.at; N = len(at); H = at.numbers == 1
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list("ijd", at, 1.2)
        hh = d[H[i] & H[j]].min() if (H[i] & H[j]).any() else float("nan"); ho = d[(H[i] != H[j])].min() if (H[i] != H[j]).any() else float("nan")
        u = find_mic(at.positions[self.idxO] - self.sites0, at.cell, True)[0]; u -= u.mean(0)
        return (f"T_kin {np.mean(self.tk[sl]):6.0f} K  <dU/dl> {np.mean(self.series[sl]):10.3f} eV  U_MLP/atom {np.mean(self.eful[sl]) / N:8.4f} eV  "
                f"E_spring/atom {np.mean(self.espr[sl]) / N * 1e3:7.2f} meV  min d(H-H) {hh:.2f} d(H-O) {ho:.2f} A  |u_O| now rms {np.sqrt((u ** 2).sum(-1).mean()):.3f} max {np.linalg.norm(u, axis=1).max():.3f} A")


def stats(series, nb=5):
    s = np.asarray(series, float)
    if len(s) < 5 * nb:
        return float("nan"), float("nan")
    return analyze.block_stats(s, nblocks=nb)


def setup(a):
    eos = import_module("17_eos_md")
    a.nrep = a.nrep or eos.CELL[a.phase][2]
    vpa = a.vpa or eos.dft_vpa(a.from_dft, a.phase, a.T, [a.P])[0]
    tag = a.tag or f"{a.phase}_{a.P:.0f}GPa_{a.T:.0f}K"
    os.makedirs(a.outdir, exist_ok=True)
    at0 = eos.build(a.phase, vpa, a.nrep, a.seed)
    iO = np.where(at0.numbers == 8)[0]; iH = np.where(at0.numbers == 1)[0]
    ctx = dict(a=a, tag=tag, vpa=vpa, at0=at0, N=len(at0), iO=iO, iH=iH, nO=len(iO), nH=len(iH), V=at0.get_volume(), sitesO=at0.positions[iO].copy(), kt=refs.kT(a.T))
    ctx["calc"] = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    ctx["full"] = engine.full_component(ctx["calc"])
    ctx["steps"] = lambda ps: int(round(ps * 1000 / a.dt))
    ctx["lam"], ctx["w"] = analyze.gauss_legendre(a.nodes, a.power)
    print(f"{tag}: N {ctx['N']} (O {ctx['nO']}, H {ctx['nH']}), V/atom {vpa:.4f} A^3 (V {ctx['V']:.1f}), kT {ctx['kt'] * 1e3:.1f} meV, dt {a.dt} fs, Langevin tau {a.tau} fs", flush=True)
    print("quadrature: Gauss-Legendre in x on (0,1), lambda = x^%g, weight_i = w_i * %g x_i^%g;  %d nodes:" % (a.power, a.power, a.power - 1, a.nodes))
    for k, (l, ww) in enumerate(zip(ctx["lam"], ctx["w"])):
        print(f"   node {k}: lambda {l:.3e}  weight {ww:.3e}")
    print(f"   sum of weights {ctx['w'].sum():.6f} (must be 1)", flush=True)
    return ctx


def stage0(c):
    a = c["a"]; at0, iO, N, V, kt = c["at0"], c["iO"], c["N"], c["V"], c["kt"]; t0 = time.time()
    from ase.neighborlist import neighbor_list
    d1_0 = float(neighbor_list("d", at0[iO], 0.6 * V ** (1 / 3)).min())
    r1 = LambdaRun(at0, {"full": c["full"], "E": engine.einstein_component(c["sitesO"], iO, 1.0)}, 1.0, a.T, iO, a.dt, a.tau, a.seed, a.every, c["steps"](a.equil_ps))
    r1.start(); print(f"lambda=1 equilibrated ({a.equil_ps} ps), {(time.time() - t0) / 60:.1f} min   {r1.diag.__doc__ and ''}", flush=True)
    nq = int(round(a.ps1 / 0.25))
    for q in range(nq):
        r1.advance(c["steps"](0.25))
        uu = np.array([find_mic(p[iO] - c["sitesO"], at0.cell, True)[0] for p in r1.snaps]); uu -= uu.mean(1, keepdims=True)
        print(f"  lambda=1 {0.25 * (q + 1):.2f} ps  {r1.diag(last=c['steps'](0.25) // a.every)}  Lindemann {np.sqrt((uu ** 2).sum(-1).mean()) / d1_0:.3f}  <|u|^2>_O {(uu ** 2).sum(-1).mean():.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    u = np.array([find_mic(p[iO] - c["sitesO"], at0.cell, True)[0] for p in r1.snaps]); u -= u.mean(1, keepdims=True)
    msd = float((u ** 2).sum(-1).mean()); kE = 3 * kt / msd
    lab, c1, c2, d1 = import_module("09_label_phase").o_structure(c["sitesO"] + u.mean(0), at0.cell.array)
    sig = []
    for p in r1.snaps[len(r1.snaps) // 2:][::2]:
        b = at0.copy(); b.positions = p; b.calc = c["calc"]; sig.append(-b.get_stress(voigt=True)[:3].mean())
    P_mlp = float(np.mean(sig)) * EV_A3_GPA + N * kt / V * EV_A3_GPA
    mO, mH = float(at0.get_masses()[iO[0]]), float(at0.get_masses()[c["iH"][0]])
    f_ref_cons = refs.f_einstein_cons(c["nO"], kE, a.T) + refs.f_ideal_gas(c["nH"], V, a.T)
    f_kin = refs.kinetic_free_energy([(c["nO"], mO, False), (c["nH"], mH, True)], a.T)
    PV = a.P / EV_A3_GPA * V
    print(f"lambda=1 summary ({a.ps1} ps, {(time.time() - t0) / 60:.1f} min): T_kin {np.mean(r1.tk):.0f} K, <|u|^2>_O {msd:.4f} A^2 (Lindemann {np.sqrt(msd) / d1:.3f}), Einstein k = {kE:.3f} eV/A^2, "
          f"O structure {lab} (CN1 {c1:.1f}, CN2 {c2:.1f}), MLP pressure {P_mlp:.1f} GPa (target {a.P:.0f})", flush=True)
    print(f"reference pieces (eV, whole cell): F_Einstein(O)+F_ideal(H) config {f_ref_cons:.3f}  -kT ln V {-kt * np.log(V):.3f}  kinetic {f_kin:.3f}  P_target V {PV:.3f}", flush=True)
    if lab != a.phase:
        print(f"WARNING: O sublattice is '{lab}', not {a.phase}: the reference lattice does not describe this system")
    d = dict(tag=c["tag"], phase=a.phase, T=a.T, P_target=a.P, vpa=c["vpa"], V=V, N=N, nO=c["nO"], nH=c["nH"], kE=kE, P_mlp=P_mlp, O_struct=lab, lindemann=float(np.sqrt(msd) / d1),
             f_ref_cons_eV=f_ref_cons, f_kin_eV=f_kin, PV_eV=PV, args=vars(a))
    json.dump(d, open(os.path.join(a.outdir, f"{c['tag']}_stage0.json"), "w"), indent=1)
    return d


def run_node(c, st, k):
    """node k alone; the series of <U_MLP - E> is saved to <outdir>/<tag>_node<k>.npy after every chunk"""
    a = c["a"]; l = float(c["lam"][k]); t0 = time.time()
    comps = {"full": c["full"], "E": engine.einstein_component(c["sitesO"], c["iO"], st["kE"])}
    r = LambdaRun(c["at0"], comps, l, a.T, c["iO"], a.dt, a.tau, a.seed + 1 + k, a.every, c["steps"](a.equil_ps))
    r.start(); print(f"node {k} lam {l:.3e} equilibrated ({a.equil_ps} ps): {r.diag()} {(time.time() - t0) / 60:.1f} min", flush=True)
    f = os.path.join(a.outdir, f"{c['tag']}_node{k}.npy")
    nchunk = int(round(a.ps / a.chunk_ps)); n_per = c["steps"](a.chunk_ps) // a.every
    for ch in range(1, nchunk + 1):
        r.advance(c["steps"](a.chunk_ps))
        np.save(f, np.array(r.series))
        m, e = stats(r.series)
        print(f"node {k} lam {l:.3e} t {ch * a.chunk_ps:5.2f} ps: last chunk <dU/dl> {np.mean(r.series[-n_per:]):10.3f}  all {m:10.3f} +- {e:7.3f} eV | {r.diag(last=n_per)} | {(time.time() - t0) / 60:.1f} min", flush=True)


def G_from_nodes(st, lam, w, series_list, N):
    ms = [stats(s) for s in series_list]; m = np.array([x[0] for x in ms]); s = np.array([x[1] for x in ms])
    dF, dFe = analyze.integrate(m, s, w)
    kt = refs.kT(st["T"])
    return (st["f_ref_cons_eV"] + dF - kt * np.log(st["V"]) + st["f_kin_eV"] + st["PV_eV"]) / N * 1e3, dFe / N * 1e3, dF / N * 1e3, m, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--phase", choices=["bcc", "fcc"], required=True)
    ap.add_argument("--T", type=float, default=3000); ap.add_argument("--P", type=float, default=400, help="target pressure (GPa)")
    ap.add_argument("--from-dft", default=None); ap.add_argument("--vpa", type=float, default=None, help="A^3/atom (instead of --from-dft)")
    ap.add_argument("--nrep", type=int, default=None); ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--power", type=float, default=4.0)
    ap.add_argument("--ps", type=float, default=4.0, help="production per lambda node"); ap.add_argument("--chunk-ps", type=float, default=0.5)
    ap.add_argument("--equil-ps", type=float, default=1.0); ap.add_argument("--ps1", type=float, default=3.0, help="length of the lambda=1 run")
    ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--tau", type=float, default=100.0); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default="runs/tdi"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage0", action="store_true"); ap.add_argument("--node", type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    c = setup(a)
    if a.stage0:
        stage0(c); sys.exit(0)
    if a.node is not None:
        st = json.load(open(os.path.join(a.outdir, f"{c['tag']}_stage0.json")))
        run_node(c, st, a.node); sys.exit(0)
    # all in one process
    st = stage0(c); t0 = time.time(); lam, w, N = c["lam"], c["w"], c["N"]
    comps = {"full": c["full"], "E": engine.einstein_component(c["sitesO"], c["iO"], st["kE"])}
    runs = [LambdaRun(c["at0"], comps, float(l), a.T, c["iO"], a.dt, a.tau, a.seed + 1 + k, a.every, c["steps"](a.equil_ps)) for k, l in enumerate(lam)]
    for k, r in enumerate(runs):
        r.start(); print(f"   node {k} lam {r.lam:.2e} equilibrated: {r.diag()}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print("   t/node (ps)   G (meV/atom)   dF_TDI (meV/atom)   wall (min)", flush=True)
    hist = []; n_per = c["steps"](a.chunk_ps) // a.every
    for ch in range(1, int(round(a.ps / a.chunk_ps)) + 1):
        for k, r in enumerate(runs):
            r.advance(c["steps"](a.chunk_ps))
            print(f"     chunk {ch} node {k} lam {r.lam:.2e}  {r.diag(last=n_per)}  {(time.time() - t0) / 60:.1f} min", flush=True)
        G, Ge, dF, m, s = G_from_nodes(st, lam, w, [r.series for r in runs], N)
        hist.append(dict(t_ps=ch * a.chunk_ps, G_meV_atom=G, G_err_meV_atom=Ge, dF_meV_atom=dF))
        print(f"   {ch * a.chunk_ps:6.2f}      {G:12.2f} +- {Ge:5.2f}   {dF:12.2f}        {(time.time() - t0) / 60:7.1f}", flush=True)
        json.dump(dict(st, lam=lam.tolist(), weights=w.tolist(), dudl_mean=m.tolist(), dudl_sem=s.tolist(), hist=hist, G_meV_atom=G, G_err_meV_atom=Ge), open(os.path.join(a.outdir, f"tdi_{c['tag']}.json"), "w"), indent=1)
    print(f"FINAL {c['tag']}: G = {G:.2f} +- {Ge:.2f} meV/atom (statistical error of the integral only)")
