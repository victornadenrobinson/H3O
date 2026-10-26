"""Thermodynamic integration for crystalline H2O (O on bcc/fcc sites, mobile protons) with a GRACE model: one phase, one volume per job.

Path ('direct', Wilson et al. 2013 type):  U(lam) = (1-lam) E_Einstein(O) + lam U_MLP,   reference = Einstein crystal for O + ideal gas for H.
  F = F_ref + int_0^1 <U_MLP - E_Einstein>_lam dlam      (+ kinetic terms; classical nuclei)       G = F + P_target V
All lambda nodes run side by side in chunks, so the estimate of G is printed as a function of simulation time (per lambda).
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
        self.calc = engine.TDICalculator(components, {"E": 1 - lam, "full": lam}, needed=("full", "E"))
        self.at = atoms0.copy(); self.at.calc = self.calc
        self.at.set_constraint(engine.FixGroupCom(idxO))
        rng = np.random.default_rng(seed)
        MaxwellBoltzmannDistribution(self.at, temperature_K=T, rng=rng)
        self.at.set_momenta(self.at.get_momenta(), apply_constraint=True)
        self.dyn = Langevin(self.at, dt * units.fs, temperature_K=T, friction=1.0 / (tau * units.fs), fixcm=False, rng=rng)
        self.ndof = self.at.get_number_of_degrees_of_freedom(); self.started = False; self.steps = 0

    def _rec(self):
        self.at.get_forces()
        p = self.calc.parts
        self.series.append(p["full"] - p["E"])
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


def stats(series, nb=5):
    s = np.asarray(series, float)
    if len(s) < 5 * nb:
        return float("nan"), float("nan")
    return analyze.block_stats(s, nblocks=nb)


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
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE))
    eos = import_module("17_eos_md")
    a.nrep = a.nrep or eos.CELL[a.phase][2]
    vpa = a.vpa or eos.dft_vpa(a.from_dft, a.phase, a.T, [a.P])[0]
    tag = a.tag or f"{a.phase}_{a.P:.0f}GPa_{a.T:.0f}K"
    os.makedirs(a.outdir, exist_ok=True); kt = refs.kT(a.T)
    at0 = eos.build(a.phase, vpa, a.nrep, a.seed)
    N = len(at0); iO = np.where(at0.numbers == 8)[0]; iH = np.where(at0.numbers == 1)[0]; nO, nH = len(iO), len(iH); V = at0.get_volume()
    sitesO = at0.positions[iO].copy()
    from ase.neighborlist import neighbor_list
    d1_0 = float(neighbor_list('d', at0[iO], 0.6 * V ** (1 / 3)).min())   # nearest O-O distance of the ideal lattice
    calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    full = engine.full_component(calc)
    print(f"{tag}: N {N} (O {nO}, H {nH}), V/atom {vpa:.4f} A^3 (V {V:.1f}), kT {kt * 1e3:.1f} meV, nodes {a.nodes} power {a.power}, dt {a.dt} fs", flush=True)
    dt_steps = lambda ps: int(round(ps * 1000 / a.dt))
    t0 = time.time()

    # (1) lambda = 1: MLP dynamics, Einstein constant, pressure, structure
    comps1 = {"full": full, "E": engine.einstein_component(sitesO, iO, 1.0)}
    r1 = LambdaRun(at0, comps1, 1.0, a.T, iO, a.dt, a.tau, a.seed, a.every, dt_steps(a.equil_ps))
    r1.start(); print(f"lambda=1 equilibrated ({a.equil_ps} ps), {(time.time() - t0) / 60:.1f} min, T_kin now {2 * r1.at.get_kinetic_energy() / (r1.ndof * units.kB):.0f} K", flush=True)
    for q in range(int(round(a.ps1 / 0.25))):
        r1.advance(dt_steps(0.25))
        uu = np.array([find_mic(p[iO] - sitesO, at0.cell, True)[0] for p in r1.snaps]); uu -= uu.mean(1, keepdims=True)
        print(f"  lambda=1 {0.25 * (q + 1):.2f} ps  T_kin {np.mean(r1.tk):.0f} K  <U_MLP-E>/atom {np.mean(r1.series) / N * 1e3:.1f} meV (dummy spring)  "
              f"O Lindemann {np.sqrt((uu ** 2).sum(-1).mean()) / d1_0:.3f}  <|u|^2>_O {(uu ** 2).sum(-1).mean():.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    P1 = np.array(r1.snaps)
    u = np.array([find_mic(p[iO] - sitesO, at0.cell, True)[0] for p in P1]); u -= u.mean(1, keepdims=True)
    msd = float((u ** 2).sum(-1).mean()); kE = 3 * kt / msd
    mean_sites = (sitesO + u.mean(0))
    lab, c1, c2, d1 = import_module("09_label_phase").o_structure(mean_sites, at0.cell.array)
    sig = []
    for p in P1[len(P1) // 2:][::2]:
        b = at0.copy(); b.positions = p; b.calc = calc; sig.append(-b.get_stress(voigt=True)[:3].mean())
    P_mlp = float(np.mean(sig)) * EV_A3_GPA + N * kt / V * EV_A3_GPA
    print(f"lambda=1 ({a.ps1} ps, {(time.time() - t0) / 60:.1f} min): T_kin {np.mean(r1.tk):.0f} K, <|u|^2>_O {msd:.4f} A^2 (Lindemann {np.sqrt(msd) / d1:.3f}), Einstein k = {kE:.3f} eV/A^2, "
          f"O structure {lab} (CN1 {c1:.1f}, CN2 {c2:.1f}), MLP pressure {P_mlp:.1f} GPa (target {a.P:.0f})", flush=True)
    if lab != a.phase:
        print(f"WARNING: O sublattice is '{lab}', not {a.phase}: the reference lattice does not describe this system")

    # (2) reference free energy (eV, whole cell)
    mO, mH = float(at0.get_masses()[iO[0]]), float(at0.get_masses()[iH[0]])
    f_ref_cons = refs.f_einstein_cons(nO, kE, a.T) + refs.f_ideal_gas(nH, V, a.T)
    f_kin = refs.kinetic_free_energy([(nO, mO, False), (nH, mH, True)], a.T)
    PV = a.P / EV_A3_GPA * V
    lam, w = analyze.gauss_legendre(a.nodes, a.power)
    comps = {"full": full, "E": engine.einstein_component(sitesO, iO, kE)}
    runs = [LambdaRun(at0, comps, float(l), a.T, iO, a.dt, a.tau, a.seed + 1 + k, a.every, dt_steps(a.equil_ps)) for k, l in enumerate(lam)]
    hist = []; nchunk = int(round(a.ps / a.chunk_ps))
    print("   t/node (ps)   G (meV/atom)   dF_TDI (meV/atom)   wall (min)      [G = F_ref + dF + kinetic + P_target V; per-atom values]", flush=True)
    for k, r in enumerate(runs):   # equilibrate every node first (nothing recorded) and report
        r.start(); print(f"   node {k} lam {r.lam:.2e} equilibrated: T_kin {2 * r.at.get_kinetic_energy() / (r.ndof * units.kB):.0f} K   {(time.time() - t0) / 60:.1f} min", flush=True)
    for c in range(1, nchunk + 1):
        for k, r in enumerate(runs):
            r.advance(dt_steps(a.chunk_ps))
            print(f"     chunk {c} node {k} lam {r.lam:.2e}  <dU/dl> {np.mean(r.series):10.3f} eV  T_kin {np.mean(r.tk):.0f} K  {(time.time() - t0) / 60:.1f} min", flush=True)
        ms = [stats(r.series) for r in runs]
        m = np.array([x[0] for x in ms]); s = np.array([x[1] for x in ms])
        dF, dFe = analyze.integrate(m, s, w)
        F_true = f_ref_cons + dF - kt * np.log(V) + f_kin
        G = F_true + PV
        hist.append(dict(t_ps=c * a.chunk_ps, G_meV_atom=G / N * 1e3, G_err_meV_atom=dFe / N * 1e3, dF_meV_atom=dF / N * 1e3))
        print(f"   {c * a.chunk_ps:6.2f}      {G / N * 1e3:12.2f} +- {dFe / N * 1e3:5.2f}   {dF / N * 1e3:12.2f}        {(time.time() - t0) / 60:7.1f}", flush=True)
        out = dict(tag=tag, phase=a.phase, T=a.T, P_target=a.P, vpa=vpa, V=V, N=N, nO=nO, nH=nH, kE=kE, P_mlp=P_mlp, O_struct=lab, lindemann=float(np.sqrt(msd) / d1),
                   f_ref_cons_eV=f_ref_cons, f_kin_eV=f_kin, PV_eV=PV, lam=lam.tolist(), weights=w.tolist(), dudl_mean=m.tolist(), dudl_sem=s.tolist(),
                   T_kin=[float(np.mean(r.tk)) for r in runs], hist=hist, G_meV_atom=hist[-1]["G_meV_atom"], G_err_meV_atom=hist[-1]["G_err_meV_atom"], args=vars(a))
        json.dump(out, open(os.path.join(a.outdir, f"tdi_{tag}.json"), "w"), indent=1)
    print("per-node <U_MLP - E_Einstein> (eV, whole cell):")
    for l, mm, ss, r in zip(lam, m, s, runs):
        print(f"   lam {l:.2e}  {mm:12.3f} +- {ss:8.3f}   T_kin {np.mean(r.tk):.0f} K")
    print(f"FINAL {tag}: G = {hist[-1]['G_meV_atom']:.2f} +- {hist[-1]['G_err_meV_atom']:.2f} meV/atom (statistical error of the integral only)")
