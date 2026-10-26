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
import pandas as pd
from ase import units
from ase.geometry import find_mic
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tdi import refs, engine, analyze

EV_A3_GPA = 160.21766


class Unstable(RuntimeError):
    pass


class LambdaRun:
    def __init__(self, atoms0, components, lam, T, idxO, dt, tau, seed, every, equil_steps):
        self.lam, self.T, self.every, self.equil, self.series, self.tk, self.snaps = lam, T, every, equil_steps, [], [], []
        self.eful, self.espr, self.thermo, self.pfn, self.dt = [], [], [], None, dt
        self.calc = engine.TDICalculator(components, {"E": 1 - lam, "full": lam}, needed=("full", "E"))
        self.at = atoms0.copy(); self.at.calc = self.calc; self.idxO = idxO
        self.sites0 = atoms0.positions[idxO].copy() if len(idxO) else atoms0.positions.copy()   # no O: displacement of every H from its start (liquid check)
        if len(idxO):
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
        if not np.isfinite(self.tk[-1]) or self.tk[-1] > 20 * self.T:   # runaway (e.g. a hard collision at too large a time step): stop at once
            raise Unstable(f"T_kin {self.tk[-1]:.0f} K (target {self.T:.0f} K) after {len(self.series)} recorded samples")
        if len(self.series) % 20 == 0:
            self.snaps.append(self.at.get_positions().copy())
            self.thermo.append(self._thermo_row())

    def _thermo_row(self):
        """one row of the thermo log (every 20 recorded samples): temperature, energies, MLP virial pressure, O (or H) displacement from the start"""
        at = self.at; N = len(at); T = 2 * at.get_kinetic_energy() / (self.ndof * units.kB)
        if len(self.idxO):
            u = find_mic(at.positions[self.idxO] - self.sites0, at.cell, True)[0]; u = u - u.mean(0)
        else:
            u = at.positions - self.sites0
        P = float("nan")
        if self.pfn is not None:
            P = self.pfn(at, T)
        return dict(t_ps=len(self.series) * self.every * self.dt / 1000, T_inst_K=T, U_MLP_eV=self.eful[-1], E_spring_eV=self.espr[-1], dUdl_eV=self.series[-1], P_MLP_GPa=P, u_rms_A=float(np.sqrt((u ** 2).sum(-1).mean())), u_max_A=float(np.linalg.norm(u, axis=1).max()))

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
        if not self.series:   # nothing recorded yet (right after equilibration)
            self.series, self.tk, self.eful, self.espr = [float("nan")], [2 * self.at.get_kinetic_energy() / (self.ndof * units.kB)], [float("nan")], [float("nan")]
            txt = self.diag(); self.series, self.tk, self.eful, self.espr = [], [], [], []
            return txt
        at = self.at; N = len(at); H = at.numbers == 1
        from ase.neighborlist import neighbor_list
        i, j, d = neighbor_list("ijd", at, 1.2)
        hh = d[H[i] & H[j]].min() if (H[i] & H[j]).any() else float("nan"); ho = d[(H[i] != H[j])].min() if (H[i] != H[j]).any() else float("nan")
        if len(self.idxO):
            u = find_mic(at.positions[self.idxO] - self.sites0, at.cell, True)[0]; u -= u.mean(0)
        else:
            u = at.positions - self.sites0
        return (f"T_kin {np.mean(self.tk[sl]):6.0f} K  <dU/dl> {np.mean(self.series[sl]):10.3f} eV  U_MLP/atom {np.mean(self.eful[sl]) / N:8.4f} eV  "
                f"E_spring/atom {np.mean(self.espr[sl]) / N * 1e3:7.2f} meV  min d(H-H) {hh:.2f} d(H-O) {ho:.2f} A  |u| now rms {np.sqrt((u ** 2).sum(-1).mean()):.3f} max {np.linalg.norm(u, axis=1).max():.3f} A")


BARS = "▁▂▃▄▅▆▇█"


def trace(series, n_per, maxn=20):
    """sparkline of the per-chunk means of the integrand (time evolution of this lambda)"""
    b = [float(np.mean(series[i * n_per:(i + 1) * n_per])) for i in range(len(series) // n_per)][-maxn:]
    if not b:
        return ""
    lo, hi = min(b), max(b)
    return "".join(BARS[int(round((v - lo) / (hi - lo + 1e-12) * 7))] for v in b)


def meta():
    """provenance written into every stage0 json"""
    import platform, subprocess, ase
    try:
        git = os.environ.get("H3O_GIT") or subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        git = "unknown"
    return dict(git=git, host=platform.node(), date=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), ase=ase.__version__, numpy=np.__version__)


def E_component(c, kE):
    """Einstein springs on the O atoms; for pure hydrogen (no O) the reference potential is zero"""
    if c.get("host_idx") is not None:   # insertion: the reference potential is the MLP of the host atoms alone
        return engine.subset_component(c["calc"], c["host_idx"])
    if c["nO"] == 0:
        return lambda atoms: (0.0, np.zeros((len(atoms), 3)))
    return engine.einstein_component(c["sitesO"], c["iO"], kE)


def prog(t_start, done, total, t_last=None, n_last=None):
    """' | 123 ms/step | 40% | ETA 13:42 (in 25 min)' from wall time since t_start; the speed is over the last piece if t_last/n_last are given"""
    now = time.time(); el = now - t_start
    sp = (now - t_last) / n_last if t_last and n_last else el / max(done, 1)
    rem = (total - done) * sp
    return f" | {sp * 1000:.0f} ms/step | {100 * done / total:.0f}% | ETA {time.strftime('%H:%M', time.localtime(now + rem))} (in {rem / 60:.0f} min)"


def stats(series, nb=5):
    s = np.asarray(series, float)
    if len(s) < 5 * nb:
        return float("nan"), float("nan")
    return analyze.block_stats(s, nblocks=nb)


def banner(c):
    a = c["a"]; lam, w = c["lam"], c["w"]; L = c["at0"].cell.lengths()
    mode = "stage 0 only" if a.stage0 else (f"lambda {a.node} only" if a.node is not None else "all in one process")
    neq, per = c["steps"](a.equil_ps), c["steps"](a.chunk_ps); nprod = int(round(a.ps / a.chunk_ps)) * per
    if a.phase == "insert":
        print("\n".join(["=" * 100, f"TDI PLAN  {c['tag']}   ({mode})", "=" * 100,
            f"system      : fcc H2O host ({c['nO_real']} O + {c['nH_host']} H) + {a.nghost} added H (ghosts), cubic cell {L[0]:.3f} A, V = {c['V']:.1f} A^3 (host volume from the DFT EOS at {a.T:.0f} K, {a.P:.0f} GPa)",
            f"conditions  : T = {a.T:.0f} K (kT = {c['kt'] * 1e3:.1f} meV), NVT at the host volume",
            f"potential   : {a.model}",
            "path        : U(lam) = (1-lam) U_MLP(host atoms only) + lam U_MLP(all atoms); ghosts are an ideal gas at lam = 0; dU/dlam = U_full - U_host (the interaction energy of the added H)",
            "result      : mu_H(in fcc H2O) = <integral> + kT[ln((N_H+n)!/N_H!) - n ln V] + 3 n kT ln Lambda_H, per added H (no P V term); compare with G per atom of pure H at the same P, T",
            f"dynamics    : Langevin, friction 1/{a.tau:.0f} fs, dt {a.dt} fs, no constraints; each lambda {a.equil_ps} ps equilibration + {a.ps} ps production",
            f"quadrature  : Gauss-Legendre in x on (0,1), lambda = x^{a.power:g}, {a.nodes} lambda values", "   index  lambda      weight"]
            + [f"   {k:4d}   {l:.3e}   {ww:.3e}" for k, (l, ww) in enumerate(zip(lam, w))] + ["=" * 100]), flush=True); return
    if a.phase in ("H", "liquid"):
        nOr = c.get("nO_real", 0)
        lines = ["=" * 100, f"TDI PLAN  {c['tag']}   ({mode})", "=" * 100,
                 f"system      : {nOr} O + {c['nH']} H = {c['N']} atoms, {'liquid H%gO' % a.hper if a.phase == 'liquid' else 'pure hydrogen fluid'}, start lattice that should melt (check the displacement column); cubic cell {L[0]:.3f} A, V = {c['V']:.1f} A^3 = {c['vpa']:.4f} A^3/atom",
                 f"conditions  : T = {a.T:.0f} K (kT = {c['kt'] * 1e3:.1f} meV), NVT, target pressure {a.P:.0f} GPa -> G = F + P_target V (per H atom below)",
                 f"potential   : {a.model} (classical nuclei)",
                 "path        : U(lam) = lam U_MLP ; reference (lam=0) = ideal gas of N indistinguishable H atoms: F_ref = -kT N ln V + kinetic terms + kT ln N! (analytic); F = F_ref + integral_0^1 <U_MLP>_lam dlam",
                 f"dynamics    : Langevin, friction 1/{a.tau:.0f} fs, dt {a.dt} fs; no centre-of-mass constraint",
                 f"each lambda : {a.equil_ps} ps equilibration + {a.ps} ps production, reports every {a.chunk_ps} ps; the 'rms |u|' in the logs is the displacement of H from its start (liquid if it grows beyond the lattice spacing)",
                 f"quadrature  : Gauss-Legendre in x on (0,1), lambda = x^{a.power:g}, {a.nodes} lambda values", "   index  lambda      weight"]
        lines += [f"   {k:4d}   {l:.3e}   {ww:.3e}" for k, (l, ww) in enumerate(zip(lam, w))] + ["=" * 100]
        print("\n".join(lines), flush=True); return
    lines = [
        "=" * 100,
        f"TDI PLAN  {c['tag']}   ({mode})",
        "=" * 100,
        f"system      : {c['nO']} O + {c['nH']} H = {c['N']} atoms (H{a.hper:g}O), {a.phase} O sublattice ({a.nrep}^3 conventional cells), protons start on random O-O bond midpoints",
        f"cell        : cubic {L[0]:.3f} A, V = {c['V']:.1f} A^3 = {c['vpa']:.4f} A^3/atom (from the DFT {a.phase} EOS at {a.P:.0f} GPa, {a.T:.0f} K); periodic",
        f"conditions  : T = {a.T:.0f} K (kT = {c['kt'] * 1e3:.1f} meV), NVT; target pressure {a.P:.0f} GPa -> G = F + P_target V",
        f"potential   : {a.model} (fine-tuned GRACE; classical nuclei, no quantum corrections)",
        "path        : direct coupling-constant integration  U(lam) = (1-lam) E_Einstein(O) + lam U_MLP",
        "              E_Einstein = k/2 sum_O |u_i|^2 (u = displacement from the ideal lattice site, minimum image), O centre of mass held fixed (FixGroupCom)",
        "              reference system (lam=0): O on springs + H ideal gas;   F(lam=1) = F_ref + integral_0^1 <U_MLP - E_Einstein>_lam dlam",
        "              F_ref = F_Einstein(O, COM fixed) + F_ideal(H) + kinetic terms - kT ln V (translation invariance), all analytic (tdi/refs.py)",
        f"Einstein k  : {('%.3f eV/A^2 (given: skips the lambda=1 run)' % a.k) if a.k else 'measured from a lambda=1 run (k = 3 kT / <|u|^2>), also gives MLP pressure and O structure'}",
        f"dynamics    : Langevin, friction 1/{a.tau:.0f} fs, dt {a.dt} fs, forces/energies from the same MLP calls (one GPU evaluation per step per lambda)",
        f"each lambda : {a.equil_ps} ps equilibration (not used) + {a.ps} ps production = {neq + nprod} steps; sampling every {a.every} steps; reports every {a.chunk_ps} ps",
        f"quadrature  : Gauss-Legendre in x on (0,1), lambda = x^{a.power:g}, {a.nodes} lambda values, weight_i = w_i {a.power:g} x_i^{a.power - 1:g}",
        "   index  lambda      weight     steps   what it samples",
    ]
    for k, (l, ww) in enumerate(zip(lam, w)):
        what = "H nearly ideal gas, O on springs" if l < 1e-2 else ("mixed" if l < 0.3 else "mostly the full MLP")
        lines.append(f"   {k:4d}   {l:.3e}   {ww:.3e}   {neq + nprod:6d}   {what}")
    lines += [f"   sum of weights = {w.sum():.6f} (must be 1)",
              f"total cost  : {a.nodes} lambda values x {neq + nprod} steps = {a.nodes * (neq + nprod)} MD steps; at about 150-175 ms/step (384 atoms) that is {a.nodes * (neq + nprod) * 0.16 / 3600:.1f} GPU-hours of work (divided by the number of processes sharing the GPU)",
              f"outputs     : {a.outdir}/{c['tag']}_stage0.json, {c['tag']}_node<k>.npy / .log, tdi_{c['tag']}.json (from tdi/collect.py)",
              "=" * 100]
    print("\n".join(lines), flush=True)


def h_vpa(inv, T, P):
    """V/atom of pure hydrogen (comp_snap == 'h') at pressure P from the DFT inventory (log P quadratic in V/atom, as dft_vpa). T on a 1000 K bin with >= 5 runs: that bin; any other T between two such bins:
    linear interpolation in T between the neighbouring bin fits (approximate, printed)."""
    df0 = import_module("01_bin_inventory").load_inventory(inv)
    df0 = df0[df0.comp_snap == "h"].assign(vpa=lambda d: d.True_V_A3 / d.N_ions).dropna(subset=["vpa"])
    n = df0.groupby("T_bin").size(); good = sorted(int(b) for b in n[n >= 5].index)

    def at(Tb):
        df = df0[df0.T_bin == Tb]
        c = np.polyfit(df.vpa, np.log(df.True_P_GPa.clip(lower=1)), 2); grid = np.linspace(df.vpa.min(), df.vpa.max(), 400); Pg = np.exp(np.polyval(c, grid))
        print(f"DFT pure H {Tb} K: {len(df)} runs, P {df.True_P_GPa.min():.0f}-{df.True_P_GPa.max():.0f} GPa, V/atom {df.vpa.min():.3f}-{df.vpa.max():.3f}")
        if not df.True_P_GPa.min() <= P <= df.True_P_GPa.max():
            print(f"WARNING: {P} GPa is outside the DFT range: volume clipped to the edge of the data")
        return float(np.interp(-P, -Pg, grid))
    Tb = int(round(T / 1000.0)) * 1000
    if abs(T - Tb) < 1 and Tb in good:
        return at(Tb)
    lo = [b for b in good if b <= T]; hi = [b for b in good if b >= T]
    if not lo or not hi:
        raise SystemExit(f"pure-H DFT runs: T = {T:.0f} K is outside the bins with >= 5 runs ({good}) in {inv}: give --vpa")
    lo, hi = max(lo), min(hi); vl, vh = at(lo), at(hi)
    print(f"NOTE: pure-H volume at {T:.0f} K interpolated between the {lo} K and {hi} K DFT fits")
    return vl + (vh - vl) * (T - lo) / (hi - lo)


def hull_vpa(csv_path, hper, T, P):
    """V/atom (A^3) of the fluid H(2+n)O at P, T from the EOS columns of data-h3o-convex-hull.csv (tabulated at 5000, 7000, 9000 K; V per O for mixtures).
    Other temperatures between 5000 and 9000 K: linear interpolation in T between the two neighbouring columns (approximate)."""
    d = pd.read_csv(csv_path)
    comp = {2.0: "h2o", 2.25: "h2.25", 3.0: "h3o", 5.0: "h5o", 7.0: "h7o"}.get(float(hper))
    if comp is None:
        raise SystemExit(f"no hull EOS column for {hper} H per O: give --vpa")

    def at(Tc):
        p, v = d[f"P {int(Tc)}K {comp}"], d[f"V {int(Tc)}K {comp} norm"]; m = p.notna() & v.notna(); p, v = p[m].values, v[m].values; o = np.argsort(p)
        if not p.min() <= P <= p.max():
            print(f"WARNING: {P} GPa is outside the hull EOS range {p.min():.0f}-{p.max():.0f} GPa for {comp} at {Tc} K: volume clipped")
        return float(np.interp(P, p[o], v[o])) / (1 + hper)
    Ts = [5000, 7000, 9000]
    if T in Ts:
        return at(T)
    if not Ts[0] <= T <= Ts[-1]:
        raise SystemExit(f"hull EOS only covers {Ts[0]}-{Ts[-1]} K: give --vpa")
    lo = max(t for t in Ts if t <= T); hi = min(t for t in Ts if t >= T)
    print(f"NOTE: volume at {T} K interpolated between the {lo} K and {hi} K hull columns")
    return at(lo) + (at(hi) - at(lo)) * (T - lo) / (hi - lo)


def solid_vpa(a):
    """V/atom of the H(hper)O crystal at (T, P): the DFT EOS of that composition if the inventory has it, else ideal mixing of the DFT H2O crystal (3 atoms/formula) and the DFT pure-H fluid
    (n = hper - 2 added H per O). Excess volume is neglected: with the PdV_excess of about 0.1-0.2 A^3 per added H (docs/mixing_pv.py) the pressure error is below 1 GPa for a few added H in 108 O,
    and the free energy error is second order in it."""
    eos = import_module("17_eos_md")
    try:
        return eos.dft_vpa(a.from_dft, a.phase, a.T, [a.P], a.hper)[0]
    except SystemExit:
        if a.hper <= 2.0:
            raise
        n = a.hper - 2.0; v_w = eos.dft_vpa(a.from_dft, a.phase, a.T, [a.P], 2.0)[0]; v_h = h_vpa(a.from_dft, a.T, a.P)
        v = (3 * v_w + n * v_h) / (3 + n)
        print(f"NOTE: no DFT EOS for H{a.hper:g}O {a.phase}: volume by ideal mixing of H2O ({v_w:.4f} A^3/atom) and pure H ({v_h:.4f}): {v:.4f} A^3/atom", flush=True)
        return v


def add_ghosts(host, n, seed=0, dmin=0.75):
    """append n H atoms at random positions at least dmin (A) from every atom already present (minimum image); they are ghosts at lambda = 0 and ordinary protons at lambda = 1"""
    from ase import Atoms
    from ase.geometry import get_distances
    rng = np.random.default_rng(seed); at = host.copy()
    for _ in range(n):
        for _try in range(20000):
            p = rng.random(3) @ at.cell.array
            _, d = get_distances(p[None, :], at.positions, cell=at.cell, pbc=True)
            if d.min() >= dmin:
                at += Atoms("H", positions=[p], cell=at.cell, pbc=True); break
        else:
            raise SystemExit(f"could not place ghost {len(at) - len(host) + 1} with dmin {dmin} A")
    return at


def build_h_n(vpa, natoms):
    """pure hydrogen with natoms atoms on a cubic lattice (sc n^3, bcc 2n^3 or fcc 4n^3: whichever matches); the lattice melts, only the density matters"""
    from ase import Atoms
    for k, basis in ((1, [(0, 0, 0)]), (2, [(0, 0, 0), (.5, .5, .5)]), (4, [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)])):
        n = round((natoms / k) ** (1 / 3))
        if k * n ** 3 == natoms:
            a = (natoms * vpa / n ** 3) ** (1 / 3)
            sp = [((np.array(b) + (i, j, l)) / n) for i in range(n) for j in range(n) for l in range(n) for b in basis]
            return Atoms("H" * natoms, scaled_positions=sp, cell=[a * n] * 3, pbc=True)
    raise SystemExit(f"{natoms} atoms is not n^3, 2n^3 or 4n^3")


def build_h(vpa, nrep):
    """pure hydrogen: bcc lattice of 2 nrep^3 atoms (it melts at these conditions: check the displacement column), cubic cell at V/atom = vpa"""
    from ase import Atoms
    n = 2 * nrep ** 3; a_conv = (n * vpa / nrep ** 3) ** (1 / 3)
    sp = [((np.array(b) + (i, j, k)) / nrep) for i in range(nrep) for j in range(nrep) for k in range(nrep) for b in ((0, 0, 0), (.5, .5, .5))]
    return Atoms("H" * n, scaled_positions=sp, cell=[a_conv * nrep] * 3, pbc=True)


def setup(a):
    eos = import_module("17_eos_md")
    tag = a.tag or f"{a.phase}_{a.P:.0f}GPa_{a.T:.0f}K" + (f"_n{a.nghost}" if a.phase == "insert" else "") + (f"_h{a.hper:g}o" if (a.hper != 2.0 or a.phase == "liquid") and a.phase != "H" else "") + (f"_N{a.natoms}" if a.phase == "H" and a.natoms else "") + (f"_rep{a.nrep}" if a.nrep and a.phase in ("bcc", "fcc") else "")
    os.makedirs(a.outdir, exist_ok=True)
    if a.phase == "H":
        a.nrep = a.nrep or 5
        vpa = a.vpa or h_vpa(a.from_dft, a.T, a.P)
        at0 = build_h_n(vpa, a.natoms) if a.natoms else build_h(vpa, a.nrep)
    elif a.phase == "insert":   # fcc H2O host (volume from the DFT EOS at this T, P) + n ghost protons
        a.nrep = a.nrep or 3
        vpa = a.vpa or eos.dft_vpa(a.from_dft, "fcc", a.T, [a.P], 2.0)[0]
        host = eos.build("fcc", vpa, a.nrep, a.seed, 2.0); at0 = add_ghosts(host, a.nghost, a.seed + 7)
    elif a.phase == "liquid":   # fluid H(2+n)O: fcc start (32 O for nrep 2) that should melt; volume from the DFT hull EOS
        a.nrep = a.nrep or 2
        vpa = a.vpa or hull_vpa(a.hull, a.hper, a.T, a.P)
        at0 = eos.build("fcc", vpa, a.nrep, a.seed, a.hper)
    else:
        a.nrep = a.nrep or eos.CELL[a.phase][2]
        vpa = a.vpa or solid_vpa(a)
        at0 = eos.build(a.phase, vpa, a.nrep, a.seed, a.hper)
    iO = np.where(at0.numbers == 8)[0]; iH = np.where(at0.numbers == 1)[0]
    ctx = dict(a=a, tag=tag, vpa=vpa, at0=at0, N=len(at0), iO=iO, iH=iH, nO=len(iO), nH=len(iH), V=at0.get_volume(), sitesO=at0.positions[iO].copy(), kt=refs.kT(a.T))
    if a.phase == "insert":
        nh = len(host); ctx.update(nO_real=int((host.numbers == 8).sum()), host_idx=np.arange(nh), nH_host=int((host.numbers == 1).sum()), iO=np.array([], dtype=int), nO=0, sitesO=np.zeros((0, 3)), V=at0.get_volume())
    if a.phase in ("H", "liquid"):   # no lattice: no Einstein springs, no fixed centre of mass; every atom starts as an ideal gas
        ctx.update(nO_real=len(iO), iO=np.array([], dtype=int), nO=0, sitesO=np.zeros((0, 3)))
    ctx["calc"] = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    ctx["full"] = engine.full_component(ctx["calc"])
    ctx["steps"] = lambda ps: int(round(ps * 1000 / a.dt))
    ctx["lam"], ctx["w"] = analyze.gauss_legendre(a.nodes, a.power)
    banner(ctx)
    return ctx


def stage0_given_k(c, kE):
    """skip the lambda=1 run: use an Einstein constant k (eV/A^2) known from an earlier run (any sensible k gives the same G; it only changes the statistical efficiency)"""
    a = c["a"]; kt, V, N = c["kt"], c["V"], c["N"]
    mO, mH = float(c["at0"].get_masses()[c["iO"][0]]), float(c["at0"].get_masses()[c["iH"][0]])
    f_ref_cons = refs.f_einstein_cons(c["nO"], kE, a.T) + refs.f_ideal_gas(c["nH"], V, a.T)
    f_kin = refs.kinetic_free_energy([(c["nO"], mO, False), (c["nH"], mH, True)], a.T); PV = a.P / EV_A3_GPA * V
    print(f"lambda=1 run skipped: Einstein k = {kE:.3f} eV/A^2 given; reference pieces (eV): config {f_ref_cons:.3f}  kinetic {f_kin:.3f}  P_target V {PV:.3f}", flush=True)
    d = dict(tag=c["tag"], phase=a.phase, T=a.T, P_target=a.P, vpa=c["vpa"], V=V, N=N, nO=c["nO"], nH=c["nH"], kE=kE, P_mlp=float("nan"), O_struct="not checked (k given)", lindemann=float("nan"),
             f_ref_cons_eV=f_ref_cons, f_kin_eV=f_kin, PV_eV=PV, args=vars(a), meta=meta())
    json.dump(d, open(os.path.join(a.outdir, f"{c['tag']}_stage0.json"), "w"), indent=1)
    return d


def stage0_hydrogen(c):
    """fluid (pure H or liquid H(2+n)O): nothing to measure; reference = ideal gas of the N atoms, indistinguishable within each species (no fixed centre of mass, so no -kT ln V term)"""
    a = c["a"]; V, N = c["V"], c["N"]; nOr = c.get("nO_real", 0); nH = c["nH"]; m = c["at0"].get_masses(); nums = c["at0"].numbers
    species = [(n, float(m[nums == z][0]), True) for n, z in ((nOr, 8), (nH, 1)) if n > 0]
    f_ref = refs.f_ideal_gas(N, V, a.T); f_kin = refs.kinetic_free_energy(species, a.T); PV = a.P / EV_A3_GPA * V
    print(f"fluid {nOr} O + {nH} H: reference pieces (eV, whole cell): F_ideal config {f_ref:.3f}  kinetic (incl. kT ln N_O! N_H!) {f_kin:.3f}  P_target V {PV:.3f}", flush=True)
    d = dict(tag=c["tag"], phase=a.phase, T=a.T, P_target=a.P, vpa=c["vpa"], V=V, N=N, nO=nOr, nH=nH, kE=0.0, P_mlp=float("nan"), O_struct="fluid (no lattice)", lindemann=float("nan"),
             f_ref_cons_eV=f_ref, f_kin_eV=f_kin, PV_eV=PV, no_com_term=True, args=vars(a), meta=meta())
    json.dump(d, open(os.path.join(a.outdir, f"{c['tag']}_stage0.json"), "w"), indent=1)
    return d


def stage0_insert(c):
    """F(N+n) - F(N) = int <U_full - U_host> dlambda + kT [ln((N_H+n)!/N_H!) - n ln V] + 3 n kT ln(Lambda_H);  reported per added H (the json 'N' is the number of ghosts), no P V term:
    at fixed V this is mu_H of the crystal at its pressure ((dF/dN)_V = (dG/dN)_P)"""
    a = c["a"]; n = a.nghost; V = c["V"]; kt = c["kt"]; NH = c["nH_host"]
    f_conf = kt * sum(np.log(NH + i) for i in range(1, n + 1)) - n * kt * np.log(V)
    f_kin = refs.kinetic_free_energy([(n, float(c["at0"].get_masses()[-1]), False)], a.T)
    print(f"insertion of {n} H into fcc H2O ({c['nO_real']} O + {NH} H, V = {V:.1f} A^3): reference pieces (eV): configurational {f_conf:.3f}  3 n kT ln Lambda {f_kin:.3f}", flush=True)
    d = dict(tag=c["tag"], phase="insert", T=a.T, P_target=a.P, vpa=c["vpa"], V=V, N=n, N_atoms_total=c["N"], nO=c["nO_real"], nH=NH + n, kE=0.0, P_mlp=float("nan"), O_struct="fcc host + ghosts", lindemann=float("nan"),
             f_ref_cons_eV=f_conf, f_kin_eV=f_kin, PV_eV=0.0, no_com_term=True, per="added H", args=vars(a), meta=meta())
    json.dump(d, open(os.path.join(a.outdir, f"{c['tag']}_stage0.json"), "w"), indent=1)
    return d


def stage0(c):
    if c["a"].phase == "insert":
        return stage0_insert(c)
    if c["a"].phase in ("H", "liquid"):
        return stage0_hydrogen(c)
    if c["a"].k:
        return stage0_given_k(c, c["a"].k)
    a = c["a"]; at0, iO, N, V, kt = c["at0"], c["iO"], c["N"], c["V"], c["kt"]; t0 = time.time()
    from ase.neighborlist import neighbor_list
    d1_0 = float(neighbor_list("d", at0[iO], 0.6 * V ** (1 / 3)).min())
    r1 = LambdaRun(at0, {"full": c["full"], "E": engine.einstein_component(c["sitesO"], iO, 1.0)}, 1.0, a.T, iO, a.dt, a.tau, a.seed, a.every, c["steps"](a.equil_ps))
    neq, nq = c["steps"](a.equil_ps), int(round(a.ps1 / 0.25)); st25 = c["steps"](0.25); tot = neq + nq * st25
    r1.start(); print(f"lambda=1 equilibrated ({a.equil_ps} ps), {(time.time() - t0) / 60:.1f} min{prog(t0, neq, tot)}", flush=True)
    for q in range(nq):
        tl = time.time(); r1.advance(c["steps"](0.25))
        uu = np.array([find_mic(p[iO] - c["sitesO"], at0.cell, True)[0] for p in r1.snaps]); uu -= uu.mean(1, keepdims=True)
        print(f"  lambda=1 {0.25 * (q + 1):.2f} ps  {r1.diag(last=c['steps'](0.25) // a.every)}  Lindemann {np.sqrt((uu ** 2).sum(-1).mean()) / d1_0:.3f}  <|u|^2>_O {(uu ** 2).sum(-1).mean():.4f}  {(time.time() - t0) / 60:.1f} min{prog(t0, neq + (q + 1) * st25, tot, tl, st25)}", flush=True)
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
             f_ref_cons_eV=f_ref_cons, f_kin_eV=f_kin, PV_eV=PV, args=vars(a), meta=meta())
    json.dump(d, open(os.path.join(a.outdir, f"{c['tag']}_stage0.json"), "w"), indent=1)
    return d


def run_node(c, st, k):
    """node k alone; the series of <U_MLP - E> is saved to <outdir>/<tag>_node<k>.npy after every chunk"""
    a = c["a"]; l = float(c["lam"][k]); t0 = time.time()
    comps = {"full": c["full"], "E": E_component(c, st["kE"])}
    r = LambdaRun(c["at0"], comps, l, a.T, c["iO"], a.dt, a.tau, a.seed + 1 + k, a.every, c["steps"](a.equil_ps))
    neq, per = c["steps"](a.equil_ps), c["steps"](a.chunk_ps); nchunk = int(round(a.ps / a.chunk_ps)); tot = neq + nchunk * per
    def pfn(at, T):   # MLP virial pressure (GPa) incl. the ideal-gas term N kB T / V, as in the EOS runs
        b = at.copy(); b.calc = c["calc"]; sg = b.get_stress(voigt=True)
        return float(-sg[:3].mean() * EV_A3_GPA + len(at) * units.kB * T / at.get_volume() * EV_A3_GPA)
    r.pfn = pfn
    r.at.get_forces(); t0 = time.time()   # warm-up evaluation: XLA compile / model graph build stays out of the speed and ETA numbers
    sf = os.path.join(a.outdir, f"{c['tag']}_node{k}_state.npz"); done = 0
    if os.path.exists(sf) and not a.fresh:   # restart: positions, momenta and the recorded series of the last finished chunk
        z = np.load(sf)
        for key, now in (("dt", a.dt), ("lam", l), ("T", a.T), ("natoms", len(c["at0"]))):   # a state from other settings would silently mix samples (old state files lack these keys: no check)
            if key in z.files and abs(float(z[key]) - float(now)) > 1e-9 * max(1.0, abs(float(now))):
                print(f"lambda {k}: saved state has {key} = {float(z[key])} but this run has {key} = {float(now)}: refusing to mix samples. Use FRESH=1 (deletes the old files) or move them away.", flush=True); sys.exit(6)
        r.at.positions[:] = z["pos"]; r.at.set_momenta(z["mom"], apply_constraint=False)
        r.series, r.tk, r.eful, r.espr = (list(z[x]) for x in ("series", "tk", "eful", "espr")); r.snaps = []
        r.dyn.attach(r._rec, interval=a.every); r.started = True; done = len(r.series) // (per // a.every)
        print(f"lambda {k}: RESTART from {sf}: {done} of {nchunk} chunks done ({len(r.series)} samples), continuing", flush=True)
    else:
        r.start()
        Teq = 2 * r.at.get_kinetic_energy() / (r.ndof * units.kB)
        if not np.isfinite(Teq) or Teq > 20 * a.T:
            print(f"lambda {k}: UNSTABLE DYNAMICS during equilibration: T_kin {Teq:.0f} K (target {a.T:.0f} K). Rerun with a smaller time step (DT=0.5, FRESH=1).", flush=True); sys.exit(5)
        print(f"lambda {k} = {l:.3e} equilibrated ({a.equil_ps} ps): {r.diag()} {(time.time() - t0) / 60:.1f} min{prog(t0, neq, tot)}", flush=True)
    f = os.path.join(a.outdir, f"{c['tag']}_node{k}.npy")
    n_per = per // a.every
    if done >= nchunk:
        print(f"lambda {k}: already complete"); return
    t_resume = time.time()
    for ch in range(done + 1, nchunk + 1):
        tl = time.time(); n_before = len(r.series)
        try:
            r.advance(per)
            Tavg = float(np.mean(r.tk[n_before:]))
            if Tavg > 2.0 * a.T:
                raise Unstable(f"mean T_kin {Tavg:.0f} K in the last chunk (target {a.T:.0f} K)")
        except Unstable as ex:
            print(f"lambda {k}: UNSTABLE DYNAMICS: {ex}. This chunk is discarded and NOT saved; files hold the last healthy chunk ({done + ch - 1} of {nchunk}). "
                  f"Rerun this lambda with a smaller time step (DT=0.5; use FRESH=1 so every sample has the same dt).", flush=True)
            sys.exit(5)
        np.save(f, np.array(r.series))
        np.savez(sf + ".tmp.npz", pos=r.at.get_positions(), mom=r.at.get_momenta(), series=r.series, tk=r.tk, eful=r.eful, espr=r.espr, dt=a.dt, lam=l, T=a.T, natoms=len(c["at0"])); os.replace(sf + ".tmp.npz", sf)
        pm = float(np.nanmean([x["P_MLP_GPa"] for x in r.thermo])) if r.thermo else float("nan")
        if r.thermo:
            import csv
            ft = os.path.join(a.outdir, f"{c['tag']}_node{k}_thermo.csv"); new = not os.path.exists(ft)
            with open(ft, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(r.thermo[0].keys()))
                if new:
                    wr.writeheader()
                wr.writerows(r.thermo)
            r.thermo = []
        if r.snaps:   # configurations (every 20 recorded samples) for later analysis / DFT single points
            from ase.io import write
            for p in r.snaps:
                fr = c["at0"].copy(); fr.positions = p; fr.info.update(lam=l, node=k, tag=c["tag"], t_ps=ch * a.chunk_ps)
                write(os.path.join(a.outdir, f"{c['tag']}_node{k}_traj.extxyz"), fr, append=True)
            r.snaps = []
        m, e = stats(r.series)
        print(f"lambda {k} = {l:.3e} t {ch * a.chunk_ps:5.2f} ps: last chunk <dU/dl> {np.mean(r.series[-n_per:]):10.3f}  all {m:10.3f} +- {e:7.3f} eV | {r.diag(last=n_per)} | P_MLP {pm:6.1f} GPa | trace {trace(r.series, n_per)} | {(time.time() - t0) / 60:.1f} min{prog(t_resume, (ch - done) * per, (nchunk - done) * per, tl, per)}", flush=True)


def G_total_eV(st, dF):
    """whole-cell G (eV) from the TDI integral dF; the -kT ln V translation term applies only when the O centre of mass is held fixed (not for the pure-H fluid)"""
    kt = refs.kT(st["T"]); tr = 0.0 if st.get("no_com_term") else -kt * np.log(st["V"])
    return st["f_ref_cons_eV"] + dF + tr + st["f_kin_eV"] + st["PV_eV"]


def G_from_nodes(st, lam, w, series_list, N):
    ms = [stats(s) for s in series_list]; m = np.array([x[0] for x in ms]); s = np.array([x[1] for x in ms])
    dF, dFe = analyze.integrate(m, s, w)
    return G_total_eV(st, dF) / N * 1e3, dFe / N * 1e3, dF / N * 1e3, m, s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--phase", choices=["bcc", "fcc", "H", "liquid", "insert"], required=True)
    ap.add_argument("--T", type=float, default=3000); ap.add_argument("--P", type=float, default=400, help="target pressure (GPa)")
    ap.add_argument("--from-dft", default=None); ap.add_argument("--vpa", type=float, default=None, help="A^3/atom (instead of --from-dft)")
    ap.add_argument("--nrep", type=int, default=None); ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--power", type=float, default=4.0)
    ap.add_argument("--ps", type=float, default=4.0, help="production per lambda node"); ap.add_argument("--chunk-ps", type=float, default=0.5)
    ap.add_argument("--equil-ps", type=float, default=1.0); ap.add_argument("--ps1", type=float, default=3.0, help="length of the lambda=1 run")
    ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--tau", type=float, default=100.0); ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default="runs/tdi"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=float, default=None, help="Einstein constant (eV/A^2): skip the lambda=1 run"); ap.add_argument("--fresh", action="store_true", help="ignore saved node states"); ap.add_argument("--hper", type=float, default=2.0, help="H atoms per O (2 = H2O, 2.25 = H2.25O: 128 O + 288 H in bcc)"); ap.add_argument("--natoms", type=int, default=0, help="pure H only: number of atoms (n^3, 2n^3 or 4n^3), e.g. 729"); ap.add_argument("--hull", default="data-h3o-convex-hull.csv", help="liquid mode: EOS table for the volume"); ap.add_argument("--nghost", type=int, default=1, help="insert mode: number of added H atoms (ghosts) in the fcc H2O host"); ap.add_argument("--stage0", action="store_true"); ap.add_argument("--node", type=int, default=None)
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
    comps = {"full": c["full"], "E": E_component(c, st["kE"])}
    runs = [LambdaRun(c["at0"], comps, float(l), a.T, c["iO"], a.dt, a.tau, a.seed + 1 + k, a.every, c["steps"](a.equil_ps)) for k, l in enumerate(lam)]
    neq, per = c["steps"](a.equil_ps), c["steps"](a.chunk_ps); nchunk = int(round(a.ps / a.chunk_ps)); tot = len(runs) * (neq + nchunk * per)
    for k, r in enumerate(runs):
        r.start(); print(f"   lambda {k} = {r.lam:.2e} equilibrated: {r.diag()}  {(time.time() - t0) / 60:.1f} min{prog(t0, (k + 1) * neq, tot)}", flush=True)
    print("   t/lambda (ps)   G (meV/atom)   dF_TDI (meV/atom)   wall (min)", flush=True)
    hist = []; n_per = c["steps"](a.chunk_ps) // a.every
    for ch in range(1, nchunk + 1):
        for k, r in enumerate(runs):
            tl = time.time(); r.advance(per)
            print(f"     chunk {ch} lambda {k} = {r.lam:.2e}  {r.diag(last=n_per)}  {(time.time() - t0) / 60:.1f} min{prog(t0, len(runs) * neq + ((ch - 1) * len(runs) + k + 1) * per, tot, tl, per)}", flush=True)
        G, Ge, dF, m, s = G_from_nodes(st, lam, w, [r.series for r in runs], N)
        hist.append(dict(t_ps=ch * a.chunk_ps, G_meV_atom=G, G_err_meV_atom=Ge, dF_meV_atom=dF))
        print(f"   {ch * a.chunk_ps:6.2f}      {G:12.2f} +- {Ge:5.2f}   {dF:12.2f}        {(time.time() - t0) / 60:7.1f}", flush=True)
        json.dump(dict(st, lam=lam.tolist(), weights=w.tolist(), dudl_mean=m.tolist(), dudl_sem=s.tolist(), hist=hist, G_meV_atom=G, G_err_meV_atom=Ge), open(os.path.join(a.outdir, f"tdi_{c['tag']}.json"), "w"), indent=1)
    print(f"FINAL {c['tag']}: G = {G:.2f} +- {Ge:.2f} meV/atom (statistical error of the integral only)")
