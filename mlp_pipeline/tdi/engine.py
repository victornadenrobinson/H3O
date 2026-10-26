"""Mixed Hamiltonians and one NVT run per lambda point.

Components are callables  atoms -> (energy, forces[N,3]):  'E' Einstein springs on the O atoms, 'full' the potential of the whole system,
'O' the same potential evaluated on the O sublattice alone (H removed), 'net' etc. for tests.
Stages (U = potential used in the MD at lambda; dudl = the integrand <dU/dlambda>):
  direct : U = (1-lam) E + lam full                       reference = Einstein O + ideal-gas H (the Wilson et al. path, use nonlinear lambda nodes)
  stage1 : U = E + lam (full - O)       dudl = full - O    ideal-gas H -> interacting H, O still on springs without O-O interaction
  stage2 : U = (full - O) + (1-lam) E + lam O   dudl = O - E    Einstein O -> interacting O, H fully interacting
"""
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixConstraint
from ase.geometry import find_mic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

STAGES = {
    "direct": (lambda lam: {"E": 1 - lam, "full": lam}, ("full", "E")),
    "stage1": (lambda lam: {"E": 1.0, "full": lam, "O": -lam}, ("full", "O")),
    "stage2": (lambda lam: {"full": 1.0, "O": -(1 - lam), "E": 1 - lam}, ("O", "E")),
}
_UNSET = object()


class FixGroupCom(FixConstraint):
    """Holds the centre of mass of a group of equal-mass atoms fixed (3 degrees of freedom removed): the exact holonomic constraint for the
    'COM fixed' ensemble used in refs.py. Positions, forces and momenta are projected."""

    def __init__(self, indices):
        self.index = np.asarray(indices, dtype=int)

    def get_removed_dof(self, atoms):
        return 3

    def adjust_positions(self, atoms, new):
        new[self.index] -= (new[self.index] - atoms.positions[self.index]).mean(0)

    def adjust_forces(self, atoms, forces):
        forces[self.index] -= forces[self.index].mean(0)

    def adjust_momenta(self, atoms, momenta):
        momenta[self.index] -= momenta[self.index].mean(0)

    def index_shuffle(self, atoms, ind):
        ind = np.asarray(ind)
        new = [int(np.where(ind == i)[0][0]) for i in self.index if i in ind]
        self.index = np.asarray(new, dtype=int)

    def get_indices(self):
        return self.index

    def todict(self):
        return {"name": "FixGroupCom", "kwargs": {"indices": self.index.tolist()}}


def einstein_component(sites, idx, k):
    """E = k/2 sum |u_i|^2, u = minimum-image displacement of atom idx[i] from its lattice site."""
    sites = np.asarray(sites, float); idx = np.asarray(idx)

    def fn(atoms):
        d, _ = find_mic(atoms.positions[idx] - sites, atoms.cell, atoms.pbc)
        F = np.zeros((len(atoms), 3)); F[idx] = -k * d
        return 0.5 * k * float((d ** 2).sum()), F
    return fn


def subset_component(calc, idx):
    """energy/forces of `calc` evaluated on the atoms idx alone (same cell), forces mapped back."""
    idx = np.asarray(idx)

    def fn(atoms):
        sub = atoms[idx]; sub.calc = calc
        e, f = sub.get_potential_energy(), sub.get_forces()
        F = np.zeros((len(atoms), 3)); F[idx] = f
        return float(e), F
    return fn


def full_component(calc):
    def fn(atoms):
        b = atoms.copy(); b.calc = calc
        return float(b.get_potential_energy()), b.get_forces()
    return fn


class TDICalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, components, weights, needed=()):
        super().__init__()
        self.components, self.weights, self.needed = components, weights, set(needed)
        self.parts = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        E, F, self.parts = 0.0, np.zeros((len(atoms), 3)), {}
        for name in set(self.weights) | self.needed:
            e, f = self.components[name](atoms)
            self.parts[name] = e
            w = self.weights.get(name, 0.0)
            E += w * e; F += w * f
        self.results = {"energy": E, "forces": F}


def run_lambda(atoms, components, stage, lam, T, dt_fs=1.0, steps=20000, equil=2000, every=2, com_idx=None, tau_fs=100.0, seed=0, thermostat="langevin"):
    """NVT run at one lambda; returns (series of dU/dlambda in eV after the equilibration, mean kinetic temperature using the constrained degrees of freedom).
    The atoms should start on the lattice sites. thermostat: 'langevin' (local, ergodic also for harmonic systems; the dynamics are irrelevant for free energies)
    or 'bussi' (global rescaling: does not mix the modes of a harmonic crystal)."""
    from ase.md.bussi import Bussi
    from ase.md.langevin import Langevin
    wfun, (a, b) = STAGES[stage]
    calc = TDICalculator(components, wfun(lam), needed=(a, b))
    at = atoms.copy(); at.calc = calc
    if com_idx is not None:
        at.set_constraint(FixGroupCom(com_idx))
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
    at.set_momenta(at.get_momenta(), apply_constraint=True)
    if thermostat == "bussi":
        dyn = Bussi(at, timestep=dt_fs * units.fs, temperature_K=T, taut=tau_fs * units.fs, rng=rng)
    else:
        dyn = Langevin(at, dt_fs * units.fs, temperature_K=T, friction=1.0 / (tau_fs * units.fs), fixcm=False, rng=rng)
    out, tk = [], []
    ndof = at.get_number_of_degrees_of_freedom()

    def rec():
        at.get_forces()
        out.append(calc.parts[a] - calc.parts[b])
        tk.append(2 * at.get_kinetic_energy() / (ndof * units.kB))
    dyn.run(equil)
    dyn.attach(rec, interval=every)
    dyn.run(steps)
    return np.array(out), float(np.mean(tk))
