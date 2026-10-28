"""Short-range H-H wall added to any ASE calculator: E_wall = sum over H-H pairs with r < r0 of K (r0 - r)^2 (finite at r = 0, force continuous at r0).
Guards against an MLP hole at short H-H range (protons fusing: H_500GPa_5000K_N250 / 6000K_N250). Healthy runs at 300-600 GPa never come closer than
~0.42 A, so with r0 <= 0.35 A the wall adds exactly nothing to them (same Hamiltonian as before); it only matters where the MLP is unphysical.
Opt-in: HHWALL=r0,K in the environment (e.g. HHWALL=0.35,200), read by 04_benchmark.get_calc_spec; recorded in the stage0 json (args 'hhwall').
In TDI the wall belongs to the full potential, so in insertion it also acts between a ghost and the host at small lambda (scaled by lambda like the
rest of U_full - U_host): the integral stays exact, the integrand there just gets a finite extra term instead of a hole.
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def wall_energy(r, r0, K):
    r = np.asarray(r, float); m = r < r0; e = np.zeros_like(r); e[m] = K * (r0 - r[m]) ** 2; return e


class HHWall(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, base, r0=0.35, K=200.0, **kw):
        super().__init__(**kw); self.base, self.r0, self.K = base, float(r0), float(K)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from ase.neighborlist import neighbor_list
        at = self.atoms.copy(); at.calc = self.base
        E = at.get_potential_energy(); F = at.get_forces().copy()
        want_s = "stress" in properties
        S = at.get_stress(voigt=False).copy() if want_s else None
        i, j, D = neighbor_list("ijD", at, self.r0)
        h = (at.numbers[i] == 1) & (at.numbers[j] == 1) & (i < j)
        if h.any():
            i, j, D = i[h], j[h], D[h]; r = np.linalg.norm(D, axis=1)
            E += float(wall_energy(r, self.r0, self.K).sum())
            dEdr = -2 * self.K * (self.r0 - r)   # dE/dr < 0: repulsive
            f = -(dEdr / np.maximum(r, 1e-12))[:, None] * D   # force on j (D = r_j - r_i)
            np.add.at(F, j, f); np.add.at(F, i, -f)
            if want_s:
                S += np.einsum("ni,nj->ij", D, -f) / at.get_volume()
        self.results = dict(energy=E, free_energy=E, forces=F)
        if want_s:
            self.results["stress"] = S.flat[[0, 4, 8, 5, 2, 1]]


def wrap(calc, spec):
    """spec 'r0,K' (or '' / None: no wall)"""
    if not spec: return calc
    r0, K = map(float, spec.split(","))
    print(f"H-H wall on: K (r0 - r)^2 for r < r0 = {r0} A, K = {K} eV/A^2 (tdi/wall.py)", flush=True)
    return HHWall(calc, r0, K)
