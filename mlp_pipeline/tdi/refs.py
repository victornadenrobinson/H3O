"""Analytic pieces of the free energy. Units: eV, Angstrom, K, amu. All configurational free energies are f = -kT ln Z_conf.

Conventions (see docs/TDI_PLAN.md):
  * O atoms are tied to lattice sites, so the N_O! of the partition function cancels against the N_O! site assignments (no factorial for O).
  * H atoms of the ideal-gas reference are indistinguishable: kT ln N_H! is added.
  * The centre of mass (COM) of the O atoms is held fixed in every lambda run (engine.FixGroupCom). Then
        Z_conf,true(interacting)  = V * Z_cons(interacting)         (translation invariance of the full system)
        Z_cons(Einstein, N springs k) = N^{3/2} (2 pi kT / k)^{(3N-3)/2}
"""
import numpy as np
from math import lgamma
from ase import units


def kT(T):
    return units.kB * T


def thermal_wavelength(m_amu, T):
    """de Broglie wavelength h / sqrt(2 pi m kT) in Angstrom."""
    h = 2 * np.pi * units._hbar
    return h / np.sqrt(2 * np.pi * m_amu * units._amu * units.kB * T * units._e) * 1e10


def f_einstein_cons(N, k, T):
    """-kT ln Z_cons for N atoms bound to sites by springs k (eV/A^2) with their COM fixed."""
    return -kT(T) * (1.5 * np.log(N) + 0.5 * (3 * N - 3) * np.log(2 * np.pi * kT(T) / k))


def f_ideal_gas(N, V, T):
    """-kT ln V^N: configurational free energy of N non-interacting atoms in volume V (periodic box)."""
    return -kT(T) * N * np.log(V)


def f_network_cons(hessian, T, n_zero=3):
    """-kT ln Z_cons of a harmonic system with Hessian H (eV/A^2): the n_zero translation modes are dropped, the Jacobian N^{3/2} of the COM constraint is added.
    Returns (f, eigenvalues). Raises if the number of (near) zero modes is not n_zero."""
    w = np.linalg.eigvalsh(hessian)
    zero = np.abs(w) < 1e-8 * max(1.0, w.max())
    if zero.sum() != n_zero:
        raise ValueError(f"{int(zero.sum())} zero modes, expected {n_zero}")
    N = hessian.shape[0] // 3
    lnZ = 1.5 * np.log(N) + 0.5 * np.sum(np.log(2 * np.pi * kT(T) / w[~zero]))
    return -kT(T) * lnZ, w


def kinetic_free_energy(species, T):
    """sum over species of 3 N kT ln(Lambda) (+ kT ln N! if the species is indistinguishable). species: list of (N, mass_amu, indistinguishable)."""
    F = 0.0
    for N, m, indist in species:
        F += 3 * N * kT(T) * np.log(thermal_wavelength(m, T)) + (kT(T) * lgamma(N + 1) if indist else 0.0)
    return F


def f_true_from_cons(f_cons, V, T):
    """interacting translation-invariant system: f_true = f_cons - kT ln V."""
    return f_cons - kT(T) * np.log(V)
