"""End-to-end check of the TDI machinery against an exactly solvable system (no GPU, about a minute).

System: N atoms on an fcc lattice joined by nearest-neighbour harmonic springs acting along the bond direction (exactly quadratic in the displacements).
Analytic free energy from the normal modes (refs.f_network_cons). TDI path: Einstein crystal (spring k_E) -> network, COM fixed, Bussi NVT.
Pass criterion: |TDI - analytic| < 3 sigma (+ 0.002 eV slack for the quadrature).
usage: python mlp_pipeline/tdi/test_harmonic.py [--steps 30000] [--nodes 8] [--T 1000]
"""
import argparse, os, sys, time
import numpy as np
from ase.build import bulk
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import refs, engine, analyze


def network(atoms, ks):
    """pairs within 1.1 d_nn, direction n, energy 1/2 ks sum [n.(u_j - u_i)]^2; returns (component fn, Hessian)."""
    d = neighbor_list("d", atoms, 3.0).min()
    i, j, D = neighbor_list("ijD", atoms, 1.1 * d)
    m = i < j; i, j = i[m], j[m]; n = D[m] / np.linalg.norm(D[m], axis=1)[:, None]
    sites = atoms.positions.copy(); N = len(atoms)
    H = np.zeros((3 * N, 3 * N))
    for a, b, v in zip(i, j, n):
        B = ks * np.outer(v, v)
        H[3 * a:3 * a + 3, 3 * a:3 * a + 3] += B; H[3 * b:3 * b + 3, 3 * b:3 * b + 3] += B
        H[3 * a:3 * a + 3, 3 * b:3 * b + 3] -= B; H[3 * b:3 * b + 3, 3 * a:3 * a + 3] -= B

    def fn(at):
        u, _ = find_mic(at.positions - sites, at.cell, at.pbc)
        delta = np.einsum("pk,pk->p", n, u[j] - u[i])
        F = np.zeros((len(at), 3))
        np.add.at(F, j, -ks * delta[:, None] * n); np.add.at(F, i, ks * delta[:, None] * n)
        return 0.5 * ks * float((delta ** 2).sum()), F
    return fn, H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--nodes", type=int, default=8); ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--ks", type=float, default=4.0); ap.add_argument("--kE", type=float, default=2.0); ap.add_argument("--dt", type=float, default=2.0); ap.add_argument("--thermostat", default="langevin"); ap.add_argument("--tau", type=float, default=100.0)
    a = ap.parse_args()
    at = bulk("O", "fcc", a=3.6, cubic=True).repeat(2); N = len(at); T = a.T
    net, H = network(at, a.ks)
    comps = {"E": engine.einstein_component(at.positions, np.arange(N), a.kE), "full": net}
    f_net, ev = refs.f_network_cons(H, T)
    f_E = refs.f_einstein_cons(N, a.kE, T)
    exact = f_net - f_E
    lam, w = analyze.gauss_legendre(a.nodes)
    means, sems = [], []
    t0 = time.time()
    for k, l in enumerate(lam):
        s, tkin = engine.run_lambda(at, comps, "direct", float(l), T, dt_fs=a.dt, steps=a.steps, equil=2000, every=5, com_idx=np.arange(N), seed=k, thermostat=a.thermostat, tau_fs=a.tau)
        m, e = analyze.block_stats(s, nblocks=10)
        means.append(m); sems.append(e)
        ex = 0.5 * refs.kT(T) * np.sum((ev[ev > 1e-6] - a.kE) / ((1 - l) * a.kE + l * ev[ev > 1e-6]))
        print(f"  lambda {l:.4f}  <dU/dl> = {m:9.4f} +- {e:.4f} eV   exact {ex:9.4f}   T_kin {tkin:7.1f} K   ({time.time() - t0:.0f} s)", flush=True)
    val, err = analyze.integrate(means, sems, w)
    kt = refs.kT(T)
    print(f"N = {N}, T = {T} K (kT = {kt * 1e3:.1f} meV), spring kE {a.kE} -> network ks {a.ks} eV/A^2")
    print(f"analytic  f_net - f_E = {exact:+.5f} eV  ({exact / N * 1e3:+.2f} meV/atom)")
    print(f"TDI                   = {val:+.5f} +- {err:.5f} eV  ({val / N * 1e3:+.2f} +- {err / N * 1e3:.2f} meV/atom)")
    d = val - exact
    ok = abs(d) < 3 * err + 0.002
    print(f"difference {d * 1e3:+.2f} meV total = {d / err:+.1f} sigma   ->  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
