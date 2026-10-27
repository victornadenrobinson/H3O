"""Short-range probe of the MLP: energy cost dU(r) = U(host + H) - U(host) of an extra proton placed at distance r from a host O or H atom (random directions, several frames).
The ideal-gas reference of every TDI path here (insertion ghost, and all H at small lambda in the crystal / fluid paths) makes the integrand sample such overlaps; a physical
potential is strongly repulsive there. A dU(r) that is not repulsive at r < ~0.7 A (a 'hole': lower than at the bonding distance) biases the free energies, and differently for the
insertion route (one ghost in a fully interacting host) and the crystal route (all H decoupled together), so the two estimates of the mixing free energy would disagree.
usage (repo root, GPU or CPU):  ~/grace-venv/bin/python mlp_pipeline/tdi/short_range.py runs/tdi insert_500GPa_6000K_n1 [--frames 5 --dirs 20 --device cuda]
Frames: the last --frames configurations of <tag>_node<K>_traj.extxyz for the highest lambda node; the n ghosts (last atoms, from the stage0 json) are removed first."""
import argparse, json, os, sys
import numpy as np
from ase import Atoms
from ase.io import read
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
from tdi import analyze

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tag"); ap.add_argument("--frames", type=int, default=5); ap.add_argument("--dirs", type=int, default=20)
ap.add_argument("--device", default="cuda"); ap.add_argument("--r", type=float, nargs="*", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
ap.add_argument("--uniform", type=int, default=200, help="also: dU at this many uniformly random points per frame (the lambda -> 0 integrand of the insertion route)")
ap.add_argument("--widom", type=int, default=0, help="insertion runs: Widom test-particle estimate with this many random points per frame on frames of the SMALLEST-lambda node (the unperturbed host): "
                "mu_ex = -kT ln <exp(-dU/kT)>, to compare with the TDI integral (mu_breakdown 'TDI'); a finite sample biases it HIGH (rare cavities missed), so Widom well BELOW TDI is the red flag"); a = ap.parse_args()
st = json.load(open(os.path.join(a.dir, f"{a.tag}_stage0.json"))); A = st["args"]
lam = analyze.rule_nodes(A)[0]; k = int(np.argmax(lam)); f = os.path.join(a.dir, f"{a.tag}_node{k}_traj.extxyz")
ng = int(st["N"]) if st["phase"] == "insert" else 0
calc = import_module("04_benchmark").get_calc_spec(A["model"], a.device)
rng = np.random.default_rng(0)
frames = read(f, index=":")[-a.frames:]
print(f"{a.tag}: {len(frames)} frames of node {k} (lambda {lam[k]:.3f}), {ng} ghost(s) removed; dU in eV")
rows = {"O": [], "H": []}; uni = []
for fr in frames:
    host = fr[:len(fr) - ng] if ng else fr.copy(); host.calc = calc; e0 = host.get_potential_energy()
    def dU(p):
        b = host + Atoms("H", positions=[p]); b.calc = calc
        return b.get_potential_energy() - e0
    for el, z in (("O", 8), ("H", 1)):
        idx = np.where(host.numbers == z)[0]
        if not len(idx): continue
        for _ in range(a.dirs):
            i = rng.choice(idx); v = rng.normal(size=3); v /= np.linalg.norm(v)
            rows[el].append([dU(host.positions[i] + r * v) for r in a.r])
    uni += [dU(rng.random(3) @ host.cell.array) for _ in range(a.uniform)]
for el, R in rows.items():
    if not R: continue
    R = np.array(R)
    print(f"\nproton at r from a host {el} atom ({len(R)} probes): r (A), dU median, min, max (eV)")
    for j, r in enumerate(a.r):
        print(f"  {r:4.2f}  {np.median(R[:, j]):10.3f} {R[:, j].min():10.3f} {R[:, j].max():10.3f}")
    kt = 8.617333262e-5 * st["T"]; js = [j for j, r in enumerate(a.r) if r <= 0.6]; jp = [j for j, r in enumerate(a.r) if 0.85 <= r <= 1.25]
    low_short = float(R[:, js].min()); low_phys = float(R[:, jp].min()) if jp else float("nan")
    mono = int(np.sum(np.any(np.diff(R[:, js], axis=1) > 0, axis=1)))   # dU must fall as r grows inside 0.6 A
    gap = (low_short - low_phys) / kt
    print(f"  lowest dU at r <= 0.6 A: {low_short:.3f} eV; lowest at 0.85-1.25 A (where protons sit): {low_phys:.3f} eV; gap {gap:+.1f} kT;"
          f" probes not monotonically repulsive inside 0.6 A: {mono} of {len(R)}"
          + ("   <- HOLE: short-range states compete with physical sites (gap < 3 kT)" if gap < 3 or mono else "   (repulsive enough: short range not visited at lambda = 1)"))
u = np.array(uni)
print(f"\nuniform random points ({len(u)}): dU median {np.median(u):.3f}, mean {u.mean():.3f}, min {u.min():.3f}, max {u.max():.3f} eV; fraction below 0: {np.mean(u < 0):.2f}")
if a.widom and st["phase"] == "insert":
    kt = 8.617333262e-5 * st["T"]; k0 = int(np.argmin(lam)); f0 = os.path.join(a.dir, f"{a.tag}_node{k0}_traj.extxyz")
    fr0 = read(f0, index=":")[-a.frames:]; du = []
    for fr in fr0:
        host = fr[:len(fr) - ng]; host.calc = calc; e0 = host.get_potential_energy()
        for _ in range(a.widom):
            b = host + Atoms("H", positions=[rng.random(3) @ host.cell.array]); b.calc = calc; du.append(b.get_potential_energy() - e0)
    du = np.array(du); x = -du / kt; mx = x.max(); w = np.exp(x - mx)
    mu = -kt * (np.log(w.mean()) + mx); blocks = [-kt * (np.log(np.exp(b - mx).mean()) + mx) for b in np.array_split(x, 5)]
    top = np.sort(w)[::-1]; share = top[:max(1, len(w) // 100)].sum() / w.sum()
    print(f"\nWidom on {len(fr0)} frames of node {k0} (lambda {lam[k0]:.1e}) x {a.widom} points = {len(du)} insertions:")
    print(f"  mu_ex = {mu:.3f} eV (5 blocks: {' '.join(f'{v:.3f}' for v in blocks)});  compare: TDI integral of this run (mu_breakdown 'TDI', per added H)")
    print(f"  'acceptance': fraction of points with dU < 0: {np.mean(du < 0):.4f}, < kT: {np.mean(du < kt):.4f}, < 5 kT: {np.mean(du < 5 * kt):.4f};  top 1% of points carry {share:.2f} of <exp(-dU/kT)>"
          + ("   <- dominated by a few points: Widom not converged here (expected in dense matter)" if share > 0.5 else ""))
