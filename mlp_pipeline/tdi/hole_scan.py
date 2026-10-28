"""Short-range hole of an MLP: energy of a real dense-H configuration while one proton is pulled onto its nearest neighbour (everything else fixed).
A physical potential rises steeply below ~0.5 A; a hole (energy falling again at small r) lets protons fuse, as in H_500GPa_5000K_N250 / 6000K_N250.
usage (Young, GPU or CPU, ~/grace-venv/bin/python):
  python mlp_pipeline/tdi/hole_scan.py grace-ft:runs/grace/ft_m2 runs/tdi/H_500GPa_5500K_N250_node9_traj.extxyz [--pairs 5] [--wall 0.35,200]
Prints E(r) - E(r at the start) for r = 1.0 ... 0.02 A for --pairs different (frame, pair) choices, the depth and position of any minimum below 0.6 A,
and with --wall r0,K the same curve with the H-H wall K (r0 - r)^2 added (tdi/wall.py), to choose K.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("traj"); ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--wall", default=None); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    from importlib import import_module
    from ase.io import read
    from ase.geometry import find_mic
    calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    fr = read(a.traj, index=":"); rng = np.random.default_rng(0)
    rs = np.r_[np.linspace(1.0, 0.3, 15), np.linspace(0.28, 0.02, 14)]
    wall = None
    if a.wall:
        from tdi.wall import wall_energy
        r0, K = map(float, a.wall.split(",")); wall = lambda r: wall_energy(np.array([r]), r0, K)[0]
    print(f"{'pair':>12s} " + " ".join(f"{r:6.2f}" for r in rs) + "   (eV, relative to the start configuration)")
    for p in range(a.pairs):
        at = fr[rng.integers(len(fr) // 2, len(fr))].copy(); at.calc = calc
        H = np.where(at.numbers == 1)[0]; i = int(rng.choice(H))
        d, D = find_mic(at.positions[H] - at.positions[i], at.cell, True); D[H == i] = np.inf; j = int(H[np.argmin(D)]); u = d[np.argmin(D)] / D.min()
        E0 = at.get_potential_energy(); base = at.positions.copy(); out = []
        for r in rs:
            at.positions = base.copy(); at.positions[j] = base[i] + u * r; out.append(at.get_potential_energy() - E0 + (wall(r) if wall else 0.0))
        out = np.array(out); low = rs < 0.6
        k = np.argmin(out[low]); depth = out[low][k] - out[low].max() if np.any(np.diff(out[low]) > 0) else 0.0
        print(f"{i:5d}-{j:<6d} " + " ".join(f"{e:6.2f}" for e in out) + (f"   HOLE: minimum {out[low][k]:.2f} eV at {rs[low][k]:.2f} A" if np.argmin(out[low]) > 0 and out[low][-1] < out[low].max() else ""))


if __name__ == "__main__":
    main()
