"""Volume per atom at a target (P, T) from the MLP's own equation of state (runs/eos_map, young_eos_map.sbatch: NVT pressure on a V/atom grid), for runs where the DFT EOS has no data,
e.g. pure H at 300 GPa below 5000 K (the DFT pure-H runs start at ~320-380 GPa there, so run_real.py refuses without --vpa).
Per temperature on the map: ln V interpolated linearly in ln P between the two grid volumes that bracket the target P (no extrapolation); between temperatures: linear in T.
usage:  python3 mlp_pipeline/tdi/eos_vpa.py --phase H --branch fluid --hper 2 --P 300 --T 4000 4500      (prints V/atom and the map points used)
        then e.g.  VPA=<value> NATOMS=250 T=4000 P=300 bash mlp_pipeline/young_tdi_par.sh H
"""
import argparse, csv, glob, os, re
import numpy as np


def load(root, phase, branch, hper):
    pts = {}
    for f in glob.glob(os.path.join(root, f"map_{branch}_h{hper:g}_{phase}_*K_i*.csv")):
        for r in csv.DictReader(open(f)):
            try: T, V, P = float(r["T_set"]), float(r["vpa"]), float(r["P_GPa"])
            except (KeyError, ValueError): continue
            if np.isfinite(P) and P > 0: pts.setdefault(T, []).append((P, V))
    return {T: sorted(v) for T, v in pts.items()}


def v_at(pv, P):
    Ps, Vs = np.array([p for p, _ in pv]), np.array([v for _, v in pv])
    if not (Ps.min() <= P <= Ps.max()): return float("nan")
    return float(np.exp(np.interp(np.log(P), np.log(Ps), np.log(Vs))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", default="H"); ap.add_argument("--branch", default="fluid"); ap.add_argument("--hper", type=float, default=2.0)
    ap.add_argument("--P", type=float, nargs="+", required=True); ap.add_argument("--T", type=float, nargs="+", required=True); ap.add_argument("--root", default="runs/eos_map"); a = ap.parse_args()
    pts = load(a.root, a.phase, a.branch, a.hper)
    if not pts: raise SystemExit(f"no map_{a.branch}_h{a.hper:g}_{a.phase}_*K_i*.csv in {a.root}: run young_eos_map.sbatch for this phase first")
    print(f"MLP EOS map {a.phase} {a.branch} H per O {a.hper:g}: T = {sorted(pts)} K; P range per T: " + ", ".join(f"{T:.0f} K {v[0][0]:.0f}-{v[-1][0]:.0f} GPa ({len(v)} pts)" for T, v in sorted(pts.items())))
    Ts = np.array(sorted(pts))
    for P in a.P:
        for T in a.T:
            vt = np.array([v_at(pts[t], P) for t in Ts]); ok = np.isfinite(vt)
            if ok.sum() == 0 or not (Ts[ok].min() <= T <= Ts[ok].max()):
                print(f"{P:6.0f} GPa {T:6.0f} K: outside the map (no extrapolation); temperatures with this P: {[float(t) for t in Ts[ok]]}"); continue
            V = float(np.interp(T, Ts[ok], vt[ok]))
            print(f"{P:6.0f} GPa {T:6.0f} K: V/atom = {V:.4f} A^3   (from " + ", ".join(f"{t:.0f} K {v:.4f}" for t, v in zip(Ts[ok], vt[ok])) + ")")


if __name__ == "__main__":
    main()
