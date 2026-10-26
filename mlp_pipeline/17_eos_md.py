"""Short NVT MD of crystalline H2O (O on bcc/fcc sites, protons on random O-O bond midpoints) at several volumes -> P(V) of an MLP.

usage (model's own python, GPU job, repo root):
  GRACE_UQ=1 ~/grace-venv/bin/python mlp_pipeline/17_eos_md.py grace-ft:runs/grace/ft_s --phase bcc --T 3000 --from-dft inventory_binned.csv --P 200 300 400 500 600 --ps 4
  ... --vpa 1.9 1.8 1.7        (explicit volumes in A^3/atom instead of --from-dft)
Per volume writes runs/eos_md/<tag>_v<vpa>/{thermo.csv, traj.extxyz, final.extxyz} and appends a row to runs/eos_md/<tag>.csv.
Pressure = virial + N kB T_inst / V (same convention as the DFT data: external + ideal-gas term). Bussi thermostat (falls back to Langevin).
D_H, D_O: slope of the MSD over the second half of production (A^2/fs * 0.1 = cm^2/s). Starting guess only: check the O sublattice stays put (D_O ~ 0).
"""
import argparse, os, time
from importlib import import_module
import numpy as np
import pandas as pd
from ase import Atoms, units
from ase.io import write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import neighbor_list
from common import out as outp, inp

EV_A3_GPA = 160.21766
CELL = {"bcc": (2, [(0, 0, 0), (.5, .5, .5)], 4), "fcc": (4, [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)], 3)}  # O per cell, basis, default repeats


def build(phase, vpa, nrep, seed=0, hper=2.0):
    nO_cell, basis, _ = CELL[phase]
    N = int(round((1 + hper) * nO_cell * nrep ** 3))   # all atoms (hper H per O)
    a = (N * vpa / nrep ** 3) ** (1 / 3)
    O = Atoms("O" * (nO_cell * nrep ** 3), scaled_positions=[((np.array(b) + (i, j, k)) / nrep) for i in range(nrep) for j in range(nrep) for k in range(nrep) for b in basis],
              cell=[a * nrep] * 3, pbc=True)
    nn = np.sort(neighbor_list("d", O, 0.9 * a))[:1] if False else None
    d = neighbor_list("d", O, a)                       # shortest O-O distance = nearest neighbour
    dmin = d.min()
    i, j, D = neighbor_list("ijD", O, 1.1 * dmin)
    keep = i < j                                       # one entry per bond (periodic images of the same pair are distinct bonds)
    mid = O.positions[i[keep]] + D[keep] / 2           # bond midpoints
    rng = np.random.default_rng(seed)
    nH = int(round(hper * len(O)))
    if nH > len(mid):
        raise SystemExit(f"only {len(mid)} bonds for {nH} protons")
    H = Atoms("H" * nH, positions=mid[rng.choice(len(mid), nH, replace=False)])
    at = O + H
    at.cell, at.pbc = O.cell, True
    at.wrap()
    return at


def comp_label(hper):
    return f"h{hper:g}o"


def _vpa_bin(df, phase, Tb, Ps):
    df = df[df.T_bin == Tb].assign(vpa=lambda d: d.True_V_A3 / d.N_ions).dropna(subset=["vpa"])
    c = np.polyfit(df.vpa, np.log(df.True_P_GPa.clip(lower=1)), 2)
    grid = np.linspace(df.vpa.min(), df.vpa.max(), 400)
    P = np.exp(np.polyval(c, grid))
    print(f"DFT {phase} {Tb:.0f} K: {len(df)} runs, P {df.True_P_GPa.min():.0f}-{df.True_P_GPa.max():.0f} GPa, V/atom {df.vpa.min():.3f}-{df.vpa.max():.3f}")
    out_of_range = [p for p in Ps if p < df.True_P_GPa.min() or p > df.True_P_GPa.max()]
    if out_of_range:   # np.interp clips silently: say so
        print(f"WARNING: target pressures {out_of_range} GPa are outside the DFT data range {df.True_P_GPa.min():.0f}-{df.True_P_GPa.max():.0f} GPa for {phase}; their volumes are clipped to the edge of the data")
    return np.array([float(np.interp(-p, -P, grid)) for p in Ps])   # P decreases with V: interpolate on the monotone branch


def dft_vpa(inv, phase, T, Ps, hper=2.0, min_runs=20):
    """V/atom at pressures Ps from the DFT EOS of this phase/composition at temperature T. T on a 1000 K bin with >= min_runs runs: that bin's fit;
    any other T between two such bins: linear interpolation in T between the neighbouring bin fits (approximate)."""
    df = import_module("01_bin_inventory").load_inventory(inv)
    df = df[(df.comp_snap == comp_label(hper)) & df.Path.str.contains(phase, case=False) & ~df.Path.str.contains("liquid|solid|melt", case=False)]
    n = df.groupby("T_bin").size(); good = sorted(int(b) for b in n[n >= min_runs].index)
    Tb = int(round(T / 1000.0)) * 1000
    if abs(T - Tb) < 1 and Tb in good:
        return list(_vpa_bin(df, phase, Tb, Ps))
    lo = [b for b in good if b <= T]; hi = [b for b in good if b >= T]
    if not lo or not hi:
        raise SystemExit(f"{phase} {hper}: T = {T:.0f} K is outside the DFT bins with >= {min_runs} runs ({good}): give --vpa")
    lo, hi = max(lo), min(hi)
    print(f"NOTE: {phase} volume at {T:.0f} K interpolated between the {lo} K and {hi} K DFT EOS fits")
    return list(_vpa_bin(df, phase, lo, Ps) + (_vpa_bin(df, phase, hi, Ps) - _vpa_bin(df, phase, lo, Ps)) * (T - lo) / (hi - lo))


def thermostat(atoms, T, dt, tau):
    try:
        from ase.md.bussi import Bussi
        return Bussi(atoms, timestep=dt * units.fs, temperature_K=T, taut=tau * units.fs)
    except ImportError:
        from ase.md.langevin import Langevin
        print("Bussi unavailable: Langevin, friction 1/tau")
        return Langevin(atoms, dt * units.fs, temperature_K=T, friction=1 / (tau * units.fs), fixcm=True)


def run_one(a, calc, vpa, tag):
    at = build(a.phase, vpa, a.nrep, a.seed); at.calc = calc
    V = at.get_volume(); N = len(at); nO = int((at.numbers == 8).sum())
    d = os.path.join(a.outdir, f"{tag}_v{vpa:.4f}"); os.makedirs(d, exist_ok=True)
    MaxwellBoltzmannDistribution(at, temperature_K=a.T, force_temp=True); Stationary(at)
    dyn = thermostat(at, a.T, a.dt, a.tau)
    n_eq, n_pr = int(a.equil_ps * 1000 / a.dt), int(a.ps * 1000 / a.dt)
    rows, pos, tpos = [], [], []
    t0 = time.time(); step = [0]

    def log():
        s = step[0]; ek = at.get_kinetic_energy(); T = 2 * ek / (3 * (N - 1) * units.kB)
        sig = at.get_stress(voigt=True)                       # eV/A^3, virial only (ASE sign: negative = compressed)
        P = -sig[:3].mean() * EV_A3_GPA + N * units.kB * T / V * EV_A3_GPA
        rows.append(dict(step=s, t_fs=s * a.dt, T=T, epot=at.get_potential_energy(), ekin=ek, P_GPa=P, **{f"s{k}": sig[k] * EV_A3_GPA for k in range(6)}))
        if s >= n_eq and s % int(10 / a.dt) == 0:
            pos.append(at.get_positions().copy()); tpos.append(s * a.dt)
        if s % int(500 / a.dt) == 0 and s:
            allr = pd.DataFrame(rows); prod = allr[allr.step >= n_eq]
            if len(prod) >= 10:
                w, phase = prod, "prod avg"
            else:   # still equilibrating: average over the last 250 fs rather than one instantaneous value
                w, phase = allr[allr.t_fs >= allr.t_fs.iloc[-1] - 250], "equil, last 250 fs"
            Tm, Pm = w["T"].mean(), w["P_GPa"].mean()
            Pe = np.std([x.mean() for x in np.array_split(w["P_GPa"].values, 5)], ddof=1) / np.sqrt(5) if len(w) >= 50 else float("nan")
            print(f"  {tag} vpa {vpa:.3f} step {s}/{n_eq + n_pr}  [{phase}, {len(w)} pts]  T {Tm:.0f} K  P {Pm:.1f} +- {Pe:.1f} GPa  "
                  f"(now T {T:.0f} P {P:.1f})  {(time.time() - t0) / s * 1000:.0f} ms/step", flush=True)
        step[0] += a.log_every
    dyn.attach(log, interval=a.log_every)
    dyn.attach(lambda: write(os.path.join(d, "traj.extxyz"), at, append=True) if step[0] >= n_eq else None, interval=int(20 / a.dt))
    dyn.run(n_eq + n_pr)
    th = pd.DataFrame(rows); th.to_csv(os.path.join(d, "thermo.csv"), index=False)
    write(os.path.join(d, "final.extxyz"), at)
    pr = th[th.step >= n_eq]
    blocks = np.array_split(pr.P_GPa.values, 5)
    P_blocks = [b.mean() for b in blocks]
    msd = {}
    if len(pos) > 20:
        P0 = np.array(pos); disp = P0 - P0[0]
        for el, sel in (("O", at.numbers == 8), ("H", at.numbers == 1)):
            m = (disp[:, sel] ** 2).sum(-1).mean(1); h = len(m) // 2
            msd[el] = np.polyfit(np.array(tpos)[h:], m[h:], 1)[0] / 6 * 0.1      # cm^2/s
            msd[f"msd_{el}_end"] = float(m[-1])
    struct = {}
    if len(pos) > 20:   # O-sublattice type in the 2nd half of production (time-averaged sites; COM removed), and its Lindemann ratio: did the lattice survive?
        P0 = np.array(pos); iO = at.numbers == 8; h = len(P0) // 2
        O = P0[h:, iO] - P0[h:, iO].mean(1, keepdims=True); mean = O.mean(0)
        lab, c1, c2, d1 = import_module("09_label_phase").o_structure(mean, at.cell.array)
        struct = dict(O_struct=lab, CN1=c1, CN2=c2, delta_L=float(np.sqrt(((O - mean) ** 2).sum(-1).mean()) / d1))
    row = dict(tag=tag, phase=a.phase, T_set=a.T, vpa=vpa, V=V, N=N, nO=nO, P_GPa=float(np.mean(P_blocks)), P_sem=float(np.std(P_blocks, ddof=1) / np.sqrt(5)),
               T_mean=float(pr["T"].mean()), epot_per_atom=float(pr.epot.mean() / N), D_H_cm2s=msd.get("H", np.nan), D_O_cm2s=msd.get("O", np.nan),
               msd_O_end=msd.get("msd_O_end", np.nan), msd_H_end=msd.get("msd_H_end", np.nan), **struct, ps=a.ps, ms_per_step=(time.time() - t0) / step[0] * 1000)
    print("RESULT", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="grace-ft:<dir> | grace:<name> | lj (harness test)")
    ap.add_argument("--phase", choices=["bcc", "fcc"], default="bcc"); ap.add_argument("--T", type=float, default=3000)
    ap.add_argument("--nrep", type=int, default=None, help="conventional cells per edge (bcc 4 -> 128 O, fcc 3 -> 108 O)")
    ap.add_argument("--vpa", type=float, nargs="*"); ap.add_argument("--from-dft", default=None, help="inventory csv: pick volumes from the DFT EOS")
    ap.add_argument("--P", type=float, nargs="*", default=[200, 300, 400, 500, 600], help="target pressures for --from-dft")
    ap.add_argument("--ps", type=float, default=4.0); ap.add_argument("--equil-ps", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--tau", type=float, default=100.0, help="thermostat time (fs)")
    ap.add_argument("--log-every", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", type=int, default=None, help="run only volume index k (for a Slurm array)")
    ap.add_argument("--tag", default=None); ap.add_argument("--outdir", default=None); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.nrep = a.nrep or CELL[a.phase][2]; a.outdir = a.outdir or outp("eos_md"); os.makedirs(a.outdir, exist_ok=True)
    vols = a.vpa or dft_vpa(a.from_dft, a.phase, a.T, a.P) if (a.vpa or a.from_dft) else None
    if not vols:
        raise SystemExit("give --vpa or --from-dft")
    tag = a.tag or f"{os.path.basename(a.model.split(':', 1)[-1].rstrip('/'))}_{a.phase}_{a.T:.0f}K"
    calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    print(f"{tag}: volumes (A^3/atom) {[round(v, 3) for v in vols]}  cell {a.phase} x{a.nrep}")
    for k, v in enumerate(vols):
        if a.only is not None and k != a.only:
            continue
        row = run_one(a, calc, v, tag)
        f = os.path.join(a.outdir, f"{tag}.csv")
        pd.DataFrame([row]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
