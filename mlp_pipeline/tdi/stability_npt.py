"""Long constant-pressure MD with a FULLY FLEXIBLE cell (ASE NPT, Melchionna / Nose-Hoover + Parrinello-Rahman): is superionic fcc H(2+n)O (e.g. H2.1O, H2.2O) a stable phase
over tens of ps, or does it deform / change O sublattice / melt / phase-separate? NVT (17_eos_md.py) cannot show this: a fixed cell holds the lattice in shape.
Printed every --block-ps (default 1 ps), from positions time-averaged over that block in fractional coordinates with the block-mean cell (O centre of mass removed):
  T, P, V/atom, cell lengths a b c and angles, shear strain (deviation of the cell from a uniformly scaled start cell), O sublattice type (fcc / bcc / other, 09_label_phase),
  Lindemann ratio of O, O mean-square displacement from the start (O diffusing = melting), H diffusion coefficient (superionic = fast H, O on its lattice).
Final VERDICT line: stable superionic fcc if over the second half the O sublattice is fcc in every block, the SYSTEMATIC shear strain (cell averaged over the second half) is < 3 %,
O MSD does not grow (< 0.5 A^2 per 10 ps) and H diffuses. The largest block-mean shear is printed too but is not a criterion: the block-mean cell of a small cell fluctuates by several % thermally
(toy test, 32 O: up to 4.5 % with a systematic shear of ~1 %; it falls like N^-1/2).
usage:  ~/grace-venv/bin/python -u mlp_pipeline/tdi/stability_npt.py --hper 2.1 --P 400 --T 4500 [--ps 30] [--nrep 4] [--model grace-ft:runs/grace/ft_m2] [-o runs/stability_npt]
        batch: sbatch mlp_pipeline/young_stability_npt.sbatch   (H2O, H2.1O, H2.2O at 400 GPa 4500 K and 500 GPa 5000 K)
Cell: fcc O, nrep^3 conventional cells (nrep 4 = 256 O; H2.1O -> 538 H = H2.102O, H2.2O -> 563 H = H2.199O), extra H on random O-O bond midpoints (17_eos_md.build).
Barostat: ttime 25 fs, ptime 100 fs, pfactor = ptime^2 B with B ~ 4.5 P (rough bulk modulus; only sets the barostat time scale). The cell must stay upper triangular (ASE NPT).
"""
import argparse, json, os, sys, time
import numpy as np
from importlib import import_module

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
EV_A3_GPA = 160.21766


def strain(h0, h):
    """shear / non-uniform strain: symmetric part of (h0^-1 h) / det^(1/3) minus identity, largest |eigenvalue| (0 = only uniform scaling)"""
    F = np.linalg.solve(h0, h); F = F / abs(np.linalg.det(F)) ** (1 / 3); E = 0.5 * (F + F.T) - np.eye(3)
    return float(np.abs(np.linalg.eigvalsh(E)).max())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hper", type=float, default=2.1); ap.add_argument("--P", type=float, default=400); ap.add_argument("--T", type=float, default=4500)
    ap.add_argument("--ps", type=float, default=30.0); ap.add_argument("--equil-ps", type=float, default=2.0, help="first part not used for the verdict (barostat settling)")
    ap.add_argument("--block-ps", type=float, default=1.0); ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--vpa", type=float, default=None, help="start V/atom (default: rough guess; the barostat corrects it within ~1-2 ps)")
    ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ttime", type=float, default=25.0); ap.add_argument("--ptime", type=float, default=100.0); ap.add_argument("-o", default="runs/stability_npt"); a = ap.parse_args()
    from ase import units
    from ase.io import write
    try:
        from ase.md.melchionna import MelchionnaNPT as NPT   # ase.md.npt.NPT is deprecated (FutureWarning) in new ASE versions; same arguments
    except ImportError:
        from ase.md.npt import NPT
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    md = import_module("17_eos_md"); lab = import_module("09_label_phase").o_structure
    vpa = a.vpa or 2.25 * (a.P / 400) ** -0.33 * (3 + 0.62 * (a.hper - 2)) / (3 + (a.hper - 2))   # rough: ~2.25 A^3/atom for H2O at 400 GPa, 5000 K; extra H ~0.6 of an H2O atom
    at = md.build("fcc", vpa, a.nrep, a.seed, a.hper); at.calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    nO = int((at.numbers == 8).sum()); nH = int((at.numbers == 1).sum()); N = len(at); iO = at.numbers == 8; iH = ~iO
    tag = f"fcc_H{nH / nO:.3f}O_{a.P:.0f}GPa_{a.T:.0f}K_n{a.nrep}"; d = os.path.join(a.o, tag); os.makedirs(d, exist_ok=True)
    print(f"{tag}: {nO} O + {nH} H = H{nH / nO:.3f}O, {N} atoms, start V/atom {vpa:.3f}, NPT (flexible cell) {a.ps} ps at {a.P} GPa, {a.T} K, dt {a.dt} fs, model {a.model}", flush=True)
    MaxwellBoltzmannDistribution(at, temperature_K=a.T, force_temp=True, rng=np.random.default_rng(a.seed)); Stationary(at)   # seeded: the run is reproducible up to the barostat's own (deterministic) dynamics
    B = 4.5 * a.P / EV_A3_GPA
    dyn = NPT(at, timestep=a.dt * units.fs, temperature_K=a.T, externalstress=a.P / EV_A3_GPA, ttime=a.ttime * units.fs, pfactor=(a.ptime * units.fs) ** 2 * B)
    nb = int(a.block_ps * 1000 / a.dt); nblk = int(round(a.ps / a.block_ps)); every = 10
    h0 = at.cell.array.copy(); O0 = None; t0 = time.time(); rows = []; buf = dict(O=[], H=[], T=[], P=[], V=[], h=[])
    uf = at.get_scaled_positions().copy(); last = uf.copy()   # unwrapped FRACTIONAL positions: the cell changes, so time averages are taken in fractional coordinates

    def sample():
        nonlocal last
        s = at.get_scaled_positions(wrap=False); ds = s - last; ds -= np.round(ds); last = s; uf[:] += ds
        c = uf[iO].mean(0); buf["O"].append(uf[iO] - c); buf["H"].append(uf[iH] - c); buf["h"].append(at.cell.array.copy())
        ek = at.get_kinetic_energy(); T = 2 * ek / (3 * (N - 1) * units.kB); V = at.get_volume()
        buf["T"].append(T); buf["V"].append(V / N); buf["P"].append(-at.get_stress(voigt=True)[:3].mean() * EV_A3_GPA + N * units.kB * T / V * EV_A3_GPA)
    dyn.attach(sample, interval=every)
    print(f"{'t/ps':>6s} {'T/K':>6s} {'P/GPa':>7s} {'V/atom':>7s} {'a':>6s} {'b':>6s} {'c':>6s} {'alpha':>6s} {'beta':>6s} {'gamma':>6s} {'shear':>6s} {'O_lat':>6s} {'CN1':>4s} "
          f"{'Lind_O':>6s} {'msdO/A2':>7s} {'D_H cm2/s':>9s}   ms/step  ETA", flush=True)
    for k in range(nblk):
        for v in buf.values(): v.clear()
        dyn.run(nb)
        hm = np.mean(buf["h"], 0); O = np.array(buf["O"]) @ hm; H = np.array(buf["H"]) @ hm; mean = O.mean(0)   # block-mean cell; fractional -> Cartesian
        if O0 is None: O0 = buf["O"][0].copy()
        O0c = O0 @ hm
        L, c1, c2, d1 = lab(mean, hm)
        lind = float(np.sqrt(((O - mean) ** 2).sum(-1).mean()) / d1); msdO = float(((O[-1] - O0c) ** 2).sum(-1).mean())
        dt_s = every * a.dt * 1e-15; mh = ((H - H[0]) ** 2).sum(-1).mean(1); h = len(mh) // 2
        DH = np.polyfit(np.arange(h, len(mh)) * dt_s, mh[h:], 1)[0] * 1e-16 / 6
        from ase.geometry import cell_to_cellpar
        cp = cell_to_cellpar(hm); sh = strain(h0, hm); t = (k + 1) * a.block_ps; el = time.time() - t0
        r = dict(t_ps=t, T=float(np.mean(buf["T"])), P=float(np.mean(buf["P"])), vpa=float(np.mean(buf["V"])), a=cp[0], b=cp[1], c=cp[2], alpha=cp[3], beta=cp[4], gamma=cp[5],
                 shear=sh, O_struct=L, CN1=c1, lind_O=lind, msd_O=msdO, D_H=float(DH), cell=hm.tolist())
        rows.append(r)
        print(f"{t:6.1f} {r['T']:6.0f} {r['P']:7.1f} {r['vpa']:7.4f} {cp[0]:6.2f} {cp[1]:6.2f} {cp[2]:6.2f} {cp[3]:6.1f} {cp[4]:6.1f} {cp[5]:6.1f} {100 * sh:5.1f}% {L:>6s} {c1:4.1f} "
              f"{lind:6.3f} {msdO:7.3f} {DH:9.2e}   {el / ((k + 1) * nb) * 1000:5.0f}  ~{el / (k + 1) * (nblk - k - 1) / 60:.0f} min", flush=True)
        write(os.path.join(d, "traj.extxyz"), at, append=True)
        json.dump(dict(args=vars(a), tag=tag, nO=nO, nH=nH, blocks=rows), open(os.path.join(d, "stability.json"), "w"), indent=1, default=float)
    late = [r for r in rows if r["t_ps"] > a.equil_ps + (a.ps - a.equil_ps) / 2]
    fcc_all = all(r["O_struct"] == "fcc" for r in late); shear_max = max(r["shear"] for r in late); shear_sys = strain(h0, np.mean([r["cell"] for r in late], 0))   # systematic: cell averaged over the late blocks
    grow = (late[-1]["msd_O"] - late[0]["msd_O"]) / max(late[-1]["t_ps"] - late[0]["t_ps"], 1e-9) * 10 if len(late) > 1 else float("nan")
    DH = np.mean([r["D_H"] for r in late])
    ok = fcc_all and shear_sys < 0.03 and grow < 0.5 and DH > 1e-5
    print(f"\nVERDICT {tag}: {'STABLE superionic fcc' if ok else 'NOT stable as superionic fcc'} over the 2nd half ({len(late)} blocks): O sublattice "
          f"{'fcc throughout' if fcc_all else 'NOT always fcc: ' + ','.join(sorted({r['O_struct'] for r in late}))}, systematic shear {100 * shear_sys:.1f} % (largest block-mean shear {100 * shear_max:.1f} %), O MSD growth {grow:.2f} A^2 per 10 ps, "
          f"D_H {DH:.1e} cm^2/s ({'superionic' if DH > 1e-5 else 'H not diffusing'}), <V/atom> {np.mean([r['vpa'] for r in late]):.4f}, <P> {np.mean([r['P'] for r in late]):.1f} GPa", flush=True)
    print(f"files: {d}/stability.json, traj.extxyz (one frame per block)")


if __name__ == "__main__":
    main()
