"""Run ONE snapshot through a foundation MLP and compare with the DFT labels. Checks the env/GPU/API work before real benchmarking.

usage (use the model's own python):  <env>/bin/python mlp_pipeline/smoke_test.py --model {mace,mattersim,grace,lj} [--name NAME] [--data valid_m.extxyz] [--max-atoms 260]
Model/class/function names for GRACE and MatterSim are NOT verified: the script searches the package for the calculator class and
prints what it found. Energies are printed but not compared (different reference energies): forces and stress are absolute.
"""
import argparse, importlib, os, pkgutil, sys, time
import numpy as np
from ase.io import read

EV_A3_GPA = 160.21766


def find_class(pkg_name, cls_name):
    pkg = importlib.import_module(pkg_name)
    for m in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        if hasattr(mod, cls_name):
            print(f"  found {cls_name} in {m.name}")
            return getattr(mod, cls_name)
    raise ImportError(f"{cls_name} not found in {pkg_name}")


def get_calc(model, name, device):
    if model == "mace":
        from mace.calculators import mace_mp
        nm, _, head = (name or "medium-mpa-0").partition("@")      # 'mace-mh-1@<head>' picks a head of a multihead model
        kw = {"head": head} if head else {}
        return mace_mp(model=nm, default_dtype=os.environ.get("MLP_DTYPE", "float64"), device=device, **kw)
    if model == "mattersim":
        C = find_class("mattersim", "MatterSimCalculator")
        try:
            return C(device=device) if not name else C(load_path=name, device=device)
        except TypeError:
            return C(device=device)
    if model == "grace":
        import tensorpotential.calculator as tc
        print("  tensorpotential.calculator exposes:", [x for x in dir(tc) if not x.startswith("_")][:25])
        # full names per the gracemaker docs; plain name = fp32, add -fp64 for double. GRACE_UQ=1 turns on the per-atom extrapolation grade gamma
        kw = {"enable_uq": True} if os.environ.get("GRACE_UQ") else {}
        return tc.grace_fm(name or "GRACE-2L-SMAX-OMAT-medium", **kw)
    if model == "lj":  # harness test only
        from ase.calculators.lj import LennardJones
        return LennardJones(sigma=1.0, epsilon=0.1, rc=4.0)
    raise SystemExit("unknown model")


def main(a):
    from common import inp
    frames = read(inp(a.data), index=":")
    pool = [f for f in frames if len(f) <= a.max_atoms] or frames
    f = pool[len(pool) // 2]
    ref_f, ref_s, ref_e = f.get_forces(), f.get_stress(voigt=False), f.get_potential_energy()
    from ase.neighborlist import neighbor_list
    I, J, Dd = neighbor_list("ijd", f, 3.0); sy = np.array(f.get_chemical_symbols()); md = {}
    for i_, j_, d_ in zip(I, J, Dd):
        k = "-".join(sorted((sy[i_], sy[j_]))); md[k] = min(md.get(k, 9.9), d_)
    print("  shortest distances (A): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(md.items())) + f"   volume/atom {f.get_volume() / len(f):.2f} A^3")
    print(f"snapshot: {f.info.get('config_type')}  {len(f)} atoms  T={f.info.get('T_K')} K  P_dft={f.info.get('P_GPa'):.0f} GPa  source={f.info.get('source')}")
    device = a.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("  WARNING: torch sees no GPU; using cpu"); device = "cpu"
        else:
            print("  torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    except ImportError:
        print("  (no torch in this env; fine for GRACE/TensorFlow)")
    t0 = time.time(); calc = get_calc(a.model, a.name, device); t_load = time.time() - t0; print(f"  model loaded in {t_load:.1f}s")
    g = f.copy(); g.calc = calc
    ts = []
    for i in range(6):  # call 1-2 = warm-up/graph build; steady state = median of calls 3-6 (fresh positions each time so nothing is cached)
        g.positions = f.positions + 1e-6 * i
        t0 = time.time(); e = g.get_potential_energy(); fo = g.get_forces(); s = g.get_stress(voigt=False); ts.append(time.time() - t0)
        print(f"  call {i + 1}: {ts[-1]:.2f}s")
    steady = float(np.median(ts[2:]))
    print(f"TIMING model={a.model}:{a.name or '-'} device={device} atoms={len(f)} threads={os.environ.get('OMP_NUM_THREADS', '-')} "
          f"load_s={t_load:.1f} first_s={ts[0]:.2f} steady_s={steady:.2f} ms_per_atom={1000 * steady / len(f):.1f}")
    rmse = float(np.sqrt(np.mean((fo - ref_f) ** 2))); ref_rms = float(np.sqrt(np.mean(ref_f ** 2)))
    P_m, P_d = -np.trace(s) / 3 * EV_A3_GPA, -np.trace(ref_s) / 3 * EV_A3_GPA
    print(f"  E/atom: model {e / len(f):.4f}  dft {ref_e / len(f):.4f} eV   (offset expected: different reference energies)")
    print(f"  force RMSE {rmse:.3f} eV/A  (DFT force RMS {ref_rms:.3f}; ratio {rmse / ref_rms:.2f})")
    print(f"  virial pressure: model {P_m:.1f} GPa, dft {P_d:.1f} GPa, diff {P_m - P_d:.1f}; stress MAE {np.mean(np.abs(s - ref_s)) * EV_A3_GPA:.1f} GPa")
    ok = np.isfinite(fo).all() and np.isfinite(s).all()
    print("SMOKE TEST", "OK" if ok else "FAILED (non-finite output)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mace", "mattersim", "grace", "lj"])
    ap.add_argument("--name", default=None, help="mace: medium | medium-mpa-0 | medium-omat-0 ...; grace: GRACE-2L-SMAX-OMAT-medium ...; mattersim: checkpoint")
    ap.add_argument("--data", default="valid_m.extxyz"); ap.add_argument("--max-atoms", type=int, default=260)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
