"""Zero-shot benchmark of foundation MLPs against DFT extxyz snapshots."""
import numpy as np
from ase.io import read

EV_A3_TO_GPA = 160.21766208


def get_calculator(name, device="cuda", **kw):
    name = name.lower()
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model=kw.get("model", "medium"), device=device, default_dtype="float64")
    if name == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device=device)
    if name == "grace":
        from tensorpotential.calculator import grace_fm
        return grace_fm(kw.get("model", "GRACE-2L-OMAT"))
    raise ValueError(f"unknown model {name}")


def compare(frames, calc):
    de, df, ds = [], [], []
    for a in frames:
        n = len(a)
        ref_e, ref_f = a.info["energy"], a.arrays["forces"]
        ref_s = a.info.get("stress")
        b = a.copy()
        b.calc = calc
        de.append((b.get_potential_energy() - ref_e) / n)
        df.append((b.get_forces() - ref_f).ravel())
        if ref_s is not None:
            ds.append((b.get_stress() - np.asarray(ref_s)) * EV_A3_TO_GPA)
    return metrics(de, df, ds)


def metrics(de, df, ds):
    m = {
        "energy_MAE_eV_per_atom": float(np.mean(np.abs(de))),
        "force_RMSE_eV_per_A": float(np.sqrt(np.mean(np.concatenate(df) ** 2))),
    }
    if ds:
        m["stress_MAE_GPa"] = float(np.mean(np.abs(np.concatenate([np.ravel(s) for s in ds]))))
    return m


def run(extxyz, models=("mace", "grace", "mattersim"), device="cuda"):
    frames = read(extxyz, index=":")
    return {m: compare(frames, get_calculator(m, device)) for m in models}
