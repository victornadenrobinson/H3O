"""Zero-shot / fine-tuned MLP vs DFT. usage: python 04_benchmark.py {mace|grace|mattersim|mace-ft:/path/model} data.extxyz
Energy MAE after per-element shift fit; forces/stress absolute. NOTE: GRACE/MatterSim APIs unverified."""
import os
import sys
import numpy as np
import pandas as pd
from ase.io import read
from common import progress, out, inp

EV_A3_TO_GPA = 160.21766


def get_calc(name):
    if name == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", default_dtype="float64", device="cuda")
    if name.startswith("mace-ft:"):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=name.split(":", 1)[1], device="cuda", default_dtype="float64")
    if name == "grace":
        from tensorpotential.calculator import grace_fm
        return grace_fm("GRACE-2L-OMAT")  # check name against installed version
    if name == "mattersim":
        from mattersim.forward_observer import MatterSimCalculator  # check import path
        return MatterSimCalculator(device="cuda")
    raise SystemExit("unknown model")


def main(model, path):
    frames = read(inp(path), index=":")
    calc = get_calc(model)
    rows, E_d, E_m, cnt = [], [], [], []
    for a in progress(frames, desc=f"{model} eval"):
        ref_f, ref_s, ref_e = a.get_forces(), a.get_stress(voigt=False), a.get_potential_energy()
        b = a.copy(); b.calc = calc
        f, s, e = b.get_forces(), b.get_stress(voigt=False), b.get_potential_energy()
        n = len(a); syms = a.get_chemical_symbols()
        cnt.append([syms.count("H"), syms.count("O")]); E_d.append(ref_e); E_m.append(e)
        rows.append(dict(comp=a.info.get("config_type"), T=a.info.get("T_K"), P_dft=a.info.get("P_GPa"),
                         f_se=np.mean((f - ref_f) ** 2), s_ae=np.mean(np.abs(s - ref_s)) * EV_A3_TO_GPA,
                         dP=-np.trace(s - ref_s) / 3 * EV_A3_TO_GPA, n=n))
    cnt, dE = np.array(cnt, float), np.array(E_m) - np.array(E_d)
    shift, *_ = np.linalg.lstsq(cnt, dE, rcond=None)  # per-element reference shift
    for r, e, c in zip(rows, dE - cnt @ shift, cnt):
        r["e_ae"] = abs(e) / c.sum()
    df = pd.DataFrame(rows)
    g = lambda d: pd.Series(dict(E_MAE_meV_atom=1e3 * d.e_ae.mean(), F_RMSE_eV_A=np.sqrt(d.f_se.mean()),
                                 S_MAE_GPa=d.s_ae.mean(), P_err_GPa=d.dP.mean(), N=len(d)))
    print("overall\n", g(df).to_string())
    print("\nby comp\n", df.groupby("comp").apply(g).to_string())
    print("\nby T\n", df.groupby("T").apply(g).to_string())
    df.to_csv(out(f"bench_{model.split(':')[0]}_{os.path.basename(path).rsplit('.', 1)[0]}.csv"), index=False)


if __name__ == "__main__":
    main(*sys.argv[1:3])
