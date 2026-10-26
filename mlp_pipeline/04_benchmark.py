"""Zero-shot / fine-tuned MLP vs DFT. usage: python 04_benchmark.py [--n 60 --max-atoms 220 --device cpu]   {mace[:medium-omat-0]|grace[:NAME]|mattersim|mace-ft:/path/model} data.extxyz
Energy MAE after per-element shift fit; forces/stress absolute. NOTE: GRACE/MatterSim APIs unverified."""
import os
import sys
import numpy as np
import pandas as pd
from ase.io import read
from common import progress, out, inp

EV_A3_TO_GPA = 160.21766


def get_calc_spec(spec, device="cuda"):
    """spec: 'mace' | 'mace:medium-omat-0' | 'mattersim' | 'grace:GRACE-2L-OMAT' | 'mace-ft:/path/model.model' | 'grace-ft:/path/final_model' | 'lj'"""
    if spec.startswith("mace-ft:"):
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=spec.split(":", 1)[1], device=device, default_dtype="float64")
    if spec.startswith("grace-ft:"):   # exported gracemaker model dir (e.g. runs/grace/ft_s/final_model)
        import tensorpotential.calculator as tc
        kw = {"enable_uq": True} if os.environ.get("GRACE_UQ") else {}
        import glob
        p = os.path.expanduser(spec.split(":", 1)[1])
        if not os.path.exists(os.path.join(p, "saved_model.pb")):   # gracemaker writes final_model under seed/<n>/; search below the given dir
            hit = sorted(glob.glob(os.path.join(p, "**", "saved_model.pb"), recursive=True))   # only below the given dir: never another run's model
            if not hit:
                raise SystemExit(f"no saved_model.pb under {p}: run `find <run dir> -name saved_model.pb`")
            p = os.path.dirname(hit[0]); print("using", p)
        try:
            return tc.TPCalculator(p, **kw)
        except TypeError:
            return tc.TPCalculator(p)
    from smoke_test import get_calc
    kind, _, name = spec.partition(":")
    return get_calc(kind, name or None, device)


def sample_frames(frames, n, max_atoms, seed=0, min_box=0.0):
    """Stratified by config_type (composition_T), at most max_atoms atoms; n total (0 = all)."""
    import random
    fr = [f for f in frames if len(f) <= max_atoms] if max_atoms else list(frames)
    if min_box:
        n0 = len(fr); fr = [f for f in fr if f.get_volume() ** (1 / 3) >= min_box]
        print(f"--min-box {min_box} A: kept {len(fr)} of {n0} frames")
    if not n or n >= len(fr):
        return fr
    rng = random.Random(seed); by = {}
    for f in fr:
        by.setdefault(f.info.get("config_type"), []).append(f)
    for v in by.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by.values()):
        for k in list(by):
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def main(model, path, n=0, max_atoms=0, device="cuda", min_box=0.0):
    frames = sample_frames(read(inp(path), index=":"), n, max_atoms, min_box=min_box)
    print(f"{len(frames)} frames, {len(set(f.info.get('config_type') for f in frames))} config types, device {device}")
    calc = get_calc_spec(model, device)
    rows, E_d, E_m, cnt, skipped = [], [], [], [], []
    for a in progress(frames, desc=f"{model} eval"):
        ref_f, ref_s, ref_e = a.get_forces(), a.get_stress(voigt=False), a.get_potential_energy()
        b = a.copy(); b.calc = calc
        try:
            f, s, e = b.get_forces(), b.get_stress(voigt=False), b.get_potential_energy()
        except Exception as ex:  # GPU out-of-memory on big dense cells: skip the frame, free memory, keep going
            if "out of memory" not in str(ex).lower() and type(ex).__name__ != "OutOfMemoryError":
                raise
            skipped.append((len(a), a.info.get("config_type")))
            b.calc = None
            try:
                import torch, gc; gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        gam = getattr(calc, "results", {}).get("gamma") if hasattr(calc, "results") else None
        n = len(a); syms = a.get_chemical_symbols()
        cnt.append([syms.count("H"), syms.count("O")]); E_d.append(ref_e); E_m.append(e)
        rows.append(dict(comp=a.info.get("config_type"), T=a.info.get("T_K"), P_dft=a.info.get("P_GPa"),
                         f_se=np.mean((f - ref_f) ** 2), s_ae=np.mean(np.abs(s - ref_s)) * EV_A3_TO_GPA,
                         dP=-np.trace(s - ref_s) / 3 * EV_A3_TO_GPA, n=n,
                         gamma_max=float(np.max(gam)) if gam is not None else np.nan, gamma_mean=float(np.mean(gam)) if gam is not None else np.nan))
    if skipped:
        print(f"SKIPPED {len(skipped)} frames (GPU out of memory), atoms: {sorted(x[0] for x in skipped)}; use --max-atoms or a bigger GPU")
    if not rows:
        raise SystemExit("no frame fit in memory")
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
    if df.gamma_max.notna().any():
        print("\nUQ gamma (<=1: environment similar to the model's training data): mean of per-frame max gamma by T")
        print(df.groupby("T").gamma_max.agg(["mean", "max"]).round(2).to_string())
        print("frames with gamma_max > 1:", int((df.gamma_max > 1).sum()), "of", len(df))
    df.to_csv(out(f"bench_{model.replace('/', '_').replace(':', '-')[:60]}_{os.path.basename(path).rsplit('.', 1)[0]}.csv"), index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("data")
    ap.add_argument("--n", type=int, default=0, help="stratified sample of N frames (0 = all)")
    ap.add_argument("--max-atoms", type=int, default=0, help="skip cells larger than this (0 = no limit); speeds up CPU runs")
    ap.add_argument("--min-box", type=float, default=0.0, help="only frames with cell edge V^(1/3) >= this (A); 9.0 matches the training rule")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.model, a.data, a.n, a.max_atoms, a.device, a.min_box)
