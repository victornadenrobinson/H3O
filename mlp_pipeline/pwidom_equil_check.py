#!/usr/bin/env python3
"""Equilibration / drift diagnostic for finished PWidom runs, from files the run already wrote (no GPU, no new MD).
For every host of every run dir it splits the sampled frames (time order) into blocks and prints, per block: potential energy per atom of the host frames, the MD pressure, and the Widom insertion mu_ex
(-kT ln mean exp(-w/kT) over that block's insertions). A trend across blocks means the host was not equilibrated when sampled (the energy scales of dg then depend on the run length and on the starting lattice);
a flat table means block-to-block scatter is the honest error. Also prints Widom - deletion (should be >= 0 asymptotically; a large or negative value = insertion/deletion not overlapping, estimator not converged).
   python3 mlp_pipeline/pwidom_equil_check.py runs/pwidom_endmem_new_500GPa_5000K runs/pwidom_xh_seed11_500GPa_5000K [--blocks 4]"""
import argparse, glob, json, os
import numpy as np
KB = 8.617333262e-5
ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); ap.add_argument("--blocks", type=int, default=4); a = ap.parse_args()
lse = lambda x: np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))
for run in a.runs:
    print(f"\n=== {run}")
    for ef in sorted(glob.glob(os.path.join(run, "*_energies.npz"))):
        base = ef[:-len("_energies.npz")]; jf, wfn = base + ".json", base + "_work.npz"
        if not (os.path.exists(jf) and os.path.exists(wfn)): continue
        r = json.load(open(jf)); kT = KB * r["T"]; n0 = r["n_frames_host"]; sel = r["sel"]
        E = np.load(ef)["energies"]; W = np.load(wfn, allow_pickle=True); wf = np.asarray(W["forward_work_eV"])          # (frames, sel)
        if len(E) < n0 or wf.shape[0] != n0: print(f"{os.path.basename(base)}: energies/work shapes do not match, skipped"); continue
        e0 = E[:n0] / r["natoms"] * 1e3; fid = np.asarray(W["host_frame_ids"]).astype(int)
        ff = base.rsplit("_", 2)[0] + "_frames.npz"; P = None
        if os.path.exists(ff):
            try: P = np.load(ff)["P_host"][fid]
            except Exception: P = None
        idx = np.array_split(np.arange(n0), a.blocks); tag = r["tag"]
        print(f"{tag}: {n0} host frames x {sel} insertions, T {r['T']:g} K, P target {r['P_target']:g}, P host {r['P_host_GPa']:.1f}, melted {r['melted']}, Kish {r['kish_frac_ins']:.3f}, Widom - deletion {r['mu_ex_widom_meV'] - r['mu_ex_delete_meV']:+.0f} meV, proxy gap {r['proxy_gap_std_kT_ins']:.2f} kT")
        print("   block   E_pot meV/atom    P host GPa    Widom mu_ex meV (this block)")
        for b, ii in enumerate(idx):
            w = wf[ii].ravel(); mu = -kT * (lse(-w / kT) - np.log(len(w))) * 1e3
            print(f"   {b + 1:>5d}   {e0[ii].mean():>12.1f}   {('%.1f' % P[ii].mean()) if P is not None else '   n/a':>10}   {mu:>12.0f}")
        h = n0 // 2; d = e0[h:].mean() - e0[:h].mean(); sem = np.hypot(e0[:h].std(ddof=1), e0[h:].std(ddof=1)) / np.sqrt(max(h, 1))
        print(f"   second half - first half of E_pot: {d:+.1f} +- {sem:.1f} meV/atom" + ("   DRIFT (not equilibrated)" if abs(d) > 2 * sem and abs(d) > 2 else ""))
