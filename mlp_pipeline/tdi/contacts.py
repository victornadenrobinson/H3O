"""Closest O-O, O-H and H-H contacts per lambda value, from the saved node frames (<tag>_node<k>_traj.extxyz, every 20th sample). No MLP, CPU only.
usage (repo root, a python with ASE, e.g. ~/grace-venv/bin/python):
    python mlp_pipeline/tdi/contacts.py runs/tdi liquid_500GPa_5000K_h2o liquid_400GPa_5000K_h2o [--oo 0.6 --oh 0.5 --hh 0.4]
Why: in the fluid paths (pure H, liquid H(2+x)O) every atom starts as an ideal gas, so at small lambda the atoms overlap freely. GRACE's O-O and O-H dimer curves turn over
below ~0.15-0.2 A (tdi/pair_scan.py: barriers ~680 / ~170 eV, no ZBL); lambda times such a barrier is below kT for lambda < ~1e-3, so O can cross into the collapsed core,
where the energy may be far below anything physical. That would bias G of the liquid low (the fcc path keeps O on springs at small lambda; pure H has a monotonic H-H curve).
Per lambda: frames, min / 1st-percentile of the closest contact per frame for each pair type, and the fraction of frames with a contact closer than --oo / --oh / --hh.
Frames with an O-O contact below --oo at lambda values where the weight is not negligible are the suspects: re-evaluate them with the MLP (energy per atom) to see the collapse."""
import argparse, glob, json, os, re
import numpy as np
from ase.io import iread
from ase.neighborlist import neighbor_list

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tags", nargs="+")
ap.add_argument("--oo", type=float, default=0.6); ap.add_argument("--oh", type=float, default=0.5); ap.add_argument("--hh", type=float, default=0.4)
ap.add_argument("--cut", type=float, default=1.3, help="neighbour cutoff (A); contacts beyond it are reported as > cut")
a = ap.parse_args()
TH = {"OO": a.oo, "OH": a.oh, "HH": a.hh}


def closest(at):
    i, j, d = neighbor_list("ijd", at, a.cut)
    z = at.numbers; out = {"OO": np.inf, "OH": np.inf, "HH": np.inf}
    for key, m in (("OO", (z[i] == 8) & (z[j] == 8)), ("OH", (z[i] == 8) & (z[j] == 1)), ("HH", (z[i] == 1) & (z[j] == 1))):
        if m.any():
            out[key] = float(d[m].min())
    return out


for tag in a.tags:
    stf = os.path.join(a.dir, f"{tag}_stage0.json"); st = json.load(open(stf)) if os.path.exists(stf) else {}
    files = sorted(glob.glob(os.path.join(a.dir, f"{tag}_node*_traj.extxyz")), key=lambda f: int(re.search(r"_node(\d+)_traj", f).group(1)))
    print(f"\n=== {a.dir}/{tag}: {len(files)} node trajectories" + (f", T {st.get('T')} K, P {st.get('P_target')} GPa, V {st.get('V', 0):.1f} A^3" if st else ""))
    if not files:
        print("  no *_traj.extxyz files"); continue
    print(f"  {'node':>4} {'lambda':>10} {'frames':>6} | " + " | ".join(f"{k} min   p1  f(<{TH[k]:.2f})" for k in ("OO", "OH", "HH")))
    for f in files:
        k = int(re.search(r"_node(\d+)_traj", f).group(1)); c = {"OO": [], "OH": [], "HH": []}; lam = np.nan
        for at in iread(f, index=":"):
            lam = at.info.get("lam", lam)
            for key, v in closest(at).items():
                c[key].append(v)
        n = len(c["OO"]); cols = []
        for key in ("OO", "OH", "HH"):
            v = np.array(c[key])
            fin = v[np.isfinite(v)]
            cols.append(f"{(fin.min() if len(fin) else np.inf):6.3f} {(np.percentile(fin, 1) if len(fin) else np.inf):5.2f} {np.mean(v < TH[key]):7.3f}")
        flag = "  <-- O-O OVERLAP" if np.mean(np.array(c["OO"]) < a.oo) > 0 else ""
        print(f"  {k:>4} {lam:10.3e} {n:>6} | " + " | ".join(cols) + flag)
