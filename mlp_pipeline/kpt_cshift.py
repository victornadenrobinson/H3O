#!/usr/bin/env python
"""Did the k-point correction transfer into the continued fine-tune? Compares, frame by frame on the DFT-MD composition-test frames,
the energy change the new model learned (E_new - E_base) with the correction it was trained on (N * de from kcorr.json).
Writes two Dmd-format csv files for composition_check.py (D = -shift, so their c is the shift of the composition test caused by each):
  runs/Dmd_zz_learned.csv  (D = -(E_new - E_base))   and   runs/Dmd_zz_pred.csv  (D = -N de)
  $PY mlp_pipeline/kpt_cshift.py runs/Dmd_mfin_f_lr1e-2_e50.csv runs/Dmd_mfinkc_cont.csv runs/kpt222mp/kcorr.json
  $PY mlp_pipeline/composition_check.py runs/Dmd_zz_learned.csv runs/Dmd_zz_pred.csv | grep -v "+   1 H"
Frames are read and ordered exactly as baseline_D.py does (same files, pure-H cells < 500 atoms dropped, stable sort by size); nO / nH are checked row by row.
"""
import argparse, csv, json, os, sys
import numpy as np
from ase.io import read
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
kc = import_module("kpt_correct")

ap = argparse.ArgumentParser(); ap.add_argument("base_csv"); ap.add_argument("new_csv"); ap.add_argument("kcorr")
ap.add_argument("--frames", nargs="+", default=["runs/dft_md_400.extxyz", "runs/dft_md_500.extxyz", "runs/dft_md_600.extxyz"])
ap.add_argument("--min-pureh-atoms", type=int, default=500); ap.add_argument("--out-prefix", default="runs/Dmd_zz")
a = ap.parse_args()
m = json.load(open(a.kcorr)); c = np.array(m["coef"])
fr = []
for p in a.frames:
    for f in read(p, index=":"):
        if "E_sigma0_eV" in f.info or "energy_sigma0" in f.info: fr.append(f)
fr = [f for f in fr if (f.numbers == 8).any() or len(f) >= a.min_pureh_atoms]; fr.sort(key=len)
rb = list(csv.DictReader(open(a.base_csv))); rn = list(csv.DictReader(open(a.new_csv)))
if not (len(rb) == len(rn) == len(fr)): sys.exit(f"row counts differ: base {len(rb)}, new {len(rn)}, frames {len(fr)}")
learned, pred, out_l, out_p = [], [], [], []
for f, b, n in zip(fr, rb, rn):
    nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum())
    if (int(b["nO"]), int(b["nH"])) != (nO, nH) or (int(n["nO"]), int(n["nH"])) != (nO, nH): sys.exit("frame order does not match the csv rows (nO/nH differ)")
    de = float(kc.feats(nO, nH, f.get_volume(), kc.T_el(f.info), 1.0)[0] @ c); P = len(f) * de / 1000          # eV per cell
    L = float(n["E_model"]) - float(b["E_model"]); learned.append(L / len(f) * 1000); pred.append(de)
    for rows, s in ((out_l, L), (out_p, P)):
        r = dict(b); r["E_model"] = 0.0
        for k in ("D_F", "D_E0", "D_E"): r[k] = -s
        rows.append(r)
for rows, tag in ((out_l, "learned"), (out_p, "pred")):
    w = csv.DictWriter(open(f"{a.out_prefix}_{tag}.csv", "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
L, Pp = np.array(learned), np.array(pred)
k = np.polyfit(Pp, L, 1) if Pp.std() > 0 else [np.nan, np.nan]
print(f"{len(L)} frames: learned E_new - E_base {L.mean():+.2f} +- {L.std():.2f} meV/atom, trained-on correction de {Pp.mean():+.2f} +- {Pp.std():.2f}; "
      f"learned = {k[0]:.2f} de {k[1]:+.2f}; rms(learned - de) {np.sqrt(((L - Pp) ** 2).mean()):.2f} meV/atom")
print(f"wrote {a.out_prefix}_learned.csv and {a.out_prefix}_pred.csv: run composition_check.py on both (c of each = its shift of the composition test)")
