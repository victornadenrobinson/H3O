"""Turn an extxyz of geometries (e.g. from build_insertion_set.py) into single-point VASP directories for labelling, plus manifest.json and dirs.txt.

usage (repo root): python mlp_pipeline/prepare_points.py FRAMES.extxyz --template vasp_inputs --out runs/vasp_ins_5000K_500GPa [--max 400] [--seed 1] [--per-state 20]
  --template   a directory with INCAR / KPOINTS / POTCAR / POSCAR of the training DFT (kept local: this script copies from it at run time, nothing is stored in the repo)
  --per-state  keep at most this many frames per config_type (random), so one state does not use the whole budget
Directories are written with the same single-point writer as dft_check.py (MD tags removed, SIGMA = kB T of the frame's T_K, species sorted to the POTCAR order).
The manifest follows the schema of perlmutter_vasp_check.py (mode 'vasp_smoke', points with id/path/input_sha256/natoms) so its `run` command can execute them:
  bash mlp_pipeline/perlmutter_vasp_check.sh run OUT --exe vasp_gam --ranks 32 --parallel 16 --minutes 50
Check one point first (`run OUT --index 0 --max-points 1 --minutes 20`). Frame metadata (n_add, run_id, split, P_target, ...) is saved in OUT/frames_meta.csv.
"""
import argparse, csv, hashlib, json, os, sys
import numpy as np
from ase.io import read
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdi import dft_check as dft


def input_hash(d):
    h = hashlib.sha256()
    for name in ("INCAR", "KPOINTS", "POSCAR", "POTCAR"):
        h.update(name.encode()); h.update(open(os.path.join(d, name), "rb").read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames"); ap.add_argument("--template", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--per-state", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if os.path.exists(a.out): sys.exit(f"{a.out} exists: use a fresh directory")
    tp = dft.template(a.template)
    fr = read(a.frames, index=":"); rng = np.random.default_rng(a.seed); idx = np.arange(len(fr)); rng.shuffle(idx)
    if a.per_state:
        seen = {}; keep = []
        for i in idx:
            k = str(fr[i].info.get("config_type", "?")); seen[k] = seen.get(k, 0) + 1
            if seen[k] <= a.per_state: keep.append(i)
        idx = np.array(keep)
    if a.max: idx = idx[:a.max]
    idx = np.sort(idx); pts = []; meta = []
    os.makedirs(a.out)
    for j, i in enumerate(idx):
        f = fr[i]; T = float(f.info.get("T_K", 5000.0)); name = f"p{j:05d}"; d = os.path.join(a.out, name)
        at = f.copy(); at.calc = None
        dft.write_dft(d, at, T, tp, f"{os.path.basename(a.frames)} frame {i}")
        pts.append(dict(id=name, path=name, input_sha256=input_hash(d), natoms=len(at)))
        meta.append(dict(id=name, frame=int(i), natoms=len(at), **{k: v for k, v in f.info.items() if isinstance(v, (int, float, str, np.integer, np.floating))}))
    json.dump(dict(schema=1, mode="vasp_smoke", template=os.path.abspath(a.template), source_frames=os.path.abspath(a.frames), pairs=[], points=pts), open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    open(os.path.join(a.out, "dirs.txt"), "w").write("".join(os.path.join(os.path.abspath(a.out), p["path"]) + "\n" for p in pts))
    keys = sorted({k for m in meta for k in m}); w = csv.DictWriter(open(os.path.join(a.out, "frames_meta.csv"), "w", newline=""), fieldnames=keys); w.writeheader(); w.writerows(meta)
    print(f"prepared {len(pts)} single points in {a.out} (largest cell {max(p['natoms'] for p in pts)} atoms); no VASP launched")


if __name__ == "__main__":
    main()
