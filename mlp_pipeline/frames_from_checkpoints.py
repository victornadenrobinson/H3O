#!/usr/bin/env python3
"""Build <prefix>_frames.npz from the MD checkpoints of a (possibly unfinished) pwidom worker, so the insertion/deletion trials can run on the frames sampled so far.
   pwidom.py skips the MD whenever <out>/<tag>_frames.npz exists, so after this a normal resume (dft_pwidom_runner.py PLAN, or pwidom.py --resume-from PREFIX --resume-in-place) goes straight to the trials.
     python mlp_pipeline/frames_from_checkpoints.py PREFIX [--min-host 8] [--min-full 8] [--max-frames N] [--dry]
   PREFIX = <worker dir>/pw   (reads PREFIX_host_md.npz and PREFIX_full_md.npz, writes PREFIX_frames.npz; refuses to overwrite).
   Frames are the equilibrated sampling window only (checkpoints store frames after the equilibration steps). Fewer frames = a larger statistical error bar, not a bias.
   The checkpoints are replaced atomically, so reading them while the job runs is safe; stop the job (or let it stop) before running the trials."""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--min-host", type=int, default=8); ap.add_argument("--min-full", type=int, default=8)
ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
out = a.prefix + "_frames.npz"
if os.path.exists(out): sys.exit(f"{out} exists: refusing to overwrite (move it away first)")
z = {}
for k in ("host", "full"):
    p = f"{a.prefix}_{k}_md.npz"
    if not os.path.isfile(p): sys.exit(f"missing {p}")
    with np.load(p, allow_pickle=False) as d: z[k] = {x: np.array(d[x]) for x in ("metadata", "numbers", "cell", "frames", "pressures", "step")}
md = {k: json.loads(str(z[k]["metadata"])) for k in z}
for k in ("model_fp", "T"):
    if md["host"][k] != md["full"][k]: sys.exit(f"host and full checkpoints disagree on {k}: {md['host'][k]} vs {md['full'][k]}")
if not np.array_equal(z["host"]["cell"], z["full"]["cell"]): sys.exit("host and full cells differ")
n_h, n_f = len(z["host"]["numbers"]), len(z["full"]["numbers"])
if n_f != n_h + 1 or not np.array_equal(z["full"]["numbers"][:n_h], z["host"]["numbers"]) or z["full"]["numbers"][-1] != 1: sys.exit("full system must be host + 1 H (H last)")
nh, nf = len(z["host"]["frames"]), len(z["full"]["frames"])
print(f"host: {nh} frames (step {int(z['host']['step'])}/{md['host']['equil_steps'] + md['host']['frames'] * md['host']['every']}), host + 1 H: {nf} frames (step {int(z['full']['step'])}); T {md['host']['T']} K, {n_h} + 1 atoms")
if nh < a.min_host or nf < a.min_full: sys.exit(f"not enough frames (need host >= {a.min_host}, full >= {a.min_full}): wait for the MD, or lower --min-host/--min-full")
fh, Ph, fp, Pf = z["host"]["frames"], z["host"]["pressures"], z["full"]["frames"], z["full"]["pressures"]
if a.max_frames: fh, Ph, fp, Pf = fh[:a.max_frames], Ph[:a.max_frames], fp[:a.max_frames], Pf[:a.max_frames]
cell = z["host"]["cell"]; V = float(abs(np.linalg.det(cell)))
if a.dry: sys.exit(f"dry run: would write {out} with {len(fh)} host and {len(fp)} full frames, V = {V:.3f} A^3")
tmp = out + ".tmp.npz"
np.savez(tmp, host=fh, P_host=Ph, full=fp, P_full=Pf, V=V, T=float(md["host"]["T"]), natoms=n_h, numbers=z["host"]["numbers"], cell=cell, model="from-checkpoints",
         model_fp=md["host"]["model_fp"], segment_seed=0, segments=1)
os.replace(tmp, out); print(f"wrote {out}: {len(fh)} host + {len(fp)} full frames")
