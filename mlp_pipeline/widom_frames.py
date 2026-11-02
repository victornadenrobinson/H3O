"""C2 geometries for the PWidom reverse test, WITHOUT any VASP inputs (the labelling step writes those from the user's template, like for every other set).
For each cached pwidom host trajectory (<tag>_frames.npz from tdi/pwidom.py) pick --frames-n host frames (second half of the trajectory, evenly spaced) and, per frame,
  host                      the frame itself (E_host)
  host + H, --sel sites     H placed at Rosenbluth-sampled proxy sites (weights exp(-phi/kT)), the sites a real Widom insertion would use
  host + H, lowest-phi site the proxy minimum
so a frame costs 2 + --sel DFT points (default 4) and the DFT insertion energy E(host+H) - E(host) is a direct test of the model's insertion energy (reverse test: configurations the model samples).
All geometries of one frame share run_id, whole npz files go to one side of the split. Same selection logic as tdi/dft_check.py prep-widom.
usage:  python mlp_pipeline/widom_frames.py runs/pwidom/*_frames.npz --proxy docs/data/toyfit_t4.json --frames-n 8 --sel 2 --out widom_c2.extxyz [--dry]
"""
import argparse, glob, hashlib, os, re, sys
import numpy as np
from ase import Atoms
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
KB = 8.617333262e-5
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("npz", nargs="+"); ap.add_argument("--proxy", default="none"); ap.add_argument("--K", type=int, default=2000); ap.add_argument("--sel", type=int, default=2); ap.add_argument("--frames-n", type=int, default=8)
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="widom_c2.extxyz"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
from tdi import pwidom as pw
files = sorted({p for g in a.npz for p in glob.glob(g)}); rng = np.random.default_rng(a.seed); proxy = pw.proxy_params(a.proxy); n_out = 0; tot = 0
if os.path.exists(a.out) and not a.dry: os.remove(a.out)
for path in files:
    z = np.load(path, allow_pickle=True); fh = z["host"]; numbers = z["numbers"]; V = float(z["V"]); T = float(z["T"]); kT = KB * T
    cell = np.asarray(z["cell"]) if "cell" in z.files else np.eye(3) * V ** (1 / 3); Ph = np.asarray(z["P_host"]) if "P_host" in z.files else None
    nO = int((numbers == 8).sum()); nH = int((numbers == 1).sum()); isH = numbers == 1; tag = os.path.basename(path).replace("_frames.npz", "")
    split = "valid" if int(hashlib.md5(tag.encode()).hexdigest(), 16) % 3 == 0 else "train"
    h0 = len(fh) // 2; idx = h0 + np.linspace(0, len(fh) - h0 - 1, min(a.frames_n, len(fh) - h0)).round().astype(int)
    m = re.search(r"(\d+)GPa", tag); Pt = float(m.group(1)) if m else float("nan"); per = 2 + a.sel; tot += per * len(idx)
    print(f"{tag}: {len(fh)} frames, {len(numbers)} atoms (H/O {nH}/{nO}), T {T:.0f} K, split {split}: {len(idx)} frames x {per} points", flush=True)
    if a.dry: continue
    Lw = np.diag(cell) if np.allclose(cell, np.diag(np.diag(cell))) else None
    for f in idx:
        x = fh[f]; base = dict(kind="widom", run_id=f"{tag}:{int(f)}", source=path, split=split, T_K=T, P_target=Pt, nO=nO, nH=nH, widom_frame=int(f))
        if Ph is not None: base["P_grace_GPa"] = float(Ph[f])
        host = Atoms(numbers=numbers, positions=x, cell=cell, pbc=True); host.info.update(base, widom_role="host", config_type=f"widom_host_{nH}H{nO}O_{int(T)}K"); write(a.out, host, append=True); n_out += 1
        c = rng.random((a.K, 3)) @ cell; phi = pw.proxy_energy(proxy, c, x, isH, cell); w = -phi / kT; pr = np.exp(w - pw.lse(w)); pr = pr / pr.sum()
        picks = [("prox", int(k)) for k in rng.choice(a.K, a.sel, p=pr)] + [("lowphi", int(np.argmin(phi)))]
        for tier, k in picks:
            b = Atoms(numbers=np.r_[numbers, 1], positions=np.vstack([x, c[k]]), cell=cell, pbc=True)
            b.info.update(base, widom_role=tier, phi_proxy=float(phi[k]), nH=nH + 1, config_type=f"widom_site_{nH + 1}H{nO}O_{int(T)}K"); write(a.out, b, append=True); n_out += 1
print(f"{len(files)} files; {tot} DFT points" + ("" if a.dry else f"; wrote {n_out} frames to {a.out}"))
