"""C2 geometries for the PWidom reverse test, WITHOUT any VASP inputs (the labelling step writes those from the user's template, like for every other set).
For each cached pwidom host trajectory (<tag>_frames.npz from tdi/pwidom.py) pick --frames-n host frames (second half of the trajectory, evenly spaced) and, per frame,
  host                      the frame itself (E_host)
  host + H, --sel sites     H placed at Rosenbluth-sampled proxy sites (weights exp(-phi/kT)), the sites a real Widom insertion would use
  host + H, lowest-phi site the proxy minimum
so a frame costs 2 + --sel DFT points (default 4) and the DFT insertion energy E(host+H) - E(host) is a direct test of the model's insertion energy (reverse test: configurations the model samples).
All geometries of one frame share run_id, whole npz files go to one side of the split. Same selection logic as tdi/dft_check.py prep-widom.
PAIR DATA FOR THE INSERTION ERROR (--del-n, --split frame): deletion pairs too, from the cached host + 1 H ('full') frames: the full frame (role full) and the same frame with one randomly chosen H removed
(--ndel of them, role del; all H are identical, so the 'added' one is not special). Together the pairs are exactly the structures the Widom / deletion estimators evaluate, so labelling them gives (a) a direct measurement of the
insertion-energy error per H at that state (pair_check.py: mean, sd, var/2kT-corrected c) and (b) training data for the quantity we care about. --split frame holds out a fraction of FRAMES (all structures of one frame stay together)
instead of whole cache files, so every state is represented in both train and valid. Cost per state: frames-n x (1 + sel + 1) + del-n x (1 + ndel) DFT points.
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
ap.add_argument("--del-n", type=int, default=0, help="frames per cache for deletion pairs (from the host + 1 H frames)"); ap.add_argument("--ndel", type=int, default=2, help="deletions per full frame")
ap.add_argument("--split", choices=("tag", "frame"), default="tag", help="tag: whole cache to one side (C2, old behaviour); frame: --valid-frac of the frames held out in every cache"); ap.add_argument("--valid-frac", type=float, default=0.2)
ap.add_argument("--min-pureh-atoms", type=int, default=500, help="skip pure-H caches with fewer atoms: Gamma-only labels of small metallic H cells are not converged (tested: -150/-403 meV/H shift with a 2x2x2 mesh); the plan uses 729-atom cells only")
ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", default="widom_c2.extxyz"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
from tdi import pwidom as pw
files = sorted({p for g in a.npz for p in glob.glob(g)}); rng = np.random.default_rng(a.seed); proxy = pw.proxy_params(a.proxy); n_out = 0; tot = 0
if os.path.exists(a.out) and not a.dry: os.remove(a.out)
for path in files:
    z = np.load(path, allow_pickle=True); fh = z["host"]; numbers = z["numbers"]; V = float(z["V"]); T = float(z["T"]); kT = KB * T
    cell = np.asarray(z["cell"]) if "cell" in z.files else np.eye(3) * V ** (1 / 3); Ph = np.asarray(z["P_host"]) if "P_host" in z.files else None
    nO = int((numbers == 8).sum()); nH = int((numbers == 1).sum()); isH = numbers == 1; tag = os.path.basename(path).replace("_frames.npz", "")
    if nO == 0 and len(numbers) < a.min_pureh_atoms: print(f"{tag}: SKIPPED pure-H cache with {len(numbers)} atoms (< {a.min_pureh_atoms}: Gamma-only labels not converged)"); continue
    split_tag = "valid" if int(hashlib.md5(tag.encode()).hexdigest(), 16) % 3 == 0 else "train"
    split_of = (lambda f, kind="h": split_tag) if a.split == "tag" else (lambda f, kind="h": "valid" if int(hashlib.md5(f"{tag}:{kind}{int(f)}".encode()).hexdigest(), 16) % 1000 < 1000 * a.valid_frac else "train")
    h0 = len(fh) // 2; idx = h0 + np.linspace(0, len(fh) - h0 - 1, min(a.frames_n, len(fh) - h0)).round().astype(int)
    m = re.search(r"(\d+)GPa", tag); Pt = float(m.group(1)) if m else float("nan"); per = 2 + a.sel; nd = min(a.del_n, len(z["full"]) - len(z["full"]) // 2) if ("full" in z.files and a.del_n) else 0; tot += per * len(idx) + (1 + a.ndel) * nd
    print(f"{tag}: {len(fh)} frames, {len(numbers)} atoms (H/O {nH}/{nO}), T {T:.0f} K, split {a.split}: {len(idx)} host frames x {per} points + {a.del_n if 'full' in z.files else 0} full frames x {1 + a.ndel}", flush=True)
    if a.dry: continue
    Lw = np.diag(cell) if np.allclose(cell, np.diag(np.diag(cell))) else None
    for f in idx:
        x = fh[f]; base = dict(kind="widom", run_id=f"{tag}:{int(f)}", source=path, split=split_of(f), T_K=T, P_target=Pt, nO=nO, nH=nH, widom_frame=int(f))
        if Ph is not None: base["P_grace_GPa"] = float(Ph[f])
        host = Atoms(numbers=numbers, positions=x, cell=cell, pbc=True); host.info.update(base, widom_role="host", config_type=f"widom_host_{nH}H{nO}O_{int(T)}K"); write(a.out, host, append=True); n_out += 1
        c = rng.random((a.K, 3)) @ cell; phi = pw.proxy_energy(proxy, c, x, isH, cell); w = -phi / kT; pr = np.exp(w - pw.lse(w)); pr = pr / pr.sum()
        picks = [("prox", int(k)) for k in rng.choice(a.K, a.sel, p=pr)] + [("lowphi", int(np.argmin(phi)))]
        for tier, k in picks:
            b = Atoms(numbers=np.r_[numbers, 1], positions=np.vstack([x, c[k]]), cell=cell, pbc=True)
            b.info.update(base, widom_role=tier, phi_proxy=float(phi[k]), nH=nH + 1, config_type=f"widom_site_{nH + 1}H{nO}O_{int(T)}K"); write(a.out, b, append=True); n_out += 1
    if nd:
        ff_ = z["full"]; numbers1 = np.r_[numbers, 1]; isH1 = numbers1 == 1; PF = np.asarray(z["P_full"]) if "P_full" in z.files else None
        h1 = len(ff_) // 2; idd = h1 + np.linspace(0, len(ff_) - h1 - 1, nd).round().astype(int)
        for f in idd:
            x = ff_[f]; base = dict(kind="widom", run_id=f"{tag}:full{int(f)}", source=path, split=split_of(f, "full"), T_K=T, P_target=Pt, nO=nO, nH=nH + 1, widom_frame=int(f))
            if PF is not None: base["P_grace_GPa"] = float(PF[f])
            fu = Atoms(numbers=numbers1, positions=x, cell=cell, pbc=True); fu.info.update(base, widom_role="full", config_type=f"widom_full_{nH + 1}H{nO}O_{int(T)}K"); write(a.out, fu, append=True); n_out += 1
            for t in rng.choice(np.where(isH1)[0], min(a.ndel, int(isH1.sum())), replace=False):
                keep = np.ones(len(x), bool); keep[t] = False
                d = Atoms(numbers=numbers1[keep], positions=x[keep], cell=cell, pbc=True); d.info.update(base, widom_role="del", nH=nH, removed_atom=int(t), config_type=f"widom_del_{nH}H{nO}O_{int(T)}K"); write(a.out, d, append=True); n_out += 1
print(f"{len(files)} files; {tot} DFT points" + ("" if a.dry else f"; wrote {n_out} frames to {a.out}"))
