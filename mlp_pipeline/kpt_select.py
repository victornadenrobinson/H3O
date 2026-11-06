#!/usr/bin/env python3
"""Choose ~300 training / validation frames for 2x2x2 k-point relabelling (the k-point correction of the Gamma-only labels), stratified so every regime is covered.
  $PY mlp_pipeline/kpt_select.py runs/train_mfin.extxyz runs/valid_mfin.extxyz --n 300 --out runs/kpt222/frames.extxyz
Strata = data group x H fraction (0.05 bins) x DFT pressure bin x temperature bin; frames are drawn round-robin over the strata (random within a stratum, fixed seed), so rare
regimes (low P, cold, pure H, H-rich) get frames before the big 500 GPa sets get more. Cells above --big-atoms (the 686-729 atom pure-H cells, ~1.7x the cost) are capped at --max-big.
Each chosen frame keeps its Gamma label (energy, forces, stress; info E_gamma_eV) for the correction fit, and the electronic temperature of that label: the SIGMA in the INCAR of
its original point (info point_dir) when that exists, otherwise its T_K, exactly as prepare_points.py derives SIGMA (Fermi smearing at kB*max(T, 1000 K); 5000 K when T is missing).
The k-point label must use the same SIGMA, otherwise the difference is not a k-point effect. Prints the coverage table and where each frame's SIGMA came from."""
import argparse, collections, hashlib, os, re
import numpy as np
from ase.io import read, write

KB = 8.617333e-5
GROUPS = [("AIMD tier m", {"old"}), ("insertions A/B", {"A", "A_b1", "A_part", "B"}), ("PWidom caches C", {"C", "C2"}), ("EOS/melt D,E", {"D_eos", "D_melt", "E"}),
          ("toy + pure H F,G", {"F", "F_toy", "F_H", "G", "T"}), ("H3O supercells H", {"H"}), ("pairs + trials PD,PW", {"PD", "PW"}), ("low-P LP", {"LP"})]


def setname(f):
    v = f.info.get("db_set", "old")
    return "F_toy" if v is False else "T" if v is True else str(v)


def group(f):
    s = setname(f)
    return next((g for g, ss in GROUPS if s in ss), "other:" + s)


def pbin(P): return "P<150" if P < 150 else "P150-350" if P < 350 else "P350-450" if P < 450 else "P450-550" if P < 550 else "P550-700" if P < 700 else "P>700"


def tbin(T):
    if T is None or not np.isfinite(T) or T < 1500: return "T<1500/static"
    return "T1500-3500" if T < 3500 else "T3500-5500" if T < 5500 else "T5500-7500" if T < 7500 else "T>7500"


def sigma_T(f):
    """(electronic temperature in K that reproduces the Gamma label's SIGMA via prepare_points, source)"""
    d = f.info.get("point_dir")
    if d and os.path.exists(os.path.join(str(d), "INCAR")):
        for l in open(os.path.join(str(d), "INCAR")):
            m = re.match(r"\s*SIGMA\s*=\s*([0-9.eE+-]+)", l, re.I)
            if m: return float(m.group(1)) / KB, "INCAR of the original point"
    T = f.info.get("T_K")
    if T is None or not np.isfinite(float(T)): return 5000.0, "no T_K: prepare_points default 5000 K (as the original label)"
    return max(float(T), 1000.0), "T_K (prepare_points rule)"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", default="runs/kpt222/frames.extxyz")
    ap.add_argument("--big-atoms", type=int, default=600); ap.add_argument("--max-big", type=int, default=30); ap.add_argument("--seed", default="kpt222"); a = ap.parse_args()
    fr = []
    for p in a.files:
        for i, f in enumerate(read(p, index=":")):
            f.info["kpt_src"] = f"{os.path.basename(p)}:{i}"; fr.append(f)
    strata = collections.defaultdict(list)
    for f in fr:
        nO, nH = int((f.numbers == 8).sum()), int((f.numbers == 1).sum()); P = -float(np.trace(f.get_stress(voigt=False))) / 3 * 160.21766
        T = f.info.get("T_K", f.info.get("T_set")); T = float(T) if T not in (None, "") else None
        key = (group(f), round(nH / (nH + nO) / 0.05) * 0.05, pbin(P), tbin(T)); f.info["_P"] = P; strata[key].append(f)
    rng = np.random.default_rng(int(hashlib.md5(a.seed.encode()).hexdigest()[:8], 16))
    order = sorted(strata, key=lambda k: hashlib.md5((a.seed + repr(k)).encode()).hexdigest())
    for k in order: rng.shuffle(strata[k])
    pick, big = [], 0
    while len(pick) < a.n and any(strata[k] for k in order):
        for k in order:
            if len(pick) >= a.n: break
            while strata[k]:
                f = strata[k].pop()
                if len(f) > a.big_atoms:
                    if big >= a.max_big: continue
                    big += 1
                pick.append((k, f)); break
    out, src = [], collections.Counter()
    for k, f in pick:
        Te, how = sigma_T(f); src[how] += 1
        g = f.copy(); g.calc = f.calc; g.info = dict(f.info); g.info.pop("_P", None)
        g.info.update(T_K=Te, T_label_K=f.info.get("T_K", f.info.get("T_set", "")), E_gamma_eV=float(f.get_potential_energy()), kpt_stratum="|".join(map(str, k)), db_set=setname(f))
        out.append(g)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if os.path.exists(a.out): os.remove(a.out)
    write(a.out, out)
    print(f"{len(out)} frames from {len(fr)} ({len(strata)} strata); {big} cells > {a.big_atoms} atoms (cap {a.max_big}) -> {a.out}")
    for title, fn in (("group", lambda k: k[0]), ("pressure", lambda k: k[2]), ("temperature", lambda k: k[3]), ("H fraction", lambda k: f"{k[1]:.2f}")):
        c = collections.Counter(fn(k) for k, _ in pick)
        print(f"  {title:12s}: " + ", ".join(f"{x} {n}" for x, n in sorted(c.items())))
    na = collections.Counter(len(f) for _, f in pick); print("  atoms       : " + ", ".join(f"{n} x{c}" for n, c in sorted(na.items())))
    print("  SIGMA from  : " + "; ".join(f"{h}: {n}" for h, n in src.items()))
    cost = sum((1900 if len(f) > a.big_atoms else 1150) for _, f in pick) / 3600
    print(f"  cost estimate at the measured 2x2x2 speed without KPAR: {cost:.0f} lane-hours = {cost / 4:.0f} node-hours (4 lanes x 32 ranks); KPAR should reduce it")


if __name__ == "__main__":
    main()
