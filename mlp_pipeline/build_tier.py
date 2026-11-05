#!/usr/bin/env python3
"""Turn the labelled snapshot databases into a training tier in the same convention as train_m / valid_m (energy = FREE energy TOTEN, forces, 3x3 stress, info kept), plus nested fractions.
  python mlp_pipeline/build_tier.py snapdb/labelled.extxyz snapdb_v2/labelled.extxyz --out-dir runs --tag n [--old-train runs/train_m.extxyz --old-valid runs/valid_m.extxyz] [--fractions 25 50 75]
Writes  <out-dir>/train_<tag>.extxyz  valid_<tag>.extxyz            new labels only (split from the 'split' field kept through the whole campaign)
        train_<tag><p>.extxyz                                        nested fractions of the new TRAIN part (25 < 50 < 75 < 100), cut by whole group (run / base / source) and stratified by set, by a hash order, so
                                                                      the subsets are nested, reproducible, and stay the same when frames are added later; valid_<tag> is the same for all
        train_m<tag>.extxyz valid_m<tag>.extxyz  (+ train_m<tag><p>) replay versions: the old tier plus the new frames (the new part cut to p%)
Sanity filters: finite energy / forces, |force| < --fmax eV/A. Prints a table of frames per set, split and file.
Combined (reference / production) builds:
  --exclude db_set=PD src=C3      drop every frame whose info[key] equals the value (reference model: sealed pair sets and the C3 hydrate held out); reported per rule
  --all-train PD PW               these sets go to TRAIN whatever their campaign split (production model: the sealed pairs, labelled split=valid by widom_frames, are trained on)
  --state-group-sets LP           groups of these sets = the MD state (src without the _v<vpa> volume tag), so the valid hold-out of --valid-from takes whole compositions x T, never a
                                  volume twin of a train frame (low-P set from points_collect.py --db-set LP on runs/dft_lowp3/points_*)
  --pureh-pmax 150                small pure-H cells (< --min-pureh-atoms) are kept when their DFT pressure is below this (GPa: molecular, insulating, Gamma-only fine); 0 = drop all small pure H"""
import argparse, collections, hashlib, math, os, re, sys
import numpy as np
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out-dir", default="runs"); ap.add_argument("--tag", default="n")
ap.add_argument("--old-train"); ap.add_argument("--old-valid"); ap.add_argument("--fractions", type=int, nargs="*", default=[25, 50, 75]); ap.add_argument("--fmax", type=float, default=100.0); ap.add_argument("--valid-from", nargs="*", default=["F_toy", "F_H", "G", "LP"], help="sets that have no validation frames from the campaign split: this fraction of their groups goes to valid"); ap.add_argument("--valid-frac", type=float, default=0.25)
ap.add_argument("--max-atoms", type=int, default=0, help="drop TRAIN frames with more atoms than this (unfrozen fine-tunes run out of GPU memory on the 700+ atom dense cells); valid files keep everything")
ap.add_argument("--min-pureh-atoms", type=int, default=500, help="drop pure-H frames with fewer atoms (Gamma-only labels of small metallic H cells are not converged); 0 keeps them")
ap.add_argument("--pureh-pmax", type=float, default=0.0, help="keep small pure-H frames with DFT pressure below this (GPa); 0 = drop all of them (old behaviour)")
ap.add_argument("--exclude", nargs="*", default=[], help="KEY=VALUE rules: drop frames with str(info[KEY]) == VALUE (KEY db_set also matches the F_toy/T aliases)")
ap.add_argument("--all-train", nargs="*", default=[], help="db_set names whose frames all go to train, ignoring their split field")
ap.add_argument("--state-group-sets", nargs="*", default=["LP"], help="db_set names grouped by MD state (src minus the _v<vpa> tag) instead of per frame")
ap.add_argument("--seed", default="tier1"); a = ap.parse_args()
a.state_groups_set = set(a.state_group_sets)
EXCL = [tuple(r.split("=", 1)) for r in a.exclude]
if any(len(r) != 2 or not r[0] for r in EXCL): sys.exit(f"--exclude takes KEY=VALUE rules, got {a.exclude}")
os.makedirs(a.out_dir, exist_ok=True); P = lambda n: os.path.join(a.out_dir, n + ".extxyz")


drops = collections.defaultdict(list)


def setname(f):
    v = f.info.get("db_set", "?")
    return "F_toy" if v is False else "T" if v is True else str(v)        # extxyz reads the set name F as the boolean False


excluded = collections.Counter()


def clean(f):
    for k, v in EXCL:
        x = setname(f) if k == "db_set" else f.info.get(k)
        if x is not None and str(x) == v: excluded[f"{k}={v}"] += 1; return None
    F = f.get_potential_energy(); fo = f.get_forces(); st = f.get_stress(voigt=False)
    if a.min_pureh_atoms and not (f.numbers == 8).any() and len(f) < a.min_pureh_atoms:
        P = -float(np.trace(st)) / 3 * 160.21766 if np.isfinite(st).all() else float("inf")
        if not (a.pureh_pmax > 0 and P < a.pureh_pmax): drops["pureH_small"].append(P); return None
    bad = not (np.isfinite(F) and np.isfinite(fo).all() and np.isfinite(st).all()); fm = float(np.abs(fo).max()) if np.isfinite(fo).all() else float("inf")
    if bad or fm > a.fmax: drops[setname(f)].append(fm); return None
    g = f.copy(); g.info = dict(f.info); g.info["db_set"] = setname(f)
    if "E_sigma0_eV" in g.info: g.info["energy_sigma0"] = float(g.info.pop("E_sigma0_eV"))        # same name as the old tiers
    g.calc = SinglePointCalculator(g, energy=F, forces=fo, stress=st); return g


def group(f):
    if setname(f) in a.state_groups_set and re.search(r"_v[0-9.]+$", str(f.info.get("src", ""))): return "state:" + re.sub(r"_v[0-9.]+$", "", str(f.info["src"]))   # MD volumes; hydrates fall through (per frame)
    for k in ("run_id", "base_id", "source", "host_source"):
        if f.info.get(k) is not None: return f"{k}:{f.info[k]}"
    return f"id:{f.info.get('db_id')}"


new = {"train": [], "valid": []}; drop = 0; seen = set()
for p in a.files:
    for f in read(p, index=":"):
        g = clean(f)
        if g is None: drop += 1; continue
        key = (g.info.get("db_id"), round(float(g.get_potential_energy()), 6))
        if key in seen: continue
        seen.add(key)
        if setname(g) in a.all_train and str(g.info.get("split", "train")) == "valid": g.info["split"] = "train"; g.info["split_was"] = "valid"
        new["valid" if str(g.info.get("split", "train")) == "valid" else "train"].append(g)
for k, v in excluded.items(): print(f"  excluded {v} frames by --exclude {k}")
for k, v in drops.items():
    if k == "pureH_small": print(f"  dropped {len(v)} small pure-H frames (< {a.min_pureh_atoms} atoms" + (f", P >= {a.pureh_pmax:.0f} GPa; their P {min(v):.0f}..{max(v):.0f}" if a.pureh_pmax > 0 else "; --pureh-pmax keeps molecular ones") + ")")
    else: print(f"  dropped {len(v)} frames of set {k}: largest |force| {max(v):.0f} eV/A (limit {a.fmax:.0f})")
# sets that were generated with the split fixed to train: move a fraction of their groups (by hash, whole groups) to valid
mv = collections.defaultdict(lambda: collections.defaultdict(list))
for f in new["train"]:
    if str(f.info.get("db_set")) in a.valid_from: mv[str(f.info["db_set"])][group(f)].append(f)
moved = collections.Counter()
for st_, gs in mv.items():
    order = sorted(gs, key=lambda g: hashlib.md5((a.seed + "v" + g).encode()).hexdigest()); k = max(1, int(round(a.valid_frac * len(order))))
    for g in order[:k]:
        for f in gs[g]: new["train"].remove(f); f.info["split"] = "valid"; new["valid"].append(f); moved[st_] += 1
if moved: print("  moved to valid (whole groups):", dict(moved))
print(f"{sum(map(len, new.values()))} labelled frames kept ({sum(len(v) for v in drops.values())} dropped by the sanity filter); train {len(new['train'])}, valid {len(new['valid'])}")
# nested fractions: within each set, groups ordered by hash; the first ceil(p% * n_groups) (at least 1) go in
by_set = collections.defaultdict(lambda: collections.defaultdict(list))
for f in new["train"]: by_set[setname(f)][group(f)].append(f)
rank = {}
for s, gs in by_set.items():
    order = sorted(gs, key=lambda g: hashlib.md5((a.seed + g).encode()).hexdigest())
    for i, g in enumerate(order): rank[g] = (s, i, len(order))


def frac_frames(p): return [f for f in new["train"] if rank[group(f)][1] < max(1, math.ceil(p / 100.0 * rank[group(f)][2]))]


if a.max_atoms:
    nb = len(new['train']); new['train'] = [f for f in new['train'] if len(f) <= a.max_atoms]; print(f"  --max-atoms {a.max_atoms}: dropped {nb - len(new['train'])} train frames")
old_tr = read(a.old_train, index=":") if a.old_train else []
if a.max_atoms: old_tr = [f for f in old_tr if len(f) <= a.max_atoms]
old_va = read(a.old_valid, index=":") if a.old_valid else []
outs = {f"train_{a.tag}": new["train"], f"valid_{a.tag}": new["valid"]}
for p in a.fractions: outs[f"train_{a.tag}{p}"] = frac_frames(p)
if old_tr:
    outs[f"train_m{a.tag}"] = old_tr + new["train"]; outs[f"valid_m{a.tag}"] = old_va + new["valid"]
    for p in a.fractions: outs[f"train_m{a.tag}{p}"] = old_tr + frac_frames(p)
for name, fr in outs.items():
    if os.path.exists(P(name)): os.remove(P(name))
    write(P(name), fr)
sets = sorted(set(by_set) | {setname(f) for f in new["valid"]}); print(f"\n{'file':22s}{'frames':>8s}   " + "  ".join(f"{s:>6s}" for s in sets))
for name, fr in outs.items():
    c = collections.Counter(setname(f) if "db_set" in f.info else "old" for f in fr); print(f"{name:22s}{len(fr):8d}   " + "  ".join(f"{c.get(s, 0):6d}" for s in sets) + (f"   (+{sum(v for k, v in c.items() if k not in sets)} old)" if old_tr and name.startswith(("train_m", "valid_m")) else ""))
print("\nnext: bash mlp_pipeline/grace_prep_pkl.sh " + " ".join(n.split('_', 1)[1] for n in outs if n.startswith("train_")) + "   (writes runs/grace/*.pkl.gz)")
