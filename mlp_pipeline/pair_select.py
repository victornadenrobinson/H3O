#!/usr/bin/env python3
"""Choose which candidate insertion/deletion pairs are worth a DFT label: query-by-committee on the quantity we care about, the insertion energy per H.
For every pair group (one reference frame + its variants, from widom_frames.py: host -> host+H 'prox'/'lowphi', full -> full-H 'del') each committee model gives
    dE = E(host + H) - E(host)        (insertion;  for a deletion pair: E(full) - E(full - H))
and the acquisition score of the group is the spread (sd over models) of dE. Large spread = the fine-tunes disagree about this insertion = the model error there is probably large.
Per state (cache) the top --keep groups are taken, plus --random-frac of --keep drawn at random from the rest, so the labelled set also contains an UNBIASED sample (needed to measure the
error without selection bias: pair_check.py reports on all frames; the random ones are the honest estimate). GPU, one process; models are loaded one after the other.
  python mlp_pipeline/pair_select.py widom_pairs_cand.extxyz --models grace-ft:runs/grace/ft_mn_f_lr3e-3_e5/seed/1/final_model grace-ft:runs/grace/ft_mnlEFnaive/seed/1/final_model grace-ft:runs/grace/ft_nBase/seed/1/final_model \\
        --keep 12 --random-frac 0.34 --out widom_pairs.extxyz --scores pair_scores.csv
Cost: (number of structures) x (number of models) energy evaluations (~2 per s per GPU at 330 atoms); host frames are evaluated once and shared by their sites."""
import argparse, collections, csv, os, sys
import numpy as np


def group_scores(groups, E):
    """groups: {gid: (ref_index, [variant indices], kind 'ins'|'del')}; E: (n_models, n_structs) energies -> {gid: (score, mean dE, [dE per variant per model])}"""
    out = {}
    for g, (ref, var, kind) in groups.items():
        d = []
        for v in var:
            de = (E[:, v] - E[:, ref]) if kind == "ins" else (E[:, ref] - E[:, v])      # eV per added H
            d.append(de)
        d = np.array(d)                                                                # variants x models
        spread = d.std(axis=1, ddof=1) if d.shape[1] > 1 else np.zeros(len(d)); out[g] = (float(spread.max()), float(d.mean()), d)
    return out


def select(scores, state_of, keep, random_frac, seed=1):
    rng = np.random.default_rng(seed); by = collections.defaultdict(list)
    for g, (s, m, _) in scores.items(): by[state_of(g)].append((s, g))
    chosen = {}
    for st, lst in by.items():
        lst.sort(reverse=True); n_rand = int(round(keep * random_frac)); n_top = max(keep - n_rand, 0)
        for s, g in lst[:n_top]: chosen[g] = "top"
        rest = [g for _, g in lst[n_top:]]
        for g in (rng.choice(rest, min(n_rand, len(rest)), replace=False) if rest else []): chosen[g] = "random"
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("files", nargs="+"); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--keep", type=int, default=12, help="groups per state"); ap.add_argument("--random-frac", type=float, default=0.34); ap.add_argument("--out", default="widom_pairs.extxyz"); ap.add_argument("--scores", default="pair_scores.csv")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    from ase.io import read, write
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from importlib import import_module
    frames = [f for p in a.files for f in read(p, index=":")]
    if a.limit: frames = frames[:a.limit]
    groups = {}
    for i, f in enumerate(frames):
        role = f.info.get("widom_role"); rid = f.info.get("run_id")
        if role in ("host", "full"): groups.setdefault(rid, [i, [], "ins" if role == "host" else "del"])[0] = i
    for i, f in enumerate(frames):
        role = f.info.get("widom_role"); rid = f.info.get("run_id")
        if role in ("prox", "lowphi", "del") and rid in groups: groups[rid][1].append(i)
    groups = {g: (v[0], v[1], v[2]) for g, v in groups.items() if v[1]}
    print(f"{len(frames)} structures in {len(groups)} pair groups ({sum(1 for v in groups.values() if v[2] == 'ins')} insertion, {sum(1 for v in groups.values() if v[2] == 'del')} deletion)", flush=True)
    need = sorted({i for r, vs, _ in groups.values() for i in [r] + vs}, key=lambda i: len(frames[i]))
    E = np.full((len(a.models), len(frames)), np.nan); b04 = import_module("04_benchmark")
    for mi, spec in enumerate(a.models):
        calc = b04.get_calc_spec(spec, a.device)
        for k, i in enumerate(need):
            g = frames[i].copy(); g.calc = calc; E[mi, i] = float(g.get_potential_energy())
            if k % 50 == 0: print(f"  model {mi + 1}/{len(a.models)}  {k}/{len(need)}", flush=True)
    sc = group_scores(groups, E); chosen = select(sc, lambda g: g.split(":")[0], a.keep, a.random_frac, a.seed)
    with open(a.scores, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["group", "state", "kind", "score_sd_dE_eV", "mean_dE_eV", "chosen"])
        for g, (s, m, _) in sorted(sc.items(), key=lambda kv: -kv[1][0]): w.writerow([g, g.split(":")[0], groups[g][2], f"{s:.4f}", f"{m:.4f}", chosen.get(g, "")])
    if os.path.exists(a.out): os.remove(a.out)
    n = 0
    for g, how in chosen.items():
        r, vs, _ = groups[g]
        for i in [r] + vs:
            f = frames[i]; f.info["acq_mode"] = how; f.info["acq_score"] = float(sc[g][0]); write(a.out, f, append=True); n += 1
    print(f"selected {len(chosen)} groups ({sum(1 for v in chosen.values() if v == 'top')} top-disagreement, {sum(1 for v in chosen.values() if v == 'random')} random) = {n} structures = DFT points -> {a.out}; scores -> {a.scores}")


if __name__ == "__main__":
    main()
