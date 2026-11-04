#!/usr/bin/env python3
"""Turn a PWidom --candidate-dir queue (--candidate-policy weighted) into ONE geometry file for DFT labelling, plus trials.csv linking the pairs.
   python3 mlp_pipeline/candidates_to_frames.py runs/<run>/candidates [more dirs] --out runs/reweight/frames.extxyz        (needs ase: use $PY)
Every distinct geometry is written once (a host shared by several trials is labelled once) with info geometry_id (= run_id), T_K, P_target, host tag; trials.csv (next to --out) has one row per exported trial:
request, tag, direction, small_id (host without the selected H / full minus H), large_id, draw_count, log_weight, dU_mlp_eV, T_K. Then:  TEMPLATE=... FRAMES_EXTXYZ=<out> bash mlp_pipeline/dgcheck_jobs.sh ."""
import argparse, csv, glob, json, os
from ase.io import read, write
ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True); a = ap.parse_args()
frames, rows, seen = [], [], set()
for d in a.dirs:
    for line in open(os.path.join(d, "queue.jsonl")):
        q = json.loads(line)
        if q.get("policy") != "weighted" or len(q["structures"]) != 2: continue
        ids = [s[:-len(".extxyz")] for s in q["structures"]]
        for gid, f in zip(ids, q["structures"]):
            if gid in seen: continue
            at = read(os.path.join(d, f)); at.info.update(geometry_id=gid, run_id=gid, T_K=q["T_K"], P_target=q["P_GPa"], tag=q["tag"], widom_role="cand", db_set="PW"); frames.append(at); seen.add(gid)
        rows.append(dict(request=q["request"], tag=q["tag"], direction=q["direction"], small_id=ids[0], large_id=ids[1], draw_count=q["draw_count"], log_weight=q["log_weight"], dU_mlp_eV=q["dU_mlp_eV"], T_K=q["T_K"], P_GPa=q["P_GPa"]))
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); write(a.out, frames)
tf = os.path.join(os.path.dirname(os.path.abspath(a.out)), "trials.csv"); w = csv.DictWriter(open(tf, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(frames)} distinct geometries -> {a.out}; {len(rows)} trials ({sum(r['draw_count'] for r in rows)} draws) -> {tf}")
