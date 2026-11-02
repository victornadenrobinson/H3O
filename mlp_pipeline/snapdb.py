"""Snapshot database for the DFT labelling campaign: merge all geometry sets, dedupe, chunk into ~1-hour VASP jobs, collect the labels.
  build   python mlp_pipeline/snapdb.py build --sets NAME=path.extxyz[:max[:prio]] ... --out snapdb      -> snapdb/db.extxyz (all frames, db_id) + snapdb/index.csv
  chunk   python mlp_pipeline/snapdb.py chunk snapdb --budget-min 50 --slots 4 [--prio-max 2]            -> snapdb/chunks/chunk_NNN/frames.extxyz, snapdb/submit_all.sh, snapdb/vasp_chunk.sbatch
  status  python mlp_pipeline/snapdb.py status snapdb                                                     -> counts per set / chunk / state
  collect python mlp_pipeline/snapdb.py collect snapdb                                                    -> snapdb/labelled.extxyz from the finished OUTCARs (energy, forces, stress), index.csv status updated
Cost model: seconds per point = ref_sec * (natoms / ref_natoms)^exp (default 105 s at 330 atoms, exponent 2: a 729-atom point ~ 510 s; calibrate with the first 729-atom run). One job = one node = `slots` concurrent
single points of 32 ranks each (4 slots on a 128-core node); points are packed (first-fit decreasing, highest priority first) so no lane exceeds --budget-min minutes.
--sets entries: NAME=file[:max[:prio]]; max caps the frames taken (stratified by config_type), prio 1 = label first (default 5). Frames keep all their info keys; duplicates (same species, positions to 1e-3 A, cell) are dropped.
No VASP input and no POTCAR data goes in this database: geometries and metadata only. Inputs are written per chunk on Perlmutter by prepare_points.py from your local template.
"""
import argparse, csv, glob, hashlib, json, os, sys
import numpy as np


def frame_hash(f):
    h = hashlib.sha1(); h.update(f.numbers.tobytes()); h.update(np.round(f.positions, 3).tobytes()); h.update(np.round(np.asarray(f.cell), 3).tobytes()); return h.hexdigest()


def stratified(frames, k, rng):
    if k <= 0 or len(frames) <= k: return list(range(len(frames)))
    groups = {}
    for i, f in enumerate(frames): groups.setdefault(str(f.info.get("config_type", "?")), []).append(i)
    for g in groups.values(): rng.shuffle(g)
    out = []; keys = sorted(groups)
    while len(out) < k and any(groups.values()):
        for key in keys:
            if groups[key] and len(out) < k: out.append(groups[key].pop())
    return sorted(out)


def cmd_build(a):
    from ase.io import read, write
    rng = np.random.default_rng(a.seed); os.makedirs(a.out, exist_ok=True); seen = set(); rows = []; db = os.path.join(a.out, "db.extxyz")
    if os.path.exists(db): os.remove(db)
    for spec in a.sets:
        name, rest = spec.split("=", 1); parts = rest.split(":"); path = parts[0]; mx = int(parts[1]) if len(parts) > 1 and parts[1] else 0; prio = int(parts[2]) if len(parts) > 2 else 5
        fr = read(path, index=":"); take = stratified(fr, mx, rng); n0 = len(rows); dup = 0
        for i in take:
            f = fr[i]; h = frame_hash(f)
            if h in seen: dup += 1; continue
            seen.add(h); did = f"s{len(rows):06d}"; f = f.copy(); f.calc = None
            f.info.update(db_id=did, db_set=name, db_prio=prio); write(db, f, append=True)
            rows.append(dict(db_id=did, set=name, prio=prio, split=str(f.info.get("split", "train")), natoms=len(f), nO=int((f.numbers == 8).sum()), nH=int((f.numbers == 1).sum()), T_K=f.info.get("T_K", ""), config_type=f.info.get("config_type", ""),
                             P=f.info.get("P_grace_GPa", f.info.get("P_target", f.info.get("P_static", ""))), status="pending", chunk=""))
        print(f"{name:14s} {len(fr):6d} frames in {path}: took {len(take)}, duplicates dropped {dup}, now {len(rows)}")
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"database: {len(rows)} frames -> {db}, {os.path.join(a.out, 'index.csv')}")


def cost(n, a): return a.ref_sec * (n / a.ref_natoms) ** a.exp


def cmd_chunk(a):
    import pandas as pd
    from ase.io import read, write
    ix = pd.read_csv(os.path.join(a.db, "index.csv")); fr = {f.info["db_id"]: f for f in read(os.path.join(a.db, "db.extxyz"), index=":")}
    todo = ix[(ix.status == "pending") & (ix.prio <= a.prio_max)].copy(); todo["cost"] = todo.natoms.apply(lambda n: cost(n, a)); todo = todo.sort_values(["prio", "cost"], ascending=[True, False])
    cap = a.budget_min * 60.0; chunks = []   # each chunk: list of lane loads + ids
    for r in todo.itertuples():
        for c in chunks:
            lane = int(np.argmin(c["lanes"]))
            if c["lanes"][lane] + r.cost <= cap: c["lanes"][lane] += r.cost; c["ids"].append(r.db_id); break
        else:
            c = dict(lanes=[0.0] * a.slots, ids=[r.db_id]); c["lanes"][0] = r.cost; chunks.append(c)
    base = os.path.join(a.db, "chunks"); os.makedirs(base, exist_ok=True)
    for k, c in enumerate(chunks):
        d = os.path.join(base, f"chunk_{k:03d}"); os.makedirs(d, exist_ok=True); write(os.path.join(d, "frames.extxyz"), [fr[i] for i in c["ids"]])
        ix.loc[ix.db_id.isin(c["ids"]), "chunk"] = k
    ix.to_csv(os.path.join(a.db, "index.csv"), index=False)
    open(os.path.join(a.db, "vasp_chunk.sbatch"), "w").write(f"""#!/bin/bash
#SBATCH -A {a.account} -C cpu -q {a.qos} -N 1 -t {a.budget_min + 10}:00 -J vasp_chunk
# usage: sbatch --export=ALL,CH=snapdb/chunks/chunk_000,TEMPLATE=<dir with INCAR KPOINTS POTCAR POSCAR> snapdb/vasp_chunk.sbatch   (repo root)
set -e; cd ${{SLURM_SUBMIT_DIR:-.}}
[ -d "$CH/points" ] || python3 mlp_pipeline/prepare_points.py $CH/frames.extxyz --template $TEMPLATE --out $CH/points
export VASP_MODULE=${{VASP_MODULE:-vasp/5.4.4-cpu}}
bash mlp_pipeline/perlmutter_vasp_check.sh run $CH/points --exe vasp_gam --ranks {a.ranks} --parallel {a.slots} --minutes {a.budget_min + 5}
""")
    open(os.path.join(a.db, "submit_all.sh"), "w").write("#!/bin/bash\n# TEMPLATE=<dir> bash snapdb/submit_all.sh [first last]   from the repo root\nset -e\nfor ch in " + " ".join(f"{k:03d}" for k in range(len(chunks))) + "; do\n  [ -n \"${1:-}\" ] && { [ $((10#$ch)) -lt $1 ] || [ $((10#$ch)) -gt ${2:-9999} ] && continue; }\n  sbatch --export=ALL,CH=" + os.path.join(a.db, "chunks", "chunk_$ch") + ",TEMPLATE=$TEMPLATE " + os.path.join(a.db, "vasp_chunk.sbatch") + "\ndone\n")
    tot = todo.cost.sum(); print(f"{len(todo)} points -> {len(chunks)} one-node jobs of {a.budget_min} min ({a.slots} slots x {a.ranks} ranks); total {tot / 3600:.1f} point-hours, {tot / 3600 / a.slots:.1f} node-hours at perfect packing")
    print("points per chunk: min %d, median %d, max %d; largest cell %d atoms" % (min(len(c["ids"]) for c in chunks), np.median([len(c["ids"]) for c in chunks]), max(len(c["ids"]) for c in chunks), todo.natoms.max()))


def cmd_status(a):
    import pandas as pd
    ix = pd.read_csv(os.path.join(a.db, "index.csv")); print(ix.groupby(["set", "status"]).size().unstack(fill_value=0).to_string()); print("\nchunks:", ix.chunk.nunique(), " split:", ix.split.value_counts().to_dict())


def cmd_collect(a):
    import pandas as pd
    from ase.io import read, write
    from ase.calculators.singlepoint import SinglePointCalculator
    ix = pd.read_csv(os.path.join(a.db, "index.csv")); out = os.path.join(a.db, "labelled.extxyz")
    if os.path.exists(out): os.remove(out)
    ok = bad = 0
    for mf in sorted(glob.glob(os.path.join(a.db, "chunks", "chunk_*", "points", "frames_meta.csv"))):
        d = os.path.dirname(mf); meta = pd.read_csv(mf)
        for r in meta.itertuples():
            oc = os.path.join(d, r.id, "OUTCAR")
            try:
                at = read(oc, index=-1, format="vasp-out")
                src = read(os.path.join(os.path.dirname(d), "frames.extxyz"), index=int(r.frame)); at.info = dict(src.info)
                at.calc = SinglePointCalculator(at, energy=at.get_potential_energy(), free_energy=at.get_potential_energy(force_consistent=True), forces=at.get_forces(), stress=at.get_stress())
                write(out, at, append=True); ix.loc[ix.db_id == src.info["db_id"], "status"] = "done"; ok += 1
            except Exception as e:
                bad += 1
                if hasattr(r, "db_id"): ix.loc[ix.db_id == r.db_id, "status"] = "failed"
    ix.to_csv(os.path.join(a.db, "index.csv"), index=False); print(f"collected {ok} labelled frames -> {out}; {bad} points without a usable OUTCAR (not run yet or failed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--sets", nargs="+", required=True); b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=1)
    c = sp.add_parser("chunk"); c.add_argument("db"); c.add_argument("--budget-min", type=int, default=50); c.add_argument("--slots", type=int, default=4); c.add_argument("--ranks", type=int, default=32); c.add_argument("--prio-max", type=int, default=99)
    c.add_argument("--ref-natoms", type=float, default=330.0); c.add_argument("--ref-sec", type=float, default=105.0); c.add_argument("--exp", type=float, default=2.0); c.add_argument("--account", default="m3349"); c.add_argument("--qos", default="regular")
    s = sp.add_parser("status"); s.add_argument("db"); k = sp.add_parser("collect"); k.add_argument("db")
    a = ap.parse_args(); {"build": cmd_build, "chunk": cmd_chunk, "status": cmd_status, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
