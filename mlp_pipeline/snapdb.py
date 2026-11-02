"""Snapshot database for the DFT labelling campaign: merge all geometry sets, dedupe, chunk into ~1-hour VASP jobs, collect the labels.
  build   python mlp_pipeline/snapdb.py build --sets NAME=path.extxyz[:max[:prio]] ... --out snapdb      -> snapdb/db.extxyz (all frames, db_id) + snapdb/index.csv
  chunk   python mlp_pipeline/snapdb.py chunk snapdb --budget-min 50 --slots 4 [--prio-max 2]            -> snapdb/chunks/chunk_NNN/frames.extxyz, snapdb/submit_all.sh, snapdb/vasp_chunk.sbatch
  status  python mlp_pipeline/snapdb.py status snapdb                                                     -> counts per set / chunk / state
  verify  python mlp_pipeline/snapdb.py verify snapdb [--strict]                                          -> integrity report (exit 1 on any ERROR): species, finite numbers, cell widths, closest contacts, volumes, id/index consistency, split leakage, metadata
  plot    python mlp_pipeline/snapdb.py plot snapdb [--out snapdb/coverage.png]                          -> P-T coverage by set, H/O histogram, cell sizes, kinds
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


def cmd_verify(a):
    import pandas as pd
    from ase.io import read
    from ase.neighborlist import neighbor_list
    fr = read(os.path.join(a.db, "db.extxyz"), index=":"); ix = pd.read_csv(os.path.join(a.db, "index.csv")); err = []; warn = []
    def bad(lst, msg, ids):
        if ids: lst.append(f"{msg}: {len(ids)} frames, e.g. {', '.join(ids[:4])}")
    ids = [f.info.get("db_id", "?") for f in fr]
    if len(set(ids)) != len(ids): err.append("duplicate db_id in db.extxyz")
    if set(ids) != set(ix.db_id): err.append(f"db.extxyz ({len(ids)}) and index.csv ({len(ix)}) disagree on the set of db_ids")
    B = {k: [] for k in ("species", "nonfinite", "volume", "width5", "width6", "HH", "OH", "OO", "vpa", "size", "nometa")}
    for f in fr:
        i = f.info.get("db_id", "?"); z = set(f.numbers.tolist())
        if not z <= {1, 8}: B["species"].append(i)
        if not (np.isfinite(f.positions).all() and np.isfinite(np.asarray(f.cell)).all()): B["nonfinite"].append(i); continue
        vol = np.linalg.det(np.asarray(f.cell))
        if vol <= 1e-6: B["volume"].append(i); continue
        c = np.asarray(f.cell); w = min(vol / np.linalg.norm(np.cross(c[(j + 1) % 3], c[(j + 2) % 3])) for j in range(3))
        if w < 5.0: B["width5"].append(i)
        elif w < 6.0: B["width6"].append(i)
        vpa = vol / len(f)
        if not 0.9 <= vpa <= 6.5: B["vpa"].append(i)
        if len(f) < 50 or len(f) > 800: B["size"].append(i)
        d_i, d_j, d = neighbor_list("ijd", f, 2.2); zz = f.numbers
        for name, (p_, q_), lo in (("HH", (1, 1), 0.35), ("OH", (8, 1), 0.50), ("OO", (8, 8), 1.0)):
            m = ((zz[d_i] == p_) & (zz[d_j] == q_)) | ((zz[d_i] == q_) & (zz[d_j] == p_))
            if m.any() and d[m].min() < lo: B[name].append(i)
        if not all(k in f.info for k in ("config_type", "split", "T_K")): B["nometa"].append(i)
    bad(err, "species other than H, O", B["species"]); bad(err, "non-finite positions / cell", B["nonfinite"]); bad(err, "non-positive cell volume", B["volume"]); bad(err, "cell width < 5 A", B["width5"])
    bad(err, "H-H closer than 0.35 A", B["HH"]); bad(err, "O-H closer than 0.50 A", B["OH"]); bad(err, "O-O closer than 1.0 A", B["OO"])
    bad(warn, "cell width 5-6 A (below the 6 A GRACE cutoff)", B["width6"]); bad(warn, "volume per atom outside 0.9-6.5 A^3", B["vpa"]); bad(warn, "atom count < 50 or > 800", B["size"]); bad(warn, "missing config_type / split / T_K", B["nometa"])
    for key in ("base_id", "run_id", "host_source", "source"):   # whole runs / bases / sources must stay on one side of the split
        g = {}
        for f in fr:
            k = f.info.get(key)
            if k is not None: g.setdefault(str(k) + (f":{f.info.get('host_frame')}" if key == "host_source" else ""), set()).add(str(f.info.get("split", "train")))
        leak = [k for k, v in g.items() if len(v) > 1]
        if leak: err.append(f"split leakage by {key}: {len(leak)} groups in both train and valid, e.g. {leak[:3]}")
    print(f"{len(fr)} frames, {ix.set.nunique()} sets; split {ix.split.value_counts().to_dict()}")
    t = ix.groupby("set").agg(frames=("db_id", "size"), natoms_min=("natoms", "min"), natoms_max=("natoms", "max"), train=("split", lambda x: int((x == "train").sum())), valid=("split", lambda x: int((x == "valid").sum())),
                              T_min=("T_K", lambda x: pd.to_numeric(x, errors="coerce").min()), T_max=("T_K", lambda x: pd.to_numeric(x, errors="coerce").max()), P_min=("P", lambda x: pd.to_numeric(x, errors="coerce").min()), P_max=("P", lambda x: pd.to_numeric(x, errors="coerce").max()))
    print(t.round(0).to_string())
    r = (ix.nH / ix.nO.replace(0, np.nan)).round(2); print("H/O (rounded 0.01) counts: " + ", ".join(f"{k}:{v}" for k, v in r.value_counts().sort_index().head(40).items()) + f"; pure H {int((ix.nO == 0).sum())}")
    for m in warn: print("WARN ", m)
    for m in err: print("ERROR", m)
    print("integrity:", "FAILED" if err else "ok", f"({len(err)} errors, {len(warn)} warnings)")
    if err or (a.strict and warn): sys.exit(1)


def cmd_plot(a):
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from ase.io import read
    ix = pd.read_csv(os.path.join(a.db, "index.csv")); fr = read(os.path.join(a.db, "db.extxyz"), index=":"); ix["vpa"] = [f.get_volume() / len(f) for f in fr]
    for c in ("P", "T_K"): ix[c] = pd.to_numeric(ix[c], errors="coerce")
    sets = sorted(ix.set.unique()); cm = plt.get_cmap("tab10"); col = {s_: cm(i % 10) for i, s_ in enumerate(sets)}
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    for s_ in sets:
        d = ix[ix.set == s_]; ax[0, 0].scatter(d.T_K, d.P, s=14, alpha=0.6, color=col[s_], label=f"{s_} ({len(d)})")
    ax[0, 0].set_xlabel("T (K)"); ax[0, 0].set_ylabel("P (GPa; GRACE pressure / target / static)"); ax[0, 0].set_title(f"P-T coverage; {int(ix.P.isna().sum())} frames without P not shown"); ax[0, 0].legend(fontsize=7, frameon=False)
    ax[0, 0].axhspan(300, 600, color="grey", alpha=0.08); ax[0, 0].axvspan(3000, 7000, color="grey", alpha=0.08)
    r = (ix.nH / ix.nO.replace(0, np.nan)); bins = np.arange(1.9, 3.3, 0.02); ax[0, 1].hist([r[ix.set == s_].dropna() for s_ in sets], bins=bins, stacked=True, color=[col[s_] for s_ in sets]); ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlabel("H / O"); ax[0, 1].set_title(f"composition (pure H: {int((ix.nO == 0).sum())} frames, not shown)")
    ax[1, 0].hist([ix.natoms[ix.set == s_] for s_ in sets], bins=np.arange(280, 800, 20), stacked=True, color=[col[s_] for s_ in sets]); ax[1, 0].set_xlabel("atoms per cell"); ax[1, 0].set_title("cell sizes")
    k = ix.config_type.astype(str).str.replace(r"_\d+K$", "", regex=True).value_counts().head(25)[::-1]; ax[1, 1].barh(k.index, k.values); ax[1, 1].tick_params(axis="y", labelsize=6); ax[1, 1].set_title("config types (top 25)")
    for s_ in sets:
        d = ix[ix.set == s_]; ax[0, 2].scatter(d.T_K, d.vpa, s=14, alpha=0.6, color=col[s_])
    ax[0, 2].set_xlabel("T (K)"); ax[0, 2].set_ylabel("volume per atom (A^3)"); ax[0, 2].set_title("volume-T coverage (all frames, also those without P)")
    sp = ix.groupby(["set", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "valid"], fill_value=0); sp.plot.barh(stacked=True, ax=ax[1, 2], color=["#4c78a8", "#f58518"]); ax[1, 2].set_title("frames per set and split")
    fig.tight_layout(); out = a.out or os.path.join(a.db, "coverage.png"); fig.savefig(out, dpi=120); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--sets", nargs="+", required=True); b.add_argument("--out", required=True); b.add_argument("--seed", type=int, default=1)
    c = sp.add_parser("chunk"); c.add_argument("db"); c.add_argument("--budget-min", type=int, default=50); c.add_argument("--slots", type=int, default=4); c.add_argument("--ranks", type=int, default=32); c.add_argument("--prio-max", type=int, default=99)
    c.add_argument("--ref-natoms", type=float, default=330.0); c.add_argument("--ref-sec", type=float, default=105.0); c.add_argument("--exp", type=float, default=2.0); c.add_argument("--account", default="m3349"); c.add_argument("--qos", default="regular")
    s = sp.add_parser("status"); s.add_argument("db"); k = sp.add_parser("collect"); k.add_argument("db"); v = sp.add_parser("verify"); v.add_argument("db"); v.add_argument("--strict", action="store_true")
    pl = sp.add_parser("plot"); pl.add_argument("db"); pl.add_argument("--out")
    a = ap.parse_args(); {"build": cmd_build, "chunk": cmd_chunk, "status": cmd_status, "collect": cmd_collect, "verify": cmd_verify, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
