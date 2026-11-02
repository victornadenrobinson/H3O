#!/usr/bin/env python3
"""'Pacman' mop-up for the VASP chunks: gather every point that has no DONE.json from all chunks of the given databases into N shared work lists, so many jobs can eat the same leftovers.
Each mop chunk <out>/chunk_JJJ/points/ holds symlinks m00000... to the real point directories (DONE.json, .run.lock and the attempts/ live in the real directories, so all jobs see the same locks
and finished points are skipped) plus a manifest.json / dirs.txt for perlmutter_vasp_check.py run. Lists are ordered most expensive first (729-atom cells), rotated by job so the jobs start in
different places, and every odd job walks the list backwards: the workers meet in the middle.
usage (repo root):  python3 mlp_pipeline/mop_up.py --dbs snapdb snapdb_v2 --out mop --jobs 8
   then, per job:   sbatch -J m$k --export=ALL,CH=mop/chunk_00$k,TEMPLATE=vasp_inputs snapdb_v2/vasp_chunk.sbatch     (points/ exists, so no preparation step runs)
Untested against a real 'run' with shared directories: try ONE job first and look for START lines on points that no other job is running."""
import argparse, glob, json, os, shutil

ap = argparse.ArgumentParser(); ap.add_argument("--dbs", nargs="+", default=["snapdb"]); ap.add_argument("--out", default="mop"); ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--min-atoms", type=int, default=0, help="only points with at least this many atoms (e.g. 700 for the pure-H cells)"); a = ap.parse_args()
todo = []
for db in a.dbs:
    for mf in sorted(glob.glob(os.path.join(db, "chunks", "chunk_*", "points", "manifest.json"))):
        pdir = os.path.dirname(mf); m = json.load(open(mf))
        for p in m["points"]:
            d = os.path.abspath(os.path.join(pdir, p["path"]))
            if os.path.exists(os.path.join(d, "DONE.json")) or p.get("natoms", 0) < a.min_atoms: continue
            todo.append(dict(src=d, natoms=p.get("natoms", 0), sha=p.get("input_sha256", ""), template=m.get("template", ""), source_frames=m.get("source_frames", "")))
todo.sort(key=lambda t: -t["natoms"])
print(f"{len(todo)} unfinished points; atoms {min((t['natoms'] for t in todo), default=0)}..{max((t['natoms'] for t in todo), default=0)}; est. {sum(105 * (t['natoms'] / 330.0) ** 2 for t in todo) / 4 / 3600:.1f} node-h")
if not todo: raise SystemExit
shutil.rmtree(a.out, ignore_errors=True)
for j in range(a.jobs):
    off = j * len(todo) // a.jobs; order = todo[off:] + todo[:off]
    if j % 2: order = order[::-1]
    pts = os.path.join(a.out, f"chunk_{j:03d}", "points"); os.makedirs(pts); entries = []
    for k, t in enumerate(order):
        name = f"m{k:05d}"; os.symlink(t["src"], os.path.join(pts, name)); entries.append(dict(id=name, path=name, input_sha256=t["sha"], natoms=t["natoms"]))
    json.dump(dict(schema=1, mode="vasp_smoke", template=todo[0]["template"], source_frames=todo[0]["source_frames"], pairs=[], points=entries), open(os.path.join(pts, "manifest.json"), "w"), indent=2)
    open(os.path.join(pts, "dirs.txt"), "w").write("".join(os.path.abspath(os.path.join(pts, e["path"])) + "\n" for e in entries))
print(f"wrote {a.jobs} work lists in {a.out}/ (job j starts at {100 // a.jobs}% steps apart; odd jobs run backwards)")
