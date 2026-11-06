#!/bin/bash
# Speed (and accuracy) of the 2x2x2 labels: KPAR and a shifted Monkhorst-Pack mesh, on the two k333-reference frames of runs/kpt (H2.25O 351 atoms, pure H 729 atoms).
#   bash mlp_pipeline/kpt_bench.sh prep            login node, repo root: one points folder per variant from runs/kpt/template_k222 (KPOINTS / INCAR edited), job file
#   sbatch runs/kpt_bench/job.sbatch               1 CPU node, 4 variants at once (one 32-rank lane each, 2 points per lane); budget 80 min
#   bash mlp_pipeline/kpt_bench.sh report          elapsed, SCF steps, k-points, s/SCF per variant, and the energy against the existing 3x3x3 (and Gamma-centred 2x2x2) label
# Variants (VAR to change): g222_kpar2 g222_kpar4 g222_kpar8 = Gamma-centred 2x2x2 (8 k-points, as in the k-point test) with KPAR 2/4/8; mp222_kpar4 = Monkhorst-Pack 2x2x2
# (shifted: 4 irreducible k-points with time reversal, if ISYM allows it). The KPAR=1 baseline is the existing k222 timing (kpoint_check.py). NPAR is removed; NCORE is reduced if it
# does not divide 32 / KPAR. Nothing in the repo is touched; POTCAR stays in your scratch copy of the template.
set -e; cd "$(dirname "$0")/.."; K=runs/kpt; B=runs/kpt_bench; PY=${PY:-$SCRATCH/grace-env/bin/python}; export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline
VAR=${VAR:-"g222_kpar2 g222_kpar4 g222_kpar8 mp222_kpar4"}
case "${1:-}" in
prep)
  [ -d $K/template_k222 ] && [ -e $K/frames_k333.extxyz ] || { echo "need $K/template_k222 and $K/frames_k333.extxyz (kpoint_jobs.sh / kpoint_frames.py)"; exit 1; }
  mkdir -p $B
  $PY - "$K/frames_k333.extxyz" "$B/frames.extxyz" <<'EOF'
import sys
from ase.io import read, write
fr = [f for f in read(sys.argv[1], ":") if f.info.get("kpt_sys") in ("H2p25O", "pureH")]
write(sys.argv[2], fr); print("bench frames:", [(f.info["kpt_key"], len(f)) for f in fr])
EOF
  for v in $VAR; do
    d=$B/template_$v; p=$B/${v}_points; [ -d $p ] && { echo "$v: $p exists (kept)"; continue; }
    rm -rf $d; cp -rL $K/template_k222 $d
    kp=${v#*kpar}; mesh=${v%%_*}
    if [ "$mesh" = mp222 ]; then printf "MP 2x2x2 bench\n0\nMonkhorst-Pack\n2 2 2\n0 0 0\n" > $d/KPOINTS; fi
    grep -viE "^\s*(KPAR|NPAR)\s*=" $d/INCAR > $d/INCAR.new || true
    nc=$(grep -iE "^\s*NCORE\s*=" $d/INCAR.new | head -1 | sed -E 's/[#!].*//; s/^\s*NCORE\s*=\s*([0-9]+).*/\1/I'); per=$((32 / kp))   # comments stripped first (a comment can contain '=')
    if [ -n "$nc" ] && [ $((per % nc)) -ne 0 ]; then sed -i -E "s/^\s*NCORE\s*=.*/NCORE = $per/I" $d/INCAR.new; echo "  $v: NCORE $nc -> $per (must divide 32/KPAR)"; fi
    echo "KPAR = $kp" >> $d/INCAR.new; mv $d/INCAR.new $d/INCAR
    $PY mlp_pipeline/prepare_points.py $B/frames.extxyz --template $d --out $p --min-T 1000 | tail -n 1
    echo "$v: $(grep -iE '^\s*(KPAR|NCORE|ISYM|ISMEAR|SIGMA)\s*=' $d/INCAR | tr '\n' ' ')| KPOINTS: $(sed -n 3p $d/KPOINTS) $(sed -n 4p $d/KPOINTS)"
  done
  { echo '#!/bin/bash'; echo "#SBATCH -A ${ACCT:-m3349} -C cpu -q ${QOS:-regular} -N 1 --ntasks-per-node=128 --cpus-per-task=1 --exclusive --mem=0 -t 01:30:00 -J kpt_bench -o $B/slurm_%j.out"
    echo 'set -e; cd ${SLURM_SUBMIT_DIR:-.}; export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline; export VASP_MODULE=${VASP_MODULE:-vasp/5.4.4-cpu}'
    for v in $VAR; do echo "bash mlp_pipeline/perlmutter_vasp_check.sh run $B/${v}_points --exe vasp_std --ranks 32 --parallel 1 --minutes 80 > $B/run_$v.log 2>&1 &"; done
    echo 'wait; echo all variants finished'; } > $B/job.sbatch
  echo "next: sbatch $B/job.sbatch   (then: bash mlp_pipeline/kpt_bench.sh report)";;
report)
  $PY - "$B" "$K/energies.csv" $VAR <<'EOF'
import csv, glob, json, os, re, sys
import numpy as np
from ase.io import read
B, ref, var = sys.argv[1], sys.argv[2], sys.argv[3:]
E = {}
for r in csv.DictReader(open(ref)): E[(r["mesh"], r["key"])] = (float(r["E_mesh"]), int(r["natoms"]), r.get("elapsed_s"), r.get("scf_steps"))
def tim(oc):
    t = dict(el=None, scf=0, nk=None)
    for l in open(oc, errors="ignore"):
        if "Iteration" in l and "(" in l: t["scf"] += 1
        elif "NKPTS =" in l and t["nk"] is None: t["nk"] = int(re.search(r"NKPTS =\s*(\d+)", l).group(1))
        elif "Elapsed time (sec):" in l: t["el"] = float(l.split(":")[1])
    return t
print(f"{'variant':13s} {'frame':10s} {'atoms':>5s} {'nkpts':>5s} {'elapsed s':>9s} {'SCF':>4s} {'s/SCF':>6s} {'vs k222 KPAR1 s/SCF':>20s} {'E - E(k333) meV/atom':>21s} {'E - E(g222)':>12s}")
for (key,), base in sorted({(k,): v for (m, k), v in E.items() if m == "k222"}.items()):
    if ("k333", key) not in E: continue
    el, sc = float(base[2] or "nan"), float(base[3] or "nan")
    print(f"{'k222 KPAR1':13s} {key:10s} {base[1]:5d} {'8':>5s} {el:9.0f} {sc:4.0f} {el / sc:6.1f} {'1.00':>20s} {1000 * (base[0] - E[('k333', key)][0]) / base[1]:+21.2f} {'0':>12s}")
for v in var:
    p = os.path.join(B, f"{v}_points")
    if not os.path.isdir(p): continue
    meta = list(csv.DictReader(open(os.path.join(p, "frames_meta.csv"))))
    for r in meta:
        d = os.path.join(p, r["id"]); oc = os.path.join(d, "OUTCAR"); dj = os.path.join(d, "DONE.json")
        if os.path.exists(dj):
            att = json.load(open(dj)).get("attempt"); oc = os.path.join(d, att, "OUTCAR") if att else oc
        if not os.path.exists(oc): print(f"{v:13s} {r.get('kpt_key', r['id']):10s} not run"); continue
        t = tim(oc); key = r.get("kpt_key")
        try: e = float(read(oc, index=-1, format="vasp-out").get_potential_energy(force_consistent=True))
        except Exception: print(f"{v:13s} {key:10s} unfinished (elapsed so far n/a)"); continue
        n = E[("k222", key)][1]; b = E[("k222", key)]; bs = float(b[2]) / float(b[3])
        s = t["el"] / t["scf"] if t["el"] and t["scf"] else float("nan")
        print(f"{v:13s} {key:10s} {n:5d} {t['nk'] or 0:5d} {t['el'] or float('nan'):9.0f} {t['scf']:4d} {s:6.1f} {bs / s:19.2f}x {1000 * (e - E[('k333', key)][0]) / n:+21.2f} {1000 * (e - b[0]) / n:+12.2f}")
print("speed-up = (k222 KPAR1 s/SCF) / (variant s/SCF) on the same frame; production vasp_gam labels run at ~3.6 s/SCF (324-352 atoms). Accuracy: |E - E(k333)| should stay < ~1 meV/atom.")
EOF
  ;;
*) sed -n 2,9p "$0"; exit 1;;
esac
