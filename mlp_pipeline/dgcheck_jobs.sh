#!/bin/bash
# DFT-level check of the PWidom insertion energies (what decides whether the model's dg can be trusted). Run from the repo root on a Perlmutter LOGIN node.
#   TEMPLATE=<your vasp template dir> bash mlp_pipeline/dgcheck_jobs.sh             builds the geometries + VASP inputs, prints the job plan (nothing submitted)
#   TEMPLATE=... SUBMIT=1 bash mlp_pipeline/dgcheck_jobs.sh                           also sbatches the CPU jobs
# What it labels, per host (H2O, H2p083O, H2p25O from the end-member run and Hplus6/12/18/27 from the xh run):
#   FRAMES host frames x (host + SEL proxy-site insertions + lowest-phi site) and DEL cached host+1H frames x (full + NDEL deletions)  ->  insertion pairs host -> host+H and deletion pairs full -> full-H
#   pair_check.py then gives, from E_DFT - E_model, the mean insertion-energy error per added H (c, meV/H) at the exact states the PWidom run sampled: dg_DFT ~ dg_model + c.
# All of it is SEALED (valid-frac 1: not for training). Inputs are written from YOUR template into runs/dgcheck (never into the repo); pure H (250 atoms) is skipped by widom_frames (Gamma-only labels invalid).
# FRAMES_EXTXYZ=<file> uses those geometries instead of widom_frames (e.g. the weighted trials of candidates_to_frames.py).
# env: RUNS (run dirs, default the two 500 GPa / 5000 K runs), PROXY, FRAMES (3), SEL (2), DEL (3), NDEL (2), NODES (jobs, 6), MINUTES (per job, 700), OUTD (runs/dgcheck), PY
# EXE (vasp_gam): VASP executable; vasp_std for k-point meshes (kpt_relabel.sh)
# slurm: ACCT (m3349), QOS (regular), CPT (cpus-per-task, 1 = the header of perlmutter_dft_pwidom.sbatch, which is known to submit), SBATCH_EXTRA (more args)
#   TEST=1 runs `sbatch --test-only` on each job file instead of submitting. CAUSE of "Job request does not match any supported policy" (found 2026-11-06): submitting
#   these CPU jobs from INSIDE a GPU allocation (interactive GPU node, inherited Slurm environment). Submit from a login node. Earlier note: --test-only answered "Job request does not match any supported
#   policy" for the same job header that submitted and ran fine with a plain sbatch (the dgchk_* jobs); so a test-only rejection is not conclusive: submit one job and check squeue.
set -e; cd "$(dirname "$0")/.."; T=${TEMPLATE:?TEMPLATE=<your vasp template dir with INCAR KPOINTS POTCAR POSCAR>}; PY=${PY:-/pscratch/sd/v/victornr/grace-env/bin/python}
export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline
RUNS=${RUNS:-"runs/pwidom_endmem_new_500GPa_5000K runs/pwidom_xh_seed11_500GPa_5000K"}; PROXY=${PROXY:-runs/pwidom_fit/proxy_ice.json}
FRAMES=${FRAMES:-3}; SEL=${SEL:-2}; DEL=${DEL:-3}; NDEL=${NDEL:-2}; NODES=${NODES:-6}; MINUTES=${MINUTES:-700}; OUTD=${OUTD:-runs/dgcheck}; SUBMIT=${SUBMIT:-0}; mkdir -p "$OUTD"
ACCT=${ACCT:-m3349}; QOS=${QOS:-regular}; CPT=${CPT:-1}; SBATCH_EXTRA=${SBATCH_EXTRA:-}; nsub=0; nfail=0
if [ -n "${FRAMES_EXTXYZ:-}" ]; then cp "$FRAMES_EXTXYZ" "$OUTD/dgcheck_frames.extxyz"; echo "using the given geometries $FRAMES_EXTXYZ (no widom_frames step)"; else
CACHES=(); for r in $RUNS; do for f in $r/pw_*_frames.npz; do case $f in *pureH*) ;; *) CACHES+=("$f");; esac; done; done
[ "${#CACHES[@]}" -gt 0 ] || { echo "no *_frames.npz caches in: $RUNS"; exit 1; }
printf 'cache: %s\n' "${CACHES[@]}"
$PY mlp_pipeline/widom_frames.py "${CACHES[@]}" --proxy "$PROXY" --frames-n $FRAMES --sel $SEL --del-n $DEL --ndel $NDEL --split frame --valid-frac 1 --seed 5 --out "$OUTD/dgcheck_frames.extxyz"
fi
$PY mlp_pipeline/split_frames.py "$OUTD/dgcheck_frames.extxyz" "$NODES" "$OUTD"
WALL=$(printf "%d:%02d:00" $(( (MINUTES + 10) / 60 )) $(( (MINUTES + 10) % 60 )))
for ((j = 0; j < NODES; j++)); do
  [ -e "$OUTD/frames_$j.extxyz" ] || continue
  fsha=$(sha256sum "$OUTD/frames_$j.extxyz" | cut -d' ' -f1)
  if [ -d "$OUTD/points_$j" ]; then      # never reuse inputs prepared from other geometries (a stale folder once silently labelled the wrong set)
    [ "$(cat "$OUTD/points_$j/.frames_sha" 2>/dev/null)" = "$fsha" ] || { echo "ERROR: $OUTD/points_$j exists but was prepared from other geometries (or before this check). Use a fresh OUTD, or move that folder away." >&2; exit 1; }
  else
    $PY mlp_pipeline/prepare_points.py "$OUTD/frames_$j.extxyz" --template "$T" --out "$OUTD/points_$j" --min-T 1000 | tail -n 1; echo "$fsha" > "$OUTD/points_$j/.frames_sha"
  fi
  printf '%s\n' '#!/bin/bash -l' "#SBATCH --account=$ACCT" "#SBATCH --qos=$QOS" '#SBATCH --constraint=cpu' '#SBATCH --nodes=1' '#SBATCH --ntasks-per-node=128' \
    "#SBATCH --cpus-per-task=$CPT" '#SBATCH --exclusive' '#SBATCH --mem=0' "#SBATCH --time=$WALL" "#SBATCH --job-name=dgchk_$j" "#SBATCH --output=$OUTD/slurm_dgchk_${j}_%j.out" \
    'set -e; cd ${SLURM_SUBMIT_DIR:-.}; export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline; export VASP_MODULE=${VASP_MODULE:-vasp/5.4.4-cpu}' \
    "bash mlp_pipeline/perlmutter_vasp_check.sh run $OUTD/points_$j --exe ${EXE:-vasp_gam} --ranks 32 --parallel 4 --minutes $MINUTES" > "$OUTD/job_$j.sbatch"
  echo "  job_$j: $(grep -c . "$OUTD/points_$j/dirs.txt") points -> $OUTD/job_$j.sbatch ($MINUTES min, 4 lanes x 32 ranks)"
  if [ "${TEST:-0}" = 1 ]; then echo "    test-only: $(sbatch --test-only $SBATCH_EXTRA "$OUTD/job_$j.sbatch" 2>&1 | tr '\n' ' ')"
  elif [ "$SUBMIT" = 1 ]; then
    if out=$(sbatch --parsable $SBATCH_EXTRA "$OUTD/job_$j.sbatch" 2>&1); then echo "    submitted: $out"; ((nsub++)) || true
    else echo "    SBATCH FAILED: $out"; nfail=$((nfail + 1)); fi
  fi
done
if [ "$SUBMIT" = 1 ]; then echo "submitted ${nsub:-0}, failed ${nfail:-0} (account $ACCT, qos $QOS, cpus-per-task $CPT)"; [ "${nfail:-0}" = 0 ] || exit 1
elif [ "${TEST:-0}" != 1 ]; then echo "(not submitted: add SUBMIT=1, or TEST=1 for sbatch --test-only)"; fi
