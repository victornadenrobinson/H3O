#!/bin/bash
# Prepare and submit the k-point sensitivity jobs (stock VASP, vasp_std, CPU). Run from the repo root on a Perlmutter LOGIN node.
#   TEMPLATE=vasp_inputs bash mlp_pipeline/kpoint_jobs.sh            prepare + show what would be submitted
#   TEMPLATE=vasp_inputs SUBMIT=1 bash mlp_pipeline/kpoint_jobs.sh   also sbatch the jobs
# For each mesh: a copy of YOUR template with only KPOINTS replaced (Gamma-centred mesh; INCAR, POTCAR, everything else identical), the per-point inputs via prepare_points.py, and one one-node job
# (4 lanes x 32 ranks, like the snapdb chunks). POTCAR is only copied from your template within your scratch; nothing goes to the repo.
set -e; cd "$(dirname "$0")/.."; T=${TEMPLATE:?TEMPLATE=<your vasp template dir with INCAR KPOINTS POTCAR POSCAR>}; K=runs/kpt; PY=${PY:-$SCRATCH/grace-env/bin/python}
export PYTHONPATH=${PYTHONPATH:-}:$PWD/mlp_pipeline
[ -e $K/frames_k222.extxyz ] || { echo "run first: $PY mlp_pipeline/kpoint_frames.py runs/dft_md_500.extxyz --out $K"; exit 1; }
# mesh   label  minutes (budget estimates: vasp_std costs ~2x vasp_gam per k-point, 5 irreducible k for 2x2x2 and ~14 for 3x3x3 with time reversal, so ~10x and ~28x a Gamma point; widen if points stay unfinished)
for spec in "k111 1 1 1 60" "k222 2 2 2 170" "k333 3 3 3 150"; do
  set -- $spec; m=$1; mesh="$2 $3 $4"; mins=$5; d=$K/template_$m; p=$K/${m}_points
  if [ ! -d $p ]; then
    rm -rf $d; cp -rL "$T" $d
    printf "k-point test %s\n0\nGamma\n%s\n0 0 0\n" "$m" "$mesh" > $d/KPOINTS
    $PY mlp_pipeline/prepare_points.py $K/frames_$m.extxyz --template $d --out $p --min-T 1000 | tail -n 1
  fi
  cat > $K/job_$m.sbatch <<EOF
#!/bin/bash
#SBATCH -A m3349 -C cpu -q regular -N 1 --ntasks-per-node=128 --cpus-per-task=2 -t $(printf "%d:%02d:00" $(( (mins + 10) / 60 )) $(( (mins + 10) % 60 ))) -J kpt_$m
set -e; cd \${SLURM_SUBMIT_DIR:-.}; export PYTHONPATH=\${PYTHONPATH:-}:\$PWD/mlp_pipeline; export VASP_MODULE=\${VASP_MODULE:-vasp/5.4.4-cpu}
bash mlp_pipeline/perlmutter_vasp_check.sh run $p --exe vasp_std --ranks 32 --parallel 4 --minutes $((mins + 5))
EOF
  echo "$m: mesh $mesh, $(grep -c . $p/dirs.txt) points -> $K/job_$m.sbatch ($mins min)"
  [ "${SUBMIT:-0}" = 1 ] && sbatch $K/job_$m.sbatch
done
[ "${SUBMIT:-0}" = 1 ] || echo "(not submitted: add SUBMIT=1)"
