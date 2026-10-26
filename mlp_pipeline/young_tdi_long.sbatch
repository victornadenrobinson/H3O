#!/bin/bash -l
# Long TDI of each system type, to see how G converges with time per lambda (G(t) traces, drift, discard test): 10 ps per lambda, cold start, dt 0.5 fs, results in runs/tdi_long (separate from the standard runs, so nothing is overwritten).
# Array task i = line i of the table below (phase P T H-per-O natoms nodes npar):
#   0 fcc H2O 500 GPa 6000 K | 1 fcc H2.25O 500/6000 | 2 pure H 500/6000 (250 atoms) | 3 bcc H2O 400/5000 | 4 liquid H2O 400/7000 | 5 liquid H2.25O 400/7000 | 6 insertion n=1 into fcc H2O 500/6000
# Lambda values: 9 (crystals), 8 (pure H, 4.5 GB per process), 10 (liquid, insertion), Gauss-Legendre, 8-10 processes on the one GPU. Cost per task is an ESTIMATE, not measured: crystals 9 x 11 ps = 22,000 steps at about 0.28 s/step = about 1.7 h; the others similar or faster.
# Read the result with  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi_long   (add --discard 0.5 to see the second-half-only G) and  mlp_pipeline/tdi/plot_all.py  for G(t). Compare with the standard runs at the same (P,T) (shorter, other lambda values).
# Restartable. usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_long.sbatch        subset: --array=0-2      longer: PS=20 sbatch --time=10:00:00 ...
#SBATCH --job-name=tdilong
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=05:00:00
#SBATCH --array=0-6
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
TABLE=("fcc 500 6000 2 - 9 9" "fcc 500 6000 2.25 - 9 9" "H 500 6000 2 250 8 8" "bcc 400 5000 2 - 9 9" "liquid 400 7000 2 - 10 10" "liquid 400 7000 2.25 - 10 10" "insert 500 6000 2 - 10 10")
read -r PH P T HP NAT ND NP <<< "${TABLE[${SLURM_ARRAY_TASK_ID:-0}]}"
[ -n "$PH" ] || { echo "no table entry for array task $SLURM_ARRAY_TASK_ID"; exit 2; }
echo "== task $SLURM_ARRAY_TASK_ID: $PH $P GPa $T K H per O $HP on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
P=$P T=$T HPER=$HP NATOMS=$([ "$NAT" = - ] || echo $NAT) NGHOST=1 NODES=$ND NPAR=$NP DT=0.5 EQUIL=1.0 PS=${PS:-10} PS1=1.5 CHUNK=0.5 MINRUNS=8 OUTDIR=runs/tdi_long bash mlp_pipeline/young_tdi_par.sh $PH
