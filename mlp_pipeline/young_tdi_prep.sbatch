#!/bin/bash -l
# TDI with the start-configuration protocol (O held while H thermalises, then O released; every lambda starts from that frame), 5 ps per lambda, 17 lambda values (Gauss-Kronrod G8+K17, same as the existing 500 GPa runs), results in runs/tdi_prep (same tags as the standard runs, separate folder).
# Array task i: 0 fcc H2O 500 GPa 6000 K | 1 fcc H2.25O 500/6000 | 2 fcc H2O 400/5000 | 3 fcc H2.25O 400/5000 | 4 fcc H2O 500/5500 | 5 fcc H2.25O 500/5500. Pure H is not affected by the protocol: use the standard pure-H runs at the same (P,T) for the mixing table (summarize pairs by P, T, dt, nodes).
# Settings: PREPHOLD=1 ps hold, PREPFREE=1 ps release (full MLP), EQUIL 0.25 ps per lambda afterwards, CHUNK 0.25, dt 0.5 fs, 9 processes at a time (waves 9 + 8). Cost is an ESTIMATE, not measured: prep about 5 min, then 17 x 5.25 ps = 10,500 steps per lambda at about 0.28 s/step with 9 processes = about 1 h per wave = about 2.5-3 h per task.
# NEW CODE: PREPHOLD/--prep was only tested as a 0.2 ps run (no result seen at the time of writing): look in runs/tdi_prep/<tag>_prep.log for the O rms displacement after release (much larger than the Einstein-spring rms of about 0.05-0.1 A = the lattice is melting; then the TDI start is not a crystal).
# compare with the standard runs:  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_prep     (500 GPa 6000 K H2O cold start: 1657.09+-2.44 meV/atom at 1 ps per lambda)
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_prep.sbatch
#SBATCH --job-name=tdiprep
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
TABLE=("500 6000 2" "500 6000 2.25" "400 5000 2" "400 5000 2.25" "500 5500 2" "500 5500 2.25")
read -r P T HP <<< "${TABLE[${SLURM_ARRAY_TASK_ID:-0}]}"
[ -n "$P" ] || { echo "no table entry for array task $SLURM_ARRAY_TASK_ID"; exit 2; }
echo "== task $SLURM_ARRAY_TASK_ID: fcc H${HP}O $P GPa $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
P=$P T=$T HPER=$HP NODES=17 NPAR=9 RULE=gk NGAUSS=8 DT=0.5 EQUIL=0.25 PS=${PS:-5} PS1=1.5 CHUNK=0.25 MINRUNS=8 PREPHOLD=${PREPHOLD:-1} PREPFREE=${PREPFREE:-1} OUTDIR=runs/tdi_prep bash mlp_pipeline/young_tdi_par.sh fcc
