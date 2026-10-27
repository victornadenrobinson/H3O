#!/bin/bash -l
# BIG MLP equation-of-state map (no lambda integration, only NVT MD at fixed volumes with the fine-tuned GRACE model): pressure P(V, T) of fcc H(2+n)O crystals (solid branch) and fluids (liquid branch) and of pure H, for dilute compositions too,
# to get (1) MLP-native volumes and bulk moduli for the TDI start volumes and the insertion pressure correction, (2) the partial molar volume of H in H2O from V(composition) at fixed P (compare v_H = 0.9 A^3 of the insertion test), (3) pressures far below the DFT data.
# VOLUMES ARE A V/ATOM GRID, NOT DFT-DERIVED: 16 volumes per task from V0 x 1.5 ... 0.42 with V0 = 3.6 A^3/atom (about 100 GPa for H2O) scaled per composition by s(n) = (3 + 0.62 n) / (3 + n) (pure H atoms take about 0.6 of an H2O atom's volume; the pure-H grid uses 0.75 x V0: a ROUGH cover, the scan reports the measured pressure).
# That covers roughly 20-1200 GPa; check the P range per group in the summary (python3 mlp_pipeline/22_eos_summary.py reads runs/eos_scan; it expects the DFT-volume tags, so use the csv files directly for this map: runs/eos_map/*.csv columns vpa, P_GPa, T_set, ...).
# Table (default, 77 tasks): crystal (solid branch, fcc, hold-free) for H per O = 2 2.02 2.05 2.1 2.25 2.5 3 at T = 2000 3000 4000 5000 6000 7000 K (42), liquid branch at 5000 6000 7000 9000 K (28), pure H fluid at 3000 4000 5000 6000 7000 8000 9000 K (7).
# Cost per task is an ESTIMATE, not measured: 16 volumes x 3.5 ps = 7000 steps at about 0.25 s/step with 6 processes sharing the GPU: about 1 h. --time=04:00:00. Solid runs that melt or liquid runs that did not melt are flagged by O_struct / D_O / delta_L in the csv (tdi/verdict.py).
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_eos_map.sbatch          subset: sbatch --array=0-5 ...         own table: JOBS="fcc solid 5000 2.05;fcc liquid 7000 3" sbatch --array=0-1 ...
#SBATCH --job-name=eosmap
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=0-76%8
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
table() { for h in 2 2.02 2.05 2.1 2.25 2.5 3; do for T in 2000 3000 4000 5000 6000 7000; do echo "fcc solid $T $h"; done; done; for h in 2 2.02 2.05 2.1 2.25 2.5 3; do for T in 5000 6000 7000 9000; do echo "fcc liquid $T $h"; done; done; for T in 3000 4000 5000 6000 7000 8000 9000; do echo "H fluid $T 2"; done; }
if [ -n "$JOBS" ]; then IFS=';' read -ra TABLE <<< "$JOBS"; else mapfile -t TABLE < <(table); fi
I=${SLURM_ARRAY_TASK_ID:-0}; read -r PH BR T H <<< "${TABLE[$I]}"
[ -n "$T" ] || { echo "no table entry for array task $I"; exit 2; }
export TF_FORCE_GPU_ALLOW_GROWTH=true H3O_RUNS=$PWD/runs
SNAP=runs/code/eosmap_$(date +%Y%m%d_%H%M%S)_$$; mkdir -p $SNAP runs/eos_map/logs && cp -r mlp_pipeline $SNAP/ && CODE=$SNAP/mlp_pipeline
PY=${PY:-$HOME/grace-venv/bin/python}; MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; PS=${PS:-3}; EQUIL=${EQUIL:-0.5}; BASE=map_${BR}_h${H}_${PH}_${T}K
N=16; VPA=$(python3 -c "
n = float('$H') - 2.0; s = 0.75 if '$PH' == 'H' else (3 + 0.62 * n) / (3 + n)
v0 = 3.6 * s; print(' '.join('%.4f' % (v0 * 1.5 * (0.42 / 1.5) ** (k / ($N - 1))) for k in range($N)))")
echo "== task $I: $PH $BR T=$T K H per O $H on $(hostname) $(date) code $(git rev-parse --short HEAD); V/atom grid: $VPA"
seq 0 $((N-1)) | xargs -P ${NPAR:-6} -I{} bash -c "$PY -u $CODE/17_eos_md.py $MODEL --phase $PH --branch $BR --T $T --hper $H --vpa $VPA --ps $PS --equil-ps $EQUIL --dt 0.5 --only {} --tag ${BASE}_i{} --outdir runs/eos_map > runs/eos_map/logs/${BASE}_i{}.log 2>&1"
echo "== done $(date)"; grep -h "^RESULT" runs/eos_map/logs/${BASE}_i*.log | cut -c1-300
