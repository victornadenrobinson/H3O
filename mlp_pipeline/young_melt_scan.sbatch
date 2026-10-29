#!/bin/bash -l
# Solid-stability scan with the fine-tuned MLP: NVT MD of bcc and fcc H2O at fixed temperatures (default 4000-8000 K) for each pressure; one array task = one (phase, P), with the temperatures run
# concurrently on the same GPU. Volumes: DFT EOS of that phase at the requested T (interpolated between the DFT temperature bins). Reports the O-sublattice structure label (CN of the time-averaged
# sites), the Lindemann ratio and the O and H diffusion constants; summary: python3 mlp_pipeline/21_melt_summary.py runs/melt
# CAVEAT: a small periodic crystal superheats; "stays solid" only bounds the melting point from above (Tm < T_stable x (1 + superheating margin)); "melts" is an upper bound for Tm. The free-energy
# comparison (liquid vs solid G from the TDI runs) gives the actual crossing temperature.
# default table: bcc and fcc at 300, 400, 500 GPa (tasks 0-5); T list "4000 5000 6000 7000 8000"; 10 ps per T after 1 ps equilibration, dt 0.5 fs.
# usage on Young:   cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_melt_scan.sbatch
#   other:  PLIST="200 600" TLIST="5500 6500" PS=20 sbatch --array=0-3 --time=04:00:00 mlp_pipeline/young_melt_scan.sbatch      (task i: phase = (bcc fcc)[i%2], pressure = PLIST[i/2])
#SBATCH --job-name=meltscan
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
export TF_FORCE_GPU_ALLOW_GROWTH=true
PLIST=(${PLIST:-300 400 500}); TLIST=(${TLIST:-4000 5000 6000 7000 8000}); I=${SLURM_ARRAY_TASK_ID:-0}
PHASES=(bcc fcc); PH=${PHASES[$((I % 2))]}; P=${PLIST[$((I / 2))]}
[ -n "$P" ] || { echo "no pressure for task $I"; exit 2; }
MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; PS=${PS:-10}; PY=${PY:-$HOME/grace-venv/bin/python}
echo "== task $I: $PH at $P GPa, T = ${TLIST[*]} K, $PS ps each, concurrently on $(hostname) $(date)"
mkdir -p runs/melt
for T in "${TLIST[@]}"; do
  grep -qs "^RESULT" runs/melt/melt_${PH}_${P}GPa_${T}K.log && { echo "skip ${PH} ${P} GPa ${T} K (already has a RESULT line)"; continue; }
  $PY -u mlp_pipeline/17_eos_md.py $MODEL --phase $PH --T $T --from-dft inventory_binned.csv --P $P --ps $PS --equil-ps 1.0 --dt 0.5 --tag melt_${PH}_${P}GPa_${T}K --outdir runs/melt > runs/melt/melt_${PH}_${P}GPa_${T}K.log 2>&1 &
done
wait
echo "== done $(date)"; grep -h "^RESULT" runs/melt/melt_${PH}_${P}GPa_*K.log | cut -c1-400
