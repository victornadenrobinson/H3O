#!/bin/bash -l
# Liquid mixing free energy (hull) at many (P, T): array task i -> (P, T, composition), composition fastest. Compositions: fluid H2O, H2.25O, H2.5O, H3O, H4O, H5O and pure H (250 atoms, the reference for every mixing row).
# Default PLIST = 300 400 500 600 GPa, TLIST = 4000 5000 6000 K  ->  12 (P,T) x 7 = 84 tasks (--array=0-83). 32-O fcc start cells (96-192 atoms), 10 lambda values, 2 ps each, dt 0.5 fs, lambda=0 = ideal gas of all atoms.
# LOW-T MELT STEP: below the melting line the fcc start would stay a (superheated) crystal at lambda near 1, so every lambda is first heated to HOTT=12000 K for HOT=0.5 ps and then equilibrated at T for 1 ps (NEW option --hot-ps, NOT TESTED yet: only syntax-checked).
#   Check in the node logs that the O 'u rms' grows (liquid) and compare liquid H2O with the solid at the same (P,T) before trusting low-T rows. HOT=0 disables the step (then the earlier 7000/9000 K behaviour).
# Volumes: DFT hull EOS where it exists, else ideal mixing / extrapolation with a WARNING in the log (MINRUNS=8); check the MLP pressure at the highest lambda. Cost per task is an ESTIMATE: about 25-40 min.
# Restartable (resubmit the same command). usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_liquid.sbatch
#   other:  PLIST="400" TLIST="3000 3500 4000" sbatch --array=0-20 mlp_pipeline/young_tdi_liquid.sbatch     (array size = nP * nT * 7)       then: ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py
#SBATCH --job-name=tdiliq
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --array=0-83%8
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PLIST=(${PLIST:-300 400 500 600}); TLIST=(${TLIST:-4000 5000 6000}); COMPS=(2 2.25 2.5 3 4 5 H)
I=${SLURM_ARRAY_TASK_ID:-0}; NC=${#COMPS[@]}; C=${COMPS[$((I % NC))]}; T=${TLIST[$(( (I / NC) % ${#TLIST[@]} ))]}; P=${PLIST[$(( I / (NC * ${#TLIST[@]}) ))]}
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: $([ $C = H ] && echo pure H || echo liquid H${C}O) at $P GPa, $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
if [ $C = H ]; then PHASE=H; HPER=2; NAT=250; HOTV=""; else PHASE=liquid; HPER=$C; NAT=""; HOTV=${HOT-0.5}; fi
P=$P T=$T HPER=$HPER NATOMS=$NAT HOT=$HOTV NODES=10 NPAR=10 DT=0.5 EQUIL=${EQUIL:-1.0} PS=${PS:-2} PS1=1.5 CHUNK=0.25 MINRUNS=8 bash mlp_pipeline/young_tdi_par.sh $PHASE
