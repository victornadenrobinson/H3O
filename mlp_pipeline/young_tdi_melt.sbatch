#!/bin/bash -l
# Melting crossing: TDI free energy of liquid H2O and of the solid branches (fcc, bcc) at the same (P, T), for DeltaG_melt(T) = G_liquid - G_solid per pressure.
# Array task i -> (P, T, phase), phase fastest: liquid (108 O, started from the fcc lattice, melts during equilibration), fcc (108 O), bcc (128 O); T next, P slowest, so a partly finished batch still holds complete triples.
# Default: P = 300 400 500 GPa, T = 5000 5500 6000 6500 K  ->  36 tasks (--array=0-35). 8 lambda values, 2 ps each after equilibration, dt 0.5 fs, all lambda values as concurrent processes on the one GPU.
# Restartable: resubmit the same command and unfinished lambda values continue from their last chunk. Check the liquid really is liquid (O D>1e-5 / 'u rms' in the node logs) and the solid stayed solid: TDI springs can hold a crystal that would melt at lambda=1.
# usage on Young (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_melt.sbatch
#   other pressures/temperatures (array size must be nP*nT*3):  PLIST="200 600" TLIST="5000 6000" sbatch --array=0-11 mlp_pipeline/young_tdi_melt.sbatch      subset: sbatch --array=0-11 ...   at most 6 at once: --array=0-35%6
# then:  python3 mlp_pipeline/tdi/summarize.py runs/tdi      (add extra lambda values later with RULE=gk NGAUSS=8 NODES=17 ONLYNODES="8 .. 16", same PS/EQUIL/CHUNK)
#SBATCH --job-name=tdimelt
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-35%6
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
PLIST=(${PLIST:-300 400 500}); TLIST=(${TLIST:-5000 5500 6000 6500}); PHASES=(liquid fcc bcc)
I=${SLURM_ARRAY_TASK_ID:-0}; PH=${PHASES[$((I % 3))]}; T=${TLIST[$(( (I / 3) % ${#TLIST[@]} ))]}; P=${PLIST[$(( I / (3 * ${#TLIST[@]}) ))]}
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: $PH at $P GPa, $T K on $(hostname) $(date) git $(git rev-parse --short HEAD)"
EQ=$([ $PH = liquid ] && echo 1.0 || echo 0.5)
[ $PH = liquid ] && export NREP=${NREP:-3}
P=$P T=$T NODES=8 NPAR=8 DT=0.5 EQUIL=$EQ PS=${PS:-2} PS1=1.5 CHUNK=0.25 MINRUNS=8 bash mlp_pipeline/young_tdi_par.sh $PH
echo "== task $I finished $(date)"
