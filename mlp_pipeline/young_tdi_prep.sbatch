#!/bin/bash -l
# TDI with the start-configuration protocol for H-rich fcc crystals (O held while H thermalises, then O released; every lambda starts from that frame), 5 ps per lambda, 17 lambda values (Gauss-Kronrod G8+K17), results in runs/tdi_prep.
# Array task i -> (P, T, composition), composition fastest. Default PLIST = 300 400 500 600 GPa, TLIST = 4000 5000 6000 K, CLIST = "2.25 2.5 2 H"  ->  12 (P,T) x 4 = 48 tasks (--array=0-47):
#   2.25 / 2.5 = fcc H2.25O / H2.5O WITH the protocol; 2 = fcc H2O and H = pure H (250 atoms) WITHOUT it (unaffected by the protocol; run with the same settings here so every mixing row has a matching H2O and pure-H partner: same dt, lambda grid, ps).
# Settings: PREPHOLD=1 ps, PREPFREE=1 ps, EQUIL 0.25 ps per lambda afterwards, CHUNK 0.25, dt 0.5 fs, PS 5. Cost per task is an ESTIMATE, not measured: crystals about 2.5-3 h (17 lambda, 9 processes: waves 9 + 8), pure H about 2 h (6 processes, 4.5 GB each).
# CAVEATS: (1) the prep code was only tested as a 0.2 ps run; look in runs/tdi_prep/<tag>_prep.log for the O rms displacement after release (much larger than the Einstein-spring rms = the lattice melted, the start is not a crystal). (2) Not every (P,T) is a stable solid: free NVT showed H2.25O melting at 400/6500 and 500/7000, and fcc H2O melts near 300 GPa between 6000 and 7000 K, so 300/6000 may melt; summarize.py notes flag free-NVT melts where such runs exist.
# compare with the standard runs:  ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_prep      (500 GPa 6000 K H2.25O cold start: 1572.49+-2.10 meV/atom at 1 ps per lambda)
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_tdi_prep.sbatch        other:  PLIST="500" TLIST="5500 6500" CLIST="2.25 2 H" sbatch --array=0-5 mlp_pipeline/young_tdi_prep.sbatch   (array size = nP * nT * nC)
#SBATCH --job-name=tdiprep
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-47%8
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
ARRAY_KEEP="PREPHOLD PREPFREE" . mlp_pipeline/array_env.sh   # drop run knobs exported by the submitting shell (they would apply to every task)
PLIST=(${PLIST:-300 400 500 600}); TLIST=(${TLIST:-4000 5000 6000}); CLIST=(${CLIST:-2.25 2.5 2 H})
I=${SLURM_ARRAY_TASK_ID:-0}; NC=${#CLIST[@]}; C=${CLIST[$((I % NC))]}; T=${TLIST[$(( (I / NC) % ${#TLIST[@]} ))]}; P=${PLIST[$(( I / (NC * ${#TLIST[@]}) ))]}
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
echo "== task $I: $([ $C = H ] && echo pure H || echo fcc H${C}O) at $P GPa $T K on $(hostname) $(date) code $(git rev-parse --short HEAD)"
export TF_FORCE_GPU_ALLOW_GROWTH=true
if [ $C = H ]; then PH=H; HP=2; NAT=250; NP=6; PRE=""; elif [ $C = 2 ]; then PH=fcc; HP=2; NAT=""; NP=9; PRE=""; else PH=fcc; HP=$C; NAT=""; NP=9; PRE=${PREPHOLD:-1}; fi
P=$P T=$T HPER=$HP NATOMS=$NAT NODES=17 NPAR=$NP RULE=gk NGAUSS=8 DT=0.5 EQUIL=0.25 PS=${PS:-5} PS1=1.5 CHUNK=0.25 MINRUNS=8 PREPHOLD=$PRE PREPFREE=${PREPFREE:-1} OUTDIR=runs/tdi_prep bash mlp_pipeline/young_tdi_par.sh $PH
