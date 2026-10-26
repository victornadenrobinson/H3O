#!/bin/bash -l
# Melting scan for H-rich fcc crystals, free NVT with the MLP, protocol: O held on its lattice sites while H thermalises (HOLD ps, default 1), then O released (equil 1 ps, production PS ps). fcc only (no bcc for now).
# Array task i -> (composition, pressure): composition fastest (HLIST = H per O, default 2.1 2.25 2.5 = 227 / 243 / 270 H in 108 O), pressure slowest (PLIST default 100 200 ... 700) -> 21 tasks (--array=0-20).
# Each task runs all temperatures of TLIST concurrently on the one GPU (default 3000 4000 ... 11000 K = 9 runs, 335-378 atoms each). The verdict is read from the O lattice (O_struct, Lindemann ratio, D_O) in the RESULT lines; the melting T is bracketed between the highest 'solid' and the lowest 'melted' T. 1000 K resolution: refine with TLIST afterwards.
# Volumes: DFT EOS of H2.xO at (P, T) if the inventory has it, else ideal mixing of the H2O crystal and pure H (a NOTE in the log); outside the DFT temperature bins the volume is extrapolated with a WARNING. The MLP pressure of each run is in the RESULT line (check it).
# CAVEATS: (1) NOT TESTED yet: the new --hold-O-ps option of 17_eos_md.py was only syntax-checked; run the 2-minute test below first. (2) fcc is the O lattice for every composition and pressure: below about 300 GPa it is probably not the stable O lattice, so a 'melting T' there is the stability limit of a metastable fcc, not a phase boundary.
# (3) A small periodic crystal superheats: 'stays solid' gives an upper bound on Tm only; 10 ps may not catch slow melting near the boundary. (4) H2.1O has 227 H in the 108 O cell = H2.102O.
# test (GPU job, ~2 min):   HLIST=2.25 PLIST=500 TLIST="6000" PS=0.3 HOLD=0.2 bash mlp_pipeline/young_melt_comp.sbatch      (runs as a plain script; output in runs/melt_comp, look for 'O held for' and the RESULT line)
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit && sbatch mlp_pipeline/young_melt_comp.sbatch       other:  HOLD=0 sbatch ...   (all atoms free from step 0, for the comparison); SEED=1 (another random start); tags end in _hold<ps>_s<seed>; tables: python3 mlp_pipeline/tdi/melt_table.py
# results: runs/melt_comp/*.log (RESULT lines), summary:  grep -h "^RESULT" runs/melt_comp/*.log ; plot_pt.py picks the csv files up if runs/melt_comp is added to its folder list
#SBATCH --job-name=meltcomp
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-20%6
#SBATCH --output=runs/slurm_%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export TF_FORCE_GPU_ALLOW_GROWTH=true
HLIST=(${HLIST:-2.1 2.25 2.5}); PLIST=(${PLIST:-100 200 300 400 500 600 700}); TLIST=(${TLIST:-3000 4000 5000 6000 7000 8000 9000 10000 11000})
I=${SLURM_ARRAY_TASK_ID:-0}; H=${HLIST[$((I % ${#HLIST[@]}))]}; P=${PLIST[$((I / ${#HLIST[@]}))]}
[ -n "$P" ] || { echo "no table entry for array task $I"; exit 2; }
MODEL=${MODEL:-grace-ft:runs/grace/ft_m2}; PS=${PS:-10}; HOLD=${HOLD:-1.0}; PY=${PY:-$HOME/grace-venv/bin/python}; OUT=runs/melt_comp; mkdir -p $OUT
SEED=${SEED:-0}; SUF=hold${HOLD}_s${SEED}
echo "== task $I: fcc H$H O at $P GPa, T = ${TLIST[*]} K, hold O $HOLD ps, $PS ps each, concurrently on $(hostname) $(date) code $(git rev-parse --short HEAD)"
for T in "${TLIST[@]}"; do
  tag=melt_fcc_h${H}o_${P}GPa_${T}K_$SUF
  grep -qs "^RESULT" $OUT/$tag.log && { echo "skip $tag (already has a RESULT line)"; continue; }
  $PY -u mlp_pipeline/17_eos_md.py $MODEL --phase fcc --hper $H --T $T --from-dft inventory_binned.csv --min-runs 8 --P $P --ps $PS --equil-ps 1.0 --hold-O-ps $HOLD --seed $SEED --dt 0.5 --tag $tag --outdir $OUT > $OUT/$tag.log 2>&1 &
done
wait
echo "== done $(date)"; grep -h "^RESULT" $OUT/melt_fcc_h${H}o_${P}GPa_*K_$SUF.log | cut -c1-420
