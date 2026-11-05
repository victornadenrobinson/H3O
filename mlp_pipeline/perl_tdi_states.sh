#!/bin/bash
# Full free energies G by TDI (lambda path from an ideal gas / Einstein reference to the MLP) on Perlmutter SHARED single-GPU jobs, one job per (phase, P, T, H per O): young_tdi_par.sh does the work (the lambda windows are processes on ONE GPU).
# Default: the confirmation state 600 GPa, 5000 K with the PWidom model: crystals H2O and H2.25O (108 O fcc, the PWidom cells), liquids H2O and H2.25O (hot-start melt), pure H with 729 atoms (the reservoir).
#   bash mlp_pipeline/perl_tdi_states.sh                 dry run (prints the job scripts' key lines)
#   SUBMIT=1 bash mlp_pipeline/perl_tdi_states.sh        submit
# Afterwards:  python3 mlp_pipeline/tdi/summarize.py runs/tdi   (mixing free energy per atom vs x). Cross-checks at one state:
#   crystal route   dG_mix(H2.25O) = G(H2.25O) - G(H2O) - 27 G_H      (per 108 O)   vs   PWidom: sum of dg over the 27 added H (integral of dg(n), pwidom_hull.py) at the same P, T, model
#   liquid hull     G(liquid H(2+n)O) - ... from the liquid runs;   crystal vs liquid G of the same composition -> where the melting line is
# env: TASKS ("fcc 600 5000 2;fcc 600 5000 2.25;liquid 600 5000 2;liquid 600 5000 2.25;H 600 5000 2" = phase P T HPER), MODEL (best fine-tune), HOURS (8), NODES (10 lambda values), PS (4 ps per lambda), EQUIL (1), HOTT/HOT (liquid melt), PY
cd "$(dirname "$0")/.."; PY=${PY:-/pscratch/sd/v/victornr/grace-env/bin/python}; MODEL=${MODEL:-grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model}; HOURS=${HOURS:-8}; NODES=${NODES:-10}; PS=${PS:-4}; EQUIL=${EQUIL:-1}; SUBMIT=${SUBMIT:-0}
TASKS=${TASKS:-"fcc 600 5000 2;fcc 600 5000 2.25;liquid 600 5000 2;liquid 600 5000 2.25;H 600 5000 2"}; mkdir -p logs_pw runs/tdi_jobs
[ -e inventory_binned.csv ] || { [ -e runs/inventory_binned.csv ] && ln -s runs/inventory_binned.csv inventory_binned.csv && echo "linked inventory_binned.csv -> runs/inventory_binned.csv (the TDI scripts read it from the repo root)"; }
IFS=';' read -ra TAB <<< "$TASKS"
for t in "${TAB[@]}"; do
  read -r PH P T HP <<< "$t"; name="tdi_${PH}_${P}GPa_${T}K_h${HP}"; js="runs/tdi_jobs/$name.sh"
  case $PH in H) NP=4; EXTRA="NATOMS=729";; fcc) NP=6; EXTRA="";; liquid) NP=8; EXTRA="HOT=0.5 HOTT=12000";; *) NP=6; EXTRA="";; esac   # cells over 400 atoms need NPAR <= 8 on a 40 GB card; 729-atom H: 4
  printf '%s\n' '#!/bin/bash' "cd $PWD" "export PYTHONPATH=\${PYTHONPATH:-}:$PWD/mlp_pipeline" \
    "env PY=$PY MODEL=$MODEL P=$P T=$T HPER=$HP NODES=$NODES NPAR=$NP PS=$PS PS1=1.5 EQUIL=$EQUIL DT=0.5 $EXTRA bash mlp_pipeline/young_tdi_par.sh $PH" > "$js"
  echo "$name: $PH $P GPa $T K H/O $HP  (NPAR $NP${EXTRA:+, $EXTRA}) -> $js"
  if [ "$SUBMIT" = 1 ]; then sbatch --parsable -A m3349_g -C gpu -q shared -G 1 -c 32 -N 1 -t ${HOURS}:00:00 -J $name -o logs_pw/$name.%j.out "$js"; fi
done
[ "$SUBMIT" = 1 ] || echo "(dry run: add SUBMIT=1; each task is one shared-GPU job, up to ${HOURS} h)"
