#!/bin/bash
# One entry point for the main calculations: bash run.sh <task> key=value ...   (repo root on Perlmutter; dry=1 prints the command without running it)
# Tasks and keys: docs/RECIPES.md (bash run.sh help prints the list). Each task maps the keys onto the existing script and prints the exact command it runs.
#   bash run.sh eos comp="h2o h3o h" T=7000
#   bash run.sh tdi phase=fcc P=500 T=5000 hper=2 quad=gk17 npar=6 ps=2            (interactive GPU node: runs here on gpu=0)
#   bash run.sh tdi phase=liquid P=500 T=5000 quad=gl10 where=batch hours=8         (login node: one shared-GPU batch job)
set -u; cd "$(dirname "$0")"
TASK=${1:-help}; shift || true
declare -A K
for kv in "$@"; do
  case "$kv" in *=*) K[${kv%%=*}]="${kv#*=}";; *) echo "argument '$kv' is not key=value"; exit 1;; esac
done
g() { local v="${K[$1]:-}"; echo "${v:-${2:-}}"; }                                # g key default
PYP=${PY:-/pscratch/sd/v/victornr/grace-env/bin/python}
BEST=grace-ft:runs/grace/ft_mfin_f_lr1e-2_e50
MODEL=$(g model "$BEST"); case "$MODEL" in grace*|lj*) ;; *) MODEL=grace-ft:runs/grace/ft_$MODEL;; esac   # model=mfin_f_lr1e-2_e50 is enough
run() { echo "+ $*"; [ "$(g dry 0)" = 1 ] && return 0; eval "$@"; }
gpu_here() { [ "$(g dry 0)" = 1 ] && return 0; nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible: run this task inside an interactive GPU job (or where=batch from a login node)"; exit 1; }; }
login_here() { [ "$(g dry 0)" = 1 ] && return 0; [ -n "${SLURM_JOB_ID:-}" ] && { echo "you are inside Slurm job $SLURM_JOB_ID: submit batch jobs from a LOGIN node (else: 'Job request does not match any supported policy')"; exit 1; }; return 0; }

case "$TASK" in
help|-h|--help)
  sed -n '/^## Tasks/,$p' docs/RECIPES.md | grep -E "^### |^\`bash run.sh" | sed 's/^### /\n/'; exit 0;;

eos)     # liquid / fluid EOS scans with the live dashboard (mlp_pipeline/eos_live.sh)
  gpu_here
  run "MODEL=$MODEL COMP='$(g comp h2o)' T=$(g T 7000) NV=$(g nv 16) ${K[vmax]:+VMAX=${K[vmax]}} ${K[vmin]:+VMIN=${K[vmin]}} ${K[ps]:+PS=${K[ps]}} ${K[eq]:+EQ=${K[eq]}} PER_GPU=$(g per_gpu 4) PY=$PYP bash mlp_pipeline/eos_live.sh";;

eos_plot) # all EOS scans vs DFT (+ French 2009 for H2O) and the P-T maps
  run "$PYP mlp_pipeline/plot_eos_all.py --T $(g T 7000) --model $(basename ${MODEL#*:}) ${K[pmin]:+--pmin ${K[pmin]}} ${K[pmax]:+--pmax ${K[pmax]}} ${K[loglog]:+--loglog} --dft-frames $(g frames 'runs/train_mfin.extxyz runs/valid_mfin.extxyz')"
  run "$PYP mlp_pipeline/plot_eos_maps.py --model $(basename ${MODEL#*:}) --dft-frames $(g frames 'runs/train_mfin.extxyz runs/valid_mfin.extxyz')";;

tdi)     # full TDI free energy of one phase (mlp_pipeline/young_tdi_par.sh): lambda values as processes on ONE GPU
  PH=$(g phase fcc); Q=$(g quad gl10)
  case "$Q" in gl*) QV="NODES=${Q#gl}";; gk*) n=${Q#gk}; QV="RULE=gk NGAUSS=$(( (n - 1) / 2 )) NODES=$n";; *) echo "quad = gl<N> (Gauss-Legendre, N lambda values) or gk<2n+1> (Gauss-Kronrod: n Gauss + n+1 added, e.g. gk17, gk21)"; exit 1;; esac
  [ -e inventory_binned.csv ] || { [ -e runs/inventory_binned.csv ] && ln -s runs/inventory_binned.csv inventory_binned.csv; }   # the DFT EOS the volumes come from
  ENVS="PY=$PYP MODEL=$MODEL P=$(g P 500) T=$(g T 5000) HPER=$(g hper 2) $QV POWER=$(g power 4) PS=$(g ps 4) PS1=$(g ps1 1.5) EQUIL=$(g equil 1) DT=0.5 NPAR=$(g npar 4)"   # npar 4: best throughput (tdi_bench_layout 2026-11-06: 2/4/6/8 per GPU = 12.3/23.8/22.2/22.4 steps/s per GPU, fcc 324 atoms)
  ENVS="$ENVS ${K[natoms]:+NATOMS=${K[natoms]}} ${K[nghost]:+NGHOST=${K[nghost]}} ${K[hot]:+HOT=${K[hot]}} ${K[hott]:+HOTT=${K[hott]}} ${K[k]:+K=${K[k]}} ${K[tag]:+TAG=${K[tag]}} ${K[outdir]:+OUTDIR=${K[outdir]}} ${K[nrep]:+NREP=${K[nrep]}} ${K[prephold]:+PREPHOLD=${K[prephold]}} ${K[chunk]:+CHUNK=${K[chunk]}}"
  [ "$PH" = liquid ] && [ -z "${K[hot]:-}" ] && ENVS="$ENVS HOT=0.5 HOTT=12000"
  [ "$PH" = H ] && [ -z "${K[natoms]:-}" ] && ENVS="$ENVS NATOMS=729"
  if [ "$(g where here)" = batch ]; then
    login_here; name="tdi_${PH}_$(g P 500)GPa_$(g T 5000)K_h$(g hper 2)_$Q"; mkdir -p logs_pw runs/tdi_jobs; js=runs/tdi_jobs/$name.sh
    printf '%s\n' '#!/bin/bash' "cd $PWD" "export PYTHONPATH=\${PYTHONPATH:-}:$PWD/mlp_pipeline" "env $ENVS bash mlp_pipeline/young_tdi_par.sh $PH" > $js
    run "sbatch -A m3349_g -C gpu -q shared -G 1 -c 32 -N 1 -t $(g hours 8):00:00 -J $name -o logs_pw/$name.%j.out $js"
  else
    gpu_here; run "CUDA_VISIBLE_DEVICES=$(g gpu 0) $ENVS bash mlp_pipeline/young_tdi_par.sh $PH"
  fi
  echo "afterwards: $PYP mlp_pipeline/tdi/summarize.py runs/tdi   (G per atom; for gk runs the Kronrod - Gauss difference = quadrature error)";;

tdi_bench)  # TDI throughput: different numbers of lambda processes per GPU at the same time
  gpu_here; run "MODEL=$MODEL SYSTEM=$(g system h2o) LAYOUT='$(g layout '2 4 6 8')' STEPS=$(g steps 200) ${K[vpa]:+VPA=${K[vpa]}} PY=$PYP bash mlp_pipeline/tdi_bench_layout.sh";;

finetune)   # one GRACE fine-tune as a shared-GPU batch job (ft_one.sh via ft_submit.sh); updates from epochs x frames
  login_here; NAME=$(g name); [ -n "$NAME" ] || { echo "name=<run name> needed"; exit 1; }
  TR=$(g train mfin); VA=$(g valid "$TR")
  UPD=$(g updates); [ -n "$UPD" ] || UPD=$($PYP -c "import pandas as pd; print(int($(g epochs 50) * len(pd.read_pickle('runs/grace/train_$TR.pkl.gz')) / $(g batch 1)))")
  echo "bash mlp_pipeline/ft_one.sh $NAME $TR $VA $UPD $(g wE 10) $(g wF 10) $(g wS 1) $(g mode frozen) $(g lr 0.01) $(g batch 1)" > runs/ft_$NAME.cmd
  run "GPUS=$(g gpus 2) TIME=$(g time 24:00:00) bash mlp_pipeline/ft_submit.sh runs/ft_$NAME.cmd";;

score)      # score fine-tunes: held-out frames + DFT-MD composition test (eval_round) and the sealed pair test
  gpu_here; run "PY=$PYP ${K[only]:+ONLY='${K[only]}'} bash mlp_pipeline/eval_round.sh"; run "PY=$PYP ${K[only]:+ONLY='${K[only]}'} bash mlp_pipeline/score_pairs_all.sh 2>&1 | tail -n 45";;

dft)        # VASP single points for a set of geometries (dgcheck_jobs.sh route): Gamma (default) or a k-mesh template
  login_here; FR=$(g frames); OD=$(g outd); [ -n "$FR" ] && [ -n "$OD" ] || { echo "frames=<extxyz> outd=<runs/...> needed"; exit 1; }
  run "FRAMES_EXTXYZ=$FR OUTD=$OD TEMPLATE=$(g template vasp_inputs) EXE=$(g exe vasp_gam) NODES=$(g nodes 8) MINUTES=$(g minutes 170) SUBMIT=$(g submit 0) PY=$PYP bash mlp_pipeline/dgcheck_jobs.sh"
  echo "afterwards: \$PY mlp_pipeline/points_collect.py $OD/points_* --out $OD/labelled.extxyz --db-set <SET>";;

kpt)        # k-point relabelling of a stratified frame selection (kpt_select + kpt_relabel, MP 2x2x2 by default)
  login_here; run "MESH=$(g mesh mp222) KPAR=$(g kpar 1) N=$(g n 300) OUTD=$(g outd runs/kpt222mp) NODES=$(g nodes 20) MINUTES=$(g minutes 120) SUBMIT=$(g submit 0) PY=$PYP bash mlp_pipeline/kpt_relabel.sh";;

*) echo "unknown task '$TASK': bash run.sh help"; exit 1;;
esac
