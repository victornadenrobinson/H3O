#!/bin/bash -l
# Static (0 K) H3O check with the MLP (mlp_pipeline/h3o_static.py): ice X, H references and the H3O-derived variants relaxed at constant pressure, one process per pressure sharing the GPU.
# Pressures whose runs/h3o_static/P<P>/h3o_static.csv exists are reused (interrupted interactive runs keep their finished pressures); REDO=1 reruns them.
# env: PS (pressures, default "300 400 450 500 600 700 800"), VARIANTS (H per O, default "3 2.5"; the H2.25O / H2.75O relaxations are the slow ones), PAR (processes at once, default 7), STEPS (default 2000), OUT (default runs/h3o_static)
# usage (login node):  cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && sbatch mlp_pipeline/young_h3o_static.sbatch
#        then:  grep -E "done:|dH_f|Huang|diff|finished" runs/slurm_h3ostat_<jobid>.out      results: runs/h3o_static/h3o_static.csv
#SBATCH --job-name=h3ostat
#SBATCH --partition=gpu
#SBATCH --account=allusers
#SBATCH --qos=freegpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_%x_%j.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
export TF_FORCE_GPU_ALLOW_GROWTH=true
PY=${PY:-$HOME/grace-venv/bin/python}
echo "== h3o_static: P = ${PS:=300 400 450 500 600 700 800}, variants ${VARIANTS:=3 2.5}, parallel ${PAR:=7}, steps ${STEPS:=2000}, out ${OUT:=runs/h3o_static} ($(date))"
$PY -u mlp_pipeline/h3o_static.py --P $PS --variants $VARIANTS --parallel $PAR --steps $STEPS -o $OUT ${REDO:+--redo}; exit $?
