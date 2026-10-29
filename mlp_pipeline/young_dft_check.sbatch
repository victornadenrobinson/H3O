#!/bin/bash -l
# NOTE (2026-10-29): Young's CPU nodes are SGE (qsub), not Slurm: use mlp_pipeline/young_dft_check.sge there (module recipe, chunking, h_rt guard, progress lines). This Slurm version only suits a Slurm machine with the VASP modules.
# VASP single points prepared by tdi/dft_check.py (prep-probe / prep-fep): array task i runs the i-th directory of <dir>/dirs.txt; finished directories are skipped (rerun-safe).
# usage (login node, repo root):   sbatch -p <cpu partition> --array=0-<n-1>%10 mlp_pipeline/young_dft_check.sbatch runs/dft_check/probe      (n = wc -l < runs/dft_check/probe/dirs.txt)
#        interactive (any allocation with MPI ranks):   NP=8 bash mlp_pipeline/young_dft_check.sbatch runs/dft_check/probe     (runs every directory in turn)
# env: VASP_MODULE (default vasp/6.5.1-intel-oneapi-mpi/oneapi-2024.2.1; 'module avail vasp' lists the others), VASP_EXE (default vasp_gam for a Gamma-only KPOINTS, else vasp_std),
#      NP (MPI ranks; default SLURM_NTASKS, else nproc), MPIRUN (default 'mpirun -np $NP'). The CPU builds are the safe choice; these modules are probably not GPU builds.
# Then:  ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py analyze runs/dft_check/probe
#SBATCH --job-name=dftchk
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_%x_%A_%a.out
[ -d mlp_pipeline ] || cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1   # interactive jobs set SLURM_SUBMIT_DIR to where the job was started (often $HOME): keep a repo-root cwd
D=${1:?usage: young_dft_check.sbatch <prepared dir, e.g. runs/dft_check/probe>}
[ -f $D/dirs.txt ] || { echo "no $D/dirs.txt: run tdi/dft_check.py prep-probe / prep-fep first"; exit 2; }
module load ${VASP_MODULE:-vasp/6.5.1-intel-oneapi-mpi/oneapi-2024.2.1} || { echo "module load failed: set VASP_MODULE (module avail vasp)"; exit 2; }
export OMP_NUM_THREADS=1
NP=${NP:-${SLURM_NTASKS:-$(nproc)}}; MPIRUN=${MPIRUN:-mpirun -np $NP}
run_one() {
  local d=$1
  if [ -f $d/OUTCAR ] && grep -q "General timing" $d/OUTCAR; then echo "$d: done already"; return 0; fi
  local exe=$VASP_EXE
  if [ -z "$exe" ]; then   # Gamma-only automatic mesh ("Gamma" + "1 1 1") -> the gamma-point build
    if sed -n 3p $d/KPOINTS | grep -qi "^ *g" && sed -n 4p $d/KPOINTS | grep -Eq "^ *1 +1 +1 *$"; then exe=vasp_gam; else exe=vasp_std; fi
  fi
  command -v $exe >/dev/null || { echo "$exe not found after module load (set VASP_EXE)"; return 2; }
  echo "== $d: $MPIRUN $exe ($(date +%H:%M:%S))"
  (cd $d && $MPIRUN $exe > vasp.out 2>&1); grep -q "General timing" $d/OUTCAR && echo "   finished $(date +%H:%M:%S): $(grep 'free  energy   TOTEN' $d/OUTCAR | tail -n1)" || echo "   FAILED: see $d/vasp.out"
}
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  d=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $D/dirs.txt); [ -n "$d" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $D/dirs.txt"; exit 2; }
  run_one $d
else
  while read -r d; do run_one $d; done < $D/dirs.txt
fi
