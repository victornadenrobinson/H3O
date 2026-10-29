# source this on a Young CPU node (RHEL 7, SGE), repo root:   source mlp_pipeline/young_cpu_env.sh
# Python for GRACE / our tools on the CPU nodes: module python/3.11.4-gnu-10.2.0 (provides libpython3.11 + a new libstdc++ for TensorFlow) + the venv ~/grace-venv-cpu (TF 2.20, tensorpotential 0.6.1; ~/grace-venv is RHEL 9 only).
# NOT for VASP: it loads gcc-libs/10.2.0; young_dft_check.sge resets to the verified gcc-libs/4.9.2 itself, but run VASP by hand in a shell WITHOUT this sourced (or inside ( ... ) ).
# env: CPU_VENV (default ~/grace-venv-cpu)
module unload -f compilers mpi gcc-libs
module load python/3.11.4-gnu-10.2.0 || { echo "young_cpu_env: module load python/3.11.4-gnu-10.2.0 failed (module avail python)"; return 1 2>/dev/null || exit 1; }
CPU_VENV=${CPU_VENV:-$HOME/grace-venv-cpu}
[ -f "$CPU_VENV/bin/activate" ] || { echo "young_cpu_env: no venv at $CPU_VENV (build: python3 -m venv ...; pip install --only-binary=:all: 'pandas<2.3' ase tqdm tensorpotential==0.6.1, TMPDIR on scratch)"; return 1 2>/dev/null || exit 1; }
source "$CPU_VENV/bin/activate"
export TMPDIR=${TMPDIR:-/lustre/scratch/$USER/tmp}; mkdir -p "$TMPDIR"
export TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2
echo "young_cpu_env: $(python --version 2>&1) at $(command -v python); TMPDIR $TMPDIR; TF_USE_LEGACY_KERAS=1"
