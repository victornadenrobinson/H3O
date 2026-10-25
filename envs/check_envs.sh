#!/bin/bash
# List python envs (venv + conda) and report which have the MLP packages. Run on the cluster login node (read-only).
# usage: bash envs/check_envs.sh [extra dir with venvs ...]
roots=("$HOME/venvs" "$HOME/.virtualenvs" "$HOME/envs" "$HOME/Scratch" "$HOME/miniconda3/envs" "$HOME/anaconda3/envs" "$HOME/.conda/envs" "$@")
pys=()
for r in "${roots[@]}"; do
  [ -d "$r" ] || continue
  while IFS= read -r p; do pys+=("$p"); done < <(find "$r" -maxdepth 4 -path '*/bin/python' -type f -o -maxdepth 4 -path '*/bin/python' -type l 2>/dev/null)
done
printf '%-60s %-8s %-12s %-12s %-12s %-12s %s\n' ENV PYTHON mace-torch tensorpotential mattersim torch "cuda?"
for p in $(printf '%s\n' "${pys[@]}" | sort -u); do
  "$p" - <<'PY' 2>/dev/null
import sys, importlib.metadata as m
def v(n):
    try: return m.version(n)
    except Exception: return "-"
try:
    import torch; cu = str(torch.cuda.is_available()) + ("/" + str(torch.version.cuda) if torch.version.cuda else "")
except Exception: cu = "-"
print("%-60s %-8s %-12s %-12s %-12s %-12s %s" % (sys.prefix[-60:], "%d.%d" % sys.version_info[:2], v("mace-torch"), v("tensorpotential"), v("mattersim"), v("torch"), cu))
PY
done
echo; echo "modules (python/cuda):"; (module avail python 2>&1; module avail cuda 2>&1) | grep -i -E "python|cuda" | head -15
echo; command -v conda >/dev/null && { echo "conda envs:"; conda env list; }
echo; echo "scheduler:"; command -v qsub >/dev/null && echo "qsub (Grid Engine)"; command -v sbatch >/dev/null && echo "sbatch (Slurm)"
echo "note: torch cuda? is False on login nodes without GPUs; check again inside a GPU job."
