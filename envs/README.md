# Environments (one per MLP; they conflict, so keep them separate)

Versions below are starting points from memory, not verified: check each package's current install docs.

```bash
# MACE  (zero-shot + fine-tune)
python -m venv ~/venvs/mace && source ~/venvs/mace/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121        # match the cluster's CUDA
pip install mace-torch ase pandas matplotlib tqdm

# GRACE  (tensorpotential; TensorFlow-based, own env)
python -m venv ~/venvs/grace && source ~/venvs/grace/bin/activate
pip install tensorpotential ase pandas tqdm                                   # see github.com/ICAMS/grace-tensorpotential

# MatterSim
python -m venv ~/venvs/mattersim && source ~/venvs/mattersim/bin/activate
pip install mattersim ase pandas tqdm

# CPU-only analysis (scripts 00-03, 06-11)
pip install numpy pandas ase matplotlib tqdm
```
Scripts 04 (benchmark) and 05 (fine-tune) are the only ones needing a model env; 04's GRACE/MatterSim calls are unverified.
Foundation weights are downloaded on first use, which needs internet on the node: pre-download on a login node if compute nodes are offline.

## Young (UCL MMM Hub, user mmm0037): what exists (from `envs/check_envs.sh`, 2026-10-25)

Scheduler: **Slurm** (`sbatch`). Modules: `python/3.11.6`, `python/3.13.0`, `cuda/12.6.2`, `cuda/12.9.1`, `cuda/13.3.0`.
Repo location: `/lustre/scratch/mmm0037/H3O` (= `~/work/H3O`). Scratch is purgeable: keep a copy of the data bundle elsewhere.

| Model | Environment | Versions | CUDA build |
|---|---|---|---|
| MACE | `~/work/envs/mace-ng` (helper: `~/work/envs/mace-env-ng.sh`; older env `~/work/envs/mace`, `mace-env.sh`) | mace-torch 0.3.16, torch 2.6.0+cu126, py3.11 | cu126 (use `cuda/12.6.2`) |
| MatterSim | `~/work/envs/mattersim-env` | mattersim 1.2.3, torch 2.6.0+cu126, py3.11 | cu126 |
| GRACE | `~/grace-venv` | tensorpotential 0.6.1, py3.11, no torch (TensorFlow-based) | check TF GPU in a GPU job |
| (viewers) | `~/work/envs/conda-viewer`, `viewer-tk` | not for MLPs | |

Version pins live in `~/work/envs/constraints.txt` and `constraints-ng.txt`.
`torch.cuda.is_available()` printed False on the login node (no GPU there): re-check inside a GPU job.
Analysis scripts (00-03, 06-11) need pandas/ase/matplotlib/tqdm: the base python has no pandas; use one of the envs above or make a small CPU env.
Not yet verified: that MACE-ng's CLI flags match `05_finetune_young.sbatch`, and GRACE's GPU stack.

## Which energy goes into the fine-tuning sets (researched 2026-10-25)
- **Label = VASP free energy (`free energy  TOTEN`)**, written as `energy`; E(sigma->0) is kept as `energy_sigma0` metadata.
  VASP forces are derivatives of the free energy F, not of the sigma->0 extrapolation (VASP wiki "Hellmann-Feynman forces"; VASP forum). At 5000-13000 K the smearing entropy is large, so mixing E(sigma->0) with F-consistent forces would put a state-dependent inconsistency into every label.
- Stress: ASE/extxyz virial stress (eV/A^3, ASE sign: P = -trace/3). It excludes the ions' kinetic term, as MLP training expects; add N kB T / V back only when comparing MLP-MD pressure with the hull EOS.
- **Not verified:** which VASP energy (TOTEN or sigma->0) the foundation models' own training sets used (MPtrj, OMat24, MatterSim data). Per-element reference shifts absorb constant offsets, but not the T-dependent entropy term, which is why we fine-tune.
- MACE: re-estimate E0s for the new reference (`--E0s=estimated`; `average` is discouraged for fine-tuning); consider multihead replay fine-tuning if plain fine-tuning gives unstable MD. Key names/flags must be checked against `mace_run_train --help` in mace-ng 0.3.16.
- GRACE (gracemaker): use `shift: auto` when fine-tuning a foundation model (per-element shifts fitted to the new DFT data).
- MatterSim: fine-tuning data format, energy/stress units and loss weights NOT verified; read the microsoft/mattersim finetune script before use.
