# widom-calculator: working rules (loaded automatically in every Claude Code session)

Current state, open items and next steps: **docs/HANDOFF.md** (read it first). How to run each calculation: **docs/RECIPES.md** (entry point `bash run.sh <task> key=value ...`; when the user asks for a calculation in key=value form, answer with the matching `run.sh` line, and keep RECIPES.md / run.sh up to date when a script's interface changes). This file holds only the rules that do not change from task to task.

## Who runs what
- The user runs everything on **NERSC Perlmutter** (checkout `~/work/H3O` = `/pscratch/sd/v/victornr/H3O`, user `victornr`, python `PY=/pscratch/sd/v/victornr/grace-env/bin/python`, accounts `m3349` CPU / `m3349_g` GPU; interactive 4x A100 nodes up to ~2 h; VASP 5.4.4 CPU via `VASP_MODULE=vasp/5.4.4-cpu`).
- The assistant has **no cluster access**: it writes and pushes code, the user runs it and pastes the output. Never claim a cluster run worked without the pasted output; say what was tested here (synthetic inputs, CPU) and what was not.
- Work on the branch named in the session instructions (recently `claude/youthful-brown-4qfaeg`); `main` is fast-forwarded only when the user agrees; no PRs unless asked.
- Repo: `victornadenrobinson/widom-calculator` (was `H3O`; pushes only work under the new name).

## Giving files to the user (their `git pull` fails because of local edits)
Always this form, with the branch-qualified ref and a temp file:
```
git fetch origin <branch>; git show origin/<branch>:mlp_pipeline/<f> > /tmp/<f>.new && mv /tmp/<f>.new mlp_pipeline/<f>
```
Never `git show FETCH_HEAD:...` after a plain `git fetch` (wrong branch), and never `git show ... > file` directly (the shell truncates the file even when git fails).
When a new script also needs a changed helper, list every file to fetch.

## Perlmutter rules learnt the hard way
- Submit CPU (VASP) jobs from a **login node**, not from inside an interactive GPU allocation: that gives "Job request does not match any supported policy". `sbatch --test-only` also answers that for job files that submit fine: do not rely on it.
- GPU fine-tunes (`ft_submit.sh`): the shared GPU queue gives ~55 GB host RAM per GPU and refuses more; the ~4.3k-frame tier needs `GPUS=2` (87.6 GB peak in gracemaker's batch building). 40 GB A100s are fine for GPU memory.
- 4 GRACE MD processes per GPU saturate an A100 (more is not faster).
- VASP k-meshes: shifted Monkhorst-Pack 2x2x2 (4 irreducible k with ISYM=0) is 1.8x faster than Gamma-centred 2x2x2 and as accurate; KPAR gives no gain on 32-rank lanes. INCAR lines can carry comments containing `=`: strip comments before parsing values.
- The user's `ls` prints colour codes: do not parse `ls` output in loops; name files explicitly or use globs.
- CSV group names can contain commas: use the csv module, not `awk -F,`.

## Conventions (physics / data)
- Energy labels = VASP free energy TOTEN (Fermi smearing, SIGMA = kB T of the MD, floor 1000 K); E(sigma->0) kept as `energy_sigma0`. Stress labels exclude the ionic N kB T / V term; pressures compared with DFT/hull data include it.
- Gamma-only training labels carry a composition-dependent k-point bias (-53 meV per added H); see docs/HANDOFF.md for the correction.
- dG per added H includes ideal dilution, pressure and finite-size corrections; dt 0.5 fs; thermostat damping 100 fs; do not mix results from the old 0.01-friction runs.
- Energies in meV (per atom or per added H), also in kT where useful.

## How the user likes to work
- Short answers; copy-pasteable commands, one per bash block; informative foreground output; in-place dashboards (no screen clear); quick runs; liquids over solids; simulation cells ~300-400 atoms (pure H 729).
- Be direct about what was and was not tested. Ask before expensive runs.

## Cloud sessions
- `requirements-cloud.txt`: CPU packages for tests and analysis (`pip install -r requirements-cloud.txt`; the environment's setup script does this). Model packages (tensorpotential/TensorFlow, MACE, MatterSim) are cluster-only.
- `pytest tests`: expect all pass except the known environment-specific `test_missing_environment_has_actionable_error`.
