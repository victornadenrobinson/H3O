# TDI runs: what gets written, how to read it, and the rules for new code

Applies to `mlp_pipeline/tdi/run_real.py` and its launchers (`young_tdi_par.sh`, `young_tdi_scan.sbatch`, `young_tdi_hull.sbatch`, `young_melt_scan.sbatch`). Wording rule: say **lambda / lambda values**,
never "nodes" (nodes = compute nodes). The names `NODES`, `--node` and `*_node<k>.*` are kept in code and file names only for compatibility.

## 1. One run = one tag
`<phase>_<P>GPa_<T>K[_h<x>o][_n<ghosts>][_N<atoms>]`, e.g. `bcc_400GPa_5000K_h2.25o`, `insert_400GPa_5000K_n2`, `H_400GPa_5000K_N500`, `liquid_400GPa_7000K_h3o`.
Phases: `bcc`, `fcc` (Einstein O + ideal-gas H reference), `H` (pure hydrogen fluid), `liquid` (ideal gas of all atoms), `insert` (fcc H2O + n ghost protons).
All files of a run live in `runs/tdi/` (tests: `runs/tdi_test/`, melt scan: `runs/melt/`). `runs/` is not in git; Lustre scratch may be purged: copy results to the repo (`tdi/summarize.py -o docs/data/tdi_results.csv`).

## 2. Files per run (`runs/tdi/<tag>_...`)
| file | content | written |
|---|---|---|
| `_stage0.json` | setup and constants: volume, N, Einstein k, reference free-energy pieces (`f_ref_cons_eV`, `f_kin_eV`, `PV_eV`), all arguments (`args`), provenance (`meta`: git commit, host, date, versions) | at start |
| `_node<k>.npy` | float array: every recorded sample (every `--every` steps) of the integrand `U_full - U_ref` in eV for the whole cell, for lambda index k | after every chunk |
| `_node<k>_state.npz` | restart state: positions, momenta, and the series (`series`, `tk`, `eful`, `espr`) | after every healthy chunk |
| `_node<k>_thermo.csv` | one row per 40 steps: `t_ps, T_inst_K, U_MLP_eV, E_spring_eV, dUdl_eV, P_MLP_GPa, u_rms_A, u_max_A` | after every chunk (appended) |
| `_node<k>_traj.extxyz` | configurations every 40 steps, tagged with lambda, index, time (positions only; for single points / FEP / analysis) | after every chunk (appended) |
| `_node<k>.log` | full text log of that lambda (see section 3) | appended |
| `_stage0.log` | log of the stage-0 step | at start |
| `tdi_<tag>.json` | collected result from `tdi/collect.py`: lambda, weights, per-lambda means and errors, `hist` (G vs time), `G_meV_atom`, `G_err_meV_atom` | end of each launcher run, or when `collect.py` is called |

Units: eV for whole-cell energies, meV/atom for G and the integrand tables, GPa for pressure, Angstrom, ps, K.

## 3. How to read the log lines
Banner (first thing printed by every process): system, cell, conditions, potential, path (U(lambda)), reference, Einstein k, dynamics, steps per lambda, quadrature table (index, lambda, weight, steps), total cost, output files.

Per chunk, per lambda:
```
lambda 8 = 7.562e-01 t  0.50 ps: last chunk <dU/dl> -281.886  all -281.886 +- 0.723 eV | T_kin 4973 K  <dU/dl> ...  U_MLP/atom ...  E_spring/atom ...
  min d(H-H) 0.58 d(H-O) nan A  |u| now rms 11.1 max 26.1 A | P_MLP 406.5 GPa | trace ▁█ | 2.8 min | 160 ms/step | 33% | ETA 15:01 (in 3 min)
```
- `<dU/dl>`: the integrand (U_MLP - U_ref) in eV, whole cell. `all ... +- ...` is the mean and block standard error over everything so far.
- `T_kin`: must stay near the target; more than 2x (chunk mean) or 20x (any sample) stops that lambda with `UNSTABLE DYNAMICS` (exit code 5), the bad chunk is not saved: rerun with `DT=0.5` and `FRESH=1`.
- `min d(H-H)`, `d(H-O)`: closest contacts; overlaps below about 0.5 A at large lambda are a warning sign.
- `|u|`: O displacement from the lattice sites (crystals; stays about 0.2-0.6 A), or displacement of every atom from its start (fluids: should grow far beyond the lattice spacing, which shows they melted).
- `P_MLP`: mean MLP virial pressure (GPa) incl. N kB T / V over the last chunk. Only the lambda values near 1 are meaningful (small lambda is mostly springs / ideal gas). Target is within a few GPa; G uses the target P in P V (error second order).
- `trace`: sparkline of the per-chunk means of this lambda (min-max scaled: shape, not size).
- `ms/step | % | ETA`: speed over the last chunk (a warm-up call keeps compile time out), percent of this process's steps, wall-clock ETA.

Launcher monitor (every 2 min): GPU utilisation and memory, one `lambda k: ms/step | % | ETA` line per lambda, then the brief dashboard (preview G with the number of lambda values that have data, Gauss-Legendre G(t) once every lambda has 0.25 ps, dG(t) if two phases are given).

## 4. Numbers and what they mean
- **G (meV/atom)** = (F_ref + integral + kinetic terms [- kT ln V for fixed-COM crystals] + P_target V) / N. Absolute, classical nuclei, MLP only. Quote differences (fcc - bcc, mixing), not absolute G, and only the Gauss-Legendre value.
- **Error** = block (5 blocks) standard error of each lambda propagated with the quadrature weights: statistical only; it ignores quadrature, dt, finite size, MLP vs DFT, nuclear quantum effects.
- **PREVIEW** (dashboard) = linear interpolation in x = lambda^(1/p) through the lambda values that exist. It is biased high by about +11 +- 1.5 meV/atom against the final Gauss-Legendre value (measured on 4 runs): never quote it.
- **insert** phase: the "G (meV/atom)" column is mu_H per **added H** in meV (the json `N` is the number of ghosts), with no P V term. Compare with the pure-H G per atom at the same P, T: negative difference = H dissolves.
- Mixing free energy per added H (H2O crystal + n H fluid -> H(2+n)O crystal) = [N_mix G_mix - N_H2O G_H2O - n_H G_H] / n_H with each G in meV/atom from the same P, T.

## 5. Running (copy-paste forms; always `git pull --no-rebase --no-edit` first on Young)
- Interactive job (GPU): `srun -p intergpu --qos=intergpu -A allusers --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash` (max wall 6 h; no per-user job limit found).
- One system: `T=5000 P=400 NODES=10 NPAR=10 DT=0.5 EQUIL=0.4 PS=1.5 CHUNK=0.25 bash mlp_pipeline/young_tdi_par.sh <bcc|fcc|H|liquid|insert>`; env knobs: `HPER` (H per O), `NGHOST`, `NATOMS` (pure H cell), `K` (Einstein constant; `K=0` measures it with a lambda=1 stage), `ONLYNODES="0 3"`, `FRESH=1`, `TAG=test`.
- GPU memory per process: about 2.5-3 GB for 324-416 atoms, 4.6 GB for 250 H, 6-7 GB for 500 H, 8.7 GB for 729 H. 40 GB card: `NPAR<=8` for cells above 400 atoms, `NPAR<=5` for 500 H.
- Restart: rerun the identical command (completed lambdas exit at once); extend a run with a larger `PS`. Same dt for the whole series.
- Monitor from any shell: `~/grace-venv/bin/python mlp_pipeline/tdi/live.py runs/tdi <tagA> [<tagB>] [--loop 60] [--brief] [--png]`; plots `tdi/plot_tdi.py`; table `tdi/summarize.py runs/tdi`; extrapolation check `tdi/uq_check.py` (GRACE gamma on saved snapshots; `GRACE_UQ` is off in production runs because its cost is unmeasured).
- Batch: `sbatch mlp_pipeline/young_tdi_scan.sbatch` (6 tasks), `young_tdi_hull.sbatch` (10 tasks), `young_melt_scan.sbatch` (6 tasks): tables are editable through env vars (see their headers).

## 6. Rules for any new script in this project
1. Print a plan banner before any MD (system, conditions, path, reference, lambda table, cost, output files).
2. Print progress with speed, percent and wall-clock ETA; flush every line (`flush=True`, `python -u`).
3. Save after every chunk and make runs restartable; never overwrite healthy data with a failed chunk (guard against runaway T).
4. Write plain-text outputs (CSV/JSON/extxyz) next to the run: thermo, snapshots, provenance (git commit, host, date). No results only in terminal scrollback.
5. Energies in meV (also in kT), pressure in GPa, say what is statistical error only, and state caveats (dt, size, quantum, MLP error).
6. Tag every run with phase, P, T and composition; never reuse a tag for a different cell size or dt.
7. Test locally with the LJ stand-in (`model = lj`, `--vpa`, tiny cells) before sending code to Young; check `python3.11 -m py_compile`.
8. Keep commands for the user as copy-paste blocks that say which machine to run them on.
