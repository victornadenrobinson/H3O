# TDI engine performance (GPU throughput)

Goal: more MD steps per second for the lambda runs on one A100 40 GB, with the same results. Results never change: every speed option produces the same node files
(`<tag>_node<k>.npy`, `_state.npz`, `_thermo.csv`, `_traj.extxyz`, logs), so `summarize.py`, `bar_lambda.py`, `report.py` and restarts work unchanged.

## What was measured before (2026-10-28, GRACE ft_m2, 251-atom pure-H insertion, 2 model calls per step)

| setup | ms per step (each process) | total MD steps/s |
|---|---|---|
| 1 process (stage-0 free NVT, 1 call per step) | 160 per call | ~6 calls/s |
| 2 processes, time-slicing | ~360 | ~5.6 |
| 8 processes, time-slicing | ~750 | ~10.7 |

nvidia-smi showed 100% busy, 4.5 GB per TensorFlow context, CPU mostly idle. "100% busy" only means a kernel is running at all times. With small
graphs (251 atoms) each kernel fills a small part of the GPU, and time-slicing runs the contexts one after another.

## Options

1. **NVIDIA MPS** (no code change): kernels of different processes run at the same time instead of time-sliced. Start the daemon in the job before the launcher
   (see the commands below). Needs the GPU in default or exclusive-process mode; `bench_tdi.sh` tries it and skips it if the daemon does not start.
2. **Batched engine, `ENGINE=batch`** (tdi/batched.py): all lambda values of a run in ONE process, one GRACE call per MD step for all of them (insertion: host and
   host-with-ghost in the same call, so 1 call instead of 2 per step). One TensorFlow context instead of NPAR (memory, no OOM retries), larger graphs per kernel.
   - How: every lambda value runs the unchanged `run_real.run_node` in its own thread (ASE Langevin, its own seeded RNG, chunks, files, restart checks). Only the model
     evaluations go through a `Batcher`: once every running lambda value has submitted its structure, one batched call evaluates them all.
   - Batching GRACE: tensorpotential's exported `compute` signature (`ComputeStructureEnergyAndForcesAndVirial`) sums over ALL atoms of its input as one structure,
     but it also returns per-atom energies and per-bond pair forces. Several structures are joined into one disjoint graph with the calculator's own data builders
     (bond indices offset per structure, one padding at the end). Energies are per-structure sums of the atomic energies, forces are per atom, and the stress of each
     structure comes from its own bonds' pair forces (the model's virial formula). A self-check against single calls runs before any MD.
   - CPU side: the neighbour list (matscipy) costs ~45 ms per 250-H structure at 6 A cutoff (~900 neighbours per atom) and holds the Python GIL, so the lists are
     built in a pool of worker processes (`H3O_NL_WORKERS`, default CPUs - 1, at most 16). The builder's per-bond Python loops are replaced by numpy indexing,
     checked to give identical arrays on the first structure of each size.
   - GPU memory: one call with 21 x 250 atoms has ~5 M bonds. If it runs out of memory the call is split in halves automatically. `BATCHMAX=<atoms>` starts smaller.
3. Not done: LAMMPS + GRACE (a new engine with its own validation), and the `mode="uniform"` dense engine of newer tensorpotential versions
   (documented as 1.3-1.7x per structure for uniform structures; check `pip show tensorpotential` on Young first).

## Validation (sandbox, CPU)

| test | result |
|---|---|
| toy model, insertion n = 1, 4 lambda, per-lambda vs batched | all node files bit-identical (npy, state, thermo csv, extxyz) |
| toy fcc H2O (Einstein springs, fixed O centre of mass), pure H, liquid with HOT | bit-identical |
| restarts: per-lambda then batched, batched then batched, vs per-lambda only | bit-identical |
| toy T0 bookkeeping (`toy_validate.py bookkeeping`) before / after the change | identical output, PASS (exact to 1e-13 eV) |
| young_tdi_par.sh with the toy model, ENGINE=batch vs default | same summarize.py row (G, error, drift, P) and same bar_lambda.py MBAR |
| random-weight GRACE-2L model (tensorpotential 0.6.2.1, CPU), batched vs single call | dE 3e-16 eV/atom, dF 4e-16 eV/A, dstress 2e-16 eV/A^3 |
| same model, 3 lambda x 120 steps, per-lambda vs batched engine | series differ by 1e-13 eV (round-off of the summation order) |

On GPU the single-call engine is not bit-reproducible either (atomic adds), so "the same G within errors" is the test for GRACE runs. On CPU a batched call costs
the same per structure as single calls (speed-up 1.00 at B = 4): the CPU has no idle capacity. The gain can only come from the GPU and must be measured on Young.

## Benchmark on Young (to run; results go in the table below)

```
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit
sbatch mlp_pipeline/bench_tdi.sh                       # everything, ~1-1.5 h of one GPU; output runs/slurm_tdibench_<id>.out
```
or inside an interactive GPU job (repo root):
```
SLURM_SUBMIT_DIR=$PWD SYSTEMS=insH NPROCS="1 8" NREPS="8 16" bash mlp_pipeline/bench_tdi.sh      # quick subset, ~20-30 min
```
Each configuration times 100 MD steps (`STEPS`) after model loading and a 20-step warm-up. No production run and no files in runs/tdi. Rows:
per-lambda engine with 1, 2, 4, 8 processes, time-sliced and with MPS (total steps/s, GPU memory); batched engine with 4, 8, 16 lambda values in one process; model
calls alone (one batched call of B structures vs B single calls, ms per structure). The markdown tables are printed at the end
(`python3 mlp_pipeline/tdi/bench_table.py runs/bench/bench_*.jsonl`).

### Results (2026-10-28, node-x12t-006, A100-SXM4-40GB, 8 CPUs, GRACE ft_m2, code 549096a; insH = 1 H into 250-atom pure H, 2 model calls per step)

| setup | total MD steps/s | ms/step each | GPU memory (MiB) |
|---|---|---|---|
| per-lambda engine, 1 process | 3.25 | 307 | 4569 |
| per-lambda engine, 8 processes, time-slicing (current production) | **10.85** | 737 | 36534 |
| per-lambda engine, 1 process, MPS | 3.26 | 307 | 4603 |
| per-lambda engine, 8 processes, MPS | **11.89** (+10%) | 673 | 36526 |
| batched engine, 8 lambda values in 1 process | 7.06 (-35%) | 1132 (1122 per model call) | - |
| batched engine, 16 lambda values in 1 process | 7.07 (-35%) | 2262 (2241 per model call) | - |

Model calls alone (ms per 251-atom structure; batched includes data preparation):

| B structures | single TPCalculator call | one batched call | speed-up | data preparation per batched call (ms) |
|---|---|---|---|---|
| 1 | 160 | 98 | 1.63 | 35 |
| 2 | 151 | 85 | 1.77 | 48 |
| 4 | 151 | 73 | 2.06 | 63 |
| 8 | 153 | 72 | 2.12 | 105 |
| 16 | 152 | 70 | 2.18 | 199 |

Self-check on the GPU (float32 model): batched vs single |dE| 1.4e-7 eV/atom, |dF| 1.8e-6 eV/A, |dstress| 9e-8 eV/A^3 (the GPU round-off level).

**Conclusions**
- The GPU itself is the limit. Eight time-sliced processes already push about 21.7 structures/s, or **46 ms of GPU per structure**. One batched call needs about 70 ms per
  structure (about 58 ms of GPU after the 12 ms of data preparation), so the batched engine is 35% SLOWER than the current production setup. Most of the 160 ms of a
  single call is CPU work (data preparation, about 60 ms) plus GPU latency. Eight processes hide both, and batching hides them as well, but it does not make the
  GRACE kernels themselves cheaper.
- MPS: +10% (11.9 vs 10.9 steps/s), with no code change and the same results: worth using.
- Batched engine: keep it opt-in (correct, but slower on this hardware). It would only help if the GRACE kernel cost per structure fell, for example with the dense engine.
- A faster neighbour list or data preparation would not help 8-process runs (the CPU work is already hidden). It would help the tail of a run, when fewer than NPAR
  processes are left: one process with the batched data path needs 98 instead of 160 ms per call.
- What remains: (a) more GPUs (lambda values split over GPUs or jobs: linear in wall time); (b) cheaper GRACE kernels: check the tensorpotential version on Young
  (`~/grace-venv/bin/pip show tensorpotential`) for the dense "uniform" engine, documented as 1.3-1.7x per structure, to be validated against the current numbers;
  TF32 matmuls are switched off by tensorpotential on purpose (precision), so they are not an option without a precision check; (c) fewer MD steps for the same error:
  a heavier H mass in the MD with a larger dt (configurational averages do not depend on the masses), or an adaptive length per lambda value. These change how the
  samples are generated (the expected G stays the same) and need your decision and a validation run.

## Using the faster options in production (after the benchmark)

- MPS with the current engine (interactive job, before the launcher; the daemon ends with the job or with `echo quit | nvidia-cuda-mps-control`):
  ```
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d
  ```
- Batched engine: `ENGINE=batch <usual knobs> bash mlp_pipeline/young_tdi_par.sh <phase>`. Same tag, files and restarts, so a run can switch engines between launches.
  Its own log is `runs/tdi/<tag>_batch.log` (status every minute: model calls, structures per call, ms per call, model share of the wall time).
- Before making it the default: one short GRACE run side by side (e.g. an insertion with `PS=1`, `OUTDIR=runs/tdi_engcmp/{lambda,batch}`) and `summarize.py` on
  both directories: the same G within the errors.
