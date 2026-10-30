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

## CPU nodes on Young (SGE) vs the A100 (measured 2026-10-29/30, mlp_pipeline/cpu_bench.py, GRACE ft_m2, tensorpotential 0.6.1 / TF 2.20 on CPU)
**Hardware / access.** Young CPU nodes: Xeon Gold 6248, 2 sockets x 20 cores (80 hardware threads, 40 slots), 188 GB, RHEL 7.9 (glibc 2.17); SGE (`qsub` / `qrsh`), not Slurm. `~/grace-venv` (built on the RHEL 9 GPU side) does not run there: use `source mlp_pipeline/young_cpu_env.sh`
(module `python/3.11.4-gnu-10.2.0` + `~/grace-venv-cpu`). Project / budget: `-P Gold -A UKCP_ED_P`; Free needs `-A UKCP_ED_P` too (`-P Free`), same speed and nodes, only the cost differs (check `gbalance -u $USER`). Whole node: `-pe smp 40`. Interactive: `qrsh -P Gold -A UKCP_ED_P -pe smp 8 -l mem=4G,h_rt=2:00:00 -now no` (an 8-slot request started in minutes; a 40-slot interactive request waited long at first; the 40-slot batch jobs started in 2 / 18 min, table below).

**Measured on 8 slots (one interactive job, node-c12n-006), calls = energy + forces of one cell, no MD overhead:**

| cell | layout (processes x threads) | s/call per process | node total (MD steps/s) | vs A100 |
|---|---|---|---|---|
| pure H, 250 atoms | 1 x 1 (alone) | 8.83 | 0.11 | - |
| pure H, 250 atoms | 8 x 1 | 9.72-9.85 | 0.82 | 0.04x of 22.7 |
| pure H, 250 atoms | 1 x 8 (8 threads, one process) | 3.83 | 0.26 | 0.01x |
| insertion, 325 atoms (2 calls per step) | 8 x 1 | 7.71 | 0.52 | 0.05x of 11.5 |

* Threads scale badly (8 threads = 2.5x one thread); one process per core is ~3x better. 8 processes together are ~11 % slower each than one alone (shared caches / memory).
* First call 12-20 s (warm-up), model load 8 s alone but 55-90 s with 4-8 processes starting together (Lustre): matters for short runs only.
* The 325-atom insertion cell is FASTER per call than the 250-atom pure-H cell (7.7 vs 9.8 s); reason not established (pure H is denser: more neighbours per atom is a guess).

**Measured on a whole node (jobs 2335963 Gold node-c12p-003, 2335965 Free node-c12n-007; 40 slots, 10 calls per process, label gold_* / free_* in runs/cpu_bench.jsonl):**

| cell | layout | s/call per process | node total (steps/s) | vs A100 | model load (s) |
|---|---|---|---|---|---|
| pure H, 250 atoms | 40 x 1 | 15.6 | 1.41 | 0.06x | 273-284 |
| pure H | 20 x 2 | 12.2-12.3 | 1.63-1.64 | 0.07x | 192-197 |
| pure H | 10 x 4 | 6.1 | 1.63-1.65 | 0.07x | 38-39 |
| insertion, 325 atoms (2 calls per step) | 40 x 1 | 13.5-14.1 | 0.96 | 0.08x | 96-124 |
| insertion | 20 x 2 | 9.5-9.6 | 1.04-1.05 | 0.09x | 186-187 |
| insertion | 10 x 4 | 4.7 | 1.06-1.07 | 0.09x | 35 |

* **A whole 40-core node = 6-9 % of one A100 (about 11-16 GPUs-worth of nodes per A100 for TDI-like work).** My earlier extrapolation (0.15-0.2x, from the 8-slot runs) was ~2x too optimistic: 40 one-thread processes slow each other by ~75 % (15.6 s per call vs 8.8 s alone; at 8 processes only 11 %): memory / cache bandwidth.
* **Best layout: 10 processes x 4 threads (or 20 x 2); 40 x 1 is the worst** (and its start-up, model load 100-280 s with 40 processes, is the slowest). 5 x 8 / 8 x 5 and the fcc H2O cell are not measured yet.
* **Gold = Free in speed (within 1 %)**; same node type. Account string for both: UKCP_ED_P. Both jobs ran 25.6 min wall, exit 0.
* **Queue wait (qacct, same minute of submission, 40 slots): Gold 2 min 9 s; Free 18 min 24 s** (one sample each; the queue was fairly empty).

| run | queue | node | submitted | started | wait | wall |
|---|---|---|---|---|---|---|
| 2335963 gold | Gold | node-c12p-003 | 20:32:10 | 20:34:19 | 2 min 9 s | 1536 s |
| 2335965 free | Free (-P Free -A UKCP_ED_P) | node-c12n-007 | 20:32:34 | 20:50:58 | 18 min 24 s | 1547 s |

**What this means (2026-10-30).**
* CPU nodes lose on speed (an A100 does the work of ~11-16 nodes), so they are for MANY INDEPENDENT runs on free, wide capacity, not for anything on the critical path or a single long trajectory (stabnpt) or fine-tuning.
* TDI insertion, one state point (17 lambda x 16,000 steps at 0.5 fs = 8 ps): A100 ~6.6 h; CPU at 10 x 4: 4.7 s per call = 9.4 s per step per lambda process -> ~42 h per lambda, 10 lambda per node at once -> 2 waves = ~84 node-hours (fits the 48 h limit per lambda with restarts, 2 nodes in parallel).
* EOS map (77 tasks x 16 volumes x 7000 steps = 8.6 M steps of 250-330 atoms): at 1.0-1.6 steps/s per node ~1,500-2,400 node-hours (4-5 days on 20 free nodes); the GPU estimate in young_eos_map.sbatch is ~77-110 GPU-hours. (My earlier 770 node-hour figure assumed 40 x 1 at 11-13 s per call and was wrong.)
* DFT checks (VASP, CPU-only on Young) and small-cell relaxations (h3o_static, 36-64 atoms) are the natural CPU jobs.
Reproduce / extend: `qsub -V -P Gold -A UKCP_ED_P -pe smp 40 -l h_rt=1:30:00 mlp_pipeline/young_cpu_bench.sge` with `CELLS`, `CALLS`, `LAYOUTS`, `BENCH_LABEL` exported (header of that file); table: `python mlp_pipeline/cpu_bench.py --table runs/cpu_bench.jsonl`; queue wait: `qacct -j <jobid> | grep -E "qsub_time|start_time"`.

## Using the faster options in production (after the benchmark)

- MPS with the current engine (interactive job, before the launcher; the daemon ends with the job or with `echo quit | nvidia-cuda-mps-control`):
  ```
  export CUDA_MPS_PIPE_DIRECTORY=$TMPDIR/mps_pipe CUDA_MPS_LOG_DIRECTORY=$TMPDIR/mps_log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d
  ```
- Batched engine: `ENGINE=batch <usual knobs> bash mlp_pipeline/young_tdi_par.sh <phase>`. Same tag, files and restarts, so a run can switch engines between launches.
  Its own log is `runs/tdi/<tag>_batch.log` (status every minute: model calls, structures per call, ms per call, model share of the wall time).
- Before making it the default: one short GRACE run side by side (e.g. an insertion with `PS=1`, `OUTDIR=runs/tdi_engcmp/{lambda,batch}`) and `summarize.py` on
  both directories: the same G within the errors.

## Next: tensorpotential's dense engine (Young has 0.6.1, which has it)

`GRACE_MODE=uniform` (read by `04_benchmark.get_calc_spec`, recorded in the stage0 json `meta.grace_mode`) creates the TPCalculator with `mode="uniform"`: the dense
reshape aggregation, documented as 1.3-1.7x faster per structure for MD (fixed atom count, near-constant neighbour count). Same model and weights, only the summation
layout changes (round-off-level differences; random GRACE-2L on CPU: 4e-16 eV/atom). It needs a `compute_dense` signature in the SavedModel; without one the calculator
falls back to segment_sum with a warning. Test (GPU job, repo root, about 3 min):
```
~/grace-venv/bin/python mlp_pipeline/tdi/bench.py dense insH       # 'dense engine ACTIVE / NOT available', ms per call of both engines, max differences
```
If it is not available: re-export ft_m2 from its checkpoint with 0.6.1 into a new directory (the old model stays as it is), then run the same test with
`--model grace-ft:<new dir>`:
```
find runs/grace/ft_m2 -name model.yaml; find runs/grace/ft_m2 -name "checkpoint*.index"
~/grace-venv/bin/grace_utils -p <model.yaml> -c <checkpoint .index> export -n runs/grace/ft_m2_dense/saved_model
```
If it is faster and agrees: a throughput run `GRACE_MODE=uniform SYSTEMS=insH NPROCS=8 MPS="0 1" PARTS=engine bash mlp_pipeline/bench_tdi.sh`, then a short side-by-side
GRACE run before using it in production.

### Dense engine result (2026-10-28, Young, ft_m2 as exported, tensorpotential 0.6.1)

`bench.py dense insH`: dense engine ACTIVE (ft_m2 already has `compute_dense`, no re-export needed). Single process, alternating 251 / 250-atom structures:
segment_sum 303 ms per call, dense 181 ms per call = **1.67x**. Differences dense vs segment_sum: max 5.6e-8 eV/atom, 2.3e-6 eV/A, 1.5e-7 eV/A^3 (float32 GPU
round-off, the same level as two segment_sum runs). Next: throughput with 8 processes (`GRACE_MODE=uniform ... bench_tdi.sh`), then a side-by-side G check.

**Throughput with the dense engine in the real lambda engine (2026-10-28 19:53, `GRACE_MODE=uniform bench_tdi.sh`, insH): NO gain, MORE memory.** 1 process: 336 ms/step
(segment_sum 307), 2.98 steps/s; 236 s to warm up (many XLA compiles); 6.6 GB per process (segment_sum 4.6). 8 processes: one died of GPU out of memory (allocating
2.2 GB), the other 7 gave 9.5 steps/s (segment_sum 10.85 with 8). The dense layout pads every atom to the largest neighbour count. In the H fluid that count
fluctuates, so the shapes change, recompile and keep their buffers. The 1.67x of `bench.py dense` was misleading: that test alternates 250- and 251-atom structures
through one calculator, which penalised the segment_sum side. **Decision: keep the default segment_sum engine; GRACE_MODE stays an opt-in switch, not recommended.**

**Not done (decided 2026-10-28):** Verlet neighbour list with a skin and the numpy data path for the per-lambda engine. It would only speed up single-process phases (stage 0, prep, the tail of a run; roughly 160 -> 100 ms per call), about 5-15% of a job, because with 8 processes the CPU work is already hidden behind the saturated GPU.

## How many lambda processes per GPU (bench_tdi.sh, 2026-10-28, A100 40 GB, GRACE ft_m2, segment_sum engine)

Total MD steps/s (ms per step of each process); GPU memory with time-slicing:

| processes | insH 251 atoms (2 calls/step) | insH with MPS | pure H 250 (1 call/step) | GPU memory, insH (MiB) |
|---|---|---|---|---|
| 1 | 3.25 (307) | 3.26 (307) | 6.59 (152) | 4569 |
| 2 | 6.63 (302) | 6.60 (303) | - | 9139 |
| 3 | 8.93 (336) | 7.54 (398) | - | 13702 |
| 4 | **11.48** (348) | 8.98 (445) | **22.66** (177) | 18268 |
| 6 | 10.59 (567) | 10.89 (551) | - | 27405 |
| 8 | 10.85 (737) | 11.89 (673) | 22.02 (363) | 36534 |

- **The GPU is saturated at 4 processes**: about 23 GRACE calls/s for 250 atoms (43 ms per call) for both systems. Up to 4 processes each runs at
  single-process speed. More processes only slice the same GPU time thinner, and need more memory (4.6 GB each; the 9th and 10th died of out of memory before).
- **Recommendation: NPAR=4** (the launcher's default). Same total speed as 8, half the GPU memory, no out-of-memory retries. MPS gives nothing at 4 (it was slower at 3-4)
  and +10% only at 8. Two jobs on one GPU cannot run faster in total than one job with NPAR=4.
- tdi/eta.py's tail model now uses this table.
