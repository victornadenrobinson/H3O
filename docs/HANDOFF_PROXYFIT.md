# Handoff 2026-10-31: pwidom proxy refit, cluster logistics, review findings

For whoever continues this (another assistant or a person). Read `HANDOFF.md`, `PROJECT.md`, `docs/CURRENT_ISSUES.md` and `docs/MU_METHODS.md` for the project; this file covers the session of 2026-10-30/31 only.
Branch with all of it: `claude/jolly-mccarthy-0zrnfe` (public repo `victornadenrobinson/H3O`; nothing merged to `main`, no PR opened). The user's own branch `claude/eloquent-hopper-kfjdei` on Young is separate (queue tooling); do not merge blindly, `pwidom.py` conflicts.

## 1. The goal
`tdi/pwidom.py` (route R1 in `docs/MU_METHODS.md`) estimates mu_H of one added H by proxy-guided Widom insertion + deletion + BAR. It is **unbiased for any proxy that does not depend on the frames of the estimate**; the proxy only sets the variance
(std of dU - phi in kT, Kish fraction; want <~ 1-2 kT, > ~4 kT degrades to plain Widom). The current proxy is the `toy_fit.py` OH/HH spline pair fitted to DFT frames. Plan: refit it to the REAL model's (GRACE ft_m2) insertion energies
dU = U(frame + H at x) - U(frame). For a pair proxy dU is linear in the spline coefficients, so it is one weighted least-squares solve. Decision rule: run the fit, read the held-out gap; if still > ~2 kT, next step is an EAM-style embedding term
(then a small NN / EDDP-like model; costs ~10-100x more per candidate because neighbours must be re-evaluated). Expected speed-up of the whole pwidom job is small (MD dominates, ~1.5-2x at most); the real value is robustness. If the existing proxy already gives <~1-2 kT at 500 GPa / 6000 K, none of this is needed.

## 2. What was built (all tested only on the toy model `toyice` in a CPU sandbox, never on GRACE)
| file | what | status |
|---|---|---|
| `mlp_pipeline/tdi/proxy_fit.py` | `label`: reads cached `runs/pwidom*/pw_*_frames.npz` (cubic cell = V^(1/3), so no host options needed), a few frames per state, Rosenbluth-selected (at `--temper` x T) + uniform probe points, GRACE dU; one `*_probes.npz` per source, resumable; `--select`, `--shard i/n`. `fit`: weighted LS for OH/HH splines (weight exp(-(dU-ref)/(wT kT)) per state), `--hold` keeps the production state out of the fit, every 4th frame held out elsewhere, per-state table, "overlap probes with phi too low" diagnostic; writes a toy_fit-format json for `pwidom.py --proxy` | toy: held-out gap 4.6 -> 0.10 kT (easy: pair truth) |
| `mlp_pipeline/young_proxy_fit.sbatch` | label + fit (ice-like and pure H separately, production state held out) + pwidom old-vs-new dg on Young. `PROD=0` stops after the fit; works with `bash` in an interactive job; needs only `MODEL=` when `PROD=0` | syntax-checked |
| `mlp_pipeline/archer2_proxy_label.sbatch` | CPU version: 16 independent shards x 8 threads on one ARCHER2 node, then the fit | syntax-checked; CPU GRACE path never run |
| `mlp_pipeline/perlmutter_submit.sh` | submits any Young `.sbatch` on Perlmutter (strips partition/account/qos/gres/mem lines, applies `-C gpu -A m3349_g -q regular --gpus N`); `DRYRUN=1` | dry-run only |
| `mlp_pipeline/perlmutter_lagtest.sh` | batch health test (shell startup, filesystems, GPU, TF import) | not yet run |
| `mlp_pipeline/tdi/pwidom.py` | frame cache stores a model fingerprint (mismatch refuses, old caches warn); `dg` refuses different T / P / model; results without bulk modulus flagged `pressure_corrected=false` | toy-tested |
| `mlp_pipeline/tdi/run_real.py` + `tests/test_pipeline.py` | all-in-one insertion path divided G by all atoms instead of the added H (`st["N"]`): fixed, regression test runs both routes. Production launcher (`young_tdi_par.sh`, node route + collect.py) was never affected | fixed, 6 tests pass |

## 3. How to continue (in order)
1. **Get the probe labels.** `PROD=0 PER_SOURCE=5 bash mlp_pipeline/young_proxy_fit.sbatch 2>&1 | tee runs/proxyfit.log` inside any GPU job on Young (~15-30 min of one GPU for ~20k structures; estimate from ~23 GRACE calls/s in `docs/PERFORMANCE.md`), or as a batch job with `--time=00:45:00` so it backfills. Alternatives: Perlmutter GPU, or ARCHER2 CPU (`archer2_proxy_label.sbatch`; CPU GRACE measured by the user at 0.04-0.05x an A100 per 8 slots, 40-core numbers are extrapolated, 0.15-0.2x). A subset of states near the production point is enough (`SELECT`).
2. **Read the held-out lines** in the fit output: "held-out sources" (production state never seen) and the per-state table. Want std(dU - phi) of selected probes <~ 1-2 kT and Kish >~ 0.3.
3. **Production comparison** (`PROD=1`, on Young where the production frames and stage0 jsons are): `pwidom.py` with the new proxy and the old one on the same frames, `dg` of both. Expect similar dg within error; the new proxy should show a smaller error bar or a better Kish.
4. If the gap stays > 2 kT: implement the EAM-style term (needs per-neighbour density updates; still numpy-cheap), refit with the same `label` data.

## 4. Clusters (what is known; most from the user's terminals)
- **Young** (UCL, Slurm for GPUs): repo `/lustre/scratch/mmm0037/H3O` (= `~/work/H3O`), venv `~/grace-venv` (tensorpotential 0.6.1, py3.11), model `runs/grace/ft_m2` (`grace-ft:runs/grace/ft_m2`), `--mem=120G` needed by gracemaker data conversion. GPU partition `gpu` (QOS `freegpu`, 40 GPUs); `intergpu` is ONE node with 8 GPUs and another user can hold all of it for hours. Interactive: `srun -p intergpu --qos=intergpu -A allusers --gres=gpu:1 -c 8 --mem=64G --time=02:00:00 --pty bash`. The user's local checkout is on branch `proxyfit` tracking this branch (`git checkout -b proxyfit origin/claude/jolly-mccarthy-0zrnfe`).
- **Why the Young GPU queue is slow (diagnosed from `sprio`/`sshare`)**: FairShare is a RANK (about 0.0008 per rank, x 1,000,000 = priority points; AGE weight is 0, so waiting never helps; half-life 7 d, QOS weight 10000). User mmm0037: usage 13.9M, FairShare 0.0406, priority ~45.6k vs ~55-60k for the top pending jobs; ~1000 users with zero usage sit at 0.98. Last 7 days: 416 GPU-h on `gpu` + 84 GPU-h on `intergpu`. `zcicob0` ran 26-35 GPUs at once. Interactive sessions are charged for the whole wall time. Shorter `--time` helps backfill (a short job can start on the 3 GPUs that are free at times).
- **Perlmutter** (NERSC): user `victornr`, GPU account `m3349_g` (~178 h shown, none used); `~/work` is a symlink to `/pscratch/sd/v/victornr`; the `interactive` QOS gives a FULL node (4 x A100 80GB, 128 CPUs; `--gpus 1` is ignored) so use all four GPUs or use `-q shared` / `regular`; the interactive terminal lagged badly (seconds, dropped characters on long pastes, Ctrl-C ignored) while ping from the user's machine is 148 ms with 0% loss and the login node ping is 0.1 ms, so prefer batch jobs and short commands; ssh to Perlmutter later died entirely (NERSC had just updated software after CVEs; cause unknown: check sshproxy certificate, status page, ticket at help.nersc.gov). TensorFlow/tensorpotential not yet installed there.
- **ARCHER2** (CPU only): login `ln04`, account `e89-ed_p` (from the user's CASTEP job), `/work/e89/e89/victor` (= `~/work`; compute nodes see /work, not /home), partitions `standard` (qos `standard`; `short` qos with `--reservation=shortqos`), 128 cores per node, compute nodes have no internet (install the venv on the login node). Interactive: `srun --nodes=1 --exclusive --time=01:00:00 --account=e89-ed_p --partition=standard --qos=standard --pty /bin/bash` (queues like any job); for tests the `short` qos with `--reservation=shortqos` is the usual quick route (check the ARCHER2 docs).
- Data move: Young's `~/proxyfit_frames.tgz` (58 MB of cached frames) exists; the model tarball `~/grace_ft_m2.tgz` may need creating. `scp` from outside to Young may be blocked (the earlier "ssh_exchange_identification" errors were Young scp'ing to itself); fall back to the UCL gateway or via the laptop.

## 5. Review findings from another assistant (2026-10-31) and their status
| # | finding | checked in code | status |
|---|---|---|---|
| 1 | `collect.py` does not check each node's lambda / Hamiltonian | confirmed: lam from stage0 only | open: add per-node check, allow nested Kronrod extension |
| 2 | mixed-phase (fcc + bcc) fit, different pressure-correction bases | not verified (`report.py` builds solid points per phase vs its own host) | needs audit of actual fit inputs |
| 3 | collapsed runs re-enter analysis | confirmed: `summarize.py` writes collapsed rows to the main CSV (`full`), `insertion_model.py` does not filter | open: carry a validity column |
| 4 | pwidom cache ignores the model | confirmed | fixed (fingerprint) |
| 5 | all-in-one insertion normalisation | confirmed, ratio exactly 9 on the toy | fixed + test |
| 6 | pressure / state checks in pwidom | confirmed | fixed |
| 7 | correlated pure-H errors treated as independent, error caps, MBAR diagnostics | `tx_diagram.py` caps the slope error at 0.5 x slope; rest not checked | open |
None of this shows the reported GRACE numbers are wrong; the decisive step is an audit of the actual Young run metadata (lambda grid, dt, collapse flag, phase per fit point).

## 6. Caveats / things not verified
- The label step has never run with real GRACE; the CPU path and the Perlmutter / ARCHER2 scripts have never run on a scheduler. Fix typos in the first real run.
- Frames are assumed cubic (fcc / bcc / pure H); hcp / layered hosts would need the real cell.
- Old caches (made before the fingerprint) only warn. The fitted proxy is a pool average over compositions, P and T: look at the per-state table, fit ice and pure H separately.
- A pair proxy fitted to a pair truth is the easy case (toy). Many-body effects set the floor on GRACE.
- CPU GRACE benchmark: measured by the user at 8 slots (0.04x A100 pure H, 0.05x insertion); the 40-core numbers (0.15x / 0.2x, assumes ~75-80% scaling) are extrapolated; another chat was updating `docs/PERFORMANCE.md`.
- The user's `docs/data/pure_h_gr.csv` / `pure_h_runs.csv` on Young are untracked and unrelated.

## 7. Working conventions (from `HANDOFF.md`)
One code block per place to paste, with a line saying where it runs (login node / which GPU job); login-node and GPU-job commands never in one block; per-job blocks self-contained (`cd` to the repo, `git pull`); no `tail` on long runs; keep pasted commands short (long ones were corrupted on the Perlmutter interactive node: put them in a script and run `bash script`).

## 8. Suggested first message to a new assistant
"Read docs/HANDOFF_PROXYFIT.md on branch claude/jolly-mccarthy-0zrnfe of victornadenrobinson/H3O, then HANDOFF.md and docs/MU_METHODS.md. I want to run step 1-3 of section 3 (label the probe points with GRACE, fit the proxy, compare dg old vs new) on <Young GPU | ARCHER2 CPU>. Here is my current terminal output: ..."
