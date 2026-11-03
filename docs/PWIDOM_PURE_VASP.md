# Same PWidom calculation with VASP instead of GRACE

The existing ASE Langevin dynamics, host/host+H ensembles, proxy proposal weights,
Widom/deletion/BAR estimators and pressure correction are unchanged. Model specs
`vasp:/absolute/config.json` select the VASP force engine through the shared ASE
calculator interface, so the target MD forces, stresses, and all trial energies
are DFT. The fitted pair potential only proposes trials; no GRACE prediction
enters the target Hamiltonian. The same calculator factory is available to TDI,
but a full crystal-TDI VASP run has not been integration-tested here.

## Pilot state and smoke test

500 GPa / 5000 K, 600 GPa / 5000 K and 400 GPa / 5000 K are useful initial
candidates from the supplied superionic-host hull/uptake comparisons. They are
not established DFT optimum-solubility points. Phase and pressure must be checked.
The 600 GPa / 4000 K candidate also showed strong uptake, but requires exact-4000-K
EOS coverage; do not silently substitute 5000 K volumes.

Run from the Perlmutter repository, inside the interactive CPU allocation:

```bash
git pull --ff-only origin feature/pwidom-checkpoint-resume
export GRACE_ENV="$SCRATCH/grace-env"
export VASP_MODULE=vasp/5.4.4-cpu
if ! command -v vasp_gam >/dev/null 2>&1; then module load "$VASP_MODULE"; fi
command -v vasp_gam

bash mlp_pipeline/perlmutter_pwidom.sh --cpu --P 500 --T 5000 \
  --inspect-eos runs/inventory_binned.csv --save-eos runs/dft_eos_500GPa_5000K.json
bash mlp_pipeline/perlmutter_vasp_check.sh dft-plan \
  --state-json runs/dft_eos_500GPa_5000K.json --template vasp_inputs \
  --ps 0.5 --equil-ps 1 --out runs/dft_pwidom_500GPa_5000K_1
bash mlp_pipeline/perlmutter_vasp_check.sh dft-worker \
  runs/dft_pwidom_500GPa_5000K_1 --index 0 --smoke
```

Inspect the EOS coverage and generated volumes/B values before the plan. Plan
preparation reads inputs and writes configurations, but performs no VASP/MD.
The smoke test DOES run VASP: 0.002 ps equilibration and 0.01 ps sampling in
both ensembles, with one trial/frame, isolated under H2O_rep0/smoke. This tiny
calculation tests the full chain and timing; it is not a scientific result.
Production settings and directories are separate. Repeat smoke with `--index 3`
for the pure-H worker if desired.

## Two-node, 24-hour production pilots

After smoke passes, submit the prepared 500-GPa state:

```bash
sbatch --account=m3349 --qos=regular \
  mlp_pipeline/perlmutter_dft_pwidom.sbatch runs/dft_pwidom_500GPa_5000K_1
```

Prepare the other state points (no automatic submission):

```bash
for P in 400 600; do
  bash mlp_pipeline/perlmutter_pwidom.sh --cpu --P "$P" --T 5000 \
    --inspect-eos runs/inventory_binned.csv --save-eos "runs/dft_eos_${P}GPa_5000K.json" || break
  bash mlp_pipeline/perlmutter_vasp_check.sh dft-plan \
    --state-json "runs/dft_eos_${P}GPa_5000K.json" --template vasp_inputs \
    --ps 0.5 --equil-ps 1 --out "runs/dft_pwidom_${P}GPa_5000K_1" || break
done
sbatch --account=m3349 --qos=regular mlp_pipeline/perlmutter_dft_pwidom.sbatch runs/dft_pwidom_400GPa_5000K_1
sbatch --account=m3349 --qos=regular mlp_pipeline/perlmutter_dft_pwidom.sbatch runs/dft_pwidom_600GPa_5000K_1
```

Each job reserves two entire CPU nodes and 256 tasks. Four independent 64-rank
workers cover H2O, H2.083O, H2.25O and pure H, with one seed each.
The older layout used eight 32-rank workers and two seeds; the final section
shows how to migrate existing plans. Each worker
runs its two MD ensembles sequentially, then trial evaluations. No worker spans
nodes; OMP/BLAS use one thread per process. VASP has no other concurrent GRACE
model consuming memory. `--mem=0` reserves available node memory. Each state/job
costs up to 48 node-hours. Three jobs together request up to 144 node-hours.

At 9 seconds per force call, 1 ps equilibration + 0.5 ps sampling in BOTH
ensembles at 0.5 fs is approximately 15 hours of MD/worker, before trial energies.
200 trial evaluations plus uncached host energies may take several more hours.
ASE-driven single points may cost more than native VASP MD; use smoke timings.
Completion within 24 hours is NOT guaranteed, and equilibration/convergence is
not established by finishing this budget. Sampling can be continued through
existing PWidom extension tooling after assessing pressure/phase/overlap.

## Output, restart and memory/disk behaviour

Each host/replica has worker.log, ordinary PWidom result/cache/checkpoint files,
and its isolated vasp_work directory. The engine prints per-call timing and
writes vasp_work/timings.jsonl (including warm-start status). Watch with:

```bash
tail -f runs/dft_pwidom_500GPa_5000K_1/H2O_rep0/worker.log
```

Re-submit the same sbatch command to continue missing work. Worker locks reject
concurrent use; completed results are skipped. MD saves at completed integration
steps every 300 wall seconds, including momenta and RNG. Persistent atomic
geometry caches retain completed energy/forces/stress even if the job dies
between a force call and the next MD checkpoint. Active VASP scratch files are
overwritten, not archived per integration step. Only the latest WAVECAR per
worker is kept. Cached NPZ results grow with the number of distinct geometries;
monitor scratch usage rather than retaining all OUTCARs/WAVECARs for all steps.

The same-composition/fixed-cell WAVECAR is reused; atom-count/cell changes force
a cold restart. Only exit-zero, electronically converged calculations with a
VASP timing footer are accepted. Forces are restored to the original ASE atom
order. Energy and free_energy both mean force-consistent TOTEN. Stress is the
ASE static tensile tensor, excluding kinetic stress (the MD pipeline adds its
kinetic term as before). Template/config fingerprints prevent mixing GRACE and
VASP trajectories or silently changing templates on resume. Keep the template
and executable stable throughout a campaign. Unset HHWALL for pure DFT.

The Γ-only configuration defines the sampled DFT Hamiltonian. Pure-H k-point and
finite-size convergence remain separate checks; do not call a Γ reservoir
converged from the previous two-pair test. The default DFT H cell is now 729 atoms (730 with the added H), at unchanged
EOS V/atom. This implements the requested roughly-700-atom Gamma size criterion;
convergence at every P/T still needs verification. The MLP launcher default is
unchanged. Existing 250-atom plans can be migrated with:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh dft-h-size \
  runs/dft_pwidom_500GPa_5000K_1 --natoms 729
```

This gives H workers fresh pureH729_rep0/rep1 directories and preserves all old
250-atom files. Running workers retain their original size until stopped.
`dft-stop-smokes PLAN` requests SIGTERM only for the current user’s smoke worker
processes for that exact plan; wait for their current force calls/checkpoints
before restarting. Generated engine INCARs have unique restart keys (no
conflicting LWAVE/ISTART entries).
No collected values alone establish equilibrium, phase stability, BAR overlap,
pressure-correction accuracy, or a converged convex hull.

## GRACE CPU timings, separately from VASP jobs

After pulling, run on idle CPU resources:

```bash
OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=1 \
  bash mlp_pipeline/perlmutter_vasp_check.sh timing runs/vasp_normal_40_1 \
  --model grace-ft:runs/grace/ft_m2 --device cpu --count 8 --repeats 3 \
  --out runs/grace_cpu_timings_1.json
```

Reports model initialization separately and per-geometry first-call/warm medians
for energy+forces+stress, resetting ASE's result cache between repeats. This is
static calculator timing, not a guarantee of MD throughput.

## Four systems × 64 ranks: 20-minute test and 48-hour jobs

Current defaults are one replica, 64 ranks per system, 729 H atoms. To reuse an
older eight-worker plan, stop its smoke workers and create a four-worker layout:

```bash
bash mlp_pipeline/perlmutter_vasp_check.sh dft-stop-smokes runs/dft_pwidom_500GPa_5000K_1
wait
bash mlp_pipeline/perlmutter_vasp_check.sh dft-h-size runs/dft_pwidom_500GPa_5000K_1 --natoms 729
bash mlp_pipeline/perlmutter_vasp_check.sh dft-layout runs/dft_pwidom_500GPa_5000K_1 \
  --ranks 64 --out runs/dft_pwidom_500GPa_5000K_4x64
bash mlp_pipeline/perlmutter_dft_pwidom_run.sh runs/dft_pwidom_500GPa_5000K_4x64 \
  --smoke --minutes 20
```

Requires an existing 256-task CPU allocation. The layout references the original
first-replica data directories and updates MPI layout in their configs; it
refuses changes while those workers hold simulation locks. Other replicas are
not launched. Smoke output goes to each worker directory's smoke64.log. The
runner sends checkpoint-stop requests at 18 minutes; at 20 minutes it terminates
unfinished calls, preserving prior atomic checkpoints/results. A terminated
partial SCF is not accepted. The full smoke estimator may remain unfinished.
Re-running continues it. Check for 729/730 H and warm=True; do not infer warm
speedup until measured. Production has separate MD/checkpoints from the smoke.

For production, from a login shell in the repo root:

```bash
sbatch --account=m3349 --qos=regular mlp_pipeline/perlmutter_dft_pwidom_48h.sbatch \
  runs/dft_pwidom_500GPa_5000K_4x64
```

The 48-hour script passes a 47-hour runner budget explicitly; it does not inherit
the old 23-hour limit. Each job requests two full nodes and 256 MPI tasks, at
most 96 node-hours. Existing 0.5 ps sampling / 1 ps equilibration budgets are
unchanged; 48 hours is a wall-time allowance, not a convergence guarantee.
Submit the same command to resume if unfinished. Do not run two jobs referencing
the same worker data directories concurrently.

Other candidate P/Ts, with exact-bin EOS checks and fresh four-worker plans:

```bash
for P in 400 600; do
  bash mlp_pipeline/perlmutter_pwidom.sh --cpu --P "$P" --T 5000 \
    --inspect-eos runs/inventory_binned.csv --save-eos "runs/dft_eos_${P}GPa_5000K_48h.json" || break
  bash mlp_pipeline/perlmutter_vasp_check.sh dft-plan \
    --state-json "runs/dft_eos_${P}GPa_5000K_48h.json" --template vasp_inputs \
    --replicates 1 --ranks 64 --h-natoms 729 \
    --out "runs/dft_pwidom_${P}GPa_5000K_4x64" || break
  sbatch --account=m3349 --qos=regular mlp_pipeline/perlmutter_dft_pwidom_48h.sbatch \
    "runs/dft_pwidom_${P}GPa_5000K_4x64" || break
done
```

Use after assessing the real-VASP smoke test. Missing EOS coverage stops plan
creation. Site QoS/account wall-time limits are enforced by sbatch; if your
account rejects 48 hours, request its permitted duration and use the matching
runner budget via DFT_MINUTES. No automatic submission is performed by helpers.
