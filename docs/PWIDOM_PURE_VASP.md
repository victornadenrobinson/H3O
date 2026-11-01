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

Each job reserves two entire CPU nodes and 256 tasks. Eight independent 32-rank
workers cover H2O, H2.083O, H2.25O and pure H, with two seeds each. Each worker
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
converged from the previous two-pair test. The default H cell is 250 atoms,
matching the prior workflow, rather than the larger Γ-converged cell discussed.
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
