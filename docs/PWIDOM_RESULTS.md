# Collect completed Perlmutter PWidom runs

Cloud does not currently have the Perlmutter filesystem mounted or an authenticated
SSH connection. Run the read-only collector on Perlmutter to scan the actual data.
It scans `runs/pwidom*` directories, including `pwidom_pm_*`, `pwidom_scan*`,
`pwidom_nscan*`, reference/extra-H stages and extensions. No trajectories,
checkpoints, models or DFT inventories are needed. Archived `_history` and
`_resume_history` snapshots are excluded to avoid repeating old records.

After updating the source checkout with the reviewed code:

```bash
cd /pscratch/sd/v/victornr/H3O
# Use the existing environment containing NumPy and matplotlib.
python mlp_pipeline/collect_pwidom_results.py \
  --root runs --out runs/plots/pwidom_summary_1 \
  --pressure-mode robust --include-flagged
```

Use a new output directory name for each export. This command cannot overwrite
existing reports and does not modify any input file or submit simulations.
`robust` matches the existing scan/dashboard partial-volume pressure convention.
`measured` reproduces the production JSON's `mu_P_meV` and its error. Both transfer
dG conventions are included in the numerical table when sufficient fields exist.
No undocumented pressure-correction switch is made.

Output:

- `summary.md`: readable table of every finished host and H reservoir, with
  chemical potential, transfer dG, statistical error, composition, run and flags.
- `finished_results.csv`: full precision, excess/fixed-V/pressure-corrected mu,
  dG in both available conventions, reservoir value/error, sample counts,
  source paths, launch status and model fingerprint.
- `all_*.png/.pdf`: all finished μ and transfer-dG points, including flagged
  points as open markers. Datasets are never pooled.
- State-specific `.png/.pdf` plus `formation_hull.csv`: the selected composition
  integral and its provisional lower convex envelope, where enough valid inputs
  exist, including an n=0 anchor. No missing compositions are manufactured.
- `results_bundle.json`: **small JSON-only export to upload to the cloud chat**.
  It contains the result records and manifests needed to reproduce the report,
  without trajectories, model weights, checkpoints or DFT inventories.
- `skipped.csv`, when needed: malformed/incomplete records and reasons.

A completed individual result is retained even if the four-host launch failed
later. Presence of a valid final result is distinct from launch completion and
from scientific convergence. Missing/ambiguous reservoirs are marked rather
than guessed across unrelated directories. Manifest pure-H snapshots are used
for extra-H stages; reservoir P/T/model compatibility is checked. Old files
without fingerprints are listed, but cannot enter the integrated hull.

## How selection, integration and uncertainty work

The report keeps all finished results. For each P/T/verified model/composition,
the integral uses the result with the largest minimum host/full frame count.
Ties prefer results named in the manifest, then deterministic source-path order.
This avoids pooling correlated parent/extension/proxy-replay records. It is an
explicit selection policy, not evidence that the longest run is converged.
Inspect `selected_for_integral` and the exact sources before drawing conclusions.

Default integration excludes all quality-flagged points. `--include-flagged`
permits low-Kish and forward/reverse disagreement points, displayed explicitly
as provisional; it still excludes melted, unverified and pressure-uncorrected
inputs. JSON-only checks cannot rule out pure-H collapse without trajectories.
No error bar here is a convergence certificate.

For H(2+n)O, the formation free energy relative to H2O plus a pure-H reservoir is
`integral_0^n transfer_dG(n) dn / (3+n)`. The code uses the existing trapezoidal
integration and finite-N reservoir correction. The convex hull is evaluated on
**atomic H fraction** `(2+n)/(3+n)`, with pure H at x=1 and formation energy zero.
It is restricted to the H2O+H composition domain and is not a complete H/O phase
hull or an independent absolute free-energy calculation.

Integration errors use the existing maximal-positive-correlation bound on
statistical sigma. This avoids assuming independent shared-reference or
parent/extension uncertainties. It does not quantify integration/model bias,
uncertain hull topology or statistical convergence.

## Replot in cloud after uploading the bundle

```bash
source /workspace/pwidom-env/bin/activate
cd /workspace/H3O
python mlp_pipeline/collect_pwidom_results.py \
  --from-bundle /path/to/results_bundle.json \
  --out /tmp/pwidom_results_review \
  --pressure-mode robust --include-flagged
```

An SSHFS/live mount would additionally require supported cloud mount capability,
network access and NERSC authentication; none is configured here. Transferring
this JSON-only export is sufficient for these figures and substantially smaller
than transferring simulation trajectories. It also avoids adding production
data to Git.
