# Cluster result snapshots

These files preserve supplied terminal summaries for future chats and analysis.
They are dated snapshots, not a live view of a cluster or proof that the run
directories/model exports are available in this checkout. Refresh them after
important runs finish or analysis/correction code changes. Keep provenance and
prior snapshots when updating; do not silently combine Young and Perlmutter data.

| Snapshot | Source | What is known |
| --- | --- | --- |
| [Young TDI summary received 2026-10-31](young_tdi_summary_received_2026-10-31.txt) | UCL Young, supplied by the user | Full pasted `summarize.py` output. Actual execution time, node and source Git revision were not supplied. |
| [Perlmutter 500 GPa / 3000 K reported results](../data/pwidom_pm_500GPa_3000K_reported.csv) | NERSC Perlmutter, allocation 60620660 | Three H/O host values from `runs/pwidom_pm_20261031_105811`; pure H was the reference. All ice-host estimates unconverged. |

## Refresh on the source cluster

On **Young**, use the working GRACE environment and checkout on that cluster,
not the Perlmutter environment path. The supplied command was:

```bash
~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py
```

Capture the complete output in a new dated text file, recording the cluster,
hostname, execution time, Git revision and command. Explicit run-root arguments
are preferable when comparing campaigns; otherwise record the defaults used.
For reusable numerical tables, `summarize.py ... -o <fresh-prefix>.csv` also
writes the corresponding difference/mixing/hull CSVs. Use a fresh prefix and
check that all intended run directories were included. Keep raw structures,
POTCARs, model exports and large trajectories outside Git.

On **Perlmutter**, use the environment discovery in the Perlmutter shell wrappers
and record the exact scan output directory. The status command is:

```bash
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/<scan-directory>
```

## Reading the Young snapshot

- `G_meV_atom` is per atom for whole-system TDI, but insertion rows are per
  **added H**. Crystal mixing costs are averages; Widom/insertion chemical
  potentials need the appropriate differential/average comparison.
- Preserve `status`, drift, quadrature difference, time step, lambda grid,
  pressure offsets/corrections, close-contact and melting notes. A `done` row
  alone does not establish scientific convergence or the intended phase.
- `dG_P2` / `P2_corr_per_H` are explicitly **not applied** in this output.
  Missing values do not mean zero correction. Older insertion rows can lack
  their pressure correction; do not mix corrected and uncorrected fits.
- The snapshot includes 300-GPa pure-H runs whose measured pressures are
  substantially higher than the target and rows marked `DIFFERENT lambda grid`.
  Retain these caveats when selecting reference energies.
- The copied report's uncertainty is statistical; GRACE model error, finite
  size and nuclear quantum effects require separate checks.
