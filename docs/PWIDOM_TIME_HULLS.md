# Cumulative chemical potentials and time-coloured hulls

Final result JSONs cannot recover a convergence history. The saved grouped work
arrays and host-pressure cache are required. The Perlmutter branch e90715a
already supplies `mlp_pipeline/tdi/pwidom_convergence.py`, which can reconstruct
prefix BAR/robust-pressure mu estimates without model calls or MD.

`export_pwidom_prefixes.py` wraps that published implementation into a JSON-only
export. It requires that module on the existing Perlmutter branch; it is not
bundled into the older audit branch. Example after making the export script
available in that checkout:

```bash
python scripts/export_pwidom_prefixes.py \
  runs/pwidom_300GPa_3000K_reference runs/pwidom_300GPa_3000K_extra \
  --out /tmp/pwidom_time_300GPa_3000K.json
```

The eight uploaded final results at this state report 50 selected host/full
frames each and record saved work-file paths. This is a suitable initial state;
old 500 GPa / 3000 K baseline/extension result JSONs do not record work paths and
may require cached-frame reanalysis if no corresponding work arrays exist.
Neither a final chemical potential nor its error is sufficient to infer history.

The exporter uses every available grouped-frame prefix after the first five
frames. It validates matching host/full frame IDs, model identity and final mu.
Legacy sampling cadence comes from the command with the published 20 fs default
when omitted. Caches and work arrays are not modified or copied into the export.
Only pressure arrays are read from the cache, not the large coordinate arrays.

Plot from the numerical export in cloud:

```bash
python mlp_pipeline/plot_pwidom_time_hulls.py /path/to/prefix_export.json \
  --out /tmp/pwidom_time_hulls --step-ps 0.1 --reference-mode cumulative
```

Outputs include μH(t), the pure-H G reference(t), transfer dG(t), formation
snapshots, time-coloured discrete hulls, CSVs and an input copy. No absolute
crystal G is calculated: formation ΔG is the composition integral of transfer
dG, relative to H2O + pure H. Hulls use atomic composition x_H=(2+n)/(3+n) and a
pure-H endpoint at x=1, ΔG=0. Their interpolation and equilibrium limitations are
unchanged from the final-state analysis.

`--reference-mode cumulative` uses the pure-H prefix at the same t as each host.
`fixed` holds the reference at its final estimate, as in the published convergence
script. Both options plot the actual cumulative reservoir curve so its evolution
is visible. The fixed-reference alternative is also retained in the time CSV.

Every hull snapshot uses the SAME host compositions, model and shared reservoir.
Only exact common times are used; missing snapshots are recorded and skipped
rather than filled with later/final values. With a 20 fs cadence, prefixes every
0.1 ps correspond to 5, 10, ..., 50 selected frames for the 1-ps example. Each
point uses all production samples up to t, not an independent 0.1-ps block.

Time means cumulative production sampled per ensemble. Host and N+1 runs are
independent; elapsed sampling time is not wall time or a synchronized physical
trajectory. Equilibration and extension re-equilibration gaps are excluded. Do
not concatenate parent and extension data as independent trajectories. No error
bands are inferred from individual correlated insertion/deletion trials. Saved
fingerprints/flags do not establish statistical convergence. Melted/other-phase
inputs are refused explicitly.

Validation: three synthetic plotting tests cover time-matched/fixed reservoirs,
composition normalization, missing exact-time refusal, output preservation and
final-value/model checks. Eleven audit tests also pass, including a corrected
reverse exponential diagnostic sign; BAR itself was unchanged. The exporter was
separately checked against the published e90715a prefix implementation using
synthetic grouped works with unequal insertion/deletion trial counts. No real
work arrays or Perlmutter prefix export were available in cloud during this step.
