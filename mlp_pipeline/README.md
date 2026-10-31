# mlp_pipeline

Run order is encoded in `build_all.sh`. Typical flows:

```bash
# Perlmutter (data lives there): after mining runs/mined_h3o.csv
bash mlp_pipeline/build_all.sh 64 2>&1 | tee runs/build.log    # inventory -> s/m/l datasets -> checks -> splits -> plots -> bundle
bash mlp_pipeline/send_bundle.sh                               # upload to Young (needs ssh key id_young), checksum-verified

# Young: models (each in its own env; use that env's python)
<env>/bin/python mlp_pipeline/smoke_test.py --model mace --name medium-omat-0
<env>/bin/python mlp_pipeline/04_benchmark.py mace:medium-omat-0 valid_m.extxyz
sbatch mlp_pipeline/05_finetune_young.sbatch                   # TAG=m selects train_m/valid_m (placeholders to fill)
```
Conventions and data manifest: see ../PROJECT.md. Inputs given as bare filenames resolve to `runs/<name>` first.

## Perlmutter: four-GPU short Widom scan

Inside an **existing** four-GPU interactive allocation, activate the GRACE Python
environment, then run from the repository (or invoke the script by absolute path):

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --dry-run
bash mlp_pipeline/perlmutter_pwidom.sh
bash mlp_pipeline/perlmutter_pwidom_status.sh runs/<output-directory> --gpu
```

`PY=/path/to/python` selects an interpreter; otherwise the active `python` is used.
The launcher submits no job. Defaults: 500 GPa, 3000 K, H2O/H2.083O/H2.25O/pure-H,
one process per GPU, 0.4 ps sampling + 0.2 ps equilibration per MD, K=2000,
8 insertions and 4 deletions per frame, batch 8, a 10-minute wall cap.
The intermediate composition uses the existing per-O volume interpolation.
State/volume/bulk-modulus metadata come from `runs/tdi*/*_stage0.json`;
`--roots DIR ...` limits discovery. The exact choices are recorded in the manifest.
Model: `grace-ft:runs/grace/ft_m2` (one unambiguous SavedModel export required).
Proxies: `runs/pwidom_fit/proxy_ice.json` and `proxy_H.json`.
Override with `--model`, `--ice-proxy`, `--h-proxy`; inspect `--help` for run knobs.

Run `--dry-run` first: it checks inputs and constructs hosts without GPU inference
or output creation. A real run imports the required model packages and invokes
the existing pwidom evaluator (including its batched/single self-check).
Every run requires a fresh `--out` or creates a timestamped directory: existing
results are never overwritten or silently reused. `manifest.json` records the
commands, source paths, resolved model, Git revision, allocation, PIDs and exits;
each host has a log and the usual pwidom frames/result files. Status is read-only;
`--gpu` runs nvidia-smi on the machine where status is invoked. On the login node,
use an overlapping allocation step for GPU monitoring rather than querying login GPUs.

The deadline, Ctrl-C, or SIGTERM stops and reaps child process groups. Completed
results remain; interrupted MD is not checkpointed. Full result completion is
reported separately from scientific convergence. The status summary uses the
existing scan's robust pressure correction and also reports measured-P dg,
Widom/deletion/BAR, Kish, melting and sampled pure-H close contacts.
3000 K is outside the supplied proxy-fit label temperatures: treat this as an
extrapolation/sampling test. Missing physical inputs cause an error; no EOS,
bulk modulus, model or proxy is invented. No GRACE result is verified by local
toy checks. For CPU workflow tests, `--cpu --model toyice` is supported.

No previous TDI at the requested state? Supply `--state-json PATH` containing
`P`, `T` and three objects: `ice`, `h225`, `H`. Each object needs measured/fitted
`vpa` (A^3/atom), `B_GPa` (isothermal bulk modulus) and a `source` description.
The two crystal objects need `nrep` (normally 3: 108 O); H needs `natoms`
(normally 250). Both crystals must use the same nrep. This avoids creating fake
stage0 files. P/T must match the command. The intermediate host retains the
existing scan convention: per-O volume interpolation and the H2O bulk modulus.
The full supplied EOS metadata is copied into the run manifest. Do not reuse an
EOS from another temperature or invent bulk moduli. Existing DFT lookup functions
are `17_eos_md.dft_vpa` and `h_vpa`; `tdi/run_real.bulk_modulus` derives B from
the same EOS slope. Their data coverage/phase filters must be reviewed first.
To inspect the inventory without launching MD, using those existing fits and
requiring exact-temperature-bin coverage (no temperature extrapolation), run:

```bash
bash mlp_pipeline/perlmutter_pwidom.sh --inspect-eos runs/inventory_binned.csv
```

This prints volume/B estimates at P and P +/- 3%, or reports insufficient
coverage. It writes no files and needs no GPU. Check that the inventory path
exists on Perlmutter; the cloud checkout does not include it.
