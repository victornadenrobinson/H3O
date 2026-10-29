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
