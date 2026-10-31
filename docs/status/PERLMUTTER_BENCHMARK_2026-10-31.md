# Perlmutter throughput — user-reported 2026-10-31

Source: pasted terminal output from Perlmutter (`nid200497`). Command:
`bash mlp_pipeline/perlmutter_bench.sh --workflow pwidom --minutes 10`.
GRACE ft_m2, H2O, 500 GPa / 3000 K, vpa=2.08908 A^3/atom;
four A100 80-GB GPUs, 128 available CPU threads, two TF threads/process.
Five warm-up and 30 timed steps/calls; neighbor worker pool disabled; no MPS changes.
The transcript reports two completed runs and the start of a third.
Actual summary.json/results.jsonl files remain on Perlmutter; not inspected here.

| Run under runs/ | MD workers/GPU | Whole-node MD steps/s | Probe batch | Whole-node structures/s |
| --- | ---: | ---: | ---: | ---: |
| bench_20261031_131111 | 4 | 79.50 | 16 | 34.06 |
| bench_20261031_131859 | 4 | 77.45 | 8 | 35.34 |

MD screening: 1/2/4/8 workers per GPU yielded approximately
5.8/11.5/21/20 aggregate steps/s per GPU. Four workers used about 9.8 GiB/GPU;
eight used 19.7 GiB without a throughput gain. Four-worker whole-node validation
reached roughly 74–98% GPU utilization in reported timing samples.

Probe screening: batch 2/4/8/16 yielded about 8.5/8.7/9.0/9.0 structures/s per
GPU. Sampled memory was 4.5/8.5/24.5/40.5 GiB, respectively. The small ranking
change between batches 8 and 16 is consistent with a throughput plateau;
batch 8 is the more memory-efficient provisional choice for this tested system.

Interpretation: four independent MD workers/GPU are promising for replicas or
many compositions. This benchmark does not yet implement a production scheduler
that packs MD then reduces concurrency for the much larger probe stage.
Do not simply launch four complete production scans per GPU: H2.25O probes
previously reached 50.5 GiB for one worker. Other compositions and pure H still
need representative peak-memory measurements before adopting packing limits.
Short throughput timings do not establish long-trajectory stability or numerical
sampling convergence. The third run, bench_20261031_133031, had only started in
the supplied transcript; its outcome is unknown.
