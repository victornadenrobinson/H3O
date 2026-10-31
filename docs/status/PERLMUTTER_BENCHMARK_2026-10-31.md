# Perlmutter throughput — user-reported 2026-10-31

Source: pasted terminal output from Perlmutter (`nid200497`). Command:
`bash mlp_pipeline/perlmutter_bench.sh --workflow pwidom --minutes 10`.
GRACE ft_m2, H2O, 500 GPa / 3000 K, vpa=2.08908 A^3/atom;
four A100 80-GB GPUs, 128 available CPU threads, two TF threads/process.
Five warm-up and 30 timed steps/calls; neighbor worker pool disabled; no MPS changes.
Two initial transcripts and a later continuation report three completed runs.
Actual summary.json/results.jsonl files remain on Perlmutter; not inspected here.

| Run under runs/ | MD workers/GPU | Whole-node MD steps/s | Probe batch | Whole-node structures/s |
| --- | ---: | ---: | ---: | ---: |
| bench_20261031_131111 | 4 | 79.50 | 16 | 34.06 |
| bench_20261031_131859 | 4 | 77.45 | 8 | 35.34 |
| bench_20261031_133031 | 8 | 79.98 | 4 | 34.99 |

All parsed reported results (66 calibration/screen/validation/whole-node records)
are in `docs/data/perlmutter_pwidom_benchmarks_20261031.csv`. Figure:
`docs/figures/perlmutter_pwidom_benchmarks_20261031.png`.

MD screening: 1/2/4/8 workers per GPU yielded approximately
5.8/11.5/21/20 aggregate steps/s per GPU. Four workers used about 9.8 GiB/GPU;
eight used 19.7 GiB without a material throughput gain. Four-worker whole-node validation
reached roughly 74–98% GPU utilization in reported timing samples.

Probe screening: batch 2/4/8/16 yielded about 8.5/8.7/9.0/9.0 structures/s per
GPU. Sampled memory was 4.5/8.5/24.5/40.5 GiB, respectively. The small ranking
changes between batches 4, 8 and 16 are consistent with a throughput plateau.
The third run screens batch 4 at 9.04 structures/s/GPU and validates it across
four GPUs at 34.99 structures/s total with only 8.5 GiB/GPU. Thus **batch 4** is
now the more memory-efficient provisional choice for this tested system: about
65% less memory than batch 8 and 79% less than batch 16, with similar throughput.
This supersedes the earlier provisional batch-8 recommendation. Batch 2 uses
4.5 GiB/GPU and is only about 6% slower than the best screened rate; concurrent
small-batch probe workers are a possible separate benchmark, not a measured win.

In the latest MD screen, four/eight workers gave 19.75/20.14 steps/s per GPU,
a 2% difference; earlier screens slightly favored four. Whole-node MD with
eight workers/GPU reached 79.98 steps/s, effectively the same range as the
earlier four-worker validations. Four workers/GPU remain the efficient starting
point: half the memory/context count, and roughly twice the speed per individual
trajectory compared with eight at similar aggregate throughput. The benchmark
selects the literal fastest short screen, so tiny fluctuations change its winner.

Interpretation: four independent MD workers/GPU are promising for replicas or
many compositions. This benchmark does not yet implement a production scheduler
that packs MD then reduces concurrency for the much larger probe stage.
Do not simply launch four complete production scans per GPU: H2.25O probes
previously reached 50.5 GiB for one worker. Other compositions and pure H still
need representative peak-memory measurements before adopting packing limits.
Short throughput timings do not establish long-trajectory stability or numerical
sampling convergence. Different layouts are screened on different GPUs while
sharing CPUs. These three reports show repeatability, not confidence intervals
or an exhaustive optimization. Defaults and scientific settings were not changed.

The intended next architecture is independent MD replicas feeding a controlled
probe queue. For one composition, pool equilibrated host and host+H snapshots
with matching T/V/model, independent seeds and no duplicate parent/extension
frames, then estimate from the grouped works. A fixed-size probe batch can
process arbitrarily many frames sequentially without VRAM scaling linearly with
the total frame count. CPU memory/storage do scale; current pwidom constructs all
probe structures in RAM, so large pooled datasets need streaming/chunked handling.
Different compositions can share the GPU queue but cannot be pooled into one
chemical-potential estimate. This pooling/scheduling workflow is not implemented.
