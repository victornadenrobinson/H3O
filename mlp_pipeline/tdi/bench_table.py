"""Markdown tables from a bench_tdi.sh result file (runs/bench/bench_<time>.jsonl) for docs/PERFORMANCE.md.  usage: python3 mlp_pipeline/tdi/bench_table.py runs/bench/bench_*.jsonl"""
import json, sys

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f) if l.strip()]
eng = [r for r in rows if r.get("mode") == "engine_sum"]
if eng:
    print("\nPer-lambda engine (separate processes on one GPU): total MD steps/s (ms per step of each process)\n")
    print("| system | atoms | processes | time-slicing | MPS | GPU memory (MiB, no MPS / MPS) |\n|---|---|---|---|---|---|")
    for key in dict.fromkeys((r["system"], r["nproc"]) for r in eng):
        x = {r["mps"]: r for r in eng if (r["system"], r["nproc"]) == key}
        cell = lambda m: f"{x[m]['total_steps_per_s']:.2f} ({x[m]['ms_per_step_each']:.0f} ms)" + ("" if x[m]["finished"] == key[1] else f" ONLY {x[m]['finished']} ran") if m in x else "-"
        n = next(iter(x.values()))["natoms"]
        print(f"| {key[0]} | {n} | {key[1]} | {cell(False)} | {cell(True)} | {x[False]['gpu_mem_MiB'] if False in x else '-'} / {x[True]['gpu_mem_MiB'] if True in x else '-'} |")
bat = [r for r in rows if r.get("mode") == "batch"]
if bat:
    print("\nBatched engine (all lambda values in one process, one model call per step)\n")
    print("| system | atoms | lambda values | total steps/s | ms per step | ms per model call | model share of wall time |\n|---|---|---|---|---|---|---|")
    for r in bat:
        print(f"| {r['system']} | {r['natoms']} | {r['nrep']} | {r['steps_per_s']:.2f} | {r['ms_per_step']:.0f} | {r['ms_per_call']:.0f} | {100 * r['model_frac']:.0f}% |")
cal = [r for r in rows if r.get("mode") == "calls"]
if cal:
    print("\nModel calls alone: B structures in one batched call vs B single TPCalculator calls (ms per structure)\n")
    print("| system | atoms | B | single | batched | speed-up | data preparation per batched call (ms) |\n|---|---|---|---|---|---|---|")
    for r in cal:
        print(f"| {r['system']} | {r['natoms']} | {r['B']} | {r['single_ms_per_structure']:.0f} | {r['batched_ms_per_structure']:.0f} | {r['speedup']:.2f} | {r['extract_ms_per_call']:.0f} |")
