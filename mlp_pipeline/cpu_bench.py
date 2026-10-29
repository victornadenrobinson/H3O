"""CPU throughput of the MLP for our TDI cells: how many MD steps per second does a whole CPU node give, compared with a GPU (A100: ~10.7 steps/s summed over
8 lambda processes of 251 atoms, docs/CURRENT_ISSUES.md 2c)? Runs P processes x T threads at the same time (each: model load, 3 warm-up calls, then --calls timed
energy + forces calls on its own copy of the cell) for every layout in --layouts, and prints s/call per process and the node total in calls (= MD steps) per second.
TDI lambda values are independent processes, so the node total is what a TDI job would get.
usage (on a CPU node, repo root):  ~/grace-venv/bin/python -u mlp_pipeline/cpu_bench.py [--model grace-ft:runs/grace/ft_m2] [--cell H2O|H|insert] [--layouts 1x128 8x16 16x8 32x4 64x2]
  e.g. Young:  srun -p <cpu partition> --qos=freecpu -N1 --exclusive -t 1:00:00 --pty bash    then the line above (--layouts up to the node's core count)
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def child(a):
    try:   # thread count before the model is built (TensorFlow reads it once)
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(a.threads); tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
    sys.path.insert(0, HERE)
    from importlib import import_module
    md = import_module("17_eos_md")
    at = md.build("H", 1.3, 5) if a.cell == "H" else md.build("fcc", 2.2, 3, a.seed, 2.0)   # pure H 250 atoms / fcc H2O 108 O = 324 atoms
    if a.cell == "insert": at = at + at[:1]; at.positions[-1] += 0.9                          # 325 atoms, like a host + one inserted H
    at.calc = import_module("04_benchmark").get_calc_spec(a.model, "cpu")
    for _ in range(3): at.positions += 0.001; at.get_forces()
    t = time.time()
    for _ in range(a.calls): at.positions += 0.001; at.get_potential_energy(); at.get_forces()
    print(json.dumps(dict(s_per_call=(time.time() - t) / a.calls, natoms=len(at))), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--cell", default="H2O", choices=["H2O", "H", "insert"])
    ap.add_argument("--layouts", nargs="+", default=["1x128", "8x16", "16x8", "32x4", "64x2"]); ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--child", action="store_true"); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    if a.child: return child(a)
    print(f"node: {os.cpu_count()} logical CPUs; model {a.model}; cell {a.cell}; {a.calls} timed calls per process (GPU reference: A100 ~10.7 steps/s for 8 x 251 atoms)", flush=True)
    for lay in a.layouts:
        P, T = map(int, lay.split("x"))
        if P * T > (os.cpu_count() or P * T): print(f"{lay}: skipped ({P * T} > {os.cpu_count()} CPUs)"); continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(T), MKL_NUM_THREADS=str(T), TF_NUM_INTRAOP_THREADS=str(T), TF_NUM_INTEROP_THREADS="1",
                   TF_CPP_MIN_LOG_LEVEL="2")
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, "-u", __file__, "--child", "--model", a.model, "--cell", a.cell, "--threads", str(T), "--calls", str(a.calls), "--seed", str(i)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env) for i in range(P)]
        res = []
        for p in ps:
            out, err = p.communicate()
            line = [l for l in out.splitlines() if l.startswith("{")]
            if line: res.append(json.loads(line[-1]))
            else: print(f"   process failed: {err.strip().splitlines()[-1] if err.strip() else 'no output'}")
        if not res: continue
        s = sum(r["s_per_call"] for r in res) / len(res)
        print(f"{lay:>6s}: {len(res)} processes x {T} threads, {res[0]['natoms']} atoms: {s:7.3f} s/call each -> node total {len(res) / s:6.2f} steps/s "
              f"({len(res) / s / 10.7:5.2f} x one A100 at ~8 x 251 atoms)   [{time.time() - t0:.0f} s incl. model loads]", flush=True)


if __name__ == "__main__":
    main()
