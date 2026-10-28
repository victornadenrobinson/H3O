"""Per-lambda status and a whole-job ETA for a running young_tdi_par.sh launch (plain python3, no numpy; called by the launcher's monitor loop).
usage: python3 mlp_pipeline/tdi/eta.py runs/tdi <tag> <nodes> <npar> ["<only these lambda indices>"]
Only log lines after the last '== launch' marker (written by the launcher into every node log) count, so a lambda waiting for a free slot shows 'queued',
not the 100% line of an earlier launch. Whole-job ETA: the running lambda values finish at their own ETAs; queued ones are assumed to need as long as a
running one needs in total; the speed per step changes with the number of processes sharing the GPU (measured on an A100, 251 atoms: 1 process 160,
2: 360, 8: 750 ms/step), so the tail of a run with fewer processes goes faster. An estimate (~), good to ~10-20%.
"""
import os, re, sys, time

MARK = re.compile(r"^(== launch|-- OOM retry)")
ETA = re.compile(r"(\d+) ms/step \| (\d+)% \| ETA (\S+) \(in (\d+) min\)")
FAIL = re.compile(r"Traceback|out of memory|ResourceExhausted|refusing to mix|UNSTABLE")
TAB = [(1, 0.16), (2, 0.36), (8, 0.75)]   # s/step vs processes on one GPU (CURRENT_ISSUES 2c)


def f(c):
    c = max(1, c)
    for (c0, t0), (c1, t1) in zip(TAB, TAB[1:]):
        if c <= c1: return t0 + (t1 - t0) * (c - c0) / (c1 - c0)
    (c0, t0), (c1, t1) = TAB[-2], TAB[-1]
    return t1 + (t1 - t0) * (c - c1) / (c1 - c0)


def status(log):
    """('queued'|'running'|'done'|'failed', ms/step, fraction of this launch's work, seconds left)"""
    if not os.path.exists(log): return "queued", None, 0.0, None
    lines = open(log, errors="replace").read().splitlines()
    i = max([j for j, s in enumerate(lines) if MARK.match(s)], default=-1)
    new = lines[i + 1:]
    if any("already complete" in s for s in new): return "done", None, 1.0, 0.0
    if any(FAIL.search(s) for s in new[-30:]): return "failed", None, 0.0, None
    m = [ETA.search(s) for s in new]; m = [x for x in m if x]
    if not m: return ("queued" if not new else "starting"), None, 0.0, None
    ms, pc, _, mins = m[-1].groups(); pc = int(pc)
    return ("done" if pc >= 100 else "running"), int(ms), pc / 100, 60.0 * int(mins)


def main():
    d, tag, nodes, npar = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ks = [int(x) for x in sys.argv[5].split()] if len(sys.argv) > 5 and sys.argv[5].strip() else range(nodes)
    st = {k: status(os.path.join(d, f"{tag}_node{k}.log")) for k in ks}
    for k, (s, ms, fr, left) in st.items():
        print(f"   lambda {k}: " + {"running": f"running   {ms} ms/step, {100 * fr:3.0f}% of this launch's steps, finishes in {left / 60:.0f} min" if s == "running" else "",
                                    "queued": "waiting   for a free GPU slot (starts when a running lambda finishes)", "starting": "starting  (loading the model / equilibrating, no speed yet)",
                                    "done": "finished", "failed": "FAILED    (see its log)"}[s])
    run = [v for v in st.values() if v[0] in ("running", "starting")]
    rr = [v for v in run if v[0] == "running" and v[2] > 0]
    nq = sum(v[0] == "queued" for v in st.values()); nf = sum(v[0] == "failed" for v in st.values())
    nd = sum(v[0] == "done" for v in st.values())
    head = f"   WHOLE JOB: {len(st)} lambda values: {len(run)} running, {nq} waiting, {nd} finished" + (f", {nf} FAILED (see their logs)" if nf else "")
    if not run and not nq:
        print(head + ": nothing left to run (summary follows)"); return
    if not rr:
        print(head + ": ETA once the running ones report progress"); return
    c = len(run); t = sum(v[1] for v in rr) / len(rr) / 1000   # s/step now at c processes
    full = sum(v[3] / (1 - v[2]) for v in rr if v[2] < 1) / max(1, sum(v[2] < 1 for v in rr))   # seconds a whole lambda takes at c processes
    steps = lambda sec: sec / t   # convert to steps at today's speed
    active = [steps(v[3]) if v[0] == "running" else steps(full) for v in run]; queue = [steps(full)] * nq; T = 0.0
    while active:   # event simulation: the per-step time follows the number of processes sharing the GPU
        tc = t * f(len(active)) / f(c); m = min(active); T += m * tc
        active = [a - m for a in active if a - m > 1e-9]
        while queue and len(active) < npar: active.append(queue.pop())
    eta = time.strftime("%H:%M", time.localtime(time.time() + T))
    print(head + f": all finished ~{eta} (in {T / 60:.0f} min" + (", incl. the waiting ones" if nq else "") + ")")


if __name__ == "__main__":
    main()
