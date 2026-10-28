"""Whole-job progress and ETA of a TDI run: all lambda values of the job, including those not started yet, from the <tag>_node<k>_progress.json files that
run_real.py rewrites every 30 s (steps done, total, recent speed) and, for lambda values without one, the saved series (<tag>_node<k>.npy) or nothing (0 steps done).
  ETA = remaining MD steps of every lambda value of the job / measured total throughput (steps/s summed over the lambda values that are running now).
The total throughput stays about constant while the GPU is shared (time-slicing / MPS / one batched process), so this is far more stable than the per-lambda ETAs.
usage:  python3 mlp_pipeline/tdi/eta.py runs/tdi <tag> [--only "0 3 5"] [--npar 8] [--per-lambda]       (numpy only; young_tdi_par.sh prints it every 30 s)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi.progress import fmt_eta

STALE_S = 180   # a progress file not rewritten for this long belongs to a process that is gone (killed, crashed, or the job ended)


def job_progress(d, tag, only=None):
    """list of dicts per lambda value of the job: k, done, total, rate (steps/s, nan unless running), state (running / done / failed / stopped / waiting)"""
    st = json.load(open(os.path.join(d, f"{tag}_stage0.json"))); A = st["args"]
    steps = lambda ps: int(round(ps * 1000 / A["dt"]))
    per = steps(A["chunk_ps"]); total = steps(A["equil_ps"]) + int(round(A["ps"] / A["chunk_ps"])) * per
    ks = only if only is not None else list(range(int(A["nodes"])))
    rows, now = [], time.time()
    for k in ks:
        r = dict(k=k, done=0, total=total, rate=float("nan"), state="waiting")
        f = os.path.join(d, f"{tag}_node{k}_progress.json")
        try:
            p = json.load(open(f))
            r.update(done=int(p["done"]), total=max(total, int(p.get("total", 0))), state=p.get("state", "running"))   # a running lambda knows its total; stage 0 may predate a longer --ps
            if r["state"] == "done" and r["done"] < r["total"]:   # finished under a shorter --ps: this job extends it
                r["state"] = "waiting"
            if r["state"] == "running":
                if now - p["t"] > STALE_S:
                    r["state"] = "stopped"
                else:
                    r["rate"] = float(p.get("rate", float("nan")))
        except (OSError, ValueError, KeyError):
            g = os.path.join(d, f"{tag}_node{k}.npy")   # no progress file (older run): finished chunks from the saved series
            try:
                n = len(np.load(g)) * int(A.get("every", 2))
                r["done"] = steps(A["equil_ps"]) + n; r["state"] = "done" if r["done"] >= total else "stopped"
            except (OSError, ValueError):
                pass
        r["done"] = min(r["done"], r["total"])
        rows.append(r)
    return rows


def summary(rows, npar=None):
    rem = sum(r["total"] - r["done"] for r in rows); tot = sum(r["total"] for r in rows)
    run = [r for r in rows if r["state"] == "running"]; rates = [r["rate"] for r in run if np.isfinite(r["rate"])]
    thr = float(np.mean(rates)) * len(run) if rates else float("nan")   # lambda values that just started have no speed yet: counted at the mean speed of the others
    # while lambda values still wait for a free slot, the running ones are replaced as they finish: the total throughput stays the same, so ETA = remaining / throughput
    eta = rem / thr if thr == thr and thr > 0 else float("nan")
    cnt = {s: sum(r["state"] == s for r in rows) for s in ("running", "waiting", "done", "stopped", "failed")}
    parts = [f"{cnt['running']} running", f"{cnt['waiting']} waiting", f"{cnt['done']} done"] + [f"{cnt[s]} {s}" for s in ("stopped", "failed") if cnt[s]]
    msg = (f"JOB {100 * (tot - rem) / max(tot, 1):.0f}% ({tot - rem} / {tot} MD steps, {len(rows)} lambda: {', '.join(parts)}) | "
           f"{thr:.2f} steps/s in total" + (f" ({1000 * len(rates) / thr:.0f} ms/step per running lambda)" if rates else "") + f" | {fmt_eta(eta) if rem else 'finished'}")
    if cnt["waiting"] and not cnt["running"]:
        msg += " | nothing running yet"
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tag"); ap.add_argument("--only", default=None, help="the lambda values of this job (ONLYNODES)")
    ap.add_argument("--npar", type=int, default=None); ap.add_argument("--per-lambda", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.dir, f"{a.tag}_stage0.json")):
        print(f"JOB {a.tag}: stage 0 not finished yet"); sys.exit(0)
    rows = job_progress(a.dir, a.tag, [int(x) for x in a.only.split()] if a.only else None)
    print(summary(rows, a.npar))
    if a.per_lambda:
        for r in rows:
            print(f"   lambda {r['k']:2d}: {r['state']:8s} {100 * r['done'] / max(r['total'], 1):3.0f}%" + (f"  {1000 / r['rate']:.0f} ms/step" if r["rate"] == r["rate"] and r["rate"] > 0 else ""))
