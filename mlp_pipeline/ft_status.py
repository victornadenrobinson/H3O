#!/usr/bin/env python3
"""Status of every fine-tune in runs/grace/ft_*: finished / running / failed, recipe, last logged train(test) errors, and the reason a crashed run stopped. Login node, no GPU. Python 3.6 safe.
   python3 mlp_pipeline/ft_status.py [runs/grace]"""
import glob, os, re, subprocess, sys, time
R = sys.argv[1] if len(sys.argv) > 1 else "runs/grace"
try: q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T %M"], stdout=subprocess.PIPE, universal_newlines=True).stdout.splitlines()
except Exception: q = []
jobs = {l.split()[0]: l.split()[1:] for l in q if l.strip()}
def last(pat, lines):
    for l in reversed(lines):
        m = re.search(pat, l)
        if m: return m
def num(txt, key):
    m = re.search(key + r":\s*([0-9.eE+-]+)\s*\(([0-9.eE+-]+)\)", txt)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
print(f"{'run':24s}{'state':9s}{'recipe':58s}{'last log point':20s}{'E rmse tr/te meV/at':>21s}{'F rmse tr/te eV/A':>20s}  note")
for d in sorted(glob.glob(R + "/ft_*")):
    name = os.path.basename(d)[3:]; seed = d + "/seed/1"
    rec = ""
    try: rec = open(d + "/recipe.txt").read().strip().replace("fine-tune " + name + ": ", "")
    except Exception: pass
    rec = re.sub(r"\s+gpu .*", "", rec).replace("train ", "").replace("valid ", "v").replace("updates ", "u").replace("weights ", "w")
    done = os.path.exists(seed + "/final_model/saved_model.pb"); lg = []
    try: lg = open(seed + "/log.txt", errors="replace").read().splitlines()
    except Exception: pass
    tl = [l for l in lg if "TRAIN(TEST)" in l]; note = ""; lastpt = ""; e = f = (None, None)
    if tl:
        m = re.search(r"I - (.*?)\s+TRAIN\(TEST\):(.*)", tl[-1]); lastpt = m.group(1).strip()[:19]; e = num(m.group(2), r"rmse/depa"); f = num(m.group(2), r"rmse/f_comp")
    jn = "ft_" + name; running = jn in jobs and jobs[jn][0] == "RUNNING"; pend = jn in jobs and jobs[jn][0] == "PENDING"
    age = ""
    try: age = (time.time() - os.path.getmtime(seed + "/log.txt")) / 60.0
    except Exception: pass
    if done: st = "DONE"
    elif running: st = "RUN"; note = "log %.0f min ago" % age if age != "" else ""
    elif pend: st = "PEND"
    elif name in jobs or (age != "" and age < 10): st = "RUN?"; note = "(job name differs; log %.0f min ago)" % age
    else:
        st = "FAILED" if lg else "NOSTART"
        try:
            so = open(d + "/stdout.txt", errors="replace").read().splitlines()
            err = last(r"(Error|error|EOFError|Out of memory|OOM|Killed|Traceback|RuntimeError).*", so); note = (err.group(0)[:70] if err else "")
        except Exception: pass
    fm = lambda x, s: "   -   " if x is None else "%.1f" % (x * s)
    print(f"{name:24s}{st:9s}{rec[:56]:58s}{lastpt:20s}{fm(e[0], 1000) + '/' + fm(e[1], 1000):>21s}{fm(f[0], 1) + '/' + fm(f[1], 1):>20s}  {note}")
