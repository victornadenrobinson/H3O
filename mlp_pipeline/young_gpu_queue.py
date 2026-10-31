#!/usr/bin/env python3
"""Young GPU queue (Slurm) in one screen, the twin of young_queue.py (SGE CPU nodes): GPU pools (nodes usable / draining / down, GPUs free), queue pressure per partition and QOS (running, pending, pending
reasons, median wait of pending jobs) and the REAL waits of jobs that started in the last days (sacct: start - submit), top users, your jobs with their priority rank inside the partition and the
scheduler's start estimate, and your QOS limits against what you hold. Read-only Slurm commands:
  scontrol show node -o | squeue -h -o '<fixed format>' | sacctmgr -n -P show qos | sacct -a -X -n -P -S <days ago>
  mlp_pipeline/gq                (wrapper, alias:  alias gq='bash -l /lustre/scratch/mmm0037/H3O/mlp_pipeline/gq')
  python mlp_pipeline/young_gpu_queue.py [--watch 30] [--no-users] [--json] [--days 2] [--save-dir DIR] [--from-dir DIR --now "2026-10-30 19:00:00"]
Pending array jobs are one squeue row ('117526_[0-4]%2'): the tasks are counted from the bracket (the %N is a throttle, shown as the reason JobArrayTaskLimit). A GPU job's GPUs = (gres gpu count per node) x nodes
x tasks. Pending jobs with no scheduler start estimate (squeue %S = N/A) simply have not been looked at by backfill yet. --from-dir reads scontrol_nodes.txt / squeue.txt / qos.txt / sacct.txt."""
import argparse, datetime, getpass, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from young_queue import HAVE_RICH, bar, hms, render_plain, render_rich, short_mem   # shared look
if HAVE_RICH:
    from rich.console import Console

SQ_FMT = "%i|%P|%u|%T|%M|%D|%C|%b|%r|%S|%V|%Q|%l|%q|%j".replace("%i", "%.40i")
FILES = dict(nodes="scontrol_nodes.txt", queue="squeue.txt", qos="qos.txt", sacct="sacct.txt")
BADSTATE = ("DOWN", "DRAIN", "FAIL", "FUTURE", "POWER_DOWN", "NOT_RESPONDING", "MAINT", "RESERVED")


def sh(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=timeout).stdout
    except Exception:
        return ""


def collect(a, now):
    d0 = (now - datetime.timedelta(days=a.days)).strftime("%Y-%m-%d"); cols = "JobID,User,Partition,QOS,State,Submit,Start,ElapsedRaw,AllocTRES"
    cmds = dict(nodes="scontrol show node -o", queue="squeue -h -o '%s'" % SQ_FMT, qos="sacctmgr -n -P show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxWall,Priority 2>/dev/null",
                sacct="sacct -a -X -n -P -S %s -o %s 2>/dev/null || sacct -X -n -P -S %s -o %s" % (d0, cols, d0, cols))
    raw = {}
    for k in FILES:
        if a.from_dir:
            p = os.path.join(a.from_dir, FILES[k]); raw[k] = open(p).read() if os.path.exists(p) else ""
        else:
            raw[k] = sh(cmds[k])
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        for k in FILES: open(os.path.join(a.save_dir, FILES[k]), "w").write(raw[k])
    return raw


# ------------------------------------------------------------------ parsers
def gres_count(s):
    """GPUs in a TRES / gres string: 'gres/gpu=4', 'gpu:a100:4', 'gres:gpu:2', 'gpu:1', 'N/A' -> 0"""
    m = re.findall(r"gpu(?::[A-Za-z0-9_.-]+)?[:=](\d+)", s or "")
    return int(m[-1]) if m else 0


def parse_nodes(text):
    nodes = []
    for line in text.splitlines():
        if not line.startswith("NodeName="): continue
        head = line.split(" Reason=")[0]; kv = dict(re.findall(r"(\w+)=(\S+)", head)); st = kv.get("State", "UNKNOWN").upper()
        gres = kv.get("Gres", "(null)"); cfg = kv.get("CfgTRES", ""); alloc = kv.get("AllocTRES", "")
        m = re.search(r"gres/gpu=(\d+)", cfg); ngpu = int(m.group(1)) if m else gres_count(gres)
        m = re.search(r"gres/gpu=(\d+)", alloc); used = int(m.group(1)) if m else 0
        gtype = (gres.split(":")[1] if gres.count(":") >= 2 else "gpu") if ngpu else "-"
        bad = any(b in st for b in BADSTATE) or st.rstrip("*~#$@") in ("UNKNOWN",)
        nodes.append(dict(name=kv["NodeName"], parts=(kv.get("Partitions") or "").split(","), state=st.rstrip("*~#$@"), gpus=ngpu, used=min(used, ngpu), gtype=gtype, cpus=int(kv.get("CPUTot", 0)), cpualloc=int(kv.get("CPUAlloc", 0)),
                          usable=not bad, draining="DRAIN" in st and "DRAINED" not in st, reason=(line.split(" Reason=")[1].split(" [")[0] if " Reason=" in line else "")))
    return nodes


def ntasks(jobid):
    """tasks of a pending array row: 117526_[0-4]%2 -> 5; 115857_[5-36%6] -> 32; 117525_[0-1,7-8,14-15] -> 6; a plain id -> 1"""
    m = re.match(r"^\d+_\[([^\]]*)\]?", jobid)
    if not m: return 1
    n = 0
    for tok in m.group(1).split("%")[0].split(","):
        r = re.match(r"^(\d+)-(\d+)(?::(\d+))?$", tok)
        n += (int(r.group(2)) - int(r.group(1))) // int(r.group(3) or 1) + 1 if r else (1 if tok.isdigit() else 0)
    return max(n, 1)


def to_dt(s):
    try: return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except Exception: return None


def dur(s):
    """squeue %M / %l: '3:26:20', '1-02:03:04', '45:10', 'UNLIMITED', 'INVALID' -> seconds"""
    m = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", s or "")
    return ((int(m.group(1) or 0) * 24 + int(m.group(2) or 0)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))) if m else None


def parse_queue(text, now):
    jobs = []
    for line in text.splitlines():
        t = line.rstrip("\n").split("|", 14)
        if len(t) < 15 or not re.match(r"^\d+", t[0].strip()): continue
        jid, part, user, state, used, nn, cpus, tres, reason, start, sub, prio, limit, qos, name = [x.strip() for x in t]
        nt = ntasks(jid); gpn = gres_count(tres)
        jobs.append(dict(id=jid, part=part, user=user, state=state, used=dur(used) or 0, nodes=int(nn) if nn.isdigit() else 1, cpus=int(cpus) if cpus.isdigit() else 0, gpn=gpn, ntask=nt,
                         gpus=gpn * (int(nn) if nn.isdigit() else 1) * nt, reason=reason, start=to_dt(start), submit=to_dt(sub), prio=int(prio) if prio.isdigit() else 0, limit=dur(limit), qos=qos, name=name,
                         running=state in ("RUNNING", "COMPLETING"), age=(now - to_dt(sub)).total_seconds() if to_dt(sub) else 0))
    return jobs


def parse_qos(text):
    out = {}
    for line in text.splitlines():
        t = line.split("|")
        if len(t) >= 6 and t[0]: out[t[0]] = dict(maxtres=t[1], maxjobs=t[2], maxsubmit=t[3], maxwall=t[4], prio=t[5], gpu_cap=gres_count(t[1]) or None)
    return out


def parse_sacct(text, now):
    rows = []
    for line in text.splitlines():
        t = line.split("|")
        if len(t) < 9: continue
        sub, st = to_dt(t[5]), to_dt(t[6])
        if sub and st and st >= sub: rows.append(dict(id=t[0], user=t[1], part=t[2], qos=t[3], state=t[4].split()[0], wait=(st - sub).total_seconds(), start=st, elapsed=int(t[7] or 0), gpus=gres_count(t[8])))
    return rows


# ------------------------------------------------------------------ summary
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else None


def summarize(nodes, jobs, qos, hist, me, now):
    S = dict(me=me); gp = defaultdict(list)
    for n in nodes:
        if n["gpus"]: gp[(n["parts"][0] if n["parts"] and n["parts"][0] else "-", n["gtype"])].append(n)
    S["pools"] = []
    for (part, gt), ns in sorted(gp.items()):
        ok = [n for n in ns if n["usable"]]
        S["pools"].append(dict(part=part, gtype=gt, nodes=len(ns), usable=len(ok), draining=sum(1 for n in ns if n["draining"]), down=sum(1 for n in ns if not n["usable"] and not n["draining"]), gpus=sum(n["gpus"] for n in ok),
                               used=sum(n["used"] for n in ok), free=sum(n["gpus"] - n["used"] for n in ok), off=sum(n["gpus"] for n in ns if not n["usable"]), free_nodes=sum(1 for n in ok if n["used"] == 0)))
    S["gpus"] = dict(total=sum(p["gpus"] for p in S["pools"]), used=sum(p["used"] for p in S["pools"]), free=sum(p["free"] for p in S["pools"]), off=sum(p["off"] for p in S["pools"]))
    run = [j for j in jobs if j["running"]]; pend = [j for j in jobs if j["state"] == "PENDING"]
    S["run"] = dict(jobs=len(run), gpus=sum(j["gpus"] for j in run)); S["pend"] = dict(jobs=sum(j["ntask"] for j in pend), gpus=sum(j["gpus"] for j in pend))
    grp = defaultdict(list)
    for j in jobs: grp[(j["part"], j["qos"])].append(j)
    S["groups"] = []
    for (part, q), js in sorted(grp.items()):
        r = [j for j in js if j["running"]]; p = [j for j in js if j["state"] == "PENDING"]; reasons = Counter()
        for j in p: reasons[j["reason"] or "?"] += j["ntask"]
        w = sorted(j["age"] for j in p); rec = [h["wait"] for h in hist if h["part"] == part and h["qos"] == q and (now - h["start"]).total_seconds() <= 86400 * 2]
        S["groups"].append(dict(part=part, qos=q, run=len(r), run_gpus=sum(j["gpus"] for j in r), pend=sum(j["ntask"] for j in p), pend_gpus=sum(j["gpus"] for j in p), med_pend=pct(w, 0.5), oldest=w[-1] if w else None,
                                reasons=reasons.most_common(3), started=len(rec), wait_med=pct(rec, 0.5), wait_p90=pct(rec, 0.9), cap=(qos.get(q) or {}).get("gpu_cap")))
    users = defaultdict(lambda: dict(run=0, gpus=0, pend=0, pgpus=0))
    for j in jobs:
        u = users[j["user"]]
        if j["running"]: u["run"] += 1; u["gpus"] += j["gpus"]
        elif j["state"] == "PENDING": u["pend"] += j["ntask"]; u["pgpus"] += j["gpus"]
    S["users"] = sorted(users.items(), key=lambda kv: (-kv[1]["gpus"], -kv[1]["pgpus"]))
    S["mine"] = sorted([j for j in jobs if j["user"] == me], key=lambda j: (not j["running"], j["submit"] or now))
    for j in S["mine"]:
        if j["state"] == "PENDING":
            ahead = [k for k in pend if k is not j and k["part"] == j["part"] and (k["prio"] > j["prio"] or (k["prio"] == j["prio"] and (k["submit"] or now) <= (j["submit"] or now)))]
            j["rank"] = len(ahead) + 1; j["gpus_ahead"] = sum(k["gpus"] for k in ahead); j["of"] = sum(1 for k in pend if k["part"] == j["part"])
    S["qos"] = qos; mine_run = defaultdict(int); mine_pend = defaultdict(int)
    for j in S["mine"]:
        (mine_run if j["running"] else mine_pend)[j["qos"]] += j["gpus"] if j["running"] else j["ntask"]
    S["mine_by_qos"] = {q: dict(run_gpus=mine_run[q], pend=mine_pend[q], **(qos.get(q) or {})) for q in set(list(mine_run) + list(mine_pend))}
    S["nodes_n"] = dict(total=len(nodes), gpu=sum(1 for n in nodes if n["gpus"]), usable=sum(1 for n in nodes if n["gpus"] and n["usable"]))
    return S


def sections(S, a, now):
    out = []; g = S["gpus"]; nn = S["nodes_n"]; fr = g["free"]
    ov = [["GPU nodes", f"{nn['gpu']} nodes with GPUs: [green]{nn['usable']} usable[/], [red]{nn['gpu'] - nn['usable']} drained / down[/] ({g['off']} GPUs out of service)"],
          ["GPUs (usable nodes)", f"[bold]{g['used']}[/] in use / {g['total']}  {bar(g['used'] / g['total'] if g['total'] else 0, 24)}   [bold]{fr} free[/]" + ("" if fr else "  [red](all busy)[/]")],
          ["Jobs", f"{S['run']['jobs']} running ({S['run']['gpus']} GPUs), {S['pend']['jobs']} pending ({S['pend']['gpus']} GPUs wanted; array tasks counted)"],
          ["Your jobs", f"{sum(1 for j in S['mine'] if j['running'])} running, {sum(j['ntask'] for j in S['mine'] if j['state'] == 'PENDING')} pending"]]
    out.append(("Overview", None, ov))
    out.append(("GPU pools (partition = first partition of the node; free GPUs only on usable nodes)", ["partition", "GPU", "nodes", "usable", "drain", "down", "GPUs", "in use", "free", "free nodes", "use"],
                [[p["part"], p["gtype"], str(p["nodes"]), str(p["usable"]), f"[yellow]{p['draining']}[/]", f"[red]{p['down']}[/]", str(p["gpus"]), str(p["used"]), f"[green]{p['free']}[/]", str(p["free_nodes"]),
                  bar(p["used"] / p["gpus"] if p["gpus"] else 0)] for p in S["pools"]] or [["[dim]no GPU nodes found[/]"] + [""] * 10]))
    rows = []
    for x in S["groups"]:
        rs = ", ".join("%s %d" % (k, v) for k, v in x["reasons"]) or "-"
        rows.append([x["part"], x["qos"], str(x["run"]), str(x["run_gpus"]), (str(x["cap"]) if x["cap"] else "-"), str(x["pend"]), str(x["pend_gpus"]), hms(x["med_pend"]) if x["med_pend"] is not None else "-",
                     "%d" % x["started"], (hms(x["wait_med"]) + " / " + hms(x["wait_p90"])) if x["wait_med"] is not None else "-", rs])
    out.append((f"Queue by partition / QOS (cap/user = QOS GPU cap per user; started / wait = jobs that started in the last {min(a.days, 2)} days: sacct start - submit)",
                ["partition", "qos", "running", "run GPUs", "cap/user", "pending", "pend GPUs", "median pend", "started", "wait med / p90", "top pending reasons"], rows))
    if not a.no_users:
        out.append(("Top users by running GPUs", ["user", "running jobs", "GPUs", "pending jobs", "pending GPUs"],
                    [[("[bold cyan]" + u + "[/]" if u == S["me"] else u), str(v["run"]), str(v["gpus"]), str(v["pend"]), str(v["pgpus"])] for u, v in S["users"][:a.top]]))
    mine = []
    for j in S["mine"]:
        sc = {"RUNNING": "green", "PENDING": "yellow"}.get(j["state"], "white"); info = ""
        if j["running"]: info = "limit " + (hms(j["limit"]) if j["limit"] else "?")
        else:
            info = "#%d/%d, %d GPUs ahead" % (j["rank"], j["of"], j["gpus_ahead"]) + (", est. start " + j["start"].strftime("%d/%m %H:%M") if j["start"] else "")
        mine.append([j["id"], j["name"][:12], f"[{sc}]{j['state'][:2]}[/]", j["part"], j["qos"], str(j["gpus"]), (hms(j["used"]) if j["running"] else "waiting " + hms(j["age"])), j["reason"][:20] if not j["running"] else "-", info])
    out.append(("Your jobs", ["job", "name", "st", "partition", "qos", "GPUs", "so far", "reason", "rank / estimate"], mine or [["[dim]none[/]"] + [""] * 8]))
    q = []
    for name, v in sorted(S["mine_by_qos"].items()):
        cap = v.get("gpu_cap"); q.append([name, str(v["run_gpus"]), str(v["pend"]), v.get("maxtres") or "-", v.get("maxjobs") or "-", v.get("maxsubmit") or "-", v.get("maxwall") or "-",
                                          "[red]at cap[/]" if cap and v["run_gpus"] >= cap else ""])
    if q: out.append(("Your QOS limits (sacctmgr)", ["qos", "your GPUs", "your pending", "MaxTRES/user", "MaxJobs/user", "MaxSubmit/user", "MaxWall", ""], q))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--watch", type=float, default=0); ap.add_argument("--no-users", action="store_true"); ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--json", action="store_true"); ap.add_argument("--days", type=int, default=2, help="history window for real waits (sacct)"); ap.add_argument("--save-dir", default=None)
    ap.add_argument("--from-dir", default=None); ap.add_argument("--now", default=None); ap.add_argument("--plain", action="store_true"); a = ap.parse_args()
    console = Console(width=max(140, Console().width)) if HAVE_RICH and not a.plain else None; host = os.uname()[1] if hasattr(os, "uname") else "?"
    while True:
        now = datetime.datetime.strptime(a.now, "%Y-%m-%d %H:%M:%S") if a.now else datetime.datetime.now(); raw = collect(a, now)
        if not raw["nodes"] and not raw["queue"]: print("no output from scontrol / squeue: not on the Young GPU (Slurm) login node; use --from-dir for a saved snapshot"); return 1
        S = summarize(parse_nodes(raw["nodes"]), parse_queue(raw["queue"], now), parse_qos(raw["qos"]), parse_sacct(raw["sacct"], now), os.environ.get("YQ_USER") or getpass.getuser(), now)
        if a.json: print(json.dumps({k: v for k, v in S.items() if k not in ("mine", "users")}, default=str, indent=1)); return 0
        secs = sections(S, a, now)
        if a.watch and console: console.clear()
        S["me"] = S["me"]
        if console:
            from rich.panel import Panel
            from rich import box
            console.print(Panel.fit(f"[bold]UCL Young[/] GPU queue (Slurm)   [dim]{now:%a %d %b %Y %H:%M:%S}   user {S['me']}   from {host}[/]", box=box.DOUBLE, border_style="magenta"))
            render_rich_sections(secs, console)
        else:
            print(f"== UCL Young GPU queue (Slurm)   {now:%a %d %b %Y %H:%M:%S}   user {S['me']}   from {host}"); render_plain_sections(secs)
        if not a.watch: return 0
        time.sleep(a.watch)


def render_rich_sections(secs, console):
    from rich.table import Table
    from rich import box
    for title, head, rows in secs:
        t = Table(title=title, title_justify="left", title_style="bold magenta", box=box.SIMPLE_HEAD, show_header=head is not None, header_style="bold", pad_edge=False)
        if head is None: t.add_column(style="bold", no_wrap=True); t.add_column()
        else:
            for i, h in enumerate(head): t.add_column(h, justify="left" if (i == 0 or h in ("qos", "name", "partition", "reason", "rank / estimate", "top pending reasons", "GPU", "so far")) else "right", no_wrap=h not in ("top pending reasons", "rank / estimate"))
        for r in rows: t.add_row(*r)
        console.print(t)


def render_plain_sections(secs):
    strip = lambda s: re.sub(r"\[/?[a-z ]*\]", "", s)
    for title, head, rows in secs:
        print("\n" + title); rows = [[strip(c) for c in r] for r in rows]
        if head is not None: rows = [head] + rows
        w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for k, r in enumerate(rows):
            print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
            if head is not None and k == 0: print("  " + "  ".join("-" * x for x in w))


if __name__ == "__main__":
    sys.exit(main())
