#!/usr/bin/env python3
"""Young CPU cluster status in one screen: node pools (usable / free / full / disabled), queue pressure (running, pending, waits by job size), top users, your jobs, your budget.
Mines SGE with four commands (`qhost -q`, `qstat -g c`, `qstat -u '*'`, `budgets`), no root, read-only. Rich tables if `rich` is importable (the venv ~/grace-venv-cpu has it), else plain text.

  mlp_pipeline/yq                 (wrapper that loads the Python module + venv; alias:  alias yq='bash -l /lustre/scratch/mmm0037/H3O/mlp_pipeline/yq')
  python mlp_pipeline/young_queue.py [--watch 30] [--no-users] [--json] [--save-dir DIR] [--from-dir DIR --now "2026-10-29 21:00:00"]

Reading the node table (from the 2026-10-29 qhost -q): a node can show load ~0 and still be unusable: when every queue instance on it carries a state flag d (disabled), u (unknown), E (error) or D, SGE will
not start anything there ("adu" on idle-looking c12h-005 / c12j-005). `free` below counts only usable nodes with every slot free; `fit N` = usable nodes with >= N free slots (can a -pe smp N job start now?).
The ~40 queues named after characters (Arya, Bran, ...) overlay the same hosts: they are collapsed to one line per distinct (used, avail, total).
--from-dir reads qhost_q.txt / qstat_gc.txt / qstat_all.txt / budgets.txt saved by --save-dir (or pasted by hand): for tests and for sharing a snapshot."""
import argparse, datetime, getpass, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAVE_RICH = True
except Exception:
    HAVE_RICH = False

FILES = dict(hosts="qhost_q.txt", gc="qstat_gc.txt", jobs="qstat_all.txt", budgets="budgets.txt", complexes="complexes.txt")
CMDS = dict(hosts="qhost -q", gc="qstat -g c", jobs="qstat -u '*'", budgets="budgets 2>/dev/null", complexes="qconf -sc 2>/dev/null")
CORES = 40       # cores (slots that count) per CPU node: a job of s slots holds ceil(s / 40) nodes when hosts are exclusive
STALE = 14 * 86400   # pending longer than this = stale (abandoned / unsatisfiable), reported separately from real waits
BAD = set("udDE")   # queue-instance state flags that stop scheduling (a alarm / s suspended / S subordinate / c, C are different and not counted here)


def sh(cmd, timeout=90):
    try:
        return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, timeout=timeout).stdout
    except Exception:
        return ""


def collect(a):
    raw = {}
    for k in FILES:
        if a.from_dir:
            p = os.path.join(a.from_dir, FILES[k]); raw[k] = open(p).read() if os.path.exists(p) else ""
        else:
            raw[k] = sh(CMDS[k])
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        for k in FILES: open(os.path.join(a.save_dir, FILES[k]), "w").write(raw[k])
    return raw


# ------------------------------------------------------------------ parsers
def parse_hosts(text):
    hosts = []; cur = None
    for line in text.splitlines():
        if not line.strip() or line.startswith("HOSTNAME") or line.startswith("---"): continue
        if line[0] not in " \t":
            t = line.split()
            if t[0] == "global": cur = None; continue
            ncpu = int(t[2]) if len(t) > 2 and t[2].isdigit() else 0
            load = float(t[6]) if len(t) > 6 and re.match(r"^[\d.]+$", t[6]) else None
            cur = dict(name=t[0], arch=t[1] if len(t) > 1 else "-", ncpu=ncpu, ncor=int(t[4]) if len(t) > 4 and t[4].isdigit() else 0, load=load,
                       memtot=t[7] if len(t) > 7 else "-", queues=[]); hosts.append(cur)
        elif cur is not None:
            m = re.match(r"^\s+(\S+)\s+(\S+)\s+(\d+)/(\d+)/(\d+)\s*(\S*)\s*$", line)
            if m: cur["queues"].append((m.group(1), int(m.group(3)), int(m.group(4)), int(m.group(5)), m.group(6)))
    for h in hosts:
        m = re.match(r"^node-([a-z0-9]+)-\d+", h["name"]); h["pool"] = m.group(1) if m else h["name"].split(".")[0].rstrip("0123456789")
        qs = h["queues"]; h["tot"] = max([q[3] for q in qs] or [0]); h["used"] = min(sum(q[2] for q in qs), h["tot"])
        flagged = bool(qs) and all(BAD & set(q[4]) for q in qs)
        h["state"] = "down" if (h["load"] is None or h["arch"] == "-" or not qs) else "disabled" if flagged else "ok"
        h["free"] = h["tot"] - h["used"] if h["state"] == "ok" else 0
    return [h for h in hosts if not (h["name"].startswith("login") or h["name"].startswith("util"))]


def parse_gc(text):
    rows = []
    for line in text.splitlines():
        t = line.split()
        if len(t) >= 8 and re.match(r"^[\d.]+$|^-NA-$", t[1]) and t[2].isdigit(): rows.append(dict(q=t[0], used=int(t[2]), res=int(t[3]), avail=int(t[4]), total=int(t[5]), bad=int(t[7]) if t[7].isdigit() else 0))
    return rows


def parse_jobs(text, now):
    jobs = []
    for line in text.splitlines():
        t = line.split()
        if len(t) < 8 or not t[0].isdigit() or not re.match(r"^\d\d/\d\d/\d{4}$", t[5]): continue
        try: when = datetime.datetime.strptime(t[5] + " " + t[6], "%m/%d/%Y %H:%M:%S")
        except ValueError: continue
        rest = t[7:]; queue = ""
        if rest and not rest[0].isdigit(): queue = rest[0]; rest = rest[1:]
        if not rest: continue
        slots = int(rest[0]); task = rest[1] if len(rest) > 1 else ""; ntask = 1
        m = re.match(r"^(\d+)-(\d+):(\d+)$", task)
        if m: ntask = (int(m.group(2)) - int(m.group(1))) // int(m.group(3)) + 1
        st = t[4]; running = "r" in st or st in ("t", "Rr")
        jobs.append(dict(id=t[0], prior=float(t[1]), name=t[2], user=t[3], state=st, when=when, queue=queue, slots=slots, task=task, ntask=ntask, running=running,
                         age=(now - when).total_seconds()))
    return jobs


def parse_exclusive(text):
    """names of complexes with relation EXCL (SGE's exclusive-host request: a job asking for it, here added to every job as exc=true, needs a host to itself)"""
    return [t[0] for t in (l.split() for l in text.splitlines()) if len(t) >= 4 and not t[0].startswith("#") and t[3] == "EXCL"]


def parse_budgets(text):
    out = []
    for line in text.splitlines():
        t = line.split()
        if len(t) == 3 and re.match(r"^-?[\d.]+$", t[2]): out.append((t[0], t[1], float(t[2])))
    return out


# ------------------------------------------------------------------ summaries
def hms(sec):
    sec = int(max(0, sec)); d, r = divmod(sec, 86400); h, r = divmod(r, 3600); m, s = divmod(r, 60)
    return (f"{d}d{h:02d}h" if d else f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s")


def bar(frac, width=10):
    n = int(round(max(0.0, min(1.0, frac)) * width)); col = "green" if frac < 0.6 else "yellow" if frac < 0.9 else "red"
    return f"[{col}]{'█' * n}[/][dim]{'░' * (width - n)}[/] {100 * frac:3.0f}%"


def short_mem(m):
    return re.sub(r"[.][0-9]+", "", str(m))


def size_class(s):
    return "1-8" if s <= 8 else "9-39" if s < 40 else "40" if s == 40 else "41+"


def summarize(hosts, gc, jobs, budgets, me, excl=(), fits=(8, 16, 40)):
    S = dict(pools=[], me=me, exclusive=list(excl))
    pools = defaultdict(list)
    for h in hosts: pools[h["pool"]].append(h)
    for name, hs in sorted(pools.items()):
        ok = [h for h in hs if h["state"] == "ok"]; cpu = Counter(h["ncpu"] for h in hs).most_common(1)[0][0]
        nfree = sum(1 for h in ok if h["used"] == 0)
        S["pools"].append(dict(pool=name, nodes=len(hs), cpus=cpu, mem=hs[0]["memtot"], usable=len(ok), free=nfree, busy=len(ok) - nfree, full=sum(1 for h in ok if h["free"] == 0),
                               part=0 if excl else sum(1 for h in ok if 0 < h["free"] < h["tot"]), disabled=sum(1 for h in hs if h["state"] == "disabled"), down=sum(1 for h in hs if h["state"] == "down"),
                               used=sum(h["used"] for h in ok), tot=sum(h["tot"] for h in ok), fit={n: (sum(1 for h in ok if h["used"] == 0 and h["tot"] >= n) if excl else sum(1 for h in ok if h["free"] >= n)) for n in fits},
                               idle_but_off=sum(1 for h in hs if h["state"] != "ok" and h["load"] is not None and h["load"] < 1.0)))
    run = [j for j in jobs if j["running"]]; pend = [j for j in jobs if not j["running"]]
    nn = lambda j: -(-j["slots"] // CORES)   # nodes a job holds: ceil(slots / 40 cores)
    S["run_nodes"] = sum(nn(j) for j in run); S["pend_nodes"] = sum(nn(j) * j["ntask"] for j in pend if j["state"] == "qw")
    S["run_jobs"] = len(run); S["run_slots"] = sum(j["slots"] for j in run); S["pend_jobs"] = sum(j["ntask"] for j in pend); S["pend_slots"] = sum(j["slots"] * j["ntask"] for j in pend)
    cls = {}
    for c in ("1-8", "9-39", "40", "41+"):
        p = [j for j in pend if size_class(j["slots"]) == c]; r = [j for j in run if size_class(j["slots"]) == c]
        w = sorted(j["age"] for j in p if j["state"] == "qw" and j["age"] < STALE)   # held (hqw) / error (Eqw) jobs are not waiting for resources; jobs older than STALE are listed separately
        cls[c] = dict(run=len(r), run_slots=sum(j["slots"] for j in r), run_nodes=sum(nn(j) for j in r), pend=sum(j["ntask"] for j in p), pend_slots=sum(j["slots"] * j["ntask"] for j in p),
                      pend_nodes=sum(nn(j) * j["ntask"] for j in p if j["state"] == "qw"), med=w[len(w) // 2] if w else None, oldest=w[-1] if w else None, recent=len(w),
                      stale=sum(1 for j in p if j["state"] == "qw" and j["age"] >= STALE), held=sum(1 for j in p if j["state"].startswith(("h", "E"))))
    S["classes"] = cls
    users = defaultdict(lambda: dict(run=0, slots=0, pend=0, pslots=0))
    for j in jobs:
        u = users[j["user"]]
        if j["running"]: u["run"] += 1; u["slots"] += j["slots"]
        else: u["pend"] += j["ntask"]; u["pslots"] += j["slots"] * j["ntask"]
    S["users"] = sorted(users.items(), key=lambda kv: (-kv[1]["slots"], -kv[1]["pslots"]))
    queued = sorted([j for j in pend if j["state"] == "qw"], key=lambda j: -j["prior"]); S["queued"] = len(queued)
    for j in jobs:
        if j in queued:
            ahead = [k for k in queued if k is not j and (k["prior"] > j["prior"] or (k["prior"] == j["prior"] and k["when"] <= j["when"]))]   # equal priority: older first
            j["rank"] = len(ahead) + 1; j["slots_ahead"] = sum(k["slots"] * k["ntask"] for k in ahead); j["nodes_ahead"] = sum(nn(k) * k["ntask"] for k in ahead)
    S["mine"] = sorted([j for j in jobs if j["user"] == me], key=lambda j: (not j["running"], j["when"]))
    seen = defaultdict(list)
    for q in gc:
        if q["total"] > 0: seen[(q["used"], q["avail"], q["total"], q["bad"])].append(q["q"])
    S["queues"] = sorted(((k, v) for k, v in seen.items()), key=lambda kv: -kv[0][2])
    S["budgets"] = budgets
    S["nodes"] = dict(total=len(hosts), usable=sum(h["state"] == "ok" for h in hosts), disabled=sum(h["state"] == "disabled" for h in hosts), down=sum(h["state"] == "down" for h in hosts))
    okh = [h for h in hosts if h["state"] == "ok"]; S["node_use"] = dict(usable=len(okh), free=sum(1 for h in okh if h["used"] == 0), busy=sum(1 for h in okh if h["used"] > 0))
    S["slots"] = dict(used=sum(h["used"] for h in hosts if h["state"] == "ok"), tot=sum(h["tot"] for h in hosts if h["state"] == "ok"))
    return S


# ------------------------------------------------------------------ rendering (sections -> rich or plain)
def sections(S, a, now):
    out = []; n = S["nodes"]; sl = S["slots"]; nu = S["node_use"]; ex = bool(S["exclusive"]); nfree = nu["free"]
    ov = [["Nodes", f"{n['total']} CPU nodes: [green]{n['usable']} usable[/], [yellow]{n['disabled']} disabled/unknown[/] (queue flags d/u/E), [red]{n['down']} down[/]"]]
    if ex:
        ov += [["Usable nodes busy", f"[bold]{nu['busy']}[/] / {nu['usable']}  {bar(nu['busy'] / nu['usable'] if nu['usable'] else 0, 24)}   [bold]{nfree} completely free[/]" + ("" if nfree else "  [red](cluster full)[/]")],
               ["Running jobs hold", f"~{S['run_nodes']} nodes (sum of ceil(slots / {CORES})) in {S['run_jobs']} jobs, {S['run_slots']:,} slots"],
               ["Queued demand", f"{S['pend_jobs']} pending jobs = ~{S['pend_nodes']} nodes of work in 'qw' ({S['pend_slots']:,} slots); [bold]{nfree}[/] free node(s) now"],
               ["Exclusive hosts", "[yellow]" + ", ".join(S["exclusive"]) + " (complex relation EXCL): every job takes whole node(s) whatever its slot count: a node with ANY job is full[/]"]]
    else:
        f40 = sum(p["fit"].get(40, 0) for p in S["pools"])
        ov += [["Slots (usable nodes)", f"{sl['used']:,} / {sl['tot']:,}  {bar(sl['used'] / sl['tot'] if sl['tot'] else 0, 24)}"],
               ["A 40-slot job fits on", f"[bold]{f40}[/] usable nodes (slot count only)" + ("  [green](can start now unless priority / reservations hold it)[/]" if f40 else "  [red](will queue)[/]")],
               ["Jobs", f"{S['run_jobs']} running ({S['run_slots']:,} slots), {S['pend_jobs']} pending ({S['pend_slots']:,} slots)"]]
    ov += [["Your jobs", f"{sum(1 for j in S['mine'] if j['running'])} running, {sum(1 for j in S['mine'] if not j['running'])} pending"]]
    ov += [["Budget " + p, f"{b:,.2f}" + ("  [red](empty)[/]" if b <= 0 else "")] for p, m, b in S["budgets"] if b > 0 or p == "UKCP_ED_P"]
    out.append(("Overview", None, ov)); rows = []
    for p in S["pools"]:
        tail = [f"[red]{p['down']}[/]", bar(p["busy"] / p["usable"] if p["usable"] else 0)] if ex else [f"[red]{p['down']}[/]", bar(p["used"] / p["tot"] if p["tot"] else 0), "/".join(str(v) for v in p["fit"].values())]
        mid = [str(p["usable"]), f"[green]{p['free']}[/]", f"[red]{p['busy']}[/]"] if ex else [f"[green]{p['free']}[/]", str(p["part"]), f"[red]{p['full']}[/]"]
        rows.append([p["pool"], "%dc/%s" % (p["cpus"], short_mem(p["mem"])), str(p["nodes"])] + mid + [f"[yellow]{p['disabled']}[/]" + ("*" if p["idle_but_off"] else "")] + tail)
    idle_off = sum(p["idle_but_off"] for p in S["pools"])
    head = (["pool", "type", "nodes", "usable", "free", "busy", "disabled", "down", "busy share"] if ex else ["pool", "type", "nodes", "free", "part", "full", "disabled", "down", "slots used", "fit 8/16/40"])
    out.append(("Node pools  (usable = not disabled / unknown / error on every queue; free = no job at all" + ("; busy = any job: exclusive hosts)" if ex else "; part / full by slots)"), head, rows))
    if idle_off: out.append(("", None, [["*", f"{idle_off} disabled/down nodes look idle (load < 1) but SGE will not schedule on them: do not count them as free"]]))
    out.append((f"Queue pressure by job size (slots per job; 'qw' only: waits of jobs pending > {STALE // 86400} days are shown as stale, held / error jobs separately)",
                ["size", "running", "run nodes", "pending", "pend nodes", "median wait", "oldest wait", "stale", "held/error"],
                [[c, str(v["run"]), f"{v['run_nodes']:,}", str(v["pend"]), f"{v['pend_nodes']:,}", hms(v["med"]) if v["med"] is not None else "-", hms(v["oldest"]) if v["oldest"] is not None else "-", str(v["stale"]), str(v["held"])] for c, v in S["classes"].items()]))
    if not a.no_users:
        out.append(("Top users by running slots", ["user", "running jobs", "slots", "pending jobs", "pending slots"],
                    [[("[bold cyan]" + u + "[/]" if u == S["me"] else u), str(v["run"]), f"{v['slots']:,}", str(v["pend"]), f"{v['pslots']:,}"] for u, v in S["users"][:a.top]]))
    out.append(("Queues (overlay the same hosts: identical rows collapsed)", ["queues", "used", "avail", "total", "unusable"],
                [[(", ".join(v[:4]) + (f" … +{len(v) - 4} more" if len(v) > 4 else "")), f"{k[0]:,}", f"{k[1]:,}", f"{k[2]:,}", f"{k[3]:,}"] for k, v in S["queues"][:6]]))
    mine = []
    for j in S["mine"]:
        sc = {"r": "green", "qw": "yellow", "Eqw": "red", "hqw": "dim"}.get(j["state"], "white")
        qh = j["queue"].split(".")[0].replace("node-", "")
        rk = ""
        if "rank" in j:
            rk = "#%s/%s, ~%s nodes ahead" % (format(j["rank"], ","), format(S["queued"], ","), format(j["nodes_ahead"], ",")) + ("; priority 0: not scored yet" if j["prior"] == 0 else "")
        elif not j["running"]: rk = "held / error: not queued"
        mine.append([j["id"] + (("." + j["task"]) if j["task"] else ""), j["name"][:12], f"[{sc}]{j['state']}[/]", qh or "-", str(j["slots"]) + (f" x{j['ntask']}" if j["ntask"] > 1 else ""),
                     j["when"].strftime("%d/%m %H:%M"), ("running " if j["running"] else "waiting ") + hms(j["age"]), rk])
    out.append(("Your jobs", ["job", "name", "st", "queue@host", "slots", "when", "so far", "priority rank"], mine or [["[dim]none[/]", "", "", "", "", "", "", ""]]))
    return out


def render_rich(secs, S, now, host, console):
    console.print(Panel.fit(f"[bold]UCL Young[/] CPU cluster   [dim]{now:%a %d %b %Y %H:%M:%S}   user {S['me']}   from {host}[/]", box=box.DOUBLE, border_style="cyan"))
    for title, head, rows in secs:
        if not title and head is None:
            console.print("[dim]" + rows[0][0] + " " + rows[0][1] + "[/]"); continue
        t = Table(title=title, title_justify="left", title_style="bold cyan", box=box.SIMPLE_HEAD, show_header=head is not None, header_style="bold", pad_edge=False)
        if head is None: t.add_column(style="bold", no_wrap=True); t.add_column()
        else:
            for i, h in enumerate(head): t.add_column(h, justify="left" if i == 0 else "right" if h not in ("queue@host", "so far", "when", "queues", "name", "priority rank") else "left", no_wrap=True)
        for r in rows: t.add_row(*r)
        console.print(t)


def render_plain(secs, S, now, host):
    strip = lambda s: re.sub(r"\[/?[a-z ]*\]", "", s)
    print(f"== UCL Young CPU cluster   {now:%a %d %b %Y %H:%M:%S}   user {S['me']}   from {host}")
    for title, head, rows in secs:
        print("\n" + title); rows = [[strip(c) for c in r] for r in rows]
        if head is not None: rows = [head] + rows
        w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for k, r in enumerate(rows):
            print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
            if head is not None and k == 0: print("  " + "  ".join("-" * x for x in w))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    ap.add_argument("--no-users", action="store_true"); ap.add_argument("--top", type=int, default=6); ap.add_argument("--json", action="store_true", help="machine-readable summary")
    ap.add_argument("--save-dir", default=None, help="also save the raw command outputs here"); ap.add_argument("--from-dir", default=None); ap.add_argument("--now", default=None, help="'YYYY-MM-DD HH:MM:SS' (with --from-dir)")
    ap.add_argument("--plain", action="store_true", help="no rich even if installed"); a = ap.parse_args()
    console = Console(width=max(100, Console().width)) if HAVE_RICH and not a.plain else None; host = os.uname()[1] if hasattr(os, "uname") else "?"
    while True:
        now = datetime.datetime.strptime(a.now, "%Y-%m-%d %H:%M:%S") if a.now else datetime.datetime.now()
        raw = collect(a); S = summarize(parse_hosts(raw["hosts"]), parse_gc(raw["gc"]), parse_jobs(raw["jobs"], now), parse_budgets(raw["budgets"]), os.environ.get("YQ_USER") or getpass.getuser(), excl=parse_exclusive(raw["complexes"]))
        if not raw["hosts"] and not raw["jobs"]: print("no output from qhost / qstat: not on Young (or SGE not on PATH); use --from-dir for a saved snapshot"); return 1
        if a.json:
            print(json.dumps({k: v for k, v in S.items() if k not in ("mine", "users", "queues")}, default=str, indent=1)); return 0
        secs = sections(S, a, now)
        if a.watch and console: console.clear()
        render_rich(secs, S, now, host, console) if console else render_plain(secs, S, now, host)
        if not a.watch: return 0
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
