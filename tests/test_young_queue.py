"""young_queue.py on fixtures in the exact formats pasted from Young (qhost -q with the 'adu' flags, qstat -g c, qstat -u '*' incl. pending array jobs, budgets): no SGE needed."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def host(name, load, used, flag, ncpu=80, mem="188.6G", ncor=40, arch="lx-amd64"):
    s = f"{name:<23} {arch:<9} {ncpu:>7} 2 {ncor:>4} {ncpu:>4} {('%.2f' % load) if load is not None else '-':>5} {mem:>7} {'8.9G' if load is not None else '-':>7}     0.0     0.0\n"
    if load is not None:
        for i, q in enumerate(["Arya", "Bran", "Bronn"]): s += f"   {q:<20} BIP   0/{used if i == 0 else 0}/{ncpu}        {flag}\n"
    return s


def write_fixture(d):
    H = "HOSTNAME                ARCH         NCPU NSOC NCOR NTHR  LOAD  MEMTOT  MEMUSE  SWAPTO  SWAPUS\n" + "-" * 94 + "\nglobal                  -               -    -    -    -     -       -       -       -       -\n"
    H += "login01                 lx-amd64       40    2   40   40  0.12  188.6G   26.8G     0.0     0.0\n"
    H += host("node-c12m-001", 40.05, 80, "") + host("node-c12m-002", 40.05, 40, "") + host("node-c12m-003", 0.05, 0, "")      # full, half used, empty
    H += host("node-c12h-005", 0.07, 0, "adu") + host("node-c12h-006", None, 0, "", ncpu=0, arch="-")                           # idle-looking but disabled, down
    open(os.path.join(d, "qhost_q.txt"), "w").write(H)
    open(os.path.join(d, "qstat_gc.txt"), "w").write("CLUSTER QUEUE                   CQLOAD   USED    RES  AVAIL  TOTAL aoACDS  cdsuE\n" + "-" * 80 + "\n"
                                                      "Arya                              0.35   5400      0   6808  26672      0  14464\nBran                              0.37      0      0   4608   8112      0   3504\nBronn                             0.37      0      0   4608   8112      0   3504\narya                              -NA-      0      0      0      0      0      0\n")
    open(os.path.join(d, "qstat_all.txt"), "w").write(
        "job-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID\n" + "-" * 113 + "\n"
        "2335952 3.45327 QRLOGIN    mmm0037      r     10/29/2026 18:59:31 Bran@node-c12n-006.ib.young.uc     8\n"
        "2335902 3.48133 QRLOGIN    mmm0037      qw    10/29/2026 16:06:57                                    8\n"
        "2335970 3.10000 dftchk     mmm0037      qw    10/29/2026 20:50:00                                    8 1-1:1\n"
        "2340001 2.90000 mdrun      alice01      r     10/29/2026 12:00:00 Arya@node-c12m-001.ib.young.uc    40\n"
        "2340011 2.70000 big        bob02        qw    10/29/2026 20:00:00                                   40 1-20:1\n"
        "2340012 2.60000 held       carol3       hqw   10/29/2026 17:00:00                                    4\n")
    open(os.path.join(d, "budgets.txt"), "w").write("Warning: untrusted X11 forwarding setup failed: xauth key data not generated\nProject         Machines   Balance\n--------------  ---------  ---------\n"
                                                    "MCC_discov_kin  ANY,young  0.00\nUKCP_ED_P       ANY        151856.71\n")


COMPLEXES = "#name               shortcut   type        relop requestable consumable default  urgency\n#------\nexc                 exc        BOOL        EXCL  YES         YES        0        1000\nh_rt                h_rt       TIME        <=    FORCED      NO         0:0:0    0\n"


def run(*args):
    r = subprocess.run([PY, os.path.join(ROOT, "mlp_pipeline", "young_queue.py"), *args], capture_output=True, text=True, env=dict(os.environ, YQ_USER="mmm0037"))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_summary_and_render(tmp_path):
    d = str(tmp_path); write_fixture(d); common = ["--from-dir", d, "--now", "2026-10-29 21:00:00"]
    S = json.loads(run(*common, "--json"))
    assert S["nodes"] == dict(total=5, usable=3, disabled=1, down=1), S["nodes"]                        # the 'adu' idle-looking node is disabled, the unreported one is down
    assert S["slots"] == dict(used=120, tot=240), S["slots"]
    assert S["run_jobs"] == 2 and S["run_slots"] == 48 and S["pend_jobs"] == 1 + 1 + 20 + 1 and S["pend_slots"] == 8 + 8 + 800 + 4, S   # array tasks counted (1-20 = 20 tasks of 40 slots)
    c12m = next(p for p in S["pools"] if p["pool"] == "c12m"); assert (c12m["free"], c12m["part"], c12m["full"]) == (1, 1, 1) and c12m["fit"]["40"] == 2 and c12m["fit"]["8"] == 2, c12m
    c12h = next(p for p in S["pools"] if p["pool"] == "c12h"); assert c12h["disabled"] == 1 and c12h["down"] == 1 and c12h["idle_but_off"] == 1 and c12h["usable"] == 0, c12h
    plain = run(*common, "--plain")
    for needle in ("UCL Young CPU cluster", "A 40-slot job fits on", "UKCP_ED_P", "151,856.71", "2335970.1-1:1", "waiting 4h53m", "running 2h00m", "do not count them as free"):
        assert needle in plain, (needle, plain[:1500])
    assert "MCC_discov_kin" not in plain                                                               # zero-balance projects are not listed (only UKCP_ED_P / positive ones)
    rich = run(*common, "--no-users"); assert "Top users" not in rich and "Node pools" in rich


def test_exclusive_mode_rank_and_held(tmp_path):
    d = str(tmp_path); write_fixture(d); open(os.path.join(d, "complexes.txt"), "w").write(COMPLEXES)
    with open(os.path.join(d, "qstat_all.txt"), "a") as f:   # a pending job of the user with the highest and a lower priority, an error-state ancient job
        f.write("2340020 9.00000 hi         zed01        qw    10/29/2026 20:55:00                                   16\n2340021 1.00000 ancient    zed01        Eqw   01/01/2023 00:00:00                                    8\n")
    S = json.loads(run("--from-dir", d, "--now", "2026-10-29 21:00:00", "--json"))
    assert S["exclusive"] == ["exc"], S["exclusive"]
    c12m = next(p for p in S["pools"] if p["pool"] == "c12m"); assert c12m["fit"]["40"] == 1 and c12m["fit"]["8"] == 1, c12m          # only the completely free node counts under EXCL
    assert S["classes"]["1-8"]["oldest"] < 10 * 3600, S["classes"]["1-8"]                                                        # the 2023 error-state job is not a queue wait
    out = run("--from-dir", d, "--now", "2026-10-29 21:00:00", "--plain")
    assert "completely free" in out and "relation EXCL" in out and "nodes ahead" in out and "Usable nodes busy" in out and "any job is full" in out.replace("a node with ANY job is full", "any job is full").lower().replace("any job is full", "any job is full"), out[:2500]
    assert "#3/" in out   # 2335902 (3.48133) and 2335970 (3.10000) queue behind the 9.0 job: ranks 2 and 3 of the 6 queued (2340011 2.7, 2340020 9.0 ... )


def test_exclusive_counting_stale_and_ties(tmp_path):
    d = str(tmp_path); write_fixture(d); open(os.path.join(d, "complexes.txt"), "w").write(COMPLEXES)
    with open(os.path.join(d, "qstat_all.txt"), "a") as f:
        f.write("2340030 0.50000 stalejob    zed01        qw    01/01/2026 00:00:00                                    8\n"          # waiting 300 days: stale, not a median
                "2340031 0.00000 newone      mmm0037      qw    10/29/2026 20:59:00                                    40\n"           # priority 0 (not scored yet), the newest
                "2340032 0.00000 zero2       zed01        qw    10/29/2026 20:58:00                                   120\n")           # also 0, older: ahead of 2340031, and 3 nodes
    S = json.loads(run("--from-dir", d, "--now", "2026-10-29 21:00:00", "--json"))
    nu = S["node_use"]; assert (nu["usable"], nu["free"], nu["busy"]) == (3, 1, 2), nu          # exclusive: the half-used node is BUSY, not 'part'
    assert S["run_nodes"] == 1 + 1 and S["pend_nodes"] >= 3 + 1, (S["run_nodes"], S["pend_nodes"])   # 8 slots = 1 node, 40 = 1, 120 = 3 (ceil(slots / 40))
    c = S["classes"]["1-8"]; assert c["stale"] == 1 and c["oldest"] < 10 * 3600, c              # the 300-day job is stale and out of the wait statistics
    out = run("--from-dir", d, "--now", "2026-10-29 21:00:00", "--plain")
    assert "priority 0: not scored yet" in out, out[-900:]
    row = next(l for l in out.splitlines() if "2340031" in l); assert "#6/6" in row and "~26 nodes ahead" in row, row   # last of 6 queued: behind every scored job and the older priority-0 job (1 + 1 + 20 + 1 + 3 nodes ahead)
