"""young_gpu_queue.py (gq) on fixtures in the exact formats of its Slurm commands: scontrol show node -o, squeue -o '%.40i|%P|%u|%T|%M|%D|%C|%b|%r|%S|%V|%Q|%l|%q|%j', sacctmgr qos, sacct. No Slurm needed."""
import json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = """NodeName=node-x12t-006 CPUAlloc=32 CPUTot=64 Gres=gpu:a100:4 Partitions=gpu,intergpu State=MIXED CfgTRES=cpu=64,mem=500000M,gres/gpu=4,gres/gpu:a100=4 AllocTRES=cpu=32,mem=240G,gres/gpu=4,gres/gpu:a100=4
NodeName=node-x12t-007 CPUAlloc=8 CPUTot=64 Gres=gpu:a100:4 Partitions=gpu State=MIXED CfgTRES=cpu=64,gres/gpu=4,gres/gpu:a100=4 AllocTRES=cpu=8,gres/gpu=1,gres/gpu:a100=1
NodeName=node-x12t-008 CPUAlloc=0 CPUTot=64 Gres=gpu:a100:4 Partitions=gpu State=IDLE CfgTRES=cpu=64,gres/gpu=4 AllocTRES=
NodeName=node-x12t-009 CPUTot=64 Gres=gpu:a100:4 Partitions=gpu State=DOWN* CfgTRES=cpu=64,gres/gpu=4 AllocTRES= Reason=Not responding [slurm@2026-10-29T10:00:00]
NodeName=node-x12t-010 CPUAlloc=16 CPUTot=64 Gres=gpu:a100:4 Partitions=gpu State=MIXED+DRAIN CfgTRES=cpu=64,gres/gpu=4 AllocTRES=cpu=16,gres/gpu=2 Reason=maintenance [root@2026-10-30T09:00:00]
NodeName=node-c12a-001 CPUTot=40 Gres=(null) Partitions=cpu State=IDLE CfgTRES=cpu=40 AllocTRES=
"""
QUEUE = """                                  117564_[3]|gpu|mmm0037|PENDING|0:00|1|8|gres:gpu:1|Priority|N/A|2026-10-30T10:00:00|1200|04:00:00|freegpu|eosmap
                         117679_[0,2-3,5]%2|gpu|mmm0037|PENDING|0:00|1|8|gres:gpu:1|JobArrayTaskLimit|N/A|2026-10-30T10:05:00|1190|04:00:00|freegpu|stabnpt
                                    117563|gpu|mmm0037|PENDING|0:00|1|8|gres:gpu:1|QOSMaxGRESPerUser|2026-10-31T03:00:00|2026-10-30T10:06:00|1180|04:00:00|freegpu|Hwall_wall
                               115857_[5-36%6]|gpu|mmm0037|PENDING|0:00|1|8|gres:gpu:1|Priority|N/A|2026-10-29T09:00:00|900|08:00:00|freegpu|tdiend
                                    119208|intergpu|mmm0037|RUNNING|56:24|1|8|gres:gpu:1|None|2026-10-30T18:10:00|2026-10-30T18:09:00|5000|04:00:00|intergpu|bash
                                    120001|gpu|alice01|RUNNING|1-02:03:04|1|16|gres:gpu:2|None|2026-10-29T16:00:00|2026-10-29T15:59:00|3000|2-00:00:00|freegpu|train
                                    120002|gpu|bob02|PENDING|0:00|2|32|gres:gpu:4|Resources|N/A|2026-10-30T12:00:00|2500|1-00:00:00|freegpu|big
"""
QOS = "normal|||||0\nfreegpu|gres/gpu=2|||04:00:00|0\nintergpu|gres/gpu=2|2|2|04:00:00|100\n"
SACCT = ("130001|alice01|gpu|freegpu|COMPLETED|2026-10-30T08:00:00|2026-10-30T08:30:00|7200|billing=8,cpu=8,gres/gpu=1\n130002|bob02|gpu|freegpu|COMPLETED|2026-10-30T09:00:00|2026-10-30T11:00:00|3600|gres/gpu=1\n"
         "130004|carol3|gpu|freegpu|RUNNING|2026-10-30T12:00:00|Unknown|0|\n")


def run(d, *args):
    r = subprocess.run([sys.executable, os.path.join(ROOT, "mlp_pipeline", "young_gpu_queue.py"), "--from-dir", d, "--now", "2026-10-30 19:00:00", *args], capture_output=True, text=True, env=dict(os.environ, YQ_USER="mmm0037"))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_gpu_summary(tmp_path):
    d = str(tmp_path)
    for name, txt in (("scontrol_nodes.txt", NODES), ("squeue.txt", QUEUE), ("qos.txt", QOS), ("sacct.txt", SACCT)): open(os.path.join(d, name), "w").write(txt)
    S = json.loads(run(d, "--json"))
    assert S["gpus"] == dict(total=12, used=5, free=7, off=8), S["gpus"]                       # 3 usable nodes x 4; the drained and the down node are out of service
    assert S["nodes_n"] == dict(total=6, gpu=5, usable=3), S["nodes_n"]
    assert S["pend"] == dict(jobs=1 + 4 + 1 + 32 + 1, gpus=1 + 4 + 1 + 32 + 8), S["pend"]       # array tasks counted from the brackets; bob02: 2 nodes x 4 GPUs
    g = {(x["part"], x["qos"]): x for x in S["groups"]}["gpu", "freegpu"]
    assert g["started"] == 2 and abs(g["wait_med"] - 7200) < 1 and g["cap"] == 2 and dict(g["reasons"])["Priority"] == 33, g   # real waits from sacct, QOS cap from sacctmgr, Priority = 1 + 32 tasks
    out = run(d, "--plain")
    for needle in ("UCL Young GPU queue", "7 free", "JobArrayTaskLimit", "QOSMaxGRESPerUser", "#2/5, 8 GPUs ahead", "est. start 31/10 03:00", "wait med / p90"):
        assert needle in out, (needle, out[:2500])
    assert "at cap" not in out   # one interactive GPU held, the cap is 2
