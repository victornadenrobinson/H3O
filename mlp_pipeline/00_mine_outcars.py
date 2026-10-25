"""Mine T, P, V and composition from OUTCAR(.gz) headers/tails.

usage: python 00_mine_outcars.py old_inventory.csv new_inventory.csv [--workers 64] [--all]
Candidate paths = old inventory rows whose path parses to a H-O composition, or contains a
h3o/clathrate/c3 keyword (use --all to mine everything). Composition/elements come from the
OUTCAR itself (VRHFIN + 'ions per type'); non-H/O runs are flagged in Other.
Pressure: last 'external pressure' (kB) -> GPa. Run on a compute node.
"""
import argparse, re, subprocess
from multiprocessing import Pool
import pandas as pd
from common import parse_composition

KEYS = ("h3o", "clathrate", "calthrate", "c3-", "/c3", "h2o", "h2-")
HEAD_N, TAIL_N = 3000, 4000


def stream(path, mode):
    cmd = ["gzip", "-dc", path] if path.endswith(".gz") else ["cat", path]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen([mode[0], "-n", str(mode[1])], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close()
    out = p2.communicate()[0].decode(errors="replace")
    p1.wait()
    return out


def parse_head(txt):
    el = re.findall(r"VRHFIN\s*=\s*([A-Za-z]+)\s*:", txt)
    m = re.search(r"ions per type\s*=\s*([\d\s]+)", txt)
    cnt = [int(x) for x in m.group(1).split()] if m else []
    t = re.search(r"TEBEG\s*=\s*([\d.]+)", txt)
    return el, cnt, float(t.group(1)) if t else None


def mine(path):
    try:
        head = stream(path, ("head", HEAD_N))
        el, cnt, T = parse_head(head)
        tail = stream(path, ("tail", TAIL_N))
        P = re.findall(r"external pressure\s*=\s*(-?[\d.]+)\s*kB", tail)
        V = re.findall(r"volume of cell\s*:\s*([\d.]+)", tail)
        d = dict(zip(el, cnt)) if len(el) == len(cnt) else {}
        other = ",".join(f"{k}{v}" for k, v in d.items() if k not in ("H", "O"))
        return dict(Path=path, True_T_K=T, True_P_GPa=float(P[-1]) * 0.1 if P else None,
                    True_V_A3=float(V[-1]) if V else None, N_H=d.get("H", 0), N_O=d.get("O", 0),
                    Other=other, N_ions=sum(cnt))
    except Exception as e:
        return dict(Path=path, Other=f"ERR:{e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    old = pd.read_csv(a.old)
    paths = old.Path if a.all else old.Path[[parse_composition(p) is not None or any(k in p.lower() for k in KEYS) for p in old.Path]]
    print(f"mining {len(paths)} of {len(old)} paths")
    with Pool(a.workers) as pool:
        rows = pool.map(mine, list(paths), chunksize=20)
    df = pd.DataFrame(rows)
    df.to_csv(a.new, index=False)
    print(df.describe(include="all").T.to_string())
