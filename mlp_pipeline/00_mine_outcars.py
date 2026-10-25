"""Mine T, P, V and composition from OUTCAR(.gz) headers/tails.

usage: python 00_mine_outcars.py old_inventory.csv new_inventory.csv [--workers 64] [--all]
Candidate paths = old inventory rows whose path parses to a H-O composition, or contains a
h3o/clathrate/c3 keyword (use --all to mine everything). Composition/elements come from the
OUTCAR itself (VRHFIN + 'ions per type'); non-H/O runs are flagged in Other.
All runs are NVT, so pressure is a statistical average: P_GPa = mean of 'external pressure' (kB -> GPa)
over the 2nd half of the run (first half = equilibration), with std and step count. Volume is fixed;
V per atom is reported since density is the control variable. Reads whole files: use a compute node.
"""
import argparse, os, re, subprocess, sys
from multiprocessing import Pool
import pandas as pd
from common import parse_composition, progress, out

KEYS = ("h3o", "clathrate", "calthrate", "c3-", "/c3", "h2o", "h2-")
HEAD_N = 8000


def stream(path, mode):
    cmd = ["gzip", "-dc", path] if path.endswith(".gz") else ["cat", path]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen([mode[0], "-n", str(mode[1])], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close()
    out = p2.communicate()[0].decode(errors="replace")
    p1.wait()
    return out


def grep_lines(path):
    cmd = ["gzip", "-dc", path] if path.endswith(".gz") else ["cat", path]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(["grep", "-a", "-E", "external pressure|total pressure|volume of cell"], stdin=p1.stdout,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close()
    out = p2.communicate()[0].decode(errors="replace")
    p1.wait()
    return out


def _uniq(seq):
    return list(dict.fromkeys(seq))


def parse_head(txt):
    m = re.search(r"ions per type\s*=\s*([\d\s]+)", txt)
    cnt = [int(x) for x in m.group(1).split()] if m else []
    # element order: VRHFIN, else POTCAR lines, else TITEL; repeats collapsed in order of appearance
    el = []
    for pat in (r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", r"POTCAR:\s+\S+\s+([A-Z][a-z]?)[_\s]", r"TITEL\s*=\s*\S+\s+([A-Z][a-z]?)[_\s]"):
        el = _uniq(re.findall(pat, txt))
        if len(el) >= len(cnt) > 0:
            break
    el = el[:len(cnt)]
    t = re.search(r"TEBEG\s*=\s*([\d.]+)", txt)
    return el, cnt, float(t.group(1)) if t else None


def poscar_comp(full):
    """Composition dict from POSCAR/CONTCAR (VASP5: elements line 6, counts line 7) in the run dir or its parent."""
    import gzip
    d0 = os.path.dirname(full)
    for d in (d0, os.path.dirname(d0)):
        for name in ("CONTCAR", "POSCAR", "CONTCAR.gz", "POSCAR.gz"):
            f = os.path.join(d, name)
            if not os.path.exists(f):
                continue
            try:
                with (gzip.open(f, "rt") if f.endswith(".gz") else open(f)) as fh:
                    L = [next(fh) for _ in range(7)]
                el, cnt = L[5].split(), L[6].split()
                if len(el) == len(cnt) and all(c.isdigit() for c in cnt) and all(e.isalpha() for e in el):
                    out = {}
                    for e, c in zip(el, cnt):
                        out[e] = out.get(e, 0) + int(c)
                    return out
            except Exception:
                continue
    return {}


def comp_dict(full, el, cnt):
    d = dict(zip(el, cnt)) if len(el) == len(cnt) else {}
    if not d:
        d = poscar_comp(full)
        if d and cnt and sum(d.values()) != sum(cnt):
            d = {}  # POSCAR from a different cell size: not trustworthy
    return d


def mine_head(path, root=""):
    el, cnt, T = parse_head(stream(os.path.join(root, path), ("head", HEAD_N)))
    d = comp_dict(os.path.join(root, path), el, cnt)
    other = ",".join(f"{k}{v}" for k, v in d.items() if k not in ("H", "O")) or ("?" if cnt and not d else "")
    return dict(True_T_K=T, N_H=d.get("H", 0), N_O=d.get("O", 0), Other=other, N_ions=sum(cnt))


def mine(path, root=""):
    full = os.path.join(root, path)
    try:
        head = stream(full, ("head", HEAD_N))
        el, cnt, T = parse_head(head)
        tail = grep_lines(full)
        P = [float(x) for x in re.findall(r"external pressure\s*=\s*(-?[\d.]+)\s*kB", tail)]
        V = re.findall(r"volume of cell\s*:\s*([\d.]+)", tail)
        P2 = P[len(P) // 2:]
        PT = [float(x) for x in re.findall(r"total pressure\s*=\s*(-?[\d.]+)\s*kB", tail)]
        PT2 = PT[len(PT) // 2:]
        d = comp_dict(full, el, cnt)
        other = ",".join(f"{k}{v}" for k, v in d.items() if k not in ("H", "O")) or ("?" if cnt and not d else "")
        return dict(Path=path, True_T_K=T, True_P_GPa=sum(P2) / len(P2) * 0.1 if P2 else None,
                    P_total_GPa=sum(PT2) / len(PT2) * 0.1 if PT2 else None,
                    P_std_GPa=(pd.Series(P2).std() * 0.1) if len(P2) > 1 else None, N_steps=len(P),
                    True_V_A3=float(V[-1]) if V else None,
                    V_per_atom=float(V[-1]) / sum(cnt) if V and cnt else None, N_H=d.get("H", 0), N_O=d.get("O", 0),
                    Other=other, N_ions=sum(cnt))
    except Exception as e:
        return dict(Path=path, Other=f"ERR:{e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--count-only", action="store_true", help="with --scan: just count OUTCAR files per top-level folder and exit")
    ap.add_argument("--scan", nargs="+", default=None, metavar="DIR",
                    help="ignore the old inventory: find every OUTCAR* under these dirs (paths stored relative to --root) and mine ALL of them")
    ap.add_argument("--fill", action="store_true", help="new csv already exists: only re-parse headers (T, composition) and update it")
    ap.add_argument("--root", default=None, help="dir that relative inventory paths are under (default: dir of old csv)")
    a = ap.parse_args()
    a.new = out(a.new)
    if a.fill:
        df = pd.read_csv(a.new)
        root = os.path.abspath(a.root or os.path.dirname(os.path.abspath(a.old)))
        from functools import partial
        with Pool(a.workers) as pool:
            res = list(progress(pool.imap(partial(mine_head, root=root), list(df.Path), chunksize=50), total=len(df), desc="headers"))
        for k in ("True_T_K", "N_H", "N_O", "Other", "N_ions"):
            df[k] = [r[k] for r in res]
        df["V_per_atom"] = df.True_V_A3 / df.N_ions.replace(0, float("nan"))
        df.to_csv(a.new, index=False)
        print(f"updated {a.new}; N_H+N_O==0 for {int(((df.N_H + df.N_O) == 0).sum())} of {len(df)}")
        print(df.assign(f=df.N_H.astype(str) + "H" + df.N_O.astype(str) + "O+" + df.Other.fillna("")).f.value_counts().head(25).to_string())
        sys.exit()
    if a.scan:
        root0 = os.path.abspath(os.path.expanduser(a.root or os.path.dirname(os.path.abspath(a.new))))
        found = []
        for d in a.scan:
            d = os.path.abspath(os.path.expanduser(d))
            print("scanning", d, "...", flush=True)
            r = subprocess.run(["find", d, "-type", "f", "-name", "OUTCAR*"], capture_output=True, text=True)
            found += [os.path.relpath(p, root0) for p in r.stdout.split("\n") if p]
        old = pd.DataFrame({"Path": sorted(set(found))})
        a.all, a.root = True, root0
        print(f"found {len(old)} OUTCAR files")
        if a.count_only:
            top = old.Path.map(lambda p: "/".join(p.split("/")[:4]))
            print(top.value_counts().head(40).to_string())
            sys.exit()
    else:
        old = pd.read_csv(a.old)
    paths = old.Path if a.all else old.Path[[parse_composition(p) is not None or any(k in p.lower() for k in KEYS) for p in old.Path]]
    root = os.path.abspath(a.root or os.path.dirname(os.path.abspath(a.old)))
    paths = list(paths)
    miss = [p for p in paths[:200] if not os.path.exists(os.path.join(root, p))]
    print(f"mining {len(paths)} of {len(old)} paths under {root}")
    if len(miss) > 100:
        sys.exit(f"ERROR: {len(miss)}/200 sampled files not found under {root} (e.g. {miss[0]}). Pass --root.")
    from functools import partial
    with Pool(a.workers) as pool:
        rows = list(progress(pool.imap(partial(mine, root=root), paths, chunksize=20), total=len(paths), desc="mining OUTCARs"))
    n_empty = sum(1 for r in rows if r.get("True_T_K") is None)
    print(f"{n_empty} of {len(rows)} files gave no TEBEG (missing/unreadable/truncated)")
    df = pd.DataFrame(rows)
    df.to_csv(a.new, index=False)
    print(df.describe(include="all").T.to_string())
