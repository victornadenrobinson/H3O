"""Inventory of the MLP-generated snapshots in a runs directory (PWidom frame caches, TDI node trajectories, any extxyz): what states they cover and how many frames there are.
Needs only numpy (no ASE, no GPU): safe on a login node.

usage:  python gen_inventory.py RUNS_DIR [--csv out.csv] [--min-frames 1]
Looks recursively for
  *_frames.npz      PWidom caches: ensembles 'host' (and 'full' = host + 1 H), keys cell/numbers/T; frames = first axis of each ensemble
  *.extxyz          TDI node trajectories, insertion sets, benchmarks: frames counted by the 'Lattice=' keyword, composition from the first frame
and parses P (<n>GPa), T (<n>K) and the number of inserted H (n<k>) from the file name when present.
Prints one table per kind and totals per (P, T, composition). These are GRACE/MLP ensembles, not DFT data: they are candidates to be re-labelled with VASP.
"""
import argparse, csv, os, re, sys
import numpy as np

SYM = {1: "H", 8: "O"}


def count_frames(path, key=b"Lattice="):
    n = 0; tail = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk: break
            buf = tail + chunk; n += buf.count(key); tail = buf[-(len(key) - 1):]
    return n


def first_frame_bytes(path):
    """bytes of the first extxyz frame (atom-count line, comment line, N atom lines), or None"""
    with open(path, "rb") as f:
        l1 = f.readline()
        try: n = int(l1.split()[0])
        except (ValueError, IndexError): return None
        tot = len(l1) + len(f.readline())
        for _ in range(n): tot += len(f.readline())
    return tot


def first_frame_comp(path):
    with open(path, "r", errors="replace") as f:
        try: n = int(f.readline().split()[0])
        except (ValueError, IndexError): return None
        f.readline(); cnt = {}
        for _ in range(n):
            parts = f.readline().split()
            if parts: cnt[parts[0]] = cnt.get(parts[0], 0) + 1
    return n, cnt


def tags(name):
    P = re.search(r"(\d+)\s*GPa", name, re.I); T = re.search(r"(\d+)\s*K(?![a-z])", name); n = re.search(r"(?:^|_)n(\d+)(?:_|\.|$)", name)
    return (int(P.group(1)) if P else None, int(T.group(1)) if T else None, int(n.group(1)) if n else None)


def comp_label(cnt):
    nO, nH = cnt.get("O", 0), cnt.get("H", 0)
    return f"O{nO}H{nH}" + (f" (H/O {nH / nO:.3f})" if nO else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs"); ap.add_argument("--csv"); ap.add_argument("--min-frames", type=int, default=1)
    ap.add_argument("--exact-below-mb", type=float, default=300.0, help="extxyz files above this size get an estimated frame count (size / first-frame bytes, marked ~) instead of a full read")
    a = ap.parse_args()
    rows = []
    for d, dirs, files in os.walk(a.runs):
        dirs[:] = [x for x in dirs if x not in ("cache", "plots", "__pycache__")]
        for fn in sorted(files):
            p = os.path.join(d, fn)
            if fn.endswith((".extxyz", "_frames.npz")): print(f"scanning {os.path.relpath(p, a.runs)}", file=sys.stderr, flush=True)
            try:
                if fn.endswith("_frames.npz"):
                    with np.load(p, allow_pickle=False) as z:
                        nums = z["numbers"]; cnt = {SYM.get(int(k), str(k)): int((nums == k).sum()) for k in np.unique(nums)}
                        T = float(z["T"]) if "T" in z.files else None
                        for ens in ("host", "full"):
                            if ens in z.files:
                                nf = int(z[ens].shape[0]); c = dict(cnt)
                                if ens == "full": c["H"] = c.get("H", 0) + 1
                                P, Tn, nn = tags(fn); rows.append(dict(kind=f"pwidom-{ens}", P=P, T=T or Tn, n_ins=nn, comp=comp_label(c), natoms=sum(c.values()), frames=nf, MB=round(os.path.getsize(p) / 1e6, 1), path=os.path.relpath(p, a.runs)))
                elif fn.endswith(".extxyz"):
                    size = os.path.getsize(p); approx = size > a.exact_below_mb * 1e6
                    if approx:
                        fb = first_frame_bytes(p); nf = int(round(size / fb)) if fb else 0              # fixed-size frames: size / bytes of the first frame (within a few %)
                    else: nf = count_frames(p)
                    fc = first_frame_comp(p)
                    if fc is None: continue
                    P, Tn, nn = tags(fn); rows.append(dict(kind="extxyz", P=P, T=Tn, n_ins=nn, comp=comp_label(fc[1]), natoms=fc[0], frames=nf, approx=approx, MB=round(os.path.getsize(p) / 1e6, 1), path=os.path.relpath(p, a.runs)))
            except Exception as e:
                print(f"skip {p}: {e}", file=sys.stderr)
    rows = [r for r in rows if r["frames"] >= a.min_frames]
    if not rows: print("nothing found"); return
    rows.sort(key=lambda r: (r["kind"], r["P"] or 0, r["T"] or 0, r["comp"], r["path"]))
    print(f"{len(rows)} files, {sum(r['frames'] for r in rows)} frames in total, {sum(r['MB'] for r in rows) / 1e3:.1f} GB\n")
    print(f"{'kind':14s} {'P':>5s} {'T':>6s} {'n_ins':>5s} {'atoms':>5s} {'frames':>7s} {'MB':>9s}  {'composition':22s} path")
    for r in rows:
        print(f"{r['kind']:14s} {str(r['P'] or '-'):>5s} {str(int(r['T']) if r['T'] else '-'):>6s} {str(r['n_ins'] if r['n_ins'] is not None else '-'):>5s} {r['natoms']:5d} {r['frames']:7d}{'~' if r.get('approx') else ' '}{r['MB']:8.1f}  {r['comp']:22s} {r['path']}")
    print("\n== frames per (kind, P, T, composition) ==")
    tot = {}
    for r in rows:
        k = (r["kind"], r["P"], int(r["T"]) if r["T"] else None, r["comp"]); t = tot.setdefault(k, [0, 0]); t[0] += r["frames"]; t[1] += 1
    for k in sorted(tot, key=lambda k: (k[0], k[1] or 0, k[2] or 0, k[3])):
        print(f"  {k[0]:14s} P={k[1]} T={k[2]} {k[3]:22s} {tot[k][0]:7d} frames in {tot[k][1]} files")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
