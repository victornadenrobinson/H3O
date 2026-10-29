"""H-H collapse check of a run: the smallest H-H distance (minimum image) in the saved snapshots of its highest-lambda values (plain numpy: reads the
extxyz text itself, so summarize.py stays ASE-free). Healthy dense H runs at 300-600 GPa stay above ~0.4 A; two protons fused to ~0.05 A mean the MLP has
a hole at short range and the run's G is meaningless (H_500GPa_5000K_N250 / H_500GPa_6000K_N250, 2026-10-28).
usage:  python3 mlp_pipeline/tdi/collapse.py runs/tdi [more dirs]      (one line per run with its smallest H-H distance; COLLAPSED below COLLAPSE_A)
        python3 mlp_pipeline/tdi/collapse.py --when runs/tdi TAG         (per lambda value: from which snapshot two protons are fused)
"""
import glob, json, os, sys
import numpy as np

COLLAPSE_A = 0.25


def frames(path, every=1):
    """yield (cell 3x3, positions of H) per frame of an ASE extxyz file written by run_real.py"""
    with open(path) as fh:
        k = 0
        while True:
            line = fh.readline()
            if not line: return
            n = int(line); head = fh.readline()
            i = head.find('Lattice="'); cell = np.array(head[i + 9:head.index('"', i + 9)].split(), float).reshape(3, 3)
            body = [fh.readline().split() for _ in range(n)]
            if k % every == 0:
                yield cell, np.array([[float(x) for x in b[1:4]] for b in body if b[0] == "H"])
            k += 1


def min_hh(path, every=1):
    best = np.inf
    for cell, p in frames(path, every):
        if len(p) < 2: continue
        inv = np.linalg.inv(cell); d = p[:, None, :] - p[None, :, :]; s = d @ inv; s -= np.round(s); r = np.sqrt(((s @ cell) ** 2).sum(-1))
        np.fill_diagonal(r, np.inf); best = min(best, float(r.min()))
    return best


def run_min_hh(d, tag, lam=None, top=2, every=5):
    """smallest H-H distance over every `every`-th snapshot of the `top` highest lambda values of a run (nan without snapshots; a fused pair persists, so subsampling does not miss it)"""
    if lam is None:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tdi import analyze
        lam = analyze.rule_nodes(json.load(open(os.path.join(d, f"{tag}_stage0.json")))["args"])[0]
    ks = list(np.argsort(lam)[::-1][:top]); vals = []
    for k in ks:
        f = os.path.join(d, f"{tag}_node{k}_traj.extxyz")
        if os.path.exists(f): vals.append(min_hh(f, every))
    return min(vals) if vals else float("nan")


def when(d, tag, r_cut=COLLAPSE_A):
    """per lambda value of a run: (k, lambda, snapshot index of the first fused pair or None, number of snapshots, t_ps of that snapshot)"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tdi import analyze
    A = json.load(open(os.path.join(d, f"{tag}_stage0.json")))["args"]; lam = analyze.rule_nodes(A)[0]; out = []
    for k in range(len(lam)):
        f = os.path.join(d, f"{tag}_node{k}_traj.extxyz")
        if not os.path.exists(f): continue
        first, n = None, 0
        for i, (cell, p) in enumerate(frames(f)):
            n += 1
            if first is None and len(p) > 1:
                inv = np.linalg.inv(cell); dd = p[:, None, :] - p[None, :, :]; s = dd @ inv; s -= np.round(s); r = np.sqrt(((s @ cell) ** 2).sum(-1)); np.fill_diagonal(r, np.inf)
                if r.min() < r_cut: first = i
        out.append((k, lam[k], first, n, None if first is None else first * A.get("every", 2) * 20 * A["dt"] / 1000))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--when":   # collapse.py --when runs/tdi TAG: in which lambda values and from which snapshot two protons are fused
        for k, l, first, n, t in when(sys.argv[2], sys.argv[3]):
            print(f"lambda {k:2d} = {l:.3e}: " + (f"fused from snapshot {first} of {n} (t ~ {t:.2f} ps of the recorded samples)" if first is not None else f"no fused pair in {n} snapshots"))
        sys.exit(0)
    for d in sys.argv[1:] or ["runs/tdi"]:
        for f in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
            tag = os.path.basename(f)[:-len("_stage0.json")]; m = run_min_hh(d, tag)
            print(f"{d + '/' + tag:55s} min d(H-H) {m:6.3f} A" + ("   COLLAPSED" if m < COLLAPSE_A else ""))
