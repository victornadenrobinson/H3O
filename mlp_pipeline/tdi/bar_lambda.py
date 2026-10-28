"""Quadrature-free check of the TDI integral: MBAR over the saved lambda series (no MD, no GPU, seconds).
usage (repo root, plain python3 with numpy):  python3 mlp_pipeline/tdi/bar_lambda.py runs/tdi insert_500GPa_6000K_n1 [more tags] [--stride 5] [--glob 'insert_*']
Every path here is linear in lambda: U(lam) = U_ref + lam dU with dU = U_full - U_ref (insertion: U_ref = host alone; crystals: Einstein springs; fluids: 0), and the node
files <tag>_node<k>.npy store dU of every sample. So the reduced energy of any sample at ANY lambda is known: beta lam dU. MBAR (self-consistent, all nodes together)
then gives F(lam) on the sampled grid and, by reweighting, at the unsampled ends lam = 0 and 1: dF = F(1) - F(0), the same quantity as the TDI integral
sum_k w_k <dU>_k, but without the quadrature. Reported per added H (insertion) or per atom (other phases), in meV, next to the TDI value of the same samples.
Overlap: the MBAR overlap-matrix element between neighbouring lambda values (and to the ends); below ~0.03 the reweighting across that gap is unreliable
(typical for crystal paths at large lambda, whose whole-cell dU fluctuates by tens of eV). Errors: spread of the estimate over 5 consecutive blocks of every series.
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB = 8.617333262e-5


def lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True); return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def mbar(lams, series, beta, targets, iters=5000, tol=1e-8):
    """lams: sampled lambda values; series: list of dU arrays (eV); returns f at targets (dimensionless, f(lam) relative to the first target) and the overlap matrix of the sampled states"""
    N = np.array([len(s) for s in series], float); x = np.concatenate(series); L = np.asarray(lams)
    u = beta * L[:, None] * x[None, :]   # (K, Ntot) reduced energies (common U_ref part cancels)
    f = np.zeros(len(L))
    for _ in range(iters):
        den = lse(np.log(N)[:, None] + f[:, None] - u, axis=0)   # per sample
        fn = -lse(-u - den[None, :], axis=1); fn -= fn[0]
        if np.max(np.abs(fn - f)) < tol: f = fn; break
        f = fn
    den = lse(np.log(N)[:, None] + f[:, None] - u, axis=0)
    ft = np.array([-lse(-beta * t * x - den) for t in targets])
    W = np.exp(f[:, None] - u - den[None, :])   # (K, Ntot) normalised weights: sum over samples of W_k = 1
    O = W @ (W.T * N[None, :])   # overlap matrix O_ij = sum_n W_in W_jn N_j
    return ft, O


def analyse(d, tag, stride, gauss_only=False):
    from tdi import analyze
    st = json.load(open(os.path.join(d, f"{tag}_stage0.json"))); A = st["args"]
    lam, w, gsub = analyze.rule_nodes(dict(nodes=A["nodes"], power=A["power"], rule=A.get("rule") or "gl", ngauss=A.get("ngauss") or 0))
    if gauss_only:   # Gauss-Kronrod run: use only its Gauss values (the first ngauss node files), as a run without the extension would have
        if gsub is None: return None, "not a Gauss-Kronrod run"
        lam, w = gsub
    ser = []
    for k in range(len(lam)):
        f = os.path.join(d, f"{tag}_node{k}.npy")
        if not os.path.exists(f): return None, f"node {k} missing"
        ser.append(np.load(f))
    n = min(len(s) for s in ser)
    if n < 50: return None, f"only {n} samples"
    ser = [s[:n] for s in ser]; beta = 1 / (KB * st["T"]); per = st["N"]; o = np.argsort(lam); lam, w, ser = np.asarray(lam)[o], np.asarray(w)[o], [ser[i] for i in o]
    ti = float(sum(wk * s.mean() for wk, s in zip(w, ser)))
    def est(chunks):
        ft, O = mbar(lam, [c[::stride] for c in chunks], beta, [0.0, 1.0]); return (ft[1] - ft[0]) / beta, O
    dF, O = est(ser)
    blocks = [est([np.array_split(s, 5)[b] for s in ser])[0] for b in range(5)]
    tib = [float(sum(wk * np.array_split(s, 5)[b].mean() for wk, s in zip(w, ser))) for b in range(5)]
    return dict(tag=tag, T=st["T"], nodes=len(lam), rule=A.get("rule") or "gl", samples=n, per=per, phase=st["phase"],
                TI=ti / per * 1e3, TI_err=np.std(tib, ddof=1) / np.sqrt(5) / per * 1e3, MBAR=dF / per * 1e3, MBAR_err=np.std(blocks, ddof=1) / np.sqrt(5) / per * 1e3,
                overlap_min=float(min(O[k, k + 1] for k in range(len(lam) - 1))), overlap=[round(float(O[k, k + 1]), 3) for k in range(len(lam) - 1)]), ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("tags", nargs="*"); ap.add_argument("--glob", default=None); ap.add_argument("--gauss-only", action="store_true", help="Gauss-Kronrod runs: analyse only the Gauss subset (tests how well MBAR / TI do with the coarse grid)")
    ap.add_argument("--stride", type=int, default=10, help="use every k-th sample (MBAR cost)")
    a = ap.parse_args()
    tags = list(a.tags) + ([os.path.basename(f)[:-len("_stage0.json")] for f in sorted(glob.glob(os.path.join(a.dir, (a.glob or "") + "_stage0.json")))] if a.glob else [])
    print(f"{'tag':42s} {'nodes':>5s} {'samples':>7s} {'TI':>10s} {'+-':>6s} {'MBAR':>10s} {'+-':>6s} {'MBAR-TI':>8s} {'min overlap':>11s}   (meV per added H for insertion, per atom otherwise)")
    for t in tags:
        r, why = analyse(a.dir, t, a.stride, a.gauss_only)
        if r is None: print(f"{t:42s} skipped: {why}"); continue
        flag = "  <- poor overlap: MBAR unreliable across some gap" if r["overlap_min"] < 0.03 else ""
        print(f"{t:42s} {r['nodes']:5d} {r['samples']:7d} {r['TI']:10.1f} {r['TI_err']:6.1f} {r['MBAR']:10.1f} {r['MBAR_err']:6.1f} {r['MBAR'] - r['TI']:8.1f} {r['overlap_min']:11.3f}{flag}")
        if flag: print(f"{'':42s} neighbour overlaps: {r['overlap']}")


if __name__ == "__main__":
    main()
