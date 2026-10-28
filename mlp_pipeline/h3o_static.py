"""Static (0 K) enthalpy check of Huang's H3O with an MLP (default the fine-tuned GRACE): a benchmark outside the training data ("unknown land", like hcp).
At each pressure, constant-pressure relaxation (cell + positions, FrechetCellFilter, FIRE) of
  (1) Huang's H3O (docs/data/h3o_huang.cell: 16 O + 32 framework H + 16 guest H in channels),
  (2) ice X (Pn-3m H2O, 2x2x2 cell),
  (3) solid-H candidates: molecular hcp H2, atomic fcc and bcc, Cs-IV (beta-Sn type, I41/amd); the lowest enthalpy per atom is the H reference,
  (4) H3O with guest H removed in whole pairs (H2.75O, H2.5O, H2.25O): can the channels be partly occupied?
Output: H per atom of everything; dH_f(x) = [H(H(2+n)O) - nO h(H2O) - n_extra h(H)] / atoms vs X = n/(n+1) (Huang's axis), and the H3O enthalpy
difference H(P) - H(450 GPa) per cell next to the user's CASTEP values (docs/data/h3o_castep_enthalpy.csv): equal differences = the MLP reproduces
the DFT equation of state of H3O (dH/dP = V), whatever the absolute energy reference.
Caveat: the MLP was trained on hot superionic / fluid configurations, not on cold molecular or atomic H solids: the H reference is the weak link.
These are STATIC enthalpies (no zero-point energy); Huang's published dH(P) curve includes ZPE, so compare with a static DFT calculation of the same references.
usage (repo root, GPU node):  ~/grace-venv/bin/python mlp_pipeline/h3o_static.py [--model grace-ft:runs/grace/ft_m2] [--P 300 400 450 500 600 700 800] [--parallel 7] [-o runs/h3o_static]
"""
import argparse, csv, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", message="logm result may be inaccurate")   # FrechetCellFilter's matrix log on near-identity cells

EV_A3_GPA = 160.21766
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def read_cell(path):
    """lattice_cart + positions_frac blocks of a CASTEP .cell file -> ase.Atoms"""
    from ase import Atoms
    lines = [l.split("#")[0].strip() for l in open(path)]
    def block(name):
        i = next(k for k, l in enumerate(lines) if l.lower().startswith(f"%block {name}"))
        out = []
        for l in lines[i + 1:]:
            if l.lower().startswith("%endblock"): break
            if l: out.append(l.split())
        return out
    cell = np.array([[float(x) for x in r[:3]] for r in block("lattice_cart") if len(r) >= 3 and r[0].lower() not in ("ang", "bohr")])
    pos = block("positions_frac")
    return Atoms([r[0] for r in pos], scaled_positions=[[float(x) for x in r[1:4]] for r in pos], cell=cell, pbc=True)


def ice_x(vpa=2.3):
    from ase import Atoms
    a = (6 * vpa) ** (1 / 3)
    at = Atoms("O2H4", scaled_positions=[(0, 0, 0), (.5, .5, .5), (.25, .25, .25), (.75, .75, .25), (.75, .25, .75), (.25, .75, .75)], cell=[a, a, a], pbc=True)
    return at.repeat((2, 2, 2))


def h_candidates(vpa=1.3):
    from ase import Atoms
    from ase.build import bulk
    out = {}
    a = (2 * vpa * 2) ** (1 / 3)   # hcp of H2 molecules (2 molecules per cell), bond 0.74 A along c
    hc = Atoms("H4", cell=[[a, 0, 0], [-a / 2, a * 3 ** .5 / 2, 0], [0, 0, a * 1.63]], pbc=True,
               scaled_positions=[(1 / 3, 2 / 3, .25 - 0.37 / (a * 1.63)), (1 / 3, 2 / 3, .25 + 0.37 / (a * 1.63)), (2 / 3, 1 / 3, .75 - 0.37 / (a * 1.63)), (2 / 3, 1 / 3, .75 + 0.37 / (a * 1.63))])
    out["H2 hcp (molecular)"] = hc.repeat((3, 3, 2))
    out["H fcc"] = bulk("H", "fcc", a=(4 * vpa) ** (1 / 3), cubic=True).repeat((3, 3, 3))
    out["H bcc"] = bulk("H", "bcc", a=(2 * vpa) ** (1 / 3), cubic=True).repeat((3, 3, 3))
    ca = 2.5; aa = (4 * vpa / ca) ** (1 / 3)   # Cs-IV / beta-Sn (A5), 4 atoms per conventional cell
    cs = Atoms("H4", cell=[aa, aa, aa * ca], pbc=True, scaled_positions=[(0, 0, 0), (0, .5, .25), (.5, .5, .5), (.5, 0, .75)])
    out["H Cs-IV (I41/amd)"] = cs.repeat((3, 3, 1))
    return out


def guest_pairs(at):
    """indices of guest H (nearest O farther than 1.3 A), grouped in nearest-neighbour pairs"""
    from ase.geometry import get_distances
    O = np.where(at.numbers == 8)[0]; H = np.where(at.numbers == 1)[0]
    _, d = get_distances(at.positions[H], at.positions[O], cell=at.cell, pbc=True); g = H[d.min(1) > 1.3]
    _, dd = get_distances(at.positions[g], at.positions[g], cell=at.cell, pbc=True); np.fill_diagonal(dd, np.inf)
    pairs, used = [], set()
    for i in np.argsort(dd.min(1)):
        if i in used: continue
        j = int(np.argmin(np.where([k in used for k in range(len(g))], np.inf, dd[i]))); used |= {i, j}; pairs.append((int(g[i]), int(g[j])))
    return pairs


PROG = dict(done=0, total=1, t0=None)   # whole-run progress: relaxations done / planned, start time


def guest_count(at):
    """number of H farther than 1.3 A from every O (guest / channel protons)"""
    from ase.geometry import get_distances
    O = at.positions[at.numbers == 8]; H = at.positions[at.numbers == 1]
    _, d = get_distances(H, O, cell=at.cell, pbc=True); return int((d.min(1) > 1.3).sum())


def relax(at, calc, P, steps, fmax, label="", every=25):
    """constant-pressure relaxation with a progress line every `every` steps and a summary line (verbose on purpose: a GPU relaxation can take minutes)"""
    import time
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    if PROG["t0"] is None: PROG["t0"] = time.time()
    at = at.copy(); at.calc = calc; t = time.time(); N = len(at)
    f = FrechetCellFilter(at, scalar_pressure=P / EV_A3_GPA); opt = FIRE(f, logfile=None)
    print(f"   [{PROG['done'] + 1}/{PROG['total']}] {label} at {P:.0f} GPa ({N} atoms): relaxing (fmax {fmax} eV/A, at most {steps} steps)", flush=True)
    def report():
        k = opt.nsteps
        if k % every == 0:
            Fm = float(np.sqrt((f.get_forces() ** 2).sum(1)).max()); h = (at.get_potential_energy() + P / EV_A3_GPA * at.get_volume()) / N
            print(f"        step {k:4d}  H {h:11.5f} eV/atom  max force {Fm:8.4f}  V {at.get_volume() / N:.4f} A^3/atom  {time.time() - t:6.1f} s", flush=True)
    opt.attach(report, interval=1)
    conv = opt.run(fmax=fmax, steps=steps)
    E = at.get_potential_energy(); V = at.get_volume(); h = (E + P / EV_A3_GPA * V) / N; Fm = float(np.sqrt((f.get_forces() ** 2).sum(1)).max())
    PROG["done"] += 1; el = time.time() - PROG["t0"]; eta = el / PROG["done"] * (PROG["total"] - PROG["done"])
    print(f"        done: {opt.nsteps} steps, {'converged' if conv else 'NOT converged (raise --steps)'} (max force {Fm:.3f}), H {h:.5f} eV/atom, V {V / N:.4f} A^3/atom, {time.time() - t:.0f} s"
          f"  | run: {PROG['done']}/{PROG['total']} relaxations, {el / 60:.1f} min so far, ETA ~{eta / 60:.0f} min", flush=True)
    return at, h, V / N


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="grace-ft:runs/grace/ft_m2"); ap.add_argument("--device", default="cuda")
    ap.add_argument("--P", type=float, nargs="+", default=[300, 400, 450, 500, 600, 700, 800]); ap.add_argument("--cell", default="docs/data/h3o_huang.cell")
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--fmax", type=float, default=0.01); ap.add_argument("-o", default="runs/h3o_static")
    ap.add_argument("--parallel", type=int, default=1, help="pressures run as separate processes sharing the GPU, at most this many at once (~4.5 GB GPU memory each)"); a = ap.parse_args()
    if a.parallel > 1 and len(a.P) > 1:
        return run_parallel(a)
    from importlib import import_module
    calc = import_module("04_benchmark").get_calc_spec(a.model, a.device)
    os.makedirs(a.o, exist_ok=True)
    h3o = read_cell(a.cell); pairs = guest_pairs(h3o); nO = int((h3o.numbers == 8).sum())
    print(f"H3O cell: {h3o.get_chemical_formula()}, {len(pairs)} guest pairs; MLP: {a.model}")
    variants = {}
    for k in (0, 2, 4, 6):   # remove k guest pairs, spread over the list
        rm = sorted({i for p in pairs[::max(1, len(pairs) // k)][:k] for i in p}) if k else []
        v = h3o.copy(); del v[rm]; n = (int((v.numbers == 1).sum()) - 2 * nO) / nO; variants[n] = v
    rows = []; PROG["total"] = len(a.P) * (1 + len(h_candidates()) + len(variants))
    print(f"plan: {len(a.P)} pressures x ({1 + len(h_candidates())} references + {len(variants)} H3O variants) = {PROG['total']} relaxations", flush=True)
    for P in a.P:
        _, hw, vw = relax(ice_x(), calc, P, a.steps, a.fmax, "ice X")
        hs = {name: relax(c, calc, P, a.steps, a.fmax, name)[1:] for name, c in h_candidates().items()}
        best = min(hs, key=lambda k: hs[k][0]); hH = hs[best][0]
        print(f"\n{P:.0f} GPa: ice X h {hw:.4f} eV/atom (V {vw:.3f}); H reference = {best} ({hH:.4f} eV/atom; " + ", ".join(f"{k} {v[0]:.4f}" for k, v in hs.items()) + ")")
        for n, v in sorted(variants.items()):
            r, h, vv = relax(v, calc, P, a.steps, a.fmax, f"H{2 + n:.3g}O")
            nH = int((r.numbers == 1).sum()); extra = nH - 2 * nO; N = len(r)
            dH = (h * N - nO * 3 * hw - extra * hH) / N * 1e3
            rows.append(dict(P_GPa=P, H_per_O=2 + n, X=n / (n + 1), dH_meV_atom=dH, H_eV_cell=h * N, natoms=N, vpa=vv, H_ref=best))
            print(f"   H{2 + n:.3g}O (X {n / (n + 1):.3f}): dH_f {dH:7.1f} meV/atom   V/atom {vv:.3f}   H_cell {h * N:.4f} eV")
            r.write(os.path.join(a.o, f"H{2 + n:.3g}O_{P:.0f}GPa_relaxed.extxyz"))   # look at them: partly filled channels may collapse / rearrange
            ng = guest_count(r); v0 = v.get_volume() / len(v)
            print(f"      structure: {ng} of {extra} extra H still in channels (> 1.3 A from every O), V/atom {v0:.3f} -> {vv:.3f}; saved H{2 + n:.3g}O_{P:.0f}GPa_relaxed.extxyz", flush=True)
    with open(os.path.join(a.o, "h3o_static.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    eos_table(rows, a.o)


def eos_table(rows, out):
    cast = {float(r["P_GPa"]): float(r["H_eV_cell"]) for r in csv.DictReader(l for l in open("docs/data/h3o_castep_enthalpy.csv") if not l.startswith("#"))}
    ml = {r["P_GPa"]: r["H_eV_cell"] for r in rows if abs(r["H_per_O"] - 3) < 1e-9}
    if 450 in cast and 450 in ml:
        print("\nH3O equation of state: H(P) - H(450 GPa) per 64-atom cell, eV (equal = the MLP reproduces the DFT dH/dP = V)")
        for P in sorted(ml):
            if P in cast: print(f"   {P:5.0f} GPa   MLP {ml[P] - ml[450]:9.3f}   CASTEP {cast[P] - cast[450]:9.3f}   diff {(ml[P] - ml[450]) - (cast[P] - cast[450]):7.3f}")
    print("\nCaveat: the MLP never saw cold solid H; the H reference (and so the absolute dH_f) is the least trustworthy part. Wrote", os.path.join(out, "h3o_static.csv"))


def run_parallel(a):
    """one child process per pressure (output <o>/P<P>), at most a.parallel at once; their lines are streamed with a [P GPa] prefix; results merged at the end"""
    import subprocess, threading, time
    base = [sys.executable, "-u", os.path.abspath(__file__), "--model", a.model, "--device", a.device, "--cell", a.cell, "--steps", str(a.steps), "--fmax", str(a.fmax)]
    quiet = ("tensorflow", "xla", "cuda", "I0000", "W0000", "E0000", "ptxas", "absl", "cpu_feature", "AVX2", "Registers are spilled", "XLA service", "StreamExecutor", "rebuild TensorFlow")
    def pump(proc, tag):
        for line in proc.stdout:
            if line.strip() and not any(q.lower() in line.lower() for q in quiet): print(f"[{tag:>4s} GPa] {line.rstrip()}", flush=True)
    todo = list(a.P); running = []; t0 = time.time()
    print(f"parallel: {len(todo)} pressures, at most {a.parallel} processes on the GPU at once", flush=True)
    while todo or running:
        while todo and len(running) < a.parallel:
            P = todo.pop(0); out = os.path.join(a.o, f"P{P:.0f}")
            pr = subprocess.Popen(base + ["--P", str(P), "-o", out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            th = threading.Thread(target=pump, args=(pr, f"{P:.0f}"), daemon=True); th.start(); running.append((P, pr, th))
            print(f"== started {P:.0f} GPa ({len(running)} running, {len(todo)} waiting, {(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(5)
        for P, pr, th in list(running):
            if pr.poll() is not None:
                th.join(); running.remove((P, pr, th))
                print(f"== finished {P:.0f} GPa (exit {pr.returncode}; {len(running)} running, {len(todo)} waiting, {(time.time() - t0) / 60:.1f} min)", flush=True)
    rows = []
    for P in a.P:
        f = os.path.join(a.o, f"P{P:.0f}", "h3o_static.csv")
        if os.path.exists(f):
            for r in csv.DictReader(open(f)):
                rows.append({k: (float(v) if k not in ("H_ref",) else v) for k, v in r.items()})
    if not rows: print("no results"); return
    os.makedirs(a.o, exist_ok=True)
    with open(os.path.join(a.o, "h3o_static.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nformation dH (meV/atom) vs composition, all pressures:")
    for P in sorted({r["P_GPa"] for r in rows}):
        print(f"   {P:5.0f} GPa  " + "  ".join(f"H{r['H_per_O']:.3g}O {r['dH_meV_atom']:7.1f}" for r in sorted((r for r in rows if r["P_GPa"] == P), key=lambda r: r["H_per_O"])) + f"   (H ref: {[r for r in rows if r['P_GPa'] == P][0]['H_ref']})")
    eos_table(rows, a.o)


if __name__ == "__main__":
    main()
