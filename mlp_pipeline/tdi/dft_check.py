"""DFT checks of the MLP where the insertion free energy is sensitive: (1) the energy of an extra proton approaching a host H or O (probe scans in real host frames),
(2) free-energy perturbation (FEP) from the MLP to DFT for mu_H - G_H. The VASP settings are copied from a TEMPLATE directory of the original training DFT (INCAR, KPOINTS,
POTCAR, POSCAR for the species order) so the energies are those the model was trained on (free energy TOTEN); only single-point tags are changed.

  python mlp_pipeline/tdi/dft_check.py prep-probe runs/tdi_long/insert_500GPa_6000K_n1 --template <dft run dir> [--frames 2 --dirs 2 --elements H] -o runs/dft_check/probe
  python mlp_pipeline/tdi/dft_check.py prep-fep   runs/tdi_long/insert_500GPa_6000K_n1 --h-run runs/tdi_long/H_500GPa_6000K_N250 --template <dir> [--frames 20] -o runs/dft_check/fep
  sbatch -p <cpu partition> --array=0-<n-1> mlp_pipeline/young_dft_check.sbatch runs/dft_check/probe      (n = lines of runs/dft_check/probe/dirs.txt; or run the dirs by hand)
  python mlp_pipeline/tdi/dft_check.py prep-static runs/h3o_static --template <dft run dir> --P 300 400 500 600 [--scales 0.98 1.0 1.02] [--tel 1000] [--kspacing 0.25] -o runs/dft_check/static
  python mlp_pipeline/tdi/dft_check.py prep-selftest --template <dft run dir> -o runs/dft_check/selftest        (a 12-atom cell and the template geometry as single points: does the VASP chain work on Young CPUs, what does a real-size cell cost? --size small|template|both)
  python mlp_pipeline/tdi/dft_check.py prep-widom runs/pwidom/pw_fcc_450GPa_6000K_frames.npz --template <dir> --proxy docs/data/toyfit_t4.json [--model <mlp> --top 50] -o runs/dft_check/widom
  python mlp_pipeline/tdi/dft_check.py status runs/dft_check/probe       (any prepared directory: pending / running / done / failed per single point, SCF progress, mean time, ETA)
  ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py analyze runs/dft_check/probe     (and .../fep, .../static, .../selftest, .../widom): reads OUTCARs, evaluates the MLP on the same structures, prints the comparison

probe: per frame, the host alone plus an extra H at r = 0.5 ... 1.2 A from a host H (or O) along the direction that keeps every OTHER atom farthest away, so dE(r) = E(host + H) - E(host)
       measures the H-H (O-H) interaction: compare DFT and MLP curves (the MLP's flat H-H region at 0.6-1.0 A, docs/CURRENT_ISSUES.md P2).
selftest: the template's own POSCAR as one single point with the single-point tags: checks modules / POTCAR / INCAR on a CPU node and gives the wall time and SCF steps of the real cell size (the cost basis of every other mode);
       analyze prints TOTEN, ranks, SCF steps, seconds per SCF step, pressure and the MLP energy per atom of the same structure.
widom: DFT test of the insertion energies pwidom.py relies on. From the cached host frames of a pwidom run (<tag>_frames.npz) and the same proxy sampler as pwidom, per frame: the host, H at --sel sites drawn with the
       Rosenbluth weights exp(-phi/kT) (tier prox), H at the lowest-proxy-energy candidate (tier lowphi) and, with --model, H at the candidate with the lowest MODEL energy among the --top lowest-phi ones (tier mlpmin: the
       sites a too-attractive model would over-weight in the Widom average). analyze: dE = E(host + H) - E(host), DFT vs MLP per tier, so a model error that matters for mu_ex shows up where the weight is.
static: DFT (VASP, the user's template settings except smearing / k-points) of the RELAXED structures that mlp_pipeline/h3o_static.py saved (runs/h3o_static/P<P>/*_relaxed.extxyz: Huang's H3O and its
       partly filled variants, ice X, the solid-H candidates), at --scales linear cell scalings (3 volumes -> E(V) fit -> enthalpy at the target pressure for DFT and for the MLP alike, so the volume
       mismatch drops out), then the same formation enthalpy dH_f(H2+nO) = [H(cell) - nO H(ice X per H2O) - extra H(H reference)] / N as h3o_static.py, with DFT and with the MLP, next to Huang's static (SI) and
       ZPE-included curves, plus the H-reference-free check against the user's CASTEP enthalpies. The cells are small (36-64 atoms): the k-mesh comes from --kspacing, not from the MD template.
fep:   Delta = E_DFT - E_MLP per structure. F_DFT - F_MLP of an ensemble = -kT ln < exp(-Delta / kT) >_MLP (also its 1st and 2nd cumulant).
       mu_H:  host + n H frames (largest-lambda node) minus host frames (smallest-lambda node: the ghost is decoupled, the host unperturbed), per added H; also the paired estimate on the
       same geometries with the ghost removed (lower noise, first order). G_H: pure-H frames (largest-lambda node of the H run), per atom. Result: the DFT correction to mu_H - G_H.
       Per-element constant offsets between DFT and MLP cancel in mu_H - G_H (one H in both). The largest-lambda node is lambda ~0.95-0.99, not exactly 1 (stated in the output)."""
import argparse, csv, glob, json, os, re, shutil, sys, time
import numpy as np
from ase import Atoms
from ase.io import read, write
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze as tdi_analyze

KB = 8.617333262e-5
MD_TAGS = {"IBRION", "NSW", "POTIM", "TEBEG", "TEEND", "SMASS", "MDALGO", "ANDERSEN_PROB", "LANGEVIN_GAMMA", "LANGEVIN_GAMMA_L", "ISIF", "NBLOCK", "KBLOCK",
           "PSTRESS", "ISYM", "LWAVE", "LCHARG", "NWRITE", "ISTART", "ICHARG", "NBANDS", "SIGMA", "ML_LMLFF", "ML_MODE", "ML_ISTART"}


def node_frames(run, which, n, keep_last_half=True):
    st = json.load(open(run + "_stage0.json")); lam = tdi_analyze.rule_nodes(st["args"])[0]
    k = int(np.argmax(lam) if which == "max" else np.argmin(lam)); fr = read(f"{run}_node{k}_traj.extxyz", index=":")
    if keep_last_half:
        fr = fr[len(fr) // 2:]
    idx = np.linspace(0, len(fr) - 1, min(n, len(fr))).round().astype(int)
    return st, float(lam[k]), [fr[i] for i in idx]


def template(tdir):
    """INCAR dict (order kept), KPOINTS text, POTCAR path, species order, ZVAL per species, NBANDS and electron count of the template system"""
    inc = {}
    for line in open(os.path.join(tdir, "INCAR")):
        line = line.split("#")[0].split("!")[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1); inc[k.strip().upper()] = v.strip()
    pot = os.path.join(tdir, "POTCAR"); txt = open(pot).read()
    species = [m.split()[0].split("_")[0] for m in re.findall(r"TITEL\s*=\s*PAW\S*\s+(\S+)", txt)]
    zval = [float(z) for z in re.findall(r"ZVAL\s*=\s*([\d.]+)", txt)]
    blocks = [b for b in re.split(r"(?<=End of Dataset\n)", txt) if b.strip()]   # one POTCAR block per species, in order
    if len(blocks) != len(species):
        raise SystemExit(f"{pot}: {len(blocks)} 'End of Dataset' blocks for species {species}")
    pos = None
    for name in ("POSCAR", "CONTCAR"):
        if os.path.exists(os.path.join(tdir, name)):
            pos = read(os.path.join(tdir, name), format="vasp"); break
    nel_t = sum(dict(zip(species, zval))[s] for s in pos.get_chemical_symbols()) if pos is not None else None
    return dict(incar=inc, kpoints=open(os.path.join(tdir, "KPOINTS")).read(), potcar=pot, species=species, zval=dict(zip(species, zval)), blocks=dict(zip(species, blocks)),
                nbands=int(inc["NBANDS"]) if "NBANDS" in inc else None, nel=nel_t)


def kmesh_text(cell, spacing):
    """Gamma-centred automatic mesh with n_i = ceil(2 pi |b_i| / spacing) (spacing in 1/A; |b_i| = reciprocal vector lengths without the 2 pi)"""
    n = [max(1, int(np.ceil(2 * np.pi * np.linalg.norm(b) / spacing - 1e-9))) for b in np.asarray(cell.reciprocal())]
    return f"static single point: Gamma-centred mesh, spacing {spacing} 1/A\n0\nGamma\n {n[0]} {n[1]} {n[2]}\n 0 0 0\n"


def write_dft(d, atoms, T, tp, label, over=None):
    """one single-point directory: atoms sorted to the POTCAR species order, INCAR = template minus MD tags plus single-point tags.
    over (static mode): dict(ismear=-1, sigma=eV, kspacing=1/A) replaces the template smearing and k-points (the MD template is Gamma-only for 250-atom cells)"""
    os.makedirs(d, exist_ok=True)
    order = [i for s in tp["species"] for i in np.where(np.array(atoms.get_chemical_symbols()) == s)[0]]
    if len(order) != len(atoms):
        raise SystemExit(f"{label}: species {set(atoms.get_chemical_symbols())} not all in the template POTCAR {tp['species']}")
    at = atoms[order]; at.wrap(); write(os.path.join(d, "POSCAR"), at, format="vasp", direct=True, sort=False)
    inc = {k: v for k, v in tp["incar"].items() if k not in MD_TAGS}
    inc.update(IBRION="-1", NSW="0", ISYM="0", LWAVE=".FALSE.", LCHARG=".FALSE.")
    ismear = int(float(tp["incar"].get("ISMEAR", "1")))
    inc["SIGMA"] = f"{KB * T:.6f}" if ismear == -1 else tp["incar"].get("SIGMA", "0.2")   # Fermi smearing = electronic temperature of the MD (as in the training AIMD)
    if over:
        inc["ISMEAR"] = str(over["ismear"]); inc["SIGMA"] = f"{over['sigma']:.6f}"
        for k in ("KSPACING", "KGAMMA"): inc.pop(k, None)   # KSPACING in the INCAR would override our KPOINTS file
    nel = sum(tp["zval"][s] for s in at.get_chemical_symbols())
    if tp["nbands"] and tp["nel"]:
        inc["NBANDS"] = str(int(np.ceil(tp["nbands"] * nel / tp["nel"] / 8) * 8))   # template bands per electron, rounded up to a multiple of 8
    with open(os.path.join(d, "INCAR"), "w") as fh:
        fh.write(f"# {label}: single point (template {os.path.basename(os.path.dirname(tp['potcar']))}, MD tags removed)\n" + "".join(f"{k} = {v}\n" for k, v in inc.items()))
    present = [s_ for s_ in tp["species"] if s_ in set(at.get_chemical_symbols())]   # POTCAR = blocks of the species present, in POSCAR order (pure H: H only)
    open(os.path.join(d, "KPOINTS"), "w").write(kmesh_text(at.cell, over["kspacing"]) if over else tp["kpoints"]); open(os.path.join(d, "POTCAR"), "w").write("".join(tp["blocks"][s_] for s_ in present))
    write(os.path.join(d, "structure.extxyz"), at)   # what the MLP is evaluated on (same order)
    return d


def clear_direction(p, others, cell, rs, rng, tries=400):
    """unit vector from p along which an extra atom at every r in rs stays farthest from all atoms in `others` (minimum image)"""
    L = np.diag(cell); best, bv = -1.0, None
    for _ in range(tries):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        q = p[None, :] + np.outer(rs, v); dd = q[:, None, :] - others[None, :, :]; dd -= L * np.round(dd / L)
        m = np.sqrt((dd ** 2).sum(-1)).min()
        if m > best:
            best, bv = m, v
    return bv, best


def prep_probe(a):
    tp = template(a.template); st, lk, frames = node_frames(a.run, "max", a.frames); ng = int(st["N"]) if st["phase"] == "insert" else 0
    rng = np.random.default_rng(a.seed); rs = np.array(a.r); rows = []; dirs = []
    for f, fr in enumerate(frames):
        host = fr[:len(fr) - ng] if ng else fr.copy(); cell = host.cell.array
        if not np.allclose(cell, np.diag(np.diag(cell))):
            raise SystemExit("probe directions assume an orthorhombic cell")
        dirs.append(write_dft(os.path.join(a.o, f"f{f}_host"), host, st["T"], tp, f"frame {f} host")); rows.append(dict(dir=dirs[-1], frame=f, kind="host"))
        for el in a.elements:
            z = 1 if el == "H" else 8
            for j in range(a.dirs):
                i = int(rng.choice(np.where(host.numbers == z)[0])); others = np.delete(host.positions, i, axis=0)
                v, clear = clear_direction(host.positions[i], others, cell, rs, rng)
                for r in rs:
                    b = host + Atoms("H", positions=[host.positions[i] + r * v])
                    d = write_dft(os.path.join(a.o, f"f{f}_{el}{j}_r{r:.2f}"), b, st["T"], tp, f"frame {f}, extra H at {r:.2f} A from host {el} {i}")
                    dirs.append(d); rows.append(dict(dir=d, frame=f, kind="probe", el=el, j=j, atom=i, r=float(r), clearance=float(clear)))
    finish(a, rows, dirs, dict(mode="probe", run=a.run, T=st["T"], lam=lk))


def prep_fep(a):
    tp = template(a.template); rows = []; dirs = []
    st, lk, full = node_frames(a.run, "max", a.frames); ng = int(st["N"])
    _, l0, host0 = node_frames(a.run, "min", a.frames)
    for f, fr in enumerate(full):
        dirs.append(write_dft(os.path.join(a.o, f"ins_full_{f}"), fr, st["T"], tp, "host + n H")); rows.append(dict(dir=dirs[-1], kind="full", frame=f))
        dirs.append(write_dft(os.path.join(a.o, f"ins_pair_{f}"), fr[:len(fr) - ng], st["T"], tp, "same frame, added H removed")); rows.append(dict(dir=dirs[-1], kind="pair", frame=f))
    for f, fr in enumerate(host0):
        dirs.append(write_dft(os.path.join(a.o, f"ins_host_{f}"), fr[:len(fr) - ng], st["T"], tp, "host ensemble (smallest lambda)")); rows.append(dict(dir=dirs[-1], kind="host", frame=f))
    meta = dict(mode="fep", run=a.run, T=st["T"], n=ng, lam_full=lk, lam_host=l0)
    if a.h_run:
        sh, lh, hfr = node_frames(a.h_run, "max", a.frames)
        for f, fr in enumerate(hfr):
            dirs.append(write_dft(os.path.join(a.o, f"pureH_{f}"), fr, sh["T"], tp, "pure H")); rows.append(dict(dir=dirs[-1], kind="pureH", frame=f))
        meta.update(h_run=a.h_run, lam_H=lh, N_H=int(sh["N"]))
    finish(a, rows, dirs, meta)


def finish(a, rows, dirs, meta):
    os.makedirs(a.o, exist_ok=True)
    json.dump(dict(meta, rows=rows), open(os.path.join(a.o, "manifest.json"), "w"), indent=1)
    open(os.path.join(a.o, "dirs.txt"), "w").write("".join(d + "\n" for d in dirs))
    print(f"{len(dirs)} single-point directories in {a.o} (list: {a.o}/dirs.txt). Young CPU nodes (SGE), from the repo root, 16 ranks per directory, CHUNK directories per task:\n"
          f"  CHUNK=1 qsub -V -P Gold -A UKCP_ED_P -pe smp 16 -l mem=4G,h_rt=6:00:00 -t 1-{len(dirs)} -tc 10 mlp_pipeline/young_dft_check.sge {a.o}\n"
          f"  progress:  ~/grace-venv-cpu/bin/python mlp_pipeline/tdi/dft_check.py status {a.o}")


def toten(d):
    """VASP free energy TOTEN (the training label) of a finished single point; None if missing / unfinished / not converged"""
    f = os.path.join(d, "OUTCAR")
    if not os.path.exists(f):
        return None
    txt = open(f, errors="replace").read()
    m = re.findall(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)", txt)
    if not m or "General timing" not in txt:
        return None
    if "aborting loop because EDIFF is reached" not in txt:
        print(f"WARNING {d}: electronic loop did not reach EDIFF (NELM hit): energy unreliable")
    return float(m[-1])


def fe_shift(D, T):
    """F_DFT - F_MLP from Delta = E_DFT - E_MLP over an MLP ensemble: exponential average, 1st and 2nd cumulant; bootstrap error of the exponential one"""
    D = np.asarray(D, float); kt = KB * T; m = D.mean()
    expo = lambda x: m - kt * np.log(np.mean(np.exp(-(x - m) / kt)))
    rng = np.random.default_rng(0); bs = [expo(rng.choice(D, len(D))) for _ in range(400)]
    return dict(exp=expo(D), exp_err=float(np.std(bs)), c1=m, c2=m - D.var(ddof=1) / (2 * kt), sd=D.std(ddof=1), n=len(D))


def do_analyze(a):
    man = json.load(open(os.path.join(a.o, "manifest.json"))); rows = man["rows"]
    if man["mode"] == "static":   # no TDI run behind it: structures of h3o_static.py
        return analyze_static(a, man, rows)
    if man["mode"] == "selftest":
        return analyze_selftest(a, man, rows)
    if man["mode"] == "widom":
        return analyze_widom(a, man, rows)
    st = json.load(open(man["run"] + "_stage0.json")); model = a.model or st["args"]["model"]
    from importlib import import_module
    calc = import_module("04_benchmark").get_calc_spec(model, a.device)
    for r in rows:
        r["E_dft"] = toten(r["dir"])
        at = read(os.path.join(r["dir"], "structure.extxyz")); at.calc = calc; r["E_mlp"] = float(at.get_potential_energy())
    done = sum(r["E_dft"] is not None for r in rows); print(f"{a.o}: {done} of {len(rows)} DFT single points finished; model {model}")
    if man["mode"] == "probe":
        host = {r["frame"]: r for r in rows if r["kind"] == "host"}
        print("extra H at r from a host atom: dE = E(host + H) - E(host), eV (DFT | MLP | MLP - DFT); clearance = closest OTHER atom to the probe line")
        for key in sorted({(r["frame"], r["el"], r["j"]) for r in rows if r["kind"] == "probe"}):
            h = host[key[0]]; pr = sorted([r for r in rows if r["kind"] == "probe" and (r["frame"], r["el"], r["j"]) == key], key=lambda r: r["r"])
            print(f"\n frame {key[0]}, host {key[1]} atom {pr[0]['atom']} (direction {key[2]}, clearance {pr[0]['clearance']:.2f} A)")
            for r in pr:
                dd = None if (r["E_dft"] is None or h["E_dft"] is None) else r["E_dft"] - h["E_dft"]; dm = r["E_mlp"] - h["E_mlp"]
                print(f"   r {r['r']:.2f}  DFT {'   n/a  ' if dd is None else f'{dd:8.3f}'}  MLP {dm:8.3f}  " + ("" if dd is None else f"diff {dm - dd:+7.3f}"))
        print("\nreading: the MLP is fine where 'diff' stays within a few tenths of an eV of its value at 1.0-1.2 A (a constant per-H offset cancels in mu_H - G_H);"
              " a diff that grows as r shrinks means the MLP's short-range H-H (O-H) energy is wrong there")
        return
    T = man["T"]; n = man["n"]; ok = lambda k: [r for r in rows if r["kind"] == k and r["E_dft"] is not None]
    D = {k: [r["E_dft"] - r["E_mlp"] for r in ok(k)] for k in ("full", "pair", "host", "pureH")}
    if min(len(D["full"]), len(D["host"])) < 3:
        print("too few finished single points for the FEP estimate"); return
    sf, sh = fe_shift(D["full"], T), fe_shift(D["host"], T)
    print(f"\nDelta = E_DFT - E_MLP (eV, whole cell): host + {n} H (lambda {man['lam_full']:.3f}): mean {sf['c1']:.3f} sd {sf['sd']:.3f} (n {sf['n']}); host (lambda {man['lam_host']:.1e}): mean {sh['c1']:.3f} sd {sh['sd']:.3f} (n {sh['n']})")
    dmu = (sf["exp"] - sh["exp"]) / n; dmu_e = np.hypot(sf["exp_err"], sh["exp_err"]) / n
    print(f"mu_H(DFT) - mu_H(MLP) per added H: {dmu * 1e3:+.0f} +- {dmu_e * 1e3:.0f} meV (exponential average; 1st cumulant {(sf['c1'] - sh['c1']) / n * 1e3:+.0f}, 2nd {(sf['c2'] - sh['c2']) / n * 1e3:+.0f})")
    pairs = {r["frame"]: r for r in ok("pair")}; pd_ = [(r["E_dft"] - r["E_mlp"]) - (pairs[r["frame"]]["E_dft"] - pairs[r["frame"]]["E_mlp"]) for r in ok("full") if r["frame"] in pairs]
    if len(pd_) >= 3:
        print(f"   paired (same geometry, added H removed; first order): {np.mean(pd_) / n * 1e3:+.0f} +- {np.std(pd_, ddof=1) / np.sqrt(len(pd_)) / n * 1e3:.0f} meV (n {len(pd_)})")
    if len(D["pureH"]) >= 3:
        sH = fe_shift(D["pureH"], T); NH = man["N_H"]; dg = sH["exp"] / NH
        print(f"G_H(DFT) - G_H(MLP) per atom: {dg * 1e3:+.1f} +- {sH['exp_err'] / NH * 1e3:.1f} meV (pure H, lambda {man['lam_H']:.3f}, n {sH['n']}; Delta sd {sH['sd']:.3f} eV per cell)")
        print(f"==> DFT correction to mu_H - G_H: {(dmu - dg) * 1e3:+.0f} +- {np.hypot(dmu_e, sH['exp_err'] / NH) * 1e3:.0f} meV per added H (add to the MLP value)")
    print("caveats: exponential averages are reliable only if Delta's sd is below ~2 kT per cell; compare the cumulant estimates; frames are from lambda < 1 (nearly the full MLP).")


# ---------------------------------------------------------------- selftest / widom / status
def outcar_info(d):
    """what a (possibly unfinished) OUTCAR says: done, ranks, SCF steps, seconds per SCF step, elapsed, NBANDS, ENCUT, pressure (GPa); None if there is no OUTCAR"""
    f = os.path.join(d, "OUTCAR")
    if not os.path.exists(f): return None
    t = open(f, errors="replace").read(); g = lambda rx: (lambda m: m[-1] if m else None)(re.findall(rx, t))
    loops = [float(x) for x in re.findall(r"LOOP:\s+cpu time\s+[\d.]+: real time\s+([\d.]+)", t)]
    el = g(r"Elapsed time \(sec\):\s*([\d.]+)"); nb = g(r"NBANDS\s*=\s*(\d+)"); ec = g(r"ENCUT\s*=\s*([\d.]+)"); rk = g(r"running on\s+(\d+) total cores")
    return dict(done="General timing" in t, ranks=int(rk) if rk else None, scf=len(loops), s_scf=float(np.mean(loops)) if loops else None, elapsed=float(el) if el else None,
                nbands=int(nb) if nb else None, encut=float(ec) if ec else None, P=outcar_pressure_GPa(d))


def prep_selftest(a):
    tp = template(a.template); T = a.T or float(tp["incar"].get("TEBEG", 5000)); sets = []
    if a.size in ("small", "both"):   # 12-atom fcc H2O cell: seconds to minutes, checks modules / POTCAR / INCAR / launcher before the real-size point
        from importlib import import_module
        sets.append(("selftest_small", import_module("17_eos_md").build("fcc", a.vpa, 1, 0, 2.0), f"selftest: small fcc H2O cell, {a.vpa} A^3/atom"))
    if a.size in ("template", "both"):
        f = next(os.path.join(a.template, n) for n in ("POSCAR", "CONTCAR") if os.path.exists(os.path.join(a.template, n)))
        sets.append(("selftest_template", read(f, format="vasp"), "selftest: template geometry"))
    rows = []; dirs = []
    for name, at, label in sets:
        d = write_dft(os.path.join(a.o, name), at, T, tp, label); nel = sum(tp["zval"][s_] for s_ in at.get_chemical_symbols()); dirs.append(d); rows.append(dict(dir=d, kind="selftest", N=len(at), T=T))
        print(f"selftest {name}: {len(at)} atoms ({dict(zip(*np.unique(at.get_chemical_symbols(), return_counts=True)))}), {nel:.0f} valence electrons, electronic T {T:.0f} K, species {tp['species']}")
    print("   INCAR written (last set):\n" + "".join("      " + l for l in open(os.path.join(dirs[-1], "INCAR")).readlines()))
    finish(a, rows, dirs, dict(mode="selftest", template=a.template, T=T, model_default=a.model))


def prep_widom(a):
    from tdi import pwidom as pw
    tp = template(a.template); z = np.load(a.frames); fh = z["host"]; numbers = z["numbers"]; V = float(z["V"]); T = float(z["T"]); kT = KB * T
    L = a.L or V ** (1 / 3); cell = np.eye(3) * L
    print(f"{a.frames}: {len(fh)} host frames, {len(numbers)} atoms, T {T:.0f} K, V {V:.1f} A^3 -> cubic cell L = {L:.4f} A" + ("" if a.L else " (assumes the cubic cell of build_host; --L overrides)"))
    isH = numbers == 1; p = pw.proxy_params(a.proxy); rng = np.random.default_rng(a.seed); half = fh[len(fh) // 2:]
    frames = [half[i] for i in np.linspace(0, len(half) - 1, min(a.frames_n, len(half))).round().astype(int)]
    model = None
    if a.model: model = pw.Model(a.model, a.device, 16)
    rows = []; dirs = []; t0 = time.time()
    for f, x in enumerate(frames):
        host = Atoms(numbers=numbers, positions=x, cell=cell, pbc=True); d = write_dft(os.path.join(a.o, f"f{f}_host"), host, T, tp, f"widom frame {f} host")
        dirs.append(d); rows.append(dict(dir=d, frame=f, kind="host"))
        c = rng.random((a.K, 3)) @ cell; phi = pw.proxy_energy(p, c, x, isH, cell); w = -phi / kT; pr = np.exp(w - pw.lse(w))
        dd = c[:, None, :] - x[None, :, :]; dd -= L * np.round(dd / L); dmin = np.sqrt((dd ** 2).sum(-1)).min(1)
        picks = [("prox", j, int(k)) for j, k in enumerate(rng.choice(a.K, a.sel, p=pr))] + [("lowphi", 0, int(np.argmin(phi)))]
        if model is not None:
            top = np.argsort(phi)[:a.top]; cand = [Atoms(numbers=np.r_[numbers, 1], positions=np.vstack([x, c[k]]), cell=cell, pbc=True) for k in top]
            dU = model.energies(cand) - model.energies([host])[0]; picks.append(("mlpmin", 0, int(top[np.argmin(dU)])))
            print(f"   frame {f}: MLP dU over the {a.top} lowest-phi candidates: min {dU.min():.2f} median {np.median(dU):.2f} max {dU.max():.2f} eV", flush=True)
        for tier, j, k in picks:
            b = Atoms(numbers=np.r_[numbers, 1], positions=np.vstack([x, c[k]]), cell=cell, pbc=True)
            d = write_dft(os.path.join(a.o, f"f{f}_{tier}{j}"), b, T, tp, f"widom frame {f}, {tier} site, phi {phi[k]:.3f} eV, clearance {dmin[k]:.2f} A")
            dirs.append(d); rows.append(dict(dir=d, frame=f, kind="site", tier=tier, j=j, phi=float(phi[k]), clearance=float(dmin[k])))
        print(f"   frame {f}: {len(picks)} sites, phi {min(phi[k] for _, _, k in picks):.2f} ... {max(phi[k] for _, _, k in picks):.2f} eV, clearance {min(dmin[k] for _, _, k in picks):.2f} ... {max(dmin[k] for _, _, k in picks):.2f} A ({time.time() - t0:.0f} s)", flush=True)
    finish(a, rows, dirs, dict(mode="widom", frames=a.frames, T=T, proxy=a.proxy, model_default=a.model or "grace-ft:runs/grace/ft_m2"))


def analyze_selftest(a, man, rows):
    from importlib import import_module
    print(f"{a.o}: selftest (template {man['template']}, electronic T {man['T']:.0f} K)")
    for r in rows:
        i = outcar_info(r["dir"]); E = toten(r["dir"])
        if i is None: print(f"   {r['dir']}: no OUTCAR yet"); continue
        print(f"   {r['dir']}: {'finished' if i['done'] else 'NOT finished'}; {r['N']} atoms on {i['ranks']} ranks; NBANDS {i['nbands']}, ENCUT {i['encut']}; {i['scf']} SCF steps at {i['s_scf'] or float('nan'):.1f} s each; wall {i['elapsed'] or float('nan'):.0f} s")
        if E is None: continue
        print(f"      TOTEN {E:.5f} eV ({E / r['N']:.5f} eV/atom); external pressure {i['P'] if i['P'] is not None else float('nan'):.1f} GPa")
        try:
            calc = import_module("04_benchmark").get_calc_spec(a.model or man.get("model_default") or "grace-ft:runs/grace/ft_m2", a.device)
            at = read(os.path.join(r["dir"], "structure.extxyz")); at.calc = calc; Em = float(at.get_potential_energy())
            print(f"      MLP {Em:.5f} eV: MLP - DFT {(Em - E) / r['N'] * 1e3:+.1f} meV/atom (a per-element constant is expected; compare with the other modes)")
        except Exception as e:
            print(f"      MLP comparison skipped ({type(e).__name__}: {e})")
        if i["elapsed"]: print(f"   cost basis: one single point of this cell = {i['elapsed'] / 3600:.2f} h on {i['ranks']} ranks = {i['elapsed'] * (i['ranks'] or 1) / 3600:.1f} core-hours; 40 such points = {40 * i['elapsed'] / 3600:.0f} h at 1 per task")


def analyze_widom(a, man, rows):
    from importlib import import_module
    calc = import_module("04_benchmark").get_calc_spec(a.model or man["model_default"], a.device)
    for r in rows:
        r["E_dft"] = toten(r["dir"]); at = read(os.path.join(r["dir"], "structure.extxyz")); at.calc = calc; r["E_mlp"] = float(at.get_potential_energy())
    print(f"{a.o}: {sum(r['E_dft'] is not None for r in rows)} of {len(rows)} DFT single points finished; MLP {a.model or man['model_default']}; proxy {man['proxy']}; T {man['T']:.0f} K")
    host = {r["frame"]: r for r in rows if r["kind"] == "host"}; sites = [r for r in rows if r["kind"] == "site"]; kt = KB * man["T"]; out = {}
    print("\n frame tier     j   phi(eV) clear(A)    dE_DFT   dE_MLP   MLP-DFT   (dE = E(host + H) - E(host), eV)")
    for r in sorted(sites, key=lambda r: (r["frame"], r["tier"], r["j"])):
        h = host[r["frame"]]; dm = r["E_mlp"] - h["E_mlp"]; dd = None if (r["E_dft"] is None or h["E_dft"] is None) else r["E_dft"] - h["E_dft"]
        print(f" {r['frame']:4d}  {r['tier']:8s} {r['j']:2d} {r['phi']:9.3f} {r['clearance']:7.2f}   " + ("   n/a  " if dd is None else f"{dd:8.3f}") + f"  {dm:8.3f}  " + ("" if dd is None else f"{dm - dd:+8.3f}"))
        if dd is not None: out.setdefault(r["tier"], []).append((dd, dm))
    print("\n tier       n   mean(MLP-DFT)  rms(MLP-DFT)   mean dE_DFT   mean dE_MLP   exp-avg shift: -kT ln<exp(-(MLP-DFT)/kT)>")
    for tier, v in out.items():
        v = np.array(v); df = v[:, 1] - v[:, 0]; ex = -kt * np.log(np.mean(np.exp(-(df - df.mean()) / kt))) + df.mean()
        print(f" {tier:8s} {len(v):3d}   {df.mean():+9.3f}     {np.sqrt((df ** 2).mean()):9.3f}   {v[:, 0].mean():9.3f}   {v[:, 1].mean():9.3f}   {ex:+9.3f} eV")
    print("\nreading: pwidom's mu_ex is an average over sites weighted by exp(-dU/kT), so the tiers matter most where dU is lowest. A MLP-DFT shift that is the same in every tier is a per-H offset"
          " (cancels in mu_H - G_H against pure H only if the H reference carries it too); a shift that grows for lowphi / mlpmin sites means the model is too attractive exactly where Widom insertions are drawn.")


def do_status(a):
    dirs = [l.strip() for l in open(os.path.join(a.o, "dirs.txt")) if l.strip()]; now = time.time(); rows = []; cnt = dict(done=0, running=0, failed=0, pending=0)
    for d in dirs:
        i = outcar_info(d); vo = os.path.join(d, "vasp.out"); age = now - os.path.getmtime(vo) if os.path.exists(vo) else None
        bad = os.path.exists(vo) and re.search(r"VERY BAD NEWS|Error termination|ERROR|forrtl|segmentation|Segmentation", open(vo, errors="replace").read())
        if i and i["done"] and toten(d) is not None: st = "done"
        elif bad: st = "failed"
        elif i and age is not None and age < a.stale: st = "running"
        elif i or vo and age is not None: st = "stopped" if age is not None and age >= a.stale else "pending"
        else: st = "pending"
        cnt["running" if st == "stopped" else st] += 1; rows.append((d, st, i))
    hdr = "state     SCF  s/SCF   wall(s)   ranks  directory"
    print(f"{a.o}: {len(dirs)} single points  ({time.strftime('%H:%M:%S')})")
    for k, (d, st, i) in enumerate(rows):
        if k % 20 == 0: print(hdr)
        if st == "pending" and not a.all: continue
        print(f"{st:8s} {i['scf'] if i else 0:4d} {(i['s_scf'] or 0) if i else 0:6.1f} {(i['elapsed'] or 0) if i else 0:9.0f} {(i['ranks'] or 0) if i else 0:7d}  {d}")
    el = [i["elapsed"] for _, st, i in rows if st == "done" and i and i["elapsed"]]
    print(f"== done {cnt['done']}  running {cnt['running']}  failed {cnt['failed']}  pending {cnt['pending']}  of {len(dirs)}" + (f";  mean {np.mean(el):.0f} s per finished point -> about {np.mean(el) * (cnt['pending'] + cnt['running']) / 3600:.1f} h of single-task time left" if el else ""))


# ---------------------------------------------------------------- static (0 K) enthalpies of the h3o_static structures
EV_A3_GPA = 160.21766


def static_files(base, P):
    """relaxed structures of pressure P written by h3o_static.py: <base>/P<P>/ (parallel mode) or <base>/ (single process). Returns [(kind, name, path)], kind in ice / href / h3o"""
    out = []
    for d in (os.path.join(base, f"P{P:.0f}"), base):
        for f in sorted(glob.glob(os.path.join(d, f"*_{P:.0f}GPa_relaxed.extxyz"))):
            b = os.path.basename(f)[:-len(f"_{P:.0f}GPa_relaxed.extxyz")]
            kind = "ice" if b == "iceX" else "href" if b.startswith("Href_") else "h3o" if re.fullmatch(r"H\d(\.\d+)?O", b) else None
            if kind and not any(o[2] == f for o in out) and not any(o[1] == b for o in out):
                out.append((kind, b, f))
    return out


def prep_static(a):
    tp = template(a.template); over = dict(ismear=-1, sigma=KB * a.tel, kspacing=a.kspacing); rows = []; dirs = []
    for P in a.P:
        files = static_files(a.base, P)
        if not any(k == "ice" for k, _, _ in files) or not any(k == "h3o" for k, _, _ in files):
            print(f"{P:.0f} GPa: no iceX / H3O relaxed structures in {a.base} (run h3o_static.py first, or the pressure is not among its --P): skipped"); continue
        for kind, name, f in files:
            if a.only and not any(o in name for o in a.only): continue
            at = read(f)
            for sc in a.scales:
                b = at.copy(); b.set_cell(np.asarray(at.cell) * sc, scale_atoms=True)
                d = write_dft(os.path.join(a.o, f"static_{name}_{P:.0f}GPa_s{sc:.3f}"), b, a.tel, tp, f"{name} {P:.0f} GPa, linear scale {sc}", over=over)
                dirs.append(d); rows.append(dict(dir=d, kind=kind, name=name, P=float(P), scale=float(sc), N=len(b), V=float(b.get_volume()), nO=int((b.numbers == 8).sum()), nH=int((b.numbers == 1).sum())))
                print(f"   {P:.0f} GPa {name:18s} {len(b):3d} atoms  scale {sc:.3f}  V/atom {b.get_volume() / len(b):.4f}  k-mesh {open(os.path.join(d, 'KPOINTS')).read().split()[-6:-3]}", flush=True)
    if not dirs: raise SystemExit("nothing to prepare")
    finish(a, rows, dirs, dict(mode="static", run=a.base, tel=a.tel, kspacing=a.kspacing, scales=a.scales, P=a.P, model_default=a.model))


def outcar_pressure_GPa(d):
    """external pressure of a finished single point (VASP prints kB), None if absent"""
    f = os.path.join(d, "OUTCAR")
    if not os.path.exists(f): return None
    m = re.findall(r"external pressure =\s*(-?\d+\.\d+)\s*kB", open(f, errors="replace").read())
    return float(m[-1]) / 10 if m else None


def enthalpy_at(rows_g, P, which):
    """enthalpy (eV, whole cell) at pressure P from E(V) of one structure: quadratic fit in V through >= 3 scalings (minimum of E + P V), else E + P V at the single volume.
    which = 'E_dft' or 'E_mlp'. Returns (H, V_min, note)"""
    p = P / EV_A3_GPA; g = [r for r in rows_g if r[which] is not None]
    if not g: return None, None, "missing"
    V = np.array([r["V"] for r in g]); E = np.array([r[which] for r in g])
    if len(g) < 3:
        k = int(np.argmin(abs(np.array([r["scale"] for r in g]) - 1.0))); return float(E[k] + p * V[k]), float(V[k]), "single volume (no E(V) fit)"
    c = np.polyfit(V, E, 2)
    if c[0] <= 0: return float(E.min() + p * V[np.argmin(E)]), float(V[np.argmin(E)]), "E(V) not convex: lowest point"
    Vm = -(c[1] + p) / (2 * c[0]); note = "" if V.min() * 0.995 <= Vm <= V.max() * 1.005 else f"V* {Vm:.1f} outside the sampled {V.min():.1f}-{V.max():.1f}: extrapolated"
    return float(np.polyval(c, Vm) + p * Vm), float(Vm), note


def analyze_static(a, man, rows):
    from importlib import import_module
    calc = import_module("04_benchmark").get_calc_spec(a.model or man.get("model_default") or "grace-ft:runs/grace/ft_m2", a.device)
    for r in rows:
        r["E_dft"] = toten(r["dir"]); r["P_dft"] = outcar_pressure_GPa(r["dir"])
        at = read(os.path.join(r["dir"], "structure.extxyz")); at.calc = calc; r["E_mlp"] = float(at.get_potential_energy()); r["P_mlp"] = float(-at.get_stress(voigt=True)[:3].mean() * EV_A3_GPA)
    pr = lambda *x: print(*x, flush=True)
    pr(f"{a.o}: {sum(r['E_dft'] is not None for r in rows)} of {len(rows)} DFT single points finished; MLP {a.model or man.get('model_default')}; electronic T {man['tel']:.0f} K, k spacing {man['kspacing']}")
    H = {}   # (P, name) -> dict(mlp=, dft=, N, nO, nH, kind)
    for P in sorted({r["P"] for r in rows}):
        pr(f"\n=== {P:.0f} GPa: per structure (energies per atom, meV; H at the target pressure from E(V) over the scalings)")
        pr("   structure            N   V_mlp/atom  E_dft-E_mlp   P_dft(GPa) at GRACE V   P_mlp    H_mlp(eV/atom)  H_dft(eV/atom)  V*_mlp  V*_dft  notes")
        for name in dict.fromkeys(r["name"] for r in rows if r["P"] == P):
            g = [r for r in rows if r["P"] == P and r["name"] == name]; r1 = min(g, key=lambda r: abs(r["scale"] - 1.0)); N = r1["N"]
            hm, vm, nm = enthalpy_at(g, P, "E_mlp"); hd, vd, nd = enthalpy_at(g, P, "E_dft")
            H[(P, name)] = dict(mlp=hm, dft=hd, N=N, nO=r1["nO"], nH=r1["nH"], kind=r1["kind"])
            dE = None if r1["E_dft"] is None else (r1["E_dft"] - r1["E_mlp"]) / N * 1e3
            sdE = "   n/a" if dE is None else f"{dE:9.1f}"; spd = "n/a" if r1["P_dft"] is None else f"{r1['P_dft']:8.1f}"; shd = "n/a" if hd is None else f"{hd / N:12.5f}"; svd = "n/a" if vd is None else f"{vd / N:6.3f}"
            pr(f"   {name:18s} {N:3d}   {r1['V'] / N:8.4f}   {sdE}     {spd}      {r1['P_mlp']:8.1f}   {hm / N:12.5f}   {shd}   {vm / N:6.3f}  {svd}  {nm}{' | ' + nd if nd else ''}")
    rd = lambda f: {float(r["P_GPa"]): float(r["dH_meV_atom"]) for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))} if os.path.exists(f) else {}
    st, zp = rd("docs/data/huang_h3o_dH_static_SI.csv"), rd("docs/data/huang_h3o_dH.csv"); itp = lambda d, P: float(np.interp(P, sorted(d), [d[k] for k in sorted(d)])) if d else float("nan")
    pr("\n=== formation enthalpy dH_f = [H(H(2+n)O cell) - nO H(ice X) - extra H(H ref)] / N, meV/atom (static, no ZPE). 'ref' = lowest-enthalpy solid-H candidate OF THAT METHOD; 'GRACE ref' = DFT with the candidate GRACE picks")
    pr("   P/GPa  structure    n_extra   GRACE   VASP(own ref)   VASP(GRACE ref)   VASP-GRACE   per extra H (meV)   | Huang static  Huang+ZPE (H3O only)   H ref GRACE / VASP")
    out = []
    for P in sorted({k[0] for k in H}):
        ice = next((v for k, v in H.items() if k[0] == P and v["kind"] == "ice"), None)
        refs_ = {k[1]: v for k, v in H.items() if k[0] == P and v["kind"] == "href"}
        if ice is None or not refs_ or any(v["mlp"] is None for v in refs_.values()): pr(f"   {P:.0f} GPa: ice X / H references missing"); continue
        bm = min(refs_, key=lambda n: refs_[n]["mlp"] / refs_[n]["N"]); okd = {n: v for n, v in refs_.items() if v["dft"] is not None}; bd = min(okd, key=lambda n: okd[n]["dft"] / okd[n]["N"]) if okd else None
        for k, v in sorted(((k, v) for k, v in H.items() if k[0] == P and v["kind"] == "h3o"), key=lambda kv: kv[1]["nH"] / kv[1]["nO"]):
            ex = v["nH"] - 2 * v["nO"]; hice = lambda m: ice[m] / ice["N"]; hH = lambda n, m: refs_[n][m] / refs_[n]["N"]
            dm = (v["mlp"] - v["nO"] * 3 * hice("mlp") - ex * hH(bm, "mlp")) / v["N"] * 1e3
            dd = None if (v["dft"] is None or ice["dft"] is None or bd is None) else (v["dft"] - v["nO"] * 3 * hice("dft") - ex * hH(bd, "dft")) / v["N"] * 1e3
            dg = None if (v["dft"] is None or ice["dft"] is None or refs_[bm]["dft"] is None) else (v["dft"] - v["nO"] * 3 * hice("dft") - ex * hH(bm, "dft")) / v["N"] * 1e3
            pe = None if dd is None else (dd - dm) * v["N"] / ex
            h3 = abs(v["nH"] / v["nO"] - 3) < 1e-9; f1 = lambda x: "   n/a" if x is None else f"{x:7.1f}"
            pr(f"   {P:5.0f}  {k[1]:10s}  {ex:4d}    {dm:7.1f}  {f1(dd)}         {f1(dg)}        {f1(None if dd is None else dd - dm)}     {f1(pe)}          | {itp(st, P) if h3 else float('nan'):7.1f}      {itp(zp, P) if h3 else float('nan'):7.1f}            {bm} / {bd}")
            out.append(dict(P_GPa=P, structure=k[1], extra_H=ex, dH_grace=dm, dH_vasp=dd, dH_vasp_graceref=dg, vasp_minus_grace_per_extraH=pe))
    pr("\n=== H-reference-free check, per extra H relative to the lowest pressure: e(P) = [H(H3O) - nO H(ice X per H2O)] / n_extra (eV). Slope = partial volume of the extra H; VASP vs GRACE vs your CASTEP")
    cast = {float(r["P_GPa"]): float(r["H_eV_cell"]) for r in csv.DictReader(l for l in open("docs/data/h3o_castep_enthalpy.csv") if not l.startswith("#"))} if os.path.exists("docs/data/h3o_castep_enthalpy.csv") else {}
    ix = {float(r["P_GPa"]): float(r["H_eV_cell"]) / 2 for r in csv.DictReader(l for l in open("docs/data/icex_castep_enthalpy.csv") if not l.startswith("#"))} if os.path.exists("docs/data/icex_castep_enthalpy.csv") else {}
    ps = sorted({k[0] for k, v in H.items() if v["kind"] == "h3o" and abs(v["nH"] / v["nO"] - 3) < 1e-9}); e = {}
    for P in ps:
        v = next(v for k, v in H.items() if k[0] == P and v["kind"] == "h3o" and abs(v["nH"] / v["nO"] - 3) < 1e-9); ice = next((v for k, v in H.items() if k[0] == P and v["kind"] == "ice"), None)
        if ice is None: continue
        ex = v["nH"] - 2 * v["nO"]
        e[P] = dict(mlp=(v["mlp"] - v["nO"] * 3 * ice["mlp"] / ice["N"]) / ex, dft=None if (v["dft"] is None or ice["dft"] is None) else (v["dft"] - v["nO"] * 3 * ice["dft"] / ice["N"]) / ex,
                    cas=((cast[P] - v["nO"] * ix[P]) / ex) if P in cast and P in ix else None)
    if len(e) > 1:
        P0 = min(e); pr("   P/GPa    GRACE      VASP       CASTEP     VASP-GRACE   CASTEP-GRACE (eV per extra H, relative to %.0f GPa)" % P0)
        for P in sorted(e):
            d = lambda m: None if (e[P][m] is None or e[P0][m] is None) else e[P][m] - e[P0][m]; g, v_, c_ = d("mlp"), d("dft"), d("cas"); f4 = lambda x: "   n/a  " if x is None else f"{x:8.4f}"
            pr(f"   {P:5.0f}  {f4(g)}  {f4(v_)}  {f4(c_)}  {f4(None if v_ is None else v_ - g)}   {f4(None if c_ is None else c_ - g)}")
    pr("\nreading: dH_f(VASP) - dH_f(GRACE) at the SAME structures is the model error of the extra-H enthalpy at 0 K (x N / n_extra for the cost per extra H); the free-energy corrections to mu0 and s need the FEP (prep-fep) on hot frames."
       " A VASP number that agrees with Huang's static curve but not with GRACE confirms the +90-100 meV/atom GRACE offset; if VASP also sits above Huang the PAW/PBE settings differ (check ENCUT, POTCAR, smearing, k-mesh).")
    if out:
        with open(os.path.join(a.o, "static_results.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]), restval=""); w.writeheader(); w.writerows(out)
        pr("wrote", os.path.join(a.o, "static_results.csv"))


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
for name in ("prep-probe", "prep-fep"):
    p = sp.add_parser(name); p.add_argument("run"); p.add_argument("--template", required=True, help="a training DFT run directory with INCAR, KPOINTS, POTCAR and POSCAR")
    p.add_argument("-o", required=True); p.add_argument("--frames", type=int, default=2 if name == "prep-probe" else 20); p.add_argument("--seed", type=int, default=0)
    if name == "prep-probe":
        p.add_argument("--dirs", type=int, default=2); p.add_argument("--elements", nargs="*", default=["H"]); p.add_argument("--r", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
    else:
        p.add_argument("--h-run", default=None, help="pure-H run (<dir>/<tag>) for the G_H correction")
p = sp.add_parser("prep-static"); p.add_argument("base", help="h3o_static.py output directory (contains P<P>/ or the *_relaxed.extxyz files)"); p.add_argument("--template", required=True, help="a training DFT run directory with INCAR, KPOINTS, POTCAR and POSCAR")
p.add_argument("--P", type=float, nargs="+", required=True, help="pressures (GPa) as run by h3o_static.py"); p.add_argument("--scales", type=float, nargs="+", default=[1.0], help="linear cell scalings (e.g. 0.98 1.0 1.02: three volumes, E(V) fit, enthalpy at the target P)")
p.add_argument("--tel", type=float, default=1000.0, help="electronic temperature (K) of the Fermi smearing, ISMEAR=-1: a nearly cold static calculation (the MD template's SIGMA = kB T of 5000 K+ is NOT used)")
p.add_argument("--kspacing", type=float, default=0.25, help="k-mesh spacing in 1/A (Gamma-centred; the MD template mesh is for 250-atom cells)"); p.add_argument("--only", nargs="*", default=None, help="only structures whose name contains one of these (iceX, Href, H3O, H2.5O ...)")
p.add_argument("--model", default=None, help="MLP for analyze (default grace-ft:runs/grace/ft_m2)"); p.add_argument("-o", required=True)
p = sp.add_parser("prep-selftest"); p.add_argument("--template", required=True); p.add_argument("-o", required=True); p.add_argument("--size", choices=["small", "template", "both"], default="both", help="small = 12-atom fcc H2O cell (fast chain check), template = the template's own cell (cost basis; 100s of atoms with hard PAW: hours on 8 ranks)"); p.add_argument("--vpa", type=float, default=2.2, help="A^3/atom of the small cell"); p.add_argument("--T", type=float, default=0.0, help="electronic temperature (K) if the template has ISMEAR=-1 (default: its TEBEG)")
p.add_argument("--model", default=None)
p = sp.add_parser("prep-widom"); p.add_argument("frames", help="pwidom.py frames file <out>/<tag>_frames.npz"); p.add_argument("--template", required=True); p.add_argument("-o", required=True)
p.add_argument("--proxy", default="none", help="as pwidom.py: a toy_fit.py json, 'toyice' or 'none'"); p.add_argument("--K", type=int, default=2000); p.add_argument("--sel", type=int, default=3, help="Rosenbluth-sampled sites per frame")
p.add_argument("--frames-n", type=int, default=4, help="host frames used"); p.add_argument("--model", default=None, help="MLP: adds the tier of sites where the MLP energy is lowest"); p.add_argument("--top", type=int, default=50)
p.add_argument("--device", default="cuda"); p.add_argument("--L", type=float, default=0.0, help="cubic cell edge (A) if the frames are not from a cubic build_host cell"); p.add_argument("--seed", type=int, default=0)
p = sp.add_parser("status"); p.add_argument("o"); p.add_argument("--all", action="store_true", help="also list pending directories"); p.add_argument("--stale", type=float, default=900.0, help="seconds without output after which a started point counts as stopped")
p = sp.add_parser("analyze"); p.add_argument("o"); p.add_argument("--model", default=None); p.add_argument("--device", default="cuda")
a = ap.parse_args()
{"prep-probe": prep_probe, "prep-fep": prep_fep, "prep-static": prep_static, "prep-selftest": prep_selftest, "prep-widom": prep_widom, "status": do_status, "analyze": do_analyze}[a.cmd](a)
