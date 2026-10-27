"""DFT checks of the MLP where the insertion free energy is sensitive: (1) the energy of an extra proton approaching a host H or O (probe scans in real host frames),
(2) free-energy perturbation (FEP) from the MLP to DFT for mu_H - G_H. The VASP settings are copied from a TEMPLATE directory of the original training DFT (INCAR, KPOINTS,
POTCAR, POSCAR for the species order) so the energies are those the model was trained on (free energy TOTEN); only single-point tags are changed.

  python mlp_pipeline/tdi/dft_check.py prep-probe runs/tdi_long/insert_500GPa_6000K_n1 --template <dft run dir> [--frames 2 --dirs 2 --elements H] -o runs/dft_check/probe
  python mlp_pipeline/tdi/dft_check.py prep-fep   runs/tdi_long/insert_500GPa_6000K_n1 --h-run runs/tdi_long/H_500GPa_6000K_N250 --template <dir> [--frames 20] -o runs/dft_check/fep
  sbatch -p <cpu partition> --array=0-<n-1> mlp_pipeline/young_dft_check.sbatch runs/dft_check/probe      (n = lines of runs/dft_check/probe/dirs.txt; or run the dirs by hand)
  ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py analyze runs/dft_check/probe     (and .../fep): reads OUTCARs, evaluates the MLP on the same structures, prints the comparison

probe: per frame, the host alone plus an extra H at r = 0.5 ... 1.2 A from a host H (or O) along the direction that keeps every OTHER atom farthest away, so dE(r) = E(host + H) - E(host)
       measures the H-H (O-H) interaction: compare DFT and MLP curves (the MLP's flat H-H region at 0.6-1.0 A, docs/CURRENT_ISSUES.md P2).
fep:   Delta = E_DFT - E_MLP per structure. F_DFT - F_MLP of an ensemble = -kT ln < exp(-Delta / kT) >_MLP (also its 1st and 2nd cumulant).
       mu_H:  host + n H frames (largest-lambda node) minus host frames (smallest-lambda node: the ghost is decoupled, the host unperturbed), per added H; also the paired estimate on the
       same geometries with the ghost removed (lower noise, first order). G_H: pure-H frames (largest-lambda node of the H run), per atom. Result: the DFT correction to mu_H - G_H.
       Per-element constant offsets between DFT and MLP cancel in mu_H - G_H (one H in both). The largest-lambda node is lambda ~0.95-0.99, not exactly 1 (stated in the output)."""
import argparse, glob, json, os, re, shutil, sys
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


def write_dft(d, atoms, T, tp, label):
    """one single-point directory: atoms sorted to the POTCAR species order, INCAR = template minus MD tags plus single-point tags"""
    os.makedirs(d, exist_ok=True)
    order = [i for s in tp["species"] for i in np.where(np.array(atoms.get_chemical_symbols()) == s)[0]]
    if len(order) != len(atoms):
        raise SystemExit(f"{label}: species {set(atoms.get_chemical_symbols())} not all in the template POTCAR {tp['species']}")
    at = atoms[order]; at.wrap(); write(os.path.join(d, "POSCAR"), at, format="vasp", direct=True, sort=False)
    inc = {k: v for k, v in tp["incar"].items() if k not in MD_TAGS}
    inc.update(IBRION="-1", NSW="0", ISYM="0", LWAVE=".FALSE.", LCHARG=".FALSE.")
    ismear = int(float(tp["incar"].get("ISMEAR", "1")))
    inc["SIGMA"] = f"{KB * T:.6f}" if ismear == -1 else tp["incar"].get("SIGMA", "0.2")   # Fermi smearing = electronic temperature of the MD (as in the training AIMD)
    nel = sum(tp["zval"][s] for s in at.get_chemical_symbols())
    if tp["nbands"] and tp["nel"]:
        inc["NBANDS"] = str(int(np.ceil(tp["nbands"] * nel / tp["nel"] / 8) * 8))   # template bands per electron, rounded up to a multiple of 8
    with open(os.path.join(d, "INCAR"), "w") as fh:
        fh.write(f"# {label}: single point (template {os.path.basename(os.path.dirname(tp['potcar']))}, MD tags removed)\n" + "".join(f"{k} = {v}\n" for k, v in inc.items()))
    present = [s_ for s_ in tp["species"] if s_ in set(at.get_chemical_symbols())]   # POTCAR = blocks of the species present, in POSCAR order (pure H: H only)
    open(os.path.join(d, "KPOINTS"), "w").write(tp["kpoints"]); open(os.path.join(d, "POTCAR"), "w").write("".join(tp["blocks"][s_] for s_ in present))
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
    print(f"{len(dirs)} single-point directories in {a.o} (list: {a.o}/dirs.txt). Run:  sbatch -p <cpu partition> --array=0-{len(dirs) - 1} mlp_pipeline/young_dft_check.sbatch {a.o}")


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


ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
for name in ("prep-probe", "prep-fep"):
    p = sp.add_parser(name); p.add_argument("run"); p.add_argument("--template", required=True, help="a training DFT run directory with INCAR, KPOINTS, POTCAR and POSCAR")
    p.add_argument("-o", required=True); p.add_argument("--frames", type=int, default=2 if name == "prep-probe" else 20); p.add_argument("--seed", type=int, default=0)
    if name == "prep-probe":
        p.add_argument("--dirs", type=int, default=2); p.add_argument("--elements", nargs="*", default=["H"]); p.add_argument("--r", type=float, nargs="*", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
    else:
        p.add_argument("--h-run", default=None, help="pure-H run (<dir>/<tag>) for the G_H correction")
p = sp.add_parser("analyze"); p.add_argument("o"); p.add_argument("--model", default=None); p.add_argument("--device", default="cuda")
a = ap.parse_args()
{"prep-probe": prep_probe, "prep-fep": prep_fep, "analyze": do_analyze}[a.cmd](a)
