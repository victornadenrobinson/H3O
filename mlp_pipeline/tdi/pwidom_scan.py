"""Quick, LIGHT scan of dg = mu_H(host) - G_H(pure H) over (P, T) with the proxy-guided Widom / deletion + BAR method (tdi/pwidom.py, docs/MU_METHODS.md R1),
reusing the volumes of existing runs: no lambda path, ~10 min per (P, T) on one GPU. Rough numbers to see the shape of mu_H(P, T, n). Trust a row only if Widom and deletion agree (column del-Wid; flagged UNCONVERGED otherwise):
the first GRACE scan (K 1000, 20 frames) had Kish fractions 0.02-0.05 and Widom/deletion 300-850 meV apart in the H2O host, so the defaults are now K 2000 and --batch 8.
Per (P, T) it runs side by side (3 processes on the GPU, below the 4-process saturation): the H2O host, the H2.25O host (if a run exists) and the pure-H reservoir.
Volumes, bulk moduli and sizes come from existing stage0 jsons in runs/tdi*: H2O host = an insertion run's json (preferred, same host as the insertion TDI) or a fcc H2O crystal json;
H2.25O host = a fcc crystal json with H per O 2.25; pure H = a pure-H run json (phase H). `--list` shows what exists. Frames are cached per host in --out, so a rerun (or a
longer --ps in ANOTHER --out) does not repeat finished work; a point whose hosts are all done is skipped.
VERBOSE: plan, then per point a status line every 30 s (phase of each host: MD host / MD host + 1 H / single points / done, minutes), the three results, and the table so far.
The pure-H frames are checked for fused protons (min d(H-H) < 0.25 A: the GRACE short-range hole, CURRENT_ISSUES): such a point is flagged, its dg is not usable.
usage (GPU session, repo root):
  ~/grace-venv/bin/python -u mlp_pipeline/tdi/pwidom_scan.py --list
  ~/grace-venv/bin/python -u mlp_pipeline/tdi/pwidom_scan.py --points "400 5000;500 5000;400 6000;300 5000"      [--ps 0.4 --K 1000 --sel 8 --ndel 4 --out runs/pwidom_scan]
  ~/grace-venv/bin/python mlp_pipeline/tdi/pwidom_scan.py --table runs/pwidom_scan          (results so far)
"""
import argparse, glob, json, os, subprocess, sys, time
from math import lgamma, log
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); KB = 8.617333262e-5
NOISE = ("_handle_getgr", "I0000", "W0000", "E0000", "cuda_executor", "device_compiler", "cuDNN", "gpu_device", "tensorflow", "XLA", "absl", "ptxas")   # TensorFlow chatter, never a status


def hosts(roots):
    """{(P, T): {'ice': json, 'ice_src': ..., 'h225': json, 'H': json}} from the stage0 jsons under roots"""
    pts = {}
    for f in sorted(sum((glob.glob(os.path.join(r, "*_stage0.json")) for r in roots), [])):
        if "_rep" in f or "tdi_test" in f: continue
        try: st = json.load(open(f)); g = st["args"]
        except Exception: continue
        if not st.get("B_GPa") or not st.get("vpa"): continue
        key = (float(st["P_target"]), float(st["T"])); d = pts.setdefault(key, {}); h = float(g.get("hper", 2.0)); ph = st.get("phase")
        if ph == "insert" and g.get("host_phase", "fcc") == "fcc" and abs(h - 2) < 1e-9 and "Hhost" not in f:
            if d.get("ice_src") != "insert": d.update(ice=f, ice_src="insert")
        elif ph == "fcc" and abs(h - 2) < 1e-9 and "ice" not in d: d.update(ice=f, ice_src="fcc crystal")
        elif ph == "fcc" and abs(h - 2.25) < 1e-9 and "h225" not in d: d["h225"] = f
        elif ph == "H" and "H" not in d and not g.get("hhwall"): d["H"] = f
        elif ph == "insert" and g.get("host_phase") == "H" and "H" not in d: d["H"] = f
    return pts


def pw_args(f, kind):
    """pwidom.py run arguments that rebuild the host of stage0 json f"""
    st = json.load(open(f)); g = st["args"]
    if kind == "H":
        nat = g.get("natoms") or 250
        return ["--phase", "H", "--natoms", str(nat), "--vpa", repr(st["vpa"]), "--T", repr(st["T"]), "--P", repr(st["P_target"]), "--bulk-gpa", repr(st["B_GPa"]), "--model", g["model"]]
    return ["--phase", "fcc", "--nrep", str(g.get("nrep") or 3), "--hper", repr(float(g.get("hper", 2.0))), "--vpa", repr(st["vpa"]), "--T", repr(st["T"]), "--P", repr(st["P_target"]),
            "--bulk-gpa", repr(st["B_GPa"]), "--model", g["model"]]


def interp_args(f0, f25, hp):
    """host H(hp)O: volume per O linear in n between the H2O host (json f0) and the H2.25O crystal (json f25) at the same (P, T); bulk modulus and model of the H2O host"""
    s0, s1 = json.load(open(f0)), json.load(open(f25)); g = s0["args"]; n = hp - 2
    vO0, vO1 = s0["vpa"] * 3, s1["vpa"] * 3.25; vpa = (vO0 + n / 0.25 * (vO1 - vO0)) / (3 + n)
    return ["--phase", "fcc", "--nrep", str(g.get("nrep") or 3), "--hper", repr(hp), "--vpa", repr(vpa), "--T", repr(s0["T"]), "--P", repr(s0["P_target"]),
            "--bulk-gpa", repr(s0["B_GPa"]), "--model", g["model"]]


def min_hh(npz):
    """smallest H-H distance over the cached pure-H frames (cubic cell of the stored volume)"""
    z = np.load(npz); L = float(z["V"]) ** (1 / 3); best = np.inf
    for X in list(z["host"])[::2]:
        d = X[:, None, :] - X[None, :, :]; d -= L * np.round(d / L); r = np.sqrt((d ** 2).sum(-1)); np.fill_diagonal(r, np.inf); best = min(best, float(r.min()))
    return best


def dg(w, h):
    kT = KB * w["T"] * 1e3; N = h["N_H"]; fs = kT * (log(N + 1) - lgamma(N + 1) / N - 1)
    return w["mu_P_meV"] - (h["mu_P_meV"] - fs), float(np.hypot(w["mu_P_err_meV"], h["mu_P_err_meV"]))


def table(out):
    rows = [json.loads(l) for l in open(os.path.join(out, "scan.jsonl"))] if os.path.exists(os.path.join(out, "scan.jsonl")) else []
    print(f"\n{'P/GPa':>6s} {'T/K':>6s} {'host':>7s} {'dg/meV':>8s} {'+-':>6s} {'mu_P host':>9s} {'mu_P H':>8s} {'del-Wid':>7s} {'proxy kT':>8s} {'kish':>5s} {'O MSD':>6s}  flags")
    rows = list({(r["P"], r["T"], r["host"], r.get("ps"), r.get("K")): r for r in rows}.values())   # a rerun appends the finished hosts again: keep the last of each
    for r in sorted(rows, key=lambda r: (r["P"], r["T"], r.get("n", 0.25 if "2.25" in r["host"] else 0.0), r["host"])):
        print(f"{r['P']:6.0f} {r['T']:6.0f} {r['host']:>7s} {r['dg']:8.0f} {r['err']:6.0f} {r['mu_host']:9.0f} {r['mu_H']:8.0f} {r.get('deletion', 0) - r.get('widom', 0):7.0f} {r['proxy_kT']:8.2f} {r['kish']:5.2f} {r['msd']:6.2f}  {r['flags']}")
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", default=None); ap.add_argument("--list", action="store_true"); ap.add_argument("--table", default=None)
    ap.add_argument("--points", default=None, help='"P T;P T;..." (default: every point that has an H2O host and a pure-H run)')
    ap.add_argument("--ps", type=float, default=0.4); ap.add_argument("--equil-ps", type=float, default=0.2); ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8, help="structures per batched GRACE call (smaller = less GPU memory with 3 hosts on one GPU)")
    ap.add_argument("--sel", type=int, default=8); ap.add_argument("--ndel", type=int, default=4); ap.add_argument("--proxy", default="docs/data/toyfit_t4.json")
    ap.add_argument("--no-h225", action="store_true"); ap.add_argument("--out", default="runs/pwidom_scan"); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default=None)
    ap.add_argument("--hpers", type=float, nargs="*", default=[], help="extra H2O-host compositions (H per O), volume interpolated per O between the H2O and H2.25O runs, e.g. 2.0556 2.1111 2.1667 (= 6, 12, 18 extra H in 108 O)")
    ap.add_argument("--maxpar", type=int, default=3, help="hosts running at once on the GPU (the GPU saturates at 4 MD processes)")
    ap.add_argument("--py", default=sys.executable); a = ap.parse_args()
    if a.table: table(a.table); return
    roots = a.roots or sorted(d for d in glob.glob("runs/tdi*") if os.path.isdir(d))
    pts = hosts(roots)
    if a.list or not a.points:
        print(f"stage0 jsons in {', '.join(roots)}: (P, T) with an H2O host, an H2.25O host and a pure-H reservoir")
        for (P, T), d in sorted(pts.items()):
            print(f"  {P:5.0f} GPa {T:6.0f} K   H2O: {(d.get('ice_src') or '-'):12s} H2.25O: {'yes' if 'h225' in d else '-':4s} pure H: {'yes' if 'H' in d else '-':4s}" + ("   <- usable" if "ice" in d and "H" in d else ""))
        if a.list or not a.points: print('\nrun with --points "P T;P T" (e.g. the usable ones above)'); return
    want = [tuple(map(float, p.split())) for p in a.points.split(";") if p.strip()]
    os.makedirs(a.out, exist_ok=True); t_all = time.time()
    print(f"== pwidom scan: {len(want)} points, PS {a.ps} ps (+{a.equil_ps} equil) per MD, K {a.K}, SEL {a.sel}, NDEL {a.ndel}, proxy {a.proxy}, out {a.out}, {time.strftime('%H:%M:%S')}", flush=True)
    for i, (P, T) in enumerate(want):
        d = pts.get((P, T), {})
        if "ice" not in d or "H" not in d:
            print(f"\n-- [{i + 1}/{len(want)}] {P:g} GPa {T:g} K: SKIPPED (missing {'H2O host' if 'ice' not in d else ''} {'pure-H run' if 'H' not in d else ''})", flush=True); continue
        jobs = [("H2O", pw_args(d["ice"], "ice"))] + ([("H2.25O", pw_args(d["h225"], "fcc"))] if "h225" in d and not a.no_h225 else [])
        for hp in a.hpers:   # intermediate compositions: volume per O interpolated between the H2O and H2.25O hosts of this (P, T)
            if "h225" not in d: print(f"   H{hp:g}O: needs an H2.25O run at this (P, T) for the volume: skipped", flush=True); continue
            jobs.append((f"H{hp:.4g}O", interp_args(d["ice"], d["h225"], hp)))
        jobs.append(("pure H", pw_args(d["H"], "H")))
        print(f"\n-- [{i + 1}/{len(want)}] {P:g} GPa {T:g} K ({time.strftime('%H:%M:%S')}, scan {(time.time() - t_all) / 60:.1f} min): hosts " + ", ".join(n for n, _ in jobs)
              + f"; at most {a.maxpar} at once", flush=True)
        procs = []; cmds = {}; queue = []
        env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2", OMP_NUM_THREADS="2")
        for name, args in jobs:
            tag = f"pw_{name.replace(' ', '').replace('.', 'p')}_{P:g}GPa_{T:.0f}K"; log_ = os.path.join(a.out, f"{tag}.log")
            old = sorted(glob.glob(os.path.join(a.out, f"{tag}_*_K{a.K}.json")))
            legacy = glob.glob(os.path.join(a.out, f"pw_{name.replace(' ', '').replace('.', '')}_{P:g}GPa_{T:.0f}K_*_K{a.K}.json"))   # tags of the first scan version ('.' removed)
            if old or legacy: procs.append([name, None, log_, (old or legacy)[0], tag]); print(f"   {name}: done before ({os.path.basename((old or legacy)[0])})", flush=True); continue
            if legacy == [] and glob.glob(os.path.join(a.out, f"pw_{name.replace(' ', '').replace('.', '')}_{P:g}GPa_{T:.0f}K_frames.npz")):
                tag = f"pw_{name.replace(' ', '').replace('.', '')}_{P:g}GPa_{T:.0f}K"; log_ = os.path.join(a.out, f"{tag}.log")   # reuse the MD frames of the first scan version
            cmd = [a.py, "-u", os.path.join(HERE, "pwidom.py"), "run", *args, "--ps", repr(a.ps), "--equil-ps", repr(a.equil_ps), "--proxy", a.proxy, "--K", str(a.K),
                   "--sel", str(a.sel), "--ndel", str(a.ndel), "--out", a.out, "--tag", tag, "--device", a.device, "--batch", str(a.batch)] + (["--model", a.model] if a.model else [])
            cmds[name] = (cmd, env, log_); rec = [name, "queued", log_, None, tag]; procs.append(rec); queue.append(rec)
        t0 = time.time(); last = 0
        while queue or any(isinstance(r[1], subprocess.Popen) and r[1].poll() is None for r in procs):
            while queue and sum(isinstance(r[1], subprocess.Popen) and r[1].poll() is None for r in procs) < a.maxpar:
                r = queue.pop(0); cmd, env_, lg = cmds[r[0]]
                with open(lg, "a") as fh: fh.write(f"======== pwidom_scan {time.strftime('%F %T')} ========\n")
                r[1] = subprocess.Popen(cmd, stdout=open(lg, "a"), stderr=subprocess.STDOUT, env=env_); print(f"   started {r[0]} ({time.time() - t0:.0f} s)", flush=True); time.sleep(5)
            time.sleep(2)
            if time.time() - last >= 30:
                last = time.time(); st = []
                for name, p, log_, _, _ in procs:
                    if p is None: st.append(f"{name}: done before"); continue
                    if p == "queued": st.append(f"{name}: queued"); continue
                    lines = [l.strip() for l in open(log_).read().splitlines() if l.strip() and not any(z in l for z in NOISE)]
                    st.append(f"{name}: {'finished' if p.poll() is not None else (lines[-1][:60] if lines else 'starting')}")
                print(f"   {time.time() - t0:5.0f} s | " + " | ".join(st), flush=True)
        for name, p, log_, res, tag in procs:   # a host that died (usually GPU out of memory while several processes shared the GPU) is retried ALONE: its MD frames are cached, only the single points rerun
            if p is not None and not glob.glob(os.path.join(a.out, f"{tag}_*_K{a.K}.json")) and name in cmds:
                cmd, env_, lg = cmds[name]; print(f"   {name}: failed while sharing the GPU ({' / '.join([l for l in open(lg).read().splitlines() if l.strip()][-1:])[:120]}); retrying alone", flush=True)
                t1 = time.time(); subprocess.run(cmd, stdout=open(lg, "a"), stderr=subprocess.STDOUT, env=env_); print(f"   {name}: retry finished in {(time.time() - t1) / 60:.1f} min", flush=True)
        out = {}
        for name, p, log_, res, tag in procs:
            res = res or (sorted(glob.glob(os.path.join(a.out, f"{tag}_*_K{a.K}.json")), key=os.path.getmtime) or [None])[-1]
            if not res:
                tail = [l for l in open(log_).read().splitlines() if l.strip()][-3:]
                print(f"   {name}: FAILED (see {log_}): {' / '.join(tail)}", flush=True); continue
            out[name] = json.load(open(res))
        if "pure H" not in out: continue
        h = out["pure H"]; hh = min_hh(os.path.join(a.out, f"{[t for n, *_, t in procs if n == 'pure H'][0]}_frames.npz"))
        for name in [n for n, _ in jobs if n != "pure H" and n in out]:
            w = out[name]; g_, e_ = dg(w, h); flags = []
            if w.get("melted"): flags.append("HOST MELTED")
            if hh < 0.25: flags.append(f"PURE-H COLLAPSE (min dHH {hh:.2f} A)")
            if max(w["proxy_gap_std_kT_ins"], w["proxy_gap_std_kT_del"]) > 3: flags.append("proxy poor (>3 kT)")
            wd = w["mu_ex_delete_meV"] - w["mu_ex_widom_meV"]
            if abs(wd) > max(200.0, 3 * e_): flags.append(f"UNCONVERGED (deletion - Widom {wd:+.0f} meV)")   # insertion and deletion bracket the answer: far apart = too few effective samples
            if w["kish_frac_ins"] < 0.05: flags.append(f"few effective insertions (Kish {w['kish_frac_ins']:.2f})")
            row = dict(P=P, T=T, host=name, n=float(w.get("hper", 2.0)) - 2, dg=g_, err=e_, mu_host=w["mu_P_meV"], mu_H=h["mu_P_meV"],
                       proxy_kT=max(w["proxy_gap_std_kT_ins"], w["proxy_gap_std_kT_del"]), kish=w["kish_frac_ins"], msd=(w["msd_O_host"] or [np.nan])[-1], min_dHH=hh,
                       widom=w["mu_ex_widom_meV"], deletion=w["mu_ex_delete_meV"], bar=w["mu_ex_bar_meV"], flags=",".join(flags) or "ok", ps=a.ps, K=a.K, sel=a.sel)
            with open(os.path.join(a.out, "scan.jsonl"), "a") as fh: fh.write(json.dumps(row, default=float) + "\n")
            print(f"   == {name:6s}: dg = {g_:7.0f} +- {e_:4.0f} meV  (mu_ex Widom {w['mu_ex_widom_meV']:.0f} / deletion {w['mu_ex_delete_meV']:.0f} / BAR {w['mu_ex_bar_meV']:.0f}; "
                  f"proxy {row['proxy_kT']:.2f} kT, Kish {row['kish']:.2f}; O MSD {row['msd']:.2f} A^2)  {row['flags']}", flush=True)
        print(f"   point done in {(time.time() - t0) / 60:.1f} min; pure-H min d(H-H) {hh:.2f} A", flush=True)
        table(a.out)
    print(f"\n== scan finished in {(time.time() - t_all) / 60:.1f} min; H2.25O rows: dg is mu_H of one MORE H in H2.25O (differential), H2O rows: the dilute limit", flush=True)


if __name__ == "__main__":
    main()
