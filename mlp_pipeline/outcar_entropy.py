#!/usr/bin/env python3
"""Cheap look at the electronic-entropy term of every finished VASP single point, straight from the OUTCAR tails (no ASE, no collect; Python 3.6 and standard library only).
  python3 mlp_pipeline/outcar_entropy.py [snapdb snapdb_v2 ...] [--csv entropy_points.csv]
Per point: free energy F = TOTEN, 'energy without entropy' E, energy(sigma->0), SIGMA from the INCAR; -T*S_el = F - E (<= 0).  Prints by set and sampling temperature: -TS per cell and per atom (mean, sd),
then by composition, and the T slope. Reading it: the MEAN at a given T is what a T-independent model has to absorb through structure only; the SD within a (set, T) group is the
geometry-dependent part; the DIFFERENCE of means between temperatures (same composition) is the part no fixed-surface model can get right for both temperatures at the same geometry."""
import csv, glob, json, os, re, sys, collections
args = [x for x in sys.argv[1:] if not x.startswith("--")]; dbs = args or ["snapdb", "snapdb_v2"]
out_csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else None
if out_csv and out_csv in dbs: dbs.remove(out_csv)
NUM = r"([-+]?\d+\.\d+(?:[eE][-+]?\d+)?)"
rows = []; miss = 0


def tail(path, n=600000):
    with open(path, "rb") as f:
        f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - n)); return f.read().decode("utf8", "ignore")


for db in dbs:
    for mf in sorted(glob.glob(os.path.join(db, "chunks", "chunk_[0-9]*", "points", "frames_meta.csv"))):
        pts = os.path.dirname(mf)
        for r in csv.DictReader(open(mf)):
            d = os.path.join(pts, r["id"]); dj = os.path.join(d, "DONE.json")
            if not os.path.exists(dj): continue
            try:
                att = json.load(open(dj)).get("attempt"); t = tail(os.path.join(d, att, "OUTCAR"))
                F = float(re.findall(r"free\s+energy\s+TOTEN\s*=\s*" + NUM, t)[-1]); m = re.findall(r"energy\s+without\s+entropy\s*=\s*" + NUM + r"\s+energy\(sigma->0\)\s*=\s*" + NUM, t)[-1]
                E, E0 = float(m[0]), float(m[1]); sig = re.search(r"SIGMA\s*=\s*" + NUM, open(os.path.join(d, "INCAR")).read())
            except Exception:
                miss += 1; continue
            setn = {"False": "F", "True": "T"}.get(r.get("db_set", "?"), r.get("db_set", "?")); n = int(float(r.get("natoms", 0) or 0)); nO = int(float(r.get("nO", 0) or 0)); nH = int(float(r.get("nH", 0) or 0))
            T = float(r["T_K"]) if r.get("T_K") not in (None, "", "nan") else float("nan")
            rows.append(dict(db=db, point=os.path.join(d), set=setn, T_K=T, natoms=n, nO=nO, nH=nH, F=F, E_noentropy=E, E_sigma0=E0, ts=E - F, sigma=float(sig.group(1)) if sig else float("nan")))
print(f"{len(rows)} points read ({miss} unreadable / missing energy lines)")
if not rows: sys.exit(1)
if out_csv:
    w = csv.DictWriter(open(out_csv, "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); print("wrote", out_csv)


def ms(v):
    m = sum(v) / len(v); return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


g = collections.defaultdict(list)
for r in rows: g[(r["set"], int(round(r["T_K"] / 1000.0)) * 1000 if r["T_K"] == r["T_K"] else -1)].append(r)
print("\n-T*S_el = E_noentropy - F  (eV)   [sigma = kT of the frame]")
print("%-8s%7s%6s%14s%8s%13s%10s%12s" % ("set", "T_K", "n", "per cell", "sd", "meV/atom", "sd", "sigma eV"))
for (s, T), v in sorted(g.items()):
    pc = [x["ts"] for x in v]; pa = [1000 * x["ts"] / x["natoms"] for x in v if x["natoms"]]; a, b = ms(pc); c_, d_ = ms(pa)
    print("%-8s%7d%6d%14.2f%8.2f%13.1f%10.1f%12.3f" % (s, T, len(v), a, b, c_, d_, sum(x["sigma"] for x in v) / len(v)))
comp = collections.defaultdict(list)
for r in rows: comp["H" if r["nO"] == 0 else "H%.2fO" % (r["nH"] / r["nO"])].append(1000 * r["ts"] / max(r["natoms"], 1))
print("\nby composition (meV/atom, mean sd n), most frequent:")
for k, v in sorted(comp.items(), key=lambda kv: -len(kv[1]))[:10]: a, b = ms(v); print("  %-8s %7.1f %6.1f %5d" % (k, a, b, len(v)))
for label, sel in (("all", lambda r: True), ("H2O-like (nH/nO 1.9-2.1)", lambda r: r["nO"] and 1.9 <= r["nH"] / r["nO"] <= 2.1), ("pure H", lambda r: r["nO"] == 0)):
    pts_ = [(r["T_K"], 1000 * r["ts"] / r["natoms"]) for r in rows if sel(r) and r["T_K"] == r["T_K"] and r["natoms"]]
    if len(pts_) > 10:
        mx = sum(p[0] for p in pts_) / len(pts_); my = sum(p[1] for p in pts_) / len(pts_); sxx = sum((p[0] - mx) ** 2 for p in pts_); slope = sum((p[0] - mx) * (p[1] - my) for p in pts_) / sxx if sxx else 0
        print("%-26s slope of -TS/atom vs T: %.3f meV/atom per 1000 K  ->  3000 K to 7000 K changes it by %.0f meV/atom (about %.1f eV per 324-atom cell)" % (label, 1000 * slope, 4000 * slope, 4000 * slope * 0.324))
