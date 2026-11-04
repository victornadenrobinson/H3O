#!/usr/bin/env python3
"""DFT-over-model correction of the PWidom mu_ex, from DFT energies of importance-resampled trials (the trials that actually carry the Widom / deletion weight).
Inputs: trials.csv (candidates_to_frames.py), the labelled geometries scored with the SAME model by baseline_D.py (D csv: column run_id = geometry_id, D_F = E_DFT - E_model, free energy, eV).
For a trial, dD = D(large) - D(small) = (DFT - model) insertion energy of that exact configuration pair. Resampled with probability ~ estimator weight, so over draws:
   insertion:  d_mu_ins = -kT ln < exp(-dD/kT) >          deletion:  d_mu_del = +kT ln < exp(+dD/kT) >           (<> = mean over draws, a trial drawn k times counts k times)
Corrected mu_ex(DFT) = mu_ex(model) + d_mu. Error = bootstrap over draws. dg_DFT = dg_model + d_mu(host) - d_mu(pure-H reference); pass the reference tag with --ref (or give the correction by hand).
   python3 mlp_pipeline/reweight_dft.py runs/reweight/trials.csv runs/reweight/D_model.csv [--ref pw_pureH_500GPa_5000K] [--boot 2000]"""
import argparse, collections, csv
import numpy as np
KB = 8.617333262e-5
ap = argparse.ArgumentParser(); ap.add_argument("trials"); ap.add_argument("dcsv"); ap.add_argument("--ref", default=""); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
D = {r["run_id"]: float(r["D_F"]) for r in csv.DictReader(open(a.dcsv))}; rng = np.random.default_rng(1); by = collections.defaultdict(list); miss = 0
for r in csv.DictReader(open(a.trials)):
    if r["small_id"] in D and r["large_id"] in D: by[(r["tag"], r["direction"])].append((D[r["large_id"]] - D[r["small_id"]], int(r["draw_count"]), float(r["T_K"])))
    else: miss += 1
if miss: print(f"{miss} trials skipped: geometries not labelled / not scored yet")
res = {}
print(f"{'host tag':34s} {'dir':9s} {'draws':>5s} {'mean dD':>9s} {'sd':>6s} {'d_mu (meV)':>11s} {'+-boot':>7s}")
for (tag, dr), v in sorted(by.items()):
    dd = np.repeat([x[0] for x in v], [x[1] for x in v]); kT = KB * v[0][2]; sgn = -1.0 if dr == "insertion" else 1.0
    f = lambda x: sgn * kT * np.log(np.mean(np.exp(sgn * x / kT)))                   # ins: -kT ln<exp(-dD/kT)>; del: +kT ln<exp(+dD/kT)>
    est = f(dd); bs = np.array([f(rng.choice(dd, len(dd))) for _ in range(a.boot)]); res[(tag, dr)] = (est * 1e3, bs.std() * 1e3)
    print(f"{tag:34s} {dr:9s} {len(dd):5d} {dd.mean() * 1e3:9.0f} {dd.std(ddof=1) * 1e3:6.0f} {est * 1e3:11.0f} {bs.std() * 1e3:7.0f}")
print("\nper host: d_mu from insertion and deletion should agree (they estimate the same shift); mean and difference:")
tags = sorted({t for t, _ in res}); ref = {d: res.get((a.ref, d)) for d in ("insertion", "deletion")} if a.ref else {}
for t in tags:
    i, d = res.get((t, "insertion")), res.get((t, "deletion")); vals = [x for x in (i, d) if x]
    m = np.mean([x[0] for x in vals]); e = np.sqrt(sum(x[1] ** 2 for x in vals)) / len(vals)
    line = f"  {t:34s} d_mu ins {i[0] if i else float('nan'):7.0f}  del {d[0] if d else float('nan'):7.0f}   mean {m:7.0f} +- {e:4.0f} meV"
    if ref and t != a.ref and all(ref.values()): rm = np.mean([x[0] for x in ref.values()]); line += f"   dg shift vs reference {m - rm:+7.0f} meV"
    print(line)
print("\ndg_DFT = dg_model + 'dg shift' (use the dg of the SAME model/run); sign: negative shift = DFT makes the H more favourable in the host than the model does.")
