#!/usr/bin/env python3
"""Make a state-json (the EOS inputs of perlmutter_pwidom.py: volumes per atom and bulk moduli of H2O, H2.25O and pure H) for a (P, T) without enough DFT coverage, by a LINEAR fit in T through the state-jsons
that exist at the same P (runs/dft_eos_<P>GPa_<T>K*.json): two temperatures interpolate or extrapolate, more than two are least-squares, ONE only is copied with a loud warning (the thermal pressure then makes the volumes
wrong by tens of GPa). This is a starting point, not an EOS: PWidom reports P_host of every host and corrects to the target pressure to second order (the 'robust' correction), so check P_host (morning_review.py shows it);
a P_host more than ~15 GPa from the target means the mix is too crude for that state.
   python3 mlp_pipeline/mix_eos.py 700 4000 [--dir runs] [--out runs/dft_eos_700GPa_4000K.json]"""
import argparse, glob, json, os, re, sys
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("P", type=float); ap.add_argument("T", type=float); ap.add_argument("--dir", default="runs"); ap.add_argument("--out"); a = ap.parse_args()
out = a.out or os.path.join(a.dir, f"dft_eos_{a.P:g}GPa_{a.T:g}K.json")
if os.path.exists(out): sys.exit(f"{out} exists: refusing to overwrite")
src = {}
for f in sorted(glob.glob(os.path.join(a.dir, f"dft_eos_{a.P:g}GPa_*K*.json"))):
    try:
        j = json.load(open(f)); 
        if float(j["P"]) == a.P and "ice" in j and "H" in j: src.setdefault(float(j["T"]), (f, j))      # one file per temperature (the first, i.e. the plain name before *_48h)
    except Exception: continue
if not src: sys.exit(f"no state-json at {a.P:g} GPa in {a.dir}: make one with perlmutter_pwidom.sh --inspect-eos ... --save-eos first")
Ts = sorted(src); ref = src[min(Ts, key=lambda t: abs(t - a.T))][1]; new = {"P": a.P, "T": a.T}; names = ", ".join(os.path.basename(src[t][0]) for t in Ts)
for k in ("ice", "h225", "H"):
    d = dict(ref[k])
    for q in ("vpa", "B_GPa"):
        ys = np.array([float(src[t][1][k][q]) for t in Ts]); d[q] = float(np.polyval(np.polyfit(Ts, ys, 1), a.T)) if len(Ts) > 1 else float(ys[0])
    d["source"] = (f"linear-in-T mix to {a.T:g} K of {names} (T = {Ts}); starting-point volumes, not a DFT EOS at this T" if len(Ts) > 1 else f"COPY of {names} at {Ts[0]:g} K for {a.T:g} K: volumes are wrong by the thermal pressure")
    new[k] = d
if len(Ts) == 1: print(f"WARNING: only one temperature ({Ts[0]:g} K) available at {a.P:g} GPa: copying its volumes; P_host will be off by the thermal pressure of {a.T - Ts[0]:+g} K", file=sys.stderr)
if a.T < min(Ts) or a.T > max(Ts): print(f"note: {a.T:g} K is outside {Ts[0]:g}..{Ts[-1]:g} K: extrapolated", file=sys.stderr)
json.dump(new, open(out, "w"), indent=1); print(f"wrote {out}:", {k: (round(new[k]["vpa"], 4), round(new[k]["B_GPa"])) for k in ("ice", "h225", "H")})
