"""Pressure-correction audit: how far each run's MLP pressure is from its target, and how much each method's correction (applied or not) moves mu_H - G_H.

Why it matters: every run is NVT at a volume from the DFT EOS, so the MLP pressure misses the target P_t by a few to ~15 GPa. Three treatments:
 (1) whole-system TDI (fcc / bcc crystals, liquid, pure H):  G = F(V) + P_t V.  Exact to FIRST order (dG/dP = V); the remaining SECOND-order error is
       dG_P2 = -V (P - P_t)^2 / (2 B)  (summarize.py column dG_P2, NOT applied); per added H in a mixing value it is combined as (N_m d_m - N_w d_w - n d_H) / n.
 (2) insertion TDI (ghost protons, fixed V):  F(N+n, V) - F(N, V) is mu_H at the cell's own pressure (~ P_mid = (P_host + P_full) / 2), not at P_t. Stage-0 correction
       dG_P = -V / (2 B) [(P_full - P_t)^2 - (P_host - P_t)^2]  ~=  -v_H (P_mid - P_t),  v_H = V (P_full - P_host) / (n B)  (partial volume of the added H)
       is FIRST order: with P_host 10-15 GPa off and v_H ~ 0.9 A^3 it is ~50-90 meV per H. Runs from before the stage-0 code (or with --vpa and no B) have none
       ("uncorrected basis"); report.py puts a (P, T) group on the uncorrected basis if any of its rows lacks it.
 (3) pwidom (no lambda path, fixed V): the same formula as (2) with its own free-MD pressures, ALWAYS applied, for the host AND for the pure-H reservoir (whose
       fixed-V mu is likewise at the pure-H cell's pressure). The two corrections partly cancel in dg.
So uncorrected insertion values and pwidom / crystal values differ by about v_H (P_mid - P_t) - v_H_pureH (P_H - P_t): tens of meV per H, comparable to the effects compared.

usage (repo root, Young):  python3 mlp_pipeline/tdi/pcorr_audit.py [--points "400 5000;500 6000"] [--roots runs/tdi runs/tdi_long ...] [--pwidom "runs/pwidom*"]
"""
import argparse, glob, json, os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwidom_scan import mu_P   # robust pwidom pressure correction (v_H fixed)

EV_A3_GPA = 160.21766
VH = 1.0   # pooled partial volume of an added H in the ice hosts (A^3, insertion n >= 2), as pwidom_scan.VH_ICE


def stage0s(roots):
    for f in sorted(sum((glob.glob(os.path.join(r, "*_stage0.json")) for r in roots), [])):
        if "_rep" in f or "tdi_test" in f: continue
        try: yield f, json.load(open(f))
        except Exception: continue


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", default=None); ap.add_argument("--points", default=None); ap.add_argument("--pwidom", default="runs/pwidom*")
    a = ap.parse_args()
    roots = a.roots or sorted(d for d in glob.glob("runs/tdi*") if os.path.isdir(d))
    want = {tuple(map(float, p.split())) for p in a.points.split(";")} if a.points else None
    ins, whole = {}, {}
    for f, st in stage0s(roots):
        key = (float(st["P_target"]), float(st["T"]))
        if want and key not in want: continue
        tag = os.path.basename(f)[:-len("_stage0.json")]
        if st.get("phase") == "insert":
            n = int(st["N"]); V = st["V"]; B = st.get("B_GPa", np.nan); Ph, Pf = st.get("P_host_GPa", np.nan), st.get("P_full_GPa", np.nan)
            vH = V * (Pf - Ph) / (n * B) if np.isfinite(Pf) else np.nan
            first = -vH * ((Ph + Pf) / 2 - key[0]) / EV_A3_GPA * 1e3 if np.isfinite(vH) else np.nan
            vn = VH * n; robust = (-vn * (Ph - key[0]) - vn * vn * B / (2 * V)) / n / EV_A3_GPA * 1e3 if np.isfinite(Ph) else np.nan   # v_H fixed at the pooled value
            first = (first, robust)
            ins.setdefault(key, []).append((tag, os.path.dirname(f), n, Ph, Pf, B, vH, st.get("dG_P_eV", np.nan) / n * 1e3 if "dG_P_eV" in st else np.nan, first))
        else:
            P = st.get("P_mlp", np.nan); N = st.get("N", np.nan); B = st.get("B_GPa", np.nan)
            d2 = -st["V"] * (P - key[0]) ** 2 / (2 * B) / EV_A3_GPA * 1e3 / N if np.isfinite(P) and np.isfinite(B) else np.nan
            whole.setdefault(key, []).append((tag, os.path.dirname(f), st.get("phase"), float(st.get("args", {}).get("hper", 2.0)), P, B, d2))
    pw = {}
    for f in sorted(glob.glob(os.path.join(a.pwidom, "*.json"))):
        try: r = json.load(open(f))
        except Exception: continue
        if "dG_P_meV" not in r: continue
        key = (float(r["P_target"]), float(r["T"]))
        if want and key not in want: continue
        pw.setdefault(key, []).append((os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f), r["phase"], r.get("hper", 2.0), r["P_host_GPa"], r["P_full_GPa"], r["B_GPa"], r["mu_fixedV_meV"], r["dG_P_meV"], r))
    for key in sorted(set(ins) | set(whole) | set(pw)):
        P, T = key; print(f"\n=========== {P:.0f} GPa {T:.0f} K ===========")
        if key in whole:
            print("  whole-system TDI (G = F + P_t V, first order exact): P_mlp (stage 0, lambda = 1 or nan) and the 2nd-order term NOT applied, meV/atom")
            for tag, d, ph, h, Pm, B, d2 in whole[key]:
                print(f"     {os.path.basename(d) + '/' + tag:45s} {ph:7s} H/O {h:<5g} P_mlp {Pm:7.1f} (miss {Pm - P:+6.1f}) B {B:6.0f}  dG_P2 {d2:7.2f} meV/atom")
        if key in ins:
            print("  insertion TDI (fixed V; stage-0 correction, FIRST order): per added H, meV.  applied = dG_P in the json (nan = run on the uncorrected basis)")
            for tag, d, n, Ph, Pf, B, vH, dGP, first in ins[key]:
                print(f"     {os.path.basename(d) + '/' + tag:45s} n {n:2d}  P_host {Ph:7.1f} P_full {Pf:7.1f} (P_mid miss {(Ph + Pf) / 2 - P:+6.1f})  B {B:6.0f}  v_H {vH:5.2f} A^3  "
                      f"dG_P applied {dGP:7.1f}  (first order {first[0]:7.1f}; robust, v_H = {VH} A^3: {first[1]:7.1f})")
        if key in pw:
            print("  pwidom (fixed V -> P_t, always applied): host and pure-H corrections and the net shift of dg, meV per H")
            hs = [r for r in pw[key] if r[1] == "H"]
            for f, ph, h, Ph, Pf, B, muV, dGP, r in pw[key]:
                rob = mu_P(r)[0] - muV
                print(f"     {f:60s} {'pure H' if ph == 'H' else 'H' + format(h if h is not None else 2.0, '.4g') + 'O':9s} P_host {Ph:7.1f} P_+1H {Pf:7.1f}  B {B:6.0f}  mu fixed V {muV:8.1f}  dG_P measured {dGP:7.1f}  robust {rob:7.1f}")
            for f, ph, h, Ph, Pf, B, muV, dGP, r in [x for x in pw[key] if x[1] != "H"]:
                for hx in hs[:1]:
                    hr = hx[-1]; print(f"     net shift of dg for {os.path.basename(f)} (host - pure H): measured {dGP - hx[7]:+7.1f}   robust {(mu_P(r)[0] - r['mu_fixedV_meV']) - (mu_P(hr)[0] - hr['mu_fixedV_meV']):+7.1f} meV per H")
    print("\nReading: an insertion row with 'dG_P applied nan' is on the uncorrected basis; its mu is high by about -(first-order estimate). Crystal mixing values are"
          " corrected to first order; their residual is the dG_P2 column combined per added H (summarize P2_corr_per_H). pwidom dg already includes the net shift.")


if __name__ == "__main__":
    main()
