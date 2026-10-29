"""Corrected start volumes for crystal runs whose MLP pressure at the highest lambda missed the target (QA-excluded in report.py, e.g. the 100 GPa fcc / bcc runs:
P_hi 116-121 vs 100 GPa from an extrapolated DFT EOS):  V_new = V_old * exp((P_hi - P_target) / B), B from the run's stage0 json (else 4.5 P as a rough guess).
One step of this is good to ~1-2 % in P for misses of ~20 %; rerun into a NEW directory (another volume = another run: never mix samples).
usage (repo root):  python3 mlp_pipeline/tdi/vpa_fix.py runs/checks/summary.csv [--tol 0.03] [--only-dir tdi_fb] [--runs-root runs]
Prints one line per run and, for fcc / bcc pairs at the same (P, T), a POINTS string for young_tdi_fccbcc.sbatch:  POINTS="P T V_fcc V_bcc;..."
"""
import argparse, csv, json, math, os


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("summary"); ap.add_argument("--tol", type=float, default=0.03, help="relative pressure miss above which a run is corrected")
    ap.add_argument("--only-dir", default=None); ap.add_argument("--runs-root", default="runs"); a = ap.parse_args()
    fix = {}
    for r in csv.DictReader(open(a.summary)):
        if r["phase"] not in ("fcc", "bcc") or (a.only_dir and r["dir"] != a.only_dir): continue
        try:
            P, Phi, V = float(r["P_target_GPa"]), float(r["P_hi_GPa"]), float(r["vpa_A3"])
        except ValueError:
            continue
        if not math.isfinite(Phi) or abs(Phi - P) / P <= a.tol: continue
        B = float("nan")
        f = os.path.join(a.runs_root, r["dir"], f"{r['tag']}_stage0.json")
        if os.path.exists(f):
            st = json.load(open(f)); B = st.get("B_GPa") or st.get("args", {}).get("bulk_gpa") or float("nan")
        B = B if B and math.isfinite(B) else 4.5 * P
        Vn = V * math.exp((Phi - P) / B)
        print(f"{r['dir'] + '/' + r['tag']:40s} P {P:6.0f} T {float(r['T_K']):6.0f}  P_hi {Phi:6.1f}  B {B:6.0f}  V {V:.5f} -> {Vn:.5f} A^3/atom  ({100 * (Vn / V - 1):+.2f} %)")
        fix[(r["phase"], P, float(r["T_K"]))] = Vn
    pts = [f"{P:g} {T:g} {fix[('fcc', P, T)]:.5f} {fix[('bcc', P, T)]:.5f}" for (ph, P, T) in sorted(fix) if ph == "fcc" and ("bcc", P, T) in fix]
    if pts: print('\nPOINTS="' + ";".join(pts) + '"')


if __name__ == "__main__":
    main()
