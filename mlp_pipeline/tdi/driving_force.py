"""What drives H into superionic ice? Decomposition of the dilute transfer free energy (one H from pure H into fcc H2O, insertion TDI) at fixed P:
    dg = du + P dv - T ds           (kinetic energy cancels: 3/2 kT per H in both)
  dg(T)  : mu0 = mu_H - G_H (n -> 0) per (P, T), docs/data/insertion_regsol_fit.csv (or --mu0 csv with P_GPa,T_K,mu0,mu0_err)
  ds     : -d(dg)/dT from a weighted linear fit over the temperatures at each P (needs >= 2 T)
  dh     : dg + T ds;   P dv : P (v_H(ice) - v(pure H));   du = dh - P dv
  v_H(ice): partial volume of the added H from the insertion runs' stage 0 (V (P_full - P_host) / (n B)), mean over n >= 2 runs at that (P, T) (n = 1 is noisy)
  v(pure H): V / N of the pure-H TDI run at (P, T) (its stage0 json vpa), corrected to P_target with its bulk modulus when the json records the MLP pressure.
Reading: P dv < 0 = denser packing of H in the ice (pressure pushes H in); du < 0 = H binds more strongly in ice; -T ds > 0 = H loses entropy on entering the ice.
usage (repo root, login node):  python3 mlp_pipeline/tdi/driving_force.py [--roots runs/tdi runs/tdi_prep runs/tdi_long] [--mu0 docs/data/insertion_regsol_fit.csv]
"""
import argparse, csv, glob, json, os
import numpy as np

KB = 8.617333262e-5; EV_A3_GPA = 160.21766


def rows(f):
    return [r for r in csv.DictReader(l for l in open(f) if not l.startswith("#"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", default=None); ap.add_argument("--mu0", default="docs/data/insertion_regsol_fit.csv"); a = ap.parse_args()
    roots = a.roots or sorted(d for d in glob.glob("runs/tdi*") if os.path.isdir(d) and "test" not in d)
    vH, vpure = {}, {}
    for f in sorted(sum((glob.glob(os.path.join(r, "*_stage0.json")) for r in roots), [])):
        try: st = json.load(open(f)); g = st["args"]
        except Exception: continue
        key = (float(st["P_target"]), float(st["T"]))
        if st.get("phase") == "insert" and g.get("host_phase", "fcc") == "fcc" and abs(float(g.get("hper", 2.0)) - 2) < 1e-9 and "Hhost" not in f:
            n = int(st["N"]); Ph, Pf, B = st.get("P_host_GPa"), st.get("P_full_GPa"), st.get("B_GPa")
            if n >= 2 and all(x is not None and np.isfinite(x) for x in (Ph, Pf, B)): vH.setdefault(key, []).append(st["V"] * (Pf - Ph) / (n * B))
        elif st.get("phase") == "H" and not g.get("hhwall") and not ("_rep" in f):
            v = float(st["vpa"]); P_m, B = st.get("P_mlp", np.nan), st.get("B_GPa", np.nan)
            if np.isfinite(P_m) and np.isfinite(B): v *= np.exp((P_m - key[0]) / B)   # to the volume at P_target
            vpure.setdefault(key, []).append(v)
    mu = {}
    for r in rows(a.mu0):
        mu.setdefault(float(r["P_GPa"]), []).append((float(r["T_K"]), float(r["mu0"]), float(r["mu0_err"])))
    print("Dilute transfer of one H from pure H into superionic fcc H2O:  dg = du + P dv - T ds   (meV per H)")
    print(f"{'P/GPa':>6s} {'T/K':>6s} {'dg':>8s} {'-T ds':>8s} {'dh':>8s} {'P dv':>8s} {'du':>8s}   {'v_H ice':>8s} {'v pure H':>8s}  ds/k_B")
    for P in sorted(mu):
        d = sorted(mu[P]); T = np.array([x[0] for x in d]); g = np.array([x[1] for x in d]); e = np.array([x[2] for x in d])
        if len(T) >= 2:
            w = 1 / e ** 2; A = np.vstack([np.ones_like(T), T - T.mean()]).T; cov = np.linalg.inv(A.T @ (A * w[:, None])); b = cov @ (A.T @ (w * g))
            ds, eds = -b[1], np.sqrt(cov[1, 1])   # meV/K
        else:
            ds, eds = np.nan, np.nan
        for Ti, gi in zip(T, g):
            vh = np.mean(vH.get((P, Ti), [])) if vH.get((P, Ti)) else np.mean(sum((v for k, v in vH.items() if k[0] == P), [])) if any(k[0] == P for k in vH) else np.nan
            vp = np.mean(vpure[(P, Ti)]) if (P, Ti) in vpure else np.nan
            Pdv = P * (vh - vp) / EV_A3_GPA * 1e3; TdS = -Ti * ds; dh = gi - TdS; du = dh - Pdv
            print(f"{P:6.0f} {Ti:6.0f} {gi:8.0f} {TdS:8.0f} {dh:8.0f} {Pdv:8.0f} {du:8.0f}   {vh:8.2f} {vp:8.2f}  {ds / (KB * 1e3):+.2f} +- {eds / (KB * 1e3):.2f}")
    print("\nP dv < 0: H packs more densely in the ice (pressure pushes it in);  du < 0: H is bound more strongly in the ice;  -T ds > 0: H loses entropy on entering the ice.")
    print("ds comes from the T dependence of dg at fixed P (few temperatures, large error); v_H(ice) is the insertion runs' stage-0 partial volume (n >= 2; other T at the same P if none).")


if __name__ == "__main__":
    main()
