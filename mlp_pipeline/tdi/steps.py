"""Find lambda values whose integrand series drifts or jumps (a melting-like event, a slow relaxation, a defect forming) in every TDI run.
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/steps.py [dir ...] [--z 3]        (default: runs/tdi*, not runs/tdi_test)
For every lambda value of every run the series is cut into 4 quarters; z = (mean of the last quarter - mean of the first quarter) / its standard error (from 10-sample blocks inside each quarter). A value is listed if |z| > --z.
'step' = the largest change between consecutive quarters (eV, whole cell): one big step in a single interval (instead of a steady slope) is the signature of a sudden event, e.g. melting or a defect forming.
Columns: contrib = weight * (last-quarter mean - first-quarter mean) / N in meV/atom: how much this lambda value alone would move G if the early part were discarded; u_rms first/last quarter and u_max come from the thermo CSV of the same
lambda value (O displacement from its lattice site; for H / liquid / insertion runs it is the displacement of every atom, not informative). A run with many flagged high-lambda values and growing u_rms is melting or restructuring."""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("--z", type=float, default=3.0); a = ap.parse_args()
dirs = a.dirs or [d for d in sorted(glob.glob("runs/tdi*")) if os.path.isdir(d) and os.path.basename(d) != "tdi_test"]
rows, summary = [], []
def sem(x, nb=10):
    b = np.array([c.mean() for c in np.array_split(x, min(nb, max(len(x) // 10, 2)))]); return b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else np.nan
for d in dirs:
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; st = json.load(open(f0)); A = st.get("args", {}); lam, w, _ = analyze.rule_nodes(A); nflag = 0; big = 0.0
        for k in range(A["nodes"]):
            f = os.path.join(d, f"{tag}_node{k}.npy")
            try: x = np.load(f)
            except Exception: continue
            if len(x) < 80: continue
            q = np.array_split(x, 4); m = np.array([c.mean() for c in q]); se1, se4 = sem(q[0]), sem(q[3]); z = (m[3] - m[0]) / np.hypot(se1, se4)
            if not np.isfinite(z) or abs(z) < a.z: continue
            nflag += 1; contrib = w[k] * (m[3] - m[0]) / st["N"] * 1e3; step = np.diff(m)[np.argmax(np.abs(np.diff(m)))]; big = max(big, abs(contrib))
            u1 = u2 = umax = np.nan; ft = os.path.join(d, f"{tag}_node{k}_thermo.csv")
            if os.path.exists(ft):
                try:
                    t = pd.read_csv(ft); h = max(len(t) // 4, 1); u1, u2, umax = t.u_rms_A.iloc[:h].mean(), t.u_rms_A.iloc[-h:].mean(), t.u_max_A.max()
                except Exception: pass
            rows.append(dict(dir=os.path.basename(d), tag=tag, lam_idx=k, lam=lam[k], weight=w[k], z=z, step_eV=step, contrib_meV_atom=contrib, u_rms_q1=u1, u_rms_q4=u2, u_max=umax))
        summary.append(dict(dir=os.path.basename(d), tag=tag, phase=st["phase"], nodes=A["nodes"], flagged=nflag, max_contrib_meV_atom=big))
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 2000)
df = pd.DataFrame(rows)
if df.empty: print(f"no lambda value with |z| > {a.z}")
else:
    print(f"lambda values with |z| > {a.z} (last quarter vs first quarter of the series):"); print(df.sort_values(["tag", "lam_idx"]).round(3).to_string(index=False))
sm = pd.DataFrame(summary)
print("\nper run (flagged = number of lambda values with |z| > threshold; max_contrib = largest single contribution to G in meV/atom):")
print(sm.sort_values(["flagged", "max_contrib_meV_atom"], ascending=False).round(2).to_string(index=False))
