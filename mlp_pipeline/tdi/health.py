"""Melting / lattice-health check for every TDI run, from the saved thermo CSVs of the highest lambda values (closest to the real system).
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/health.py [dir ...] [--top 3]        (default: runs/tdi*, not runs/tdi_test)
For each tag with a lattice (bcc/fcc/insert) prints, for the last --top lambda values: u_rms (O displacement from its lattice site, centre of mass removed) averaged over the first and last quarter of the run, the largest u_max,
the Lindemann-type ratio u_rms / d_nn (d_nn = nearest-neighbour O-O distance of the ideal lattice at the run's volume) and flags: RATIO if > 0.14 (typical classical melting values 0.10-0.15), GROWTH if u_rms grew by more than 15% between the first and
the last quarter (a plateau means a stable lattice, a rise means the lattice is losing order), UMAX if u_max > 1.2 A. A flag is a warning to look closer (plain NVT run without springs), not a proof of melting: the TDI springs are still on at these lambda values."""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("--top", type=int, default=3); a = ap.parse_args()
dirs = a.dirs or [d for d in sorted(glob.glob("runs/tdi*")) if os.path.isdir(d) and os.path.basename(d) != "tdi_test"]
rows = []
for d in dirs:
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; st = json.load(open(f0)); A = st.get("args", {})
        if st["phase"] not in ("fcc", "bcc", "insert"):
            continue
        nO = st["nO"]; VO = st["N"] * st["vpa"] / nO if st["phase"] != "insert" else st["V"] / nO    # volume per O
        nn = (4 * VO) ** (1 / 3) / 2 ** 0.5 if st["phase"] in ("fcc", "insert") else (2 * VO) ** (1 / 3) * 3 ** 0.5 / 2
        for k in range(A["nodes"] - a.top, A["nodes"]):
            ft = os.path.join(d, f"{tag}_node{k}_thermo.csv")
            if not os.path.exists(ft): continue
            try: t = pd.read_csv(ft)
            except Exception: continue
            if len(t) < 8: continue
            q = max(len(t) // 4, 1); u1, u2 = t.u_rms_A.iloc[:q].mean(), t.u_rms_A.iloc[-q:].mean(); um = t.u_max_A.max()
            ratio = t.u_rms_A.mean() / nn; flags = []
            if ratio > 0.14: flags.append("RATIO")
            if u2 > 1.15 * u1: flags.append("GROWTH")
            if um > 1.2: flags.append("UMAX")
            rows.append(dict(dir=os.path.basename(d), tag=tag, lambda_idx=k, t_ps=t.t_ps.iloc[-1], T_K=st["T"], u_rms_first=u1, u_rms_last=u2, u_max=um, d_nn=nn, lindemann=ratio, flags=" ".join(flags)))
df = pd.DataFrame(rows)
if df.empty: raise SystemExit("no thermo CSVs found (run from the repo root)")
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
print(df.sort_values(["T_K", "tag", "lambda_idx"]).round(3).to_string(index=False))
bad = df[df["flags"] != ""].tag.unique()
print("\ntags with a flag:", ", ".join(bad) if len(bad) else "none")
