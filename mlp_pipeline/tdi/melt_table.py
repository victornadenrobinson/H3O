"""Free-NVT melting runs of H-rich fcc crystals compared by protocol (O held for HOLD ps first, or all atoms free), one row per (H per O, P, T), one column per protocol and seed.
usage (repo root, any python with pandas):  python3 mlp_pipeline/tdi/melt_table.py [dir ...]      default dirs: runs/melt_comp runs/melt
cell text: verdict (solid / marginal / melted; same rule as plot_pt.py: O lattice kept, Lindemann ratio < 0.2 solid, < 0.5 marginal, D_O < 1e-5 cm2/s) | D_H (cm2/s) | Lindemann ratio | measured P (GPa)
Protocol label: free = all atoms free from step 0, hold<ps> = O held that long first (needs rows written by the new 17_eos_md.py); _s<n> = seed (from the tag; runs without one are s0). Hold time 0 and old rows without a hold column count as free."""
import glob, os, re, sys
import numpy as np, pandas as pd
dirs = [d for d in sys.argv[1:] if not d.startswith("-")] or ["runs/melt_comp", "runs/melt"]
rows = []
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        try: t = pd.read_csv(f)
        except Exception: continue
        for _, r in t.iterrows():
            if r.get("branch", "solid") != "solid" or r.get("phase") not in ("fcc", "bcc") or "T_set" not in r: continue
            do, dl = r.get("D_O_cm2s", np.nan), r.get("delta_L", np.nan); keep = r.get("O_struct") == r["phase"] and bool(np.isfinite(do) and do < 1e-5)
            v = "solid" if keep and dl < 0.2 else "marginal" if keep and dl < 0.5 else "melted"
            hold = r.get("hold_O_ps", 0.0); hold = 0.0 if not np.isfinite(hold) else float(hold); m = re.search(r"_s(\d+)", str(r.get("tag", ""))); seed = m.group(1) if m else "0"
            rows.append(dict(phase=r["phase"], hper=round(float(r.get("hper", 2.0)), 3), P=round(float(r["P_GPa"]), -1), T=int(r["T_set"]), proto=("free" if hold == 0 else f"hold{hold:g}") + f"_s{seed}",
                             cell=f"{v} | {r.get('D_H_cm2s', np.nan):.1e} | {dl:.2f} | {r['P_GPa']:.0f}"))
if not rows: raise SystemExit("no free-NVT rows found (run from the repo root)")
df = pd.DataFrame(rows).drop_duplicates(["phase", "hper", "P", "T", "proto"], keep="last")
tab = df.pivot(index=["phase", "hper", "P", "T"], columns="proto", values="cell").fillna("")
print("verdict | D_H cm2/s | Lindemann | P_MLP GPa\n"); print(tab.to_string())
v = df.assign(v=df.cell.str.split(" ").str[0]); print("\nmelted / runs per protocol:"); print(v.groupby("proto").v.apply(lambda s: f"{(s == 'melted').sum()}/{len(s)}").to_string())
