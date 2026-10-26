"""G versus simulation time for EVERY TDI run at once, as a standalone SVG (small multiples, no matplotlib needed): one panel per run showing G(t) - G(final) in meV/atom with its statistical error band,
where G(t) integrates the first t ps of every lambda series (all lambda values truncated to the same time). A converged run is a flat line at zero inside its band; a drifting run is a slope. Hover a point for its numbers.
usage (repo root, on the machine with runs/):  ~/grace-venv/bin/python mlp_pipeline/tdi/plot_all.py [dir ...] [-o docs/figures/tdi_G_vs_time.svg] [--every 0.25] [--ncol 6] [--ylim 20]
Default directories: runs/tdi* (not runs/tdi_test). Insertion rows are mu_H per added H (no PV term), liquids and pure H are per atom, crystals per atom (all include P V)."""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tdi import analyze
from tdi.gcalc import G_from_nodes

ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="*"); ap.add_argument("-o", default="tdi_G_vs_time.svg"); ap.add_argument("--every", type=float, default=0.25)
ap.add_argument("--ncol", type=int, default=6); ap.add_argument("--ylim", type=float, default=20.0); a = ap.parse_args()
dirs = a.dirs or [d for d in sorted(glob.glob("runs/tdi*")) if os.path.isdir(d) and os.path.basename(d) != "tdi_test"]
runs = []
for d in dirs:
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]; st = json.load(open(f0)); A = st.get("args", {})
        lam, w, _ = analyze.rule_nodes(A); per_ps = 1000 / A["dt"] / A["every"]; ser = []
        for k in range(A["nodes"]):
            f = os.path.join(d, f"{tag}_node{k}.npy")
            try: ser.append(np.load(f) if os.path.exists(f) else np.array([]))
            except Exception: ser.append(np.array([]))
        for k in range(A["nodes"]):   # a failed lambda value with negligible weight takes the series of the next lambda value (as summarize.py does)
            if len(ser[k]) == 0 and w[k] < 1e-4:
                j = next((j for j in range(k + 1, A["nodes"]) if len(ser[j]) > 0), None)
                if j is not None: ser[k] = ser[j].copy()
        n = min(len(x) for x in ser)
        if n < 0.5 * a.every * per_ps: continue
        ts = np.arange(a.every, n / per_ps + 1e-9, a.every)
        if len(ts) < 2: continue
        G = [G_from_nodes(st, lam, w, [x[:int(round(t * per_ps))] for x in ser], st["N"])[:2] for t in ts]
        runs.append(dict(dir=os.path.basename(d), tag=tag, phase=st["phase"], P=st["P_target"], T=st["T"], h=A.get("hper", 2.0), N=st["N"], dt=A["dt"], nodes=A["nodes"], t=ts, G=np.array([g[0] for g in G]), e=np.array([g[1] for g in G])))
if not runs: raise SystemExit("no run with at least two time points found (run from the repo root)")
runs.sort(key=lambda r: (r["phase"], r["P"], r["T"], r["h"], r["dir"], r["tag"]))
COL = {"fcc": "var(--c1)", "bcc": "var(--c2)", "liquid": "var(--c3)", "H": "var(--c4)", "insert": "var(--c5)"}
ncol = a.ncol; PW, PH = 210, 128; GX, GY = 14, 50; M = 24; nrow = (len(runs) + ncol - 1) // ncol; W = M * 2 + ncol * (PW + GX); H = 134 + nrow * (PH + GY) + 20
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
     '''<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#6b6a66;--c5:#6250d6}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#a9a89f;--c5:#9085e9}}
text{fill:var(--ink2);font-size:10px}.t{fill:var(--ink);font-size:15px;font-weight:600}.pt{fill:var(--ink);font-size:10px;font-weight:600}.g{stroke:var(--grid);stroke-width:1}.z{stroke:var(--ink2);stroke-width:1}
</style>''', f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
     f'<text class="t" x="{M}" y="26">TDI free energy versus simulation time, every run</text>',
     f'<text x="{M}" y="44">G(t) minus G(final), meV per atom (insertion: per added H). t = time per lambda value. Band = statistical error of G(t). Axis range +-{a.ylim:g} meV (clipped beyond).</text>',
     f'<text x="{M}" y="58">A converged run is flat on zero inside its band; a drifting run is a slope. Hover a point for its numbers.</text>']
lx = M
for name in ("fcc", "bcc", "liquid", "H", "insert"):
    s.append(f'<rect x="{lx}" y="74" width="12" height="12" fill="{COL[name]}"/><text x="{lx + 17}" y="84">{ {"H": "pure H", "insert": "insertion"}.get(name, name) }</text>'); lx += 90
for i, r in enumerate(runs):
    cx = M + (i % ncol) * (PW + GX); cy = 134 + (i // ncol) * (PH + GY); gf = r["G"][-1]; d = r["G"] - gf; tm = r["t"][-1]; c = COL.get(r["phase"], "var(--c4)")
    X = lambda t: cx + 28 + (t / tm) * (PW - 36); Y = lambda v: cy + PH / 2 - max(-a.ylim, min(a.ylim, v)) / a.ylim * (PH / 2 - 6)
    s.append(f'<rect x="{cx}" y="{cy}" width="{PW}" height="{PH}" fill="none" stroke="{c}" stroke-width="1.5" rx="3"/>')
    s.append(f'<text class="pt" x="{cx + 4}" y="{cy - 28}">{r["tag"]}</text><text x="{cx + 4}" y="{cy - 17}">{r["dir"]} | N {r["N"]} | dt {r["dt"]:g} | {r["nodes"]} lambda</text><text x="{cx + 4}" y="{cy - 6}">G {gf:.1f} +- {r["e"][-1]:.1f} | {tm:g} ps per lambda</text>')
    for v in (-a.ylim, 0, a.ylim):
        s.append(f'<line class="{"z" if v == 0 else "g"}" x1="{cx + 28}" x2="{cx + PW - 8}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/><text x="{cx + 24}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:g}</text>')
    up = [Y(dv - e) for dv, e in zip(d, r["e"])]; lo = [Y(dv + e) for dv, e in zip(d, r["e"])]
    pts = " ".join(f"{X(t):.1f},{y:.1f}" for t, y in zip(r["t"], up)) + " " + " ".join(f"{X(t):.1f},{y:.1f}" for t, y in zip(r["t"][::-1], lo[::-1]))
    s.append(f'<polygon points="{pts}" fill="{c}" fill-opacity="0.18"/><polyline points="' + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(r["t"], d)) + f'" fill="none" stroke="{c}" stroke-width="2"/>')
    for t, v, e, g in zip(r["t"], d, r["e"], r["G"]):
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="2.6" fill="{c}" stroke="var(--surface)" stroke-width="1"><title>{r["tag"]} ({r["dir"]}) t = {t:.2f} ps: G = {g:.2f} +- {e:.2f} meV/atom, minus final = {v:+.2f}</title></circle>')
    s.append(f'<text x="{cx + 28 + (PW - 36) / 2}" y="{cy + PH + 11}" text-anchor="middle">0 - {tm:g} ps</text>')
s.append("</svg>")
os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True); open(a.o, "w").write("\n".join(s)); print(f"wrote {a.o}: {len(runs)} runs")
