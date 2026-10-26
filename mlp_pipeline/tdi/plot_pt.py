"""P-T map of every simulation we have, one panel per composition (H per O; pure H), as a standalone SVG (no matplotlib).
usage (repo root):  ~/grace-venv/bin/python mlp_pipeline/tdi/plot_pt.py [-o runs/pt_map.svg]
Sources: TDI runs (runs/tdi*/*_stage0.json: filled symbols, shape by phase: fcc square, bcc triangle, liquid circle, pure H diamond, insertion cross-hair circle; plotted at the target pressure) and free NVT runs written by 17_eos_md.py
(runs/melt_tdi, runs/melt, runs/eos_scan: rings; plotted at the MLP pressure measured in the run). Free-NVT verdict (same rule as 21_melt_summary.py, three-way): solid = O lattice kept (fcc label, Lindemann ratio < 0.2, D_O < 1e-5 cm2/s),
marginal = lattice kept but 0.2 <= ratio < 0.5, melted = anything else on the solid branch; liquid branch: liquid = melted, 'not melted' = the pre-melting step failed. Hover a symbol for its details."""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("-o", default="runs/pt_map.svg"); ap.add_argument("--ncol", type=int, default=3); a = ap.parse_args()
pts = []   # dict(kind, comp, P, T, label, status)
for d in sorted(glob.glob("runs/tdi*")):
    if not os.path.isdir(d) or os.path.basename(d) == "tdi_test": continue
    for f0 in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        tag = os.path.basename(f0)[:-len("_stage0.json")]
        try: st = json.load(open(f0))
        except Exception: continue
        if not glob.glob(os.path.join(d, f"{tag}_node*.npy")): continue
        h = "pure H" if st["phase"] == "H" else float(st.get("args", {}).get("hper", 2.0))
        pts.append(dict(src="tdi", kind=st["phase"], comp=h, P=float(st["P_target"]), T=float(st["T"]), label=f"TDI {tag} ({os.path.basename(d)}), N {st['N']}"))
def verdict(r):
    if r.get("phase") == "H": return "fluid"
    lab = r.get("O_struct"); dl = r.get("delta_L", np.nan); do = r.get("D_O_cm2s", np.nan); ph = r.get("phase", "fcc")
    keep = (lab == ph) and (do < 1e-5 if np.isfinite(do) else False)
    solid = keep and dl < 0.2; marg = keep and 0.2 <= dl < 0.5
    if r.get("branch", "solid") == "liquid": return "not melted" if (solid or marg) else "liquid"
    return "solid" if solid else "marginal" if marg else "melted"
for d in ("runs/melt_tdi", "runs/melt", "runs/eos_scan"):
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        try: t = pd.read_csv(f)
        except Exception: continue
        for _, r in t.iterrows():
            r = r.to_dict(); r.setdefault("branch", "solid")
            if "T_set" not in r or "P_GPa" not in r: continue
            h = "pure H" if r.get("phase") == "H" else float(r["hper"]) if "hper" in r and np.isfinite(r.get("hper", np.nan)) else 2.0
            v = verdict(r); pts.append(dict(src="nvt", kind=v, comp=h, P=float(r["P_GPa"]), T=float(r["T_mean"] if np.isfinite(r.get("T_mean", np.nan)) else r["T_set"]), label=f"free NVT {os.path.basename(f)[:-4]} ({d}): {v}, O {r.get('O_struct','-')}, delta_L {r.get('delta_L', float('nan')):.2f}, D_O {r.get('D_O_cm2s', float('nan')):.1e}, P {r['P_GPa']:.0f} GPa"))
if not pts: raise SystemExit("no runs found (run from the repo root)")
order = sorted({p["comp"] for p in pts}, key=lambda c: (1e9 if c == "pure H" else c))
SH = {"fcc": "sq", "bcc": "tri", "liquid": "circ", "H": "dia", "insert": "circx"}
COL = {"fcc": "var(--c1)", "bcc": "var(--c2)", "liquid": "var(--c3)", "H": "var(--c4)", "insert": "var(--c5)",
       "solid": "var(--c1)", "marginal": "var(--warn)", "melted": "var(--bad)", "liquid_n": "var(--c3)", "fluid": "var(--c4)", "not melted": "var(--c5)"}
NV = {"solid": "o", "marginal": "dia", "melted": "x", "liquid": "o", "fluid": "o", "not melted": "x"}
PW, PH = 330, 250; GX, GY = 24, 50; M = 24; ncol = a.ncol; nrow = (len(order) + ncol - 1) // ncol; W = 2 * M + ncol * (PW + GX); H = 150 + nrow * (PH + GY)
X0, X1, Y0, Y1 = 50.0, 1300.0, 2000.0, 10000.0
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
     '''<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#6b6a66;--c5:#6250d6;--warn:#eda100;--bad:#e34948}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#a9a89f;--c5:#9085e9;--warn:#c98500;--bad:#e66767}}
text{fill:var(--ink2);font-size:11px}.t{fill:var(--ink);font-size:15px;font-weight:600}.pt{fill:var(--ink);font-size:12px;font-weight:600}.g{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--ink2);stroke-width:1}
</style>''', f'<rect width="{W}" height="{H}" fill="var(--surface)"/>', f'<text class="t" x="{M}" y="26">Every simulation in P-T space, one panel per composition (H per O)</text>',
     f'<text x="{M}" y="44">Filled symbols: TDI runs at their target P and T. Rings and crosses: free NVT runs (springs off) at their measured P and T, coloured by what the O lattice did. Hover a symbol.</text>']
def mark(kind, x, y, col, fill=True, r=5.5, title=""):
    f = f'fill="{col}"' if fill else 'fill="none"'; st = f'stroke="{col}" stroke-width="{1.5 if fill else 2}"'
    if kind == "sq": m = f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" {f} {st}'
    elif kind == "tri": m = f'<path d="M{x:.1f} {y - r - 1:.1f} L{x + r + 1:.1f} {y + r:.1f} L{x - r - 1:.1f} {y + r:.1f} Z" {f} {st}'
    elif kind == "dia": m = f'<path d="M{x:.1f} {y - r - 1.5:.1f} L{x + r + 1.5:.1f} {y:.1f} L{x:.1f} {y + r + 1.5:.1f} L{x - r - 1.5:.1f} {y:.1f} Z" {f} {st}'
    elif kind == "x": return f'<path d="M{x - r:.1f} {y - r:.1f} L{x + r:.1f} {y + r:.1f} M{x - r:.1f} {y + r:.1f} L{x + r:.1f} {y - r:.1f}" stroke="{col}" stroke-width="2.4" fill="none"><title>{title}</title></path>'
    else: m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {f} {st}'
    return f'{m}><title>{title}</title></' + ("rect" if kind == "sq" else "circle" if kind in ("circ", "circx", "o") else "path") + '>'
ly = 66; lx = M
for lab, kind, col, fl in (("TDI fcc", "sq", "var(--c1)", True), ("TDI bcc", "tri", "var(--c2)", True), ("TDI liquid", "circ", "var(--c3)", True), ("TDI pure H", "dia", "var(--c4)", True), ("TDI insertion", "circx", "var(--c5)", True),
                           ("NVT solid", "o", "var(--c1)", False), ("NVT marginal", "dia", "var(--warn)", False), ("NVT melted", "x", "var(--bad)", False), ("NVT liquid branch", "o", "var(--c3)", False), ("NVT pure H", "o", "var(--c4)", False)):
    s.append(mark(kind, lx + 7, ly, col, fl, 5) + f'<text x="{lx + 17}" y="{ly + 4}">{lab}</text>'); lx += 118 if lx < 600 else 126
    if lx > W - 140: lx = M; ly += 22
for i, c in enumerate(order):
    cx = M + (i % ncol) * (PW + GX); cy = 150 + (i // ncol) * (PH + GY) + 6; X = lambda p: cx + 36 + (p - X0) / (X1 - X0) * (PW - 46); Y = lambda t: cy + PH - 28 - (t - Y0) / (Y1 - Y0) * (PH - 40)
    s.append(f'<text class="pt" x="{cx + 36}" y="{cy - 6}">{"pure H (250 atoms)" if c == "pure H" else f"H{c:g}O"}</text><rect x="{cx + 36}" y="{cy + 12 - 6 + 0}" width="0" height="0"/>')
    for t in range(2000, 10001, 2000): s.append(f'<line class="g" x1="{X(X0):.1f}" x2="{X(X1):.1f}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/><text x="{cx + 32}" y="{Y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for p in range(200, 1301, 200): s.append(f'<line class="g" x1="{X(p):.1f}" x2="{X(p):.1f}" y1="{Y(Y0):.1f}" y2="{Y(Y1):.1f}"/><text x="{X(p):.1f}" y="{cy + PH - 12}" text-anchor="middle">{p}</text>')
    s.append(f'<line class="ax" x1="{X(X0):.1f}" x2="{X(X0):.1f}" y1="{Y(Y0):.1f}" y2="{Y(Y1):.1f}"/><line class="ax" x1="{X(X0):.1f}" x2="{X(X1):.1f}" y1="{Y(Y0):.1f}" y2="{Y(Y0):.1f}"/><text x="{(X(X0) + X(X1)) / 2:.1f}" y="{cy + PH + 2}" text-anchor="middle">pressure (GPa)</text>')
    if i % ncol == 0: s.append(f'<text x="{cx + 4}" y="{cy + PH / 2}" transform="rotate(-90 {cx + 4} {cy + PH / 2})" text-anchor="middle">temperature (K)</text>')
    for p in [q for q in pts if q["comp"] == c and q["src"] == "nvt"] + [q for q in pts if q["comp"] == c and q["src"] == "tdi"]:
        x, y = X(p["P"]), Y(p["T"])
        if not (X0 <= p["P"] <= X1 and Y0 <= p["T"] <= Y1): continue
        if p["src"] == "tdi": s.append(mark(SH.get(p["kind"], "circ"), x + {"bcc": -7, "insert": 7}.get(p["kind"], 0), y, COL.get(p["kind"], "var(--c4)"), True, 5.5, p["label"]))   # small x offsets so bcc / insertion symbols do not hide under fcc
        else: s.append(mark(NV.get(p["kind"], "o"), x, y, COL["liquid_n" if p["kind"] == "liquid" else p["kind"]], False, 5.5, p["label"]))
s.append("</svg>")
os.makedirs(os.path.dirname(os.path.abspath(a.o)), exist_ok=True); open(a.o, "w").write("\n".join(s)); print(f"wrote {a.o}: {len(pts)} simulations in {len(order)} panels ({sum(p['src'] == 'tdi' for p in pts)} TDI, {sum(p['src'] == 'nvt' for p in pts)} free NVT)")
