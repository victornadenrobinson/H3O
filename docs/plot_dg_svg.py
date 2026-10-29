"""G_fcc - G_bcc (meV/atom) versus pressure, our GRACE TDI results against Wilson 2013 and Cheng 2021, as a standalone SVG (no matplotlib needed; hover titles on every marker, light/dark by the viewer's theme).
usage: python3 docs/plot_dg_svg.py [csv] [out.svg]      reads docs/data/dg_fcc_bcc.csv (G rows), plots fcc - bcc (negative = fcc stable), panels 3000 K and 5000 K (+-250 K)."""
import math, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "dg_fcc_bcc.csv")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures", "dg_fcc_bcc_vs_P.svg")
d = pd.read_csv(csv, comment="#"); d = d[d.quantity == "G"].copy()
d["dG"] = d.dG_meV_atom.where(d.sign.str.strip() == "fcc-bcc", -d.dG_meV_atom)
TEMPS = [3000, 5000]; XMIN, XMAX, YMIN, YMAX = 80.0, 1100.0, -45.0, 5.0
W, H = 980, 600; PW, PH = 400, 330; X0 = [70, 540]; Y0 = 90
mine = [("this work: GRACE TDI, 108/128 O cells (dt 1 fs)", "circle", "this work, 108/128 O cells, H2O (dt 1 fs)", True),
        ("this work: GRACE TDI, 32/54 O cells (dt 0.5 fs)", "tri", "this work, 32/54 O cells, H2O (dt 0.5 fs)", True),
        ("this work: GRACE TDI, H2.25O (108/128 O, dt 0.5 fs)", "diamond", "this work, H2.25O (not H2O)", False)]


def px(i, P): return X0[i] + (math.log(P) - math.log(XMIN)) / (math.log(XMAX) - math.log(XMIN)) * PW
def py(v): return Y0 + (YMAX - v) / (YMAX - YMIN) * PH


def marker(kind, x, y, cls, title, filled=True, r=5.5):
    fill = "" if filled else ' fill="var(--surface)"'
    if kind == "circle": m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"{fill}'
    elif kind == "tri": m = f'<path d="M{x:.1f} {y - r - 1:.1f} L{x + r + 1:.1f} {y + r:.1f} L{x - r - 1:.1f} {y + r:.1f} Z"{fill}'
    elif kind == "diamond": m = f'<path d="M{x:.1f} {y - r - 1.5:.1f} L{x + r + 1.5:.1f} {y:.1f} L{x:.1f} {y + r + 1.5:.1f} L{x - r - 1.5:.1f} {y:.1f} Z"{fill}'
    else: m = f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}"{fill}'
    return f'{m} class="{cls}"><title>{title}</title>' + {"circle": "</circle>", "rect": "</rect>"}.get(kind, "</path>")


s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">',
     '''<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--lit:#6b6a66;--mine:#2a78d6}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#34342f;--lit:#a9a89f;--mine:#3987e5}}
text{fill:var(--ink2);font-size:12px}.t{fill:var(--ink);font-size:15px;font-weight:600}.st{font-size:12px}.lab{fill:var(--ink);font-size:12px}
.grid{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--ink2);stroke-width:1}.axis{stroke:var(--ink2);stroke-width:1}
.cheng{fill:none;stroke:var(--lit);stroke-width:2}.chengdot{fill:var(--lit)}.wil{fill:var(--surface);stroke:var(--lit);stroke-width:2}.eb{stroke:var(--lit);stroke-width:1.5}
.mine{fill:var(--mine);stroke:var(--surface);stroke-width:2}.mineh{stroke:var(--mine);stroke-width:2}.mineeb{stroke:var(--mine);stroke-width:1.5}
</style>''', f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
     '<text class="t" x="24" y="30">fcc vs bcc superionic H2O: free-energy difference at 400 GPa-class pressures</text>',
     '<text class="st" x="24" y="50">G(fcc) - G(bcc), meV per atom (negative = fcc more stable). Error bars: statistical only. Hover a marker for its details.</text>']
for i, T in enumerate(TEMPS):
    x0 = X0[i]
    s.append(f'<text class="t" x="{x0}" y="{Y0 - 14}" style="font-size:13px">{T} K</text>')
    for v in range(-40, 10, 10):
        s.append(f'<line class="{"zero" if v == 0 else "grid"}" x1="{x0}" x2="{x0 + PW}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>')
        if i == 0: s.append(f'<text x="{x0 - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for P in (100, 200, 300, 500, 1000):
        s.append(f'<line class="grid" x1="{px(i, P):.1f}" x2="{px(i, P):.1f}" y1="{Y0}" y2="{Y0 + PH}"/><text x="{px(i, P):.1f}" y="{Y0 + PH + 18}" text-anchor="middle">{P}</text>')
    s.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="{Y0}" y2="{Y0 + PH}"/><line class="axis" x1="{x0}" x2="{x0 + PW}" y1="{Y0 + PH}" y2="{Y0 + PH}"/>')
    s.append(f'<text x="{x0 + PW / 2}" y="{Y0 + PH + 38}" text-anchor="middle">pressure (GPa, log scale)</text>')
    dd = d[(d.T_K - T).abs() <= 250]; labels = []
    ch = dd[dd.source.str.startswith("Cheng") & (dd.T_K == T)].sort_values("P_GPa"); ch = ch[(ch.P_GPa >= XMIN) & (ch.P_GPa <= XMAX)]
    if len(ch):
        s.append('<polyline class="cheng" points="' + " ".join(f"{px(i, p):.1f},{py(v):.1f}" for p, v in zip(ch.P_GPa, ch.dG)) + '"/>')
        s.append(f'<text class="lab" x="{px(i, ch.P_GPa.iloc[-1]) + 8:.1f}" y="{py(ch.dG.iloc[-1]) + 4:.1f}" style="fill:var(--lit)">Cheng 2021</text>')
    for _, r in dd[dd.source.str.startswith("Wilson") & (dd.T_K == T)].iterrows():
        if not (XMIN <= r.P_GPa <= XMAX): continue
        x, y = px(i, r.P_GPa), py(r.dG); e = r.err_meV_atom * PH / (YMAX - YMIN)
        s.append(f'<line class="eb" x1="{x:.1f}" x2="{x:.1f}" y1="{y - e:.1f}" y2="{y + e:.1f}"/>' + marker("rect", x, y, "wil", f"Wilson 2013 (DFT): {r.dG:.1f} +- {r.err_meV_atom:.1f} meV/atom at {r.P_GPa:.0f} GPa, {r.T_K:.0f} K", True, 5))
    for src, kind, name, filled in mine:
        for _, r in dd[dd.source == src].iterrows():
            x, y = px(i, r.P_GPa), py(r.dG); e = r.err_meV_atom * PH / (YMAX - YMIN)
            s.append(f'<line class="mineeb" x1="{x:.1f}" x2="{x:.1f}" y1="{y - e:.1f}" y2="{y + e:.1f}"/>' + marker(kind, x, y, "mine" if filled else "mineh", f"{name}: {r.dG:.1f} +- {r.err_meV_atom:.1f} meV/atom at {r.P_GPa:.0f} GPa, {r.T_K:.0f} K", filled))
            labels.append([x - 10 if r.P_GPa >= 300 else x + 10, y + 4, f"{r.dG:.1f}", "end" if r.P_GPa >= 300 else "start"])
    labels.sort(key=lambda l: l[1])
    for k in range(1, len(labels)):   # push labels apart (>= 14 px) so close points do not print on top of each other
        labels[k][1] = max(labels[k][1], labels[k - 1][1] + 14)
    for lx_, ly_, tx_, an_ in labels:
        s.append(f'<text class="lab" x="{lx_:.1f}" y="{ly_:.1f}" text-anchor="{an_}">{tx_}</text>')
s.append(f'<text x="{X0[0] - 52}" y="{Y0 + PH / 2}" transform="rotate(-90 {X0[0] - 52} {Y0 + PH / 2})" text-anchor="middle">G(fcc) - G(bcc)  (meV/atom)</text>')
ly = H - 70; lx = 70
s.append(f'<line class="cheng" x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}"/><text x="{lx + 28}" y="{ly + 4}">Cheng 2021 (NNP + FEP to PBE)</text>')
lx += 250; s.append(marker("rect", lx + 8, ly, "wil", "", True, 5) + f'<text x="{lx + 22}" y="{ly + 4}">Wilson 2013 (DFT, 32/54 molecules)</text>')
ly += 24; lx = 70
for src, kind, name, filled in mine:
    s.append(marker(kind, lx + 6, ly, "mine" if filled else "mineh", "", filled) + f'<text x="{lx + 18}" y="{ly + 4}">{name}</text>'); lx += 18 + 6.6 * len(name) + 22
s.append(f'<text x="24" y="{H - 24}" style="font-size:11px">Not shown: Wilson at 4000 GPa (3000 K: -71.3 +- 2.3). Ours: MLP level (fine-tuned GRACE), classical nuclei, no correction to PBE, statistical errors only.</text>')
s.append(f'<text x="24" y="{H - 8}" style="font-size:11px">The 32/54 O points were still drifting down by 4-5 meV/atom; the 108/128 O points at 5000 K ran 1.5 ps per lambda value.</text>')
s.append("</svg>")
open(out, "w").write("\n".join(s)); print("wrote", out)
