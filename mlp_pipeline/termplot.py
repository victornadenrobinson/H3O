#!/usr/bin/env python3
"""High-resolution plots in the terminal with braille dots (2x4 dots per character cell) and 24-bit colour. Needs numpy only; standard library otherwise.
  python mlp_pipeline/termplot.py report baseline_D.csv                       # 4 panels from baseline_D.py: force error vs force size, D vs T, D histogram per set, relative force error per set
  python mlp_pipeline/termplot.py scatter baseline_D.csv --x T_K --y D_F --color set [--ypa] [--clip 1 99] [--width 110 --height 28]
  python mlp_pipeline/termplot.py hist baseline_D.csv --y D_F --color set --bins 60 --centre         (--centre: subtract the group median of identical (set, T, nO, nH))
  columns: any numeric column of the csv; derived ones: D_F_pa (meV/atom), f_rel (force RMSE / force RMS), also D_E0, D_E, ts, f_rmse, f_rms, natoms, T_K.
As in mdview.py, the colour is one per character cell (that of the last dot written in it), so dense overlaps show the last set drawn."""
import argparse, collections, csv, math, sys
import numpy as np

PAL = [196, 39, 82, 220, 201, 51, 208, 129, 118, 213, 45, 226, 160, 99, 154, 33]
BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])


def fg(code, s): return f"\033[38;5;{code}m{s}\033[0m"


class Canvas:
    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows; self.w, self.h = cols * 2, rows * 4; self.on = np.zeros((self.h, self.w), bool); self.col = np.zeros((self.h, self.w), np.int16)

    def dots(self, x, y, c):                                   # x, y arrays in dot coordinates (y down)
        x = np.round(x).astype(int); y = np.round(y).astype(int); m = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h); self.on[y[m], x[m]] = True; self.col[y[m], x[m]] = c

    def line(self, x0, y0, x1, y1, c):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1; t = np.linspace(0, 1, n * 2); self.dots(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, c)

    def rows_text(self):
        out = []; B = self.on.reshape(self.rows, 4, self.cols, 2); C = self.col.reshape(self.rows, 4, self.cols, 2)
        for r in range(self.rows):
            line = ""
            for c in range(self.cols):
                m = int((B[r, :, c, :] * BITS).sum())
                if not m: line += " "
                else:
                    cc = C[r, :, c, :][B[r, :, c, :]]; line += fg(PAL[int(np.bincount(cc).argmax()) % len(PAL)] if cc.size else 7, chr(0x2800 + m))
            out.append(line)
        return out


def nice(lo, hi, n=5):
    if hi <= lo: hi = lo + 1
    step = 10 ** math.floor(math.log10((hi - lo) / n)); step *= min((1, 2, 5, 10), key=lambda m: abs((hi - lo) / (step * m) - n)); t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9: out.append(t); t += step
    return out


def fmt(v): return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.2e}"


def frame(cv, xlim, ylim, xlabel, ylabel, title, legend=None, tickfmt=fmt, ylog=False):
    rows = cv.rows_text(); W = max(len(fmt(t)) for t in nice(*ylim)) + 1; yt = {}
    for t in nice(*ylim, n=max(3, cv.rows // 4)):
        r = int(round((1 - (t - ylim[0]) / (ylim[1] - ylim[0])) * (cv.rows - 1)))
        if 0 <= r < cv.rows: yt[r] = fmt(t)
    out = ["\033[1m" + title + "\033[0m"] + ([legend] if legend else [])
    for r, line in enumerate(rows): out.append(yt.get(r, "").rjust(W) + "┤" + line)
    out.append(" " * W + "└" + "─" * cv.cols); ticks = nice(*xlim, n=max(3, cv.cols // 14)); lab = [" "] * (cv.cols + 12)
    for t in ticks:
        c = int(round((t - xlim[0]) / (xlim[1] - xlim[0]) * (cv.cols - 1))); s = fmt(t)
        for i, ch in enumerate(s):
            if c + i < len(lab): lab[c + i] = ch
    out.append(" " * (W + 1) + "".join(lab).rstrip() + "   " + xlabel); out.insert(2 if legend else 1, " " * W + " " + ylabel)
    return "\n".join(out)


def load(path):
    rows = list(csv.DictReader(open(path))); cols = {}
    for k in rows[0]:
        try: cols[k] = np.array([float(r[k]) for r in rows])
        except ValueError: cols[k] = np.array([r[k] for r in rows])
    cols["set"] = np.array([{"False": "F_toy", "True": "T"}.get(x, x) for x in cols["set"]])
    n = cols["natoms"]; cols["D_F_pa"] = cols["D_F"] / n * 1000; cols["D_E0_pa"] = cols["D_E0"] / n * 1000; cols["f_rel"] = cols["f_rmse"] / np.maximum(cols["f_rms"], 1e-9)
    return cols


def centred(cols, key):
    g = collections.defaultdict(list)
    for i, (s, T, nO, nH) in enumerate(zip(cols["set"], cols["T_K"], cols["nO"], cols["nH"])): g[(s, round(T / 1000), nO, nH)].append(i)
    out = cols[key].copy()
    for idx in g.values(): out[idx] = cols[key][idx] - np.median(cols[key][idx])
    return out


def categories(cols, by): u = list(dict.fromkeys(cols[by])); return u, {k: i for i, k in enumerate(u)}


def legend_line(u, cmap): return "  ".join(fg(PAL[cmap[k] % len(PAL)], "● " + str(k)) for k in u)


def scatter(cols, xk, yk, by, w, h, clip, xs=None, ys=None, title="", ref=None):
    x = cols[xk] if xs is None else xs; y = cols[yk] if ys is None else ys; u, cmap = categories(cols, by)
    lo = lambda a: np.percentile(a, clip[0]); hi = lambda a: np.percentile(a, clip[1]); xl = (lo(x), hi(x)); yl = (lo(y), hi(y)); pad = lambda l: (l[0] - 0.03 * (l[1] - l[0]), l[1] + 0.03 * (l[1] - l[0])); xl, yl = pad(xl), pad(yl)
    cv = Canvas(w, h)
    for k in u:
        m = cols[by] == k; px = (x[m] - xl[0]) / (xl[1] - xl[0]) * (cv.w - 1); py = (1 - (y[m] - yl[0]) / (yl[1] - yl[0])) * (cv.h - 1); cv.dots(px, py, cmap[k]); cv.dots(px + 1, py, cmap[k])
    for slope, c in (ref or []):
        xx = np.array([xl[0], xl[1]]); yy = slope * xx
        cv.line((xx[0] - xl[0]) / (xl[1] - xl[0]) * (cv.w - 1), (1 - (yy[0] - yl[0]) / (yl[1] - yl[0])) * (cv.h - 1), (xx[1] - xl[0]) / (xl[1] - xl[0]) * (cv.w - 1), (1 - (yy[1] - yl[0]) / (yl[1] - yl[0])) * (cv.h - 1), c)
    return frame(cv, xl, yl, xk, yk, title or f"{yk} vs {xk}", legend_line(u, cmap))


def hist(cols, yk, by, w, h, bins, centre, clip, ys=None, title=""):
    y = cols[yk] if ys is None else ys
    if centre: y = centred({**cols, "_y": y}, "_y")
    u, cmap = categories(cols, by); lo, hi = np.percentile(y, clip[0]), np.percentile(y, clip[1]); edges = np.linspace(lo, hi, bins + 1); cv = Canvas(w, h); top = 0; H = {}
    for k in u:
        v = y[cols[by] == k]; hh, _ = np.histogram(np.clip(v, lo, hi), edges); hh = hh / max(len(v), 1) / (edges[1] - edges[0]); H[k] = hh; top = max(top, hh.max())
    for k in u:
        hh = H[k]; xs_ = (edges[:-1] + edges[1:]) / 2; px = (xs_ - lo) / (hi - lo) * (cv.w - 1); py = (1 - hh / top) * (cv.h - 1)
        for i in range(len(px) - 1): cv.line(px[i], py[i], px[i + 1], py[i + 1], cmap[k])
    return frame(cv, (lo, hi), (0, top), yk + (" (centred on the group median)" if centre else ""), "density", title or f"distribution of {yk} by {by}", legend_line(u, cmap))


def ridge(cols, yk, by, w, rows_per=3, centre=True, clip=(1, 99), title="", ys=None):
    """one horizontal band per category, each a filled density outline on a shared x axis: far easier to read than 13 overlapping curves"""
    y = cols[yk] if ys is None else ys
    if centre: y = centred({**cols, "_y": y}, "_y")
    u, cmap = categories(cols, by); lo, hi = np.percentile(y, clip[0]), np.percentile(y, clip[1]); lab = 9; out = ["\033[1m" + (title or f"{yk} per cell, by {by}" + (" (centred on the group median)" if centre else "")) + "\033[0m"]
    for k in u:
        v = y[cols[by] == k]; cv = Canvas(w, rows_per); e = np.linspace(lo, hi, cv.w + 1); hh, _ = np.histogram(np.clip(v, lo, hi), e); hh = np.convolve(hh.astype(float), np.ones(3) / 3, mode="same"); hh = hh / max(hh.max(), 1e-9)
        for xdot in range(cv.w):
            n = int(round(hh[xdot] * (cv.h - 1)))
            if n > 0: cv.dots(np.full(n + 1, xdot), np.arange(cv.h - 1 - n, cv.h), cmap[k])
        z = int(round((0 - lo) / (hi - lo) * (cv.w - 1)))
        if 0 <= z < cv.w: cv.dots(np.array([z]), np.array([0]), 7)
        rows = cv.rows_text(); mid = rows_per // 2
        for r, line in enumerate(rows): out.append((str(k)[:lab - 1].ljust(lab) if r == mid else " " * lab) + line + (f"  sd {np.std(v):5.1f}  n={len(v)}" if r == mid else ""))
    out.append(" " * lab + "".join(f"{fmt(t):<{max(cv.cols // 6, 6)}}" for t in nice(lo, hi, 6)[:6]) + "  " + yk); return "\n".join(out)


def bars(cols, key, by, w, title):
    u, cmap = categories(cols, by); stats = []
    for k in u:
        v = cols[key][cols[by] == k]; stats.append((k, np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v)))
    mx = max(s[3] for s in stats) * 1.05; out = ["\033[1m" + title + "\033[0m"]
    for k, med, q1, q3, n in stats:
        a, b, c = int(q1 / mx * w), int(med / mx * w), int(q3 / mx * w); line = [" "] * (w + 1)
        for i in range(a, c): line[i] = "░"
        for i in range(0, a): line[i] = " "
        line[min(b, w)] = "█"; out.append(f"{str(k):8s}" + fg(PAL[cmap[k] % len(PAL)], "".join(line)) + f"  median {med * 100:5.1f}%  (IQR {q1 * 100:.0f}-{q3 * 100:.0f}%, n={n})")
    out.append("         " + "bar = median, shaded = interquartile range; the dashed target of the test suite is 8 %")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("mode", choices=["report", "scatter", "hist", "ridge"]); ap.add_argument("csv")
    ap.add_argument("--x", default="T_K"); ap.add_argument("--y", default="D_F_pa"); ap.add_argument("--color", default="set"); ap.add_argument("--width", type=int, default=100); ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--clip", type=float, nargs=2, default=[1, 99]); ap.add_argument("--tmin", type=float, default=500, help="report: leave out frames below this temperature in the D-vs-T panel (static 0 K frames stretch the axis)"); ap.add_argument("--bins", type=int, default=50); ap.add_argument("--centre", action="store_true"); a = ap.parse_args(); cols = load(a.csv)
    if a.mode == "scatter": print(scatter(cols, a.x, a.y, a.color, a.width, a.height, a.clip))
    elif a.mode == "ridge": print(ridge(cols, a.y, a.color, a.width - 25, 3, a.centre, a.clip))
    elif a.mode == "hist": print(hist(cols, a.y, a.color, a.width, a.height, a.bins, a.centre, a.clip))
    else:
        print(scatter(cols, "f_rms", "f_rmse", "set", a.width, a.height, [0, 99.5], title="force error against force size (lines: 8 % target, 20 %)", ref=[(0.08, 7), (0.2, 7)])); print()
        m = cols["T_K"] >= a.tmin; sub = {k: v[m] for k, v in cols.items()}
        print(scatter(sub, "T_K", "D_F", "set", a.width, a.height, a.clip, ys=centred(sub, "D_F"), title=f"D_F per cell (eV), centred on the group median, against temperature (T >= {a.tmin:.0f} K)")); print()
        print(ridge(cols, "D_F", "set", a.width - 25, 3, True, (1, 99), title="spread of D_F (eV per cell) within identical states, one band per set; thin white tick = zero")); print()
        print(bars(cols, "f_rel", "set", a.width - 35, "relative force error by set"))
