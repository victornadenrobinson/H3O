"""Colourful retro 'game' progress bar, standard library only.   from gamebar import Bar;  b = Bar(total, "scoring"); b.update(i, "set A")  ...  b.done()
A rainbow health-bar that fills up, an XP-style counter, frames per second, ETA, and a LEVEL UP flash at 25 / 50 / 75 / 100 %. Falls back to a plain line every ~5 % when not on a terminal."""
import sys, time

COLS = [196, 202, 208, 214, 220, 190, 154, 118, 82, 46, 47, 48, 49, 51, 45, 39, 33, 27, 57, 93, 129, 165, 201]


def c256(n, s): return f"\033[38;5;{n}m{s}\033[0m"


class Bar:
    def __init__(self, total, label="progress", width=36, out=sys.stderr):
        self.total = max(int(total), 1); self.label = label; self.w = width; self.out = out; self.t0 = time.time(); self.tty = out.isatty(); self.level = 0; self.last = -1.0

    def update(self, i, note=""):
        frac = min(i / self.total, 1.0); dt = time.time() - self.t0; rate = i / dt if dt > 0 else 0.0; eta = (self.total - i) / rate if rate > 0 else 0.0
        lvl = int(frac * 4)
        if not self.tty:
            if frac - self.last >= 0.05 or i >= self.total: self.last = frac; print(f"[{self.label}] {100 * frac:3.0f}%  {i}/{self.total}  {rate:.1f}/s  eta {eta:.0f}s  {note}", file=self.out, flush=True)
            return
        n = int(frac * self.w); body = "".join(c256(COLS[int(k * (len(COLS) - 1) / max(self.w - 1, 1))], "█") for k in range(n))
        head = c256(231, "▓") if n < self.w else ""; rest = "\033[2m" + "░" * max(self.w - n - len(head and "x"), 0) + "\033[0m"
        em = int(eta // 60), int(eta % 60)
        line = f"\r\033[1m{self.label:<10s}\033[0m ⟦{body}{head}{rest}⟧ {c256(226, f'{100 * frac:5.1f}%')}  \033[1m{i}\033[0m/{self.total} XP  {c256(51, f'⚡{rate:5.1f}/s')}  ETA {em[0]:02d}:{em[1]:02d}  {c256(213, note[:28]):<28s}\033[K"
        self.out.write(line); self.out.flush()
        if lvl > self.level and frac < 1.0:
            self.level = lvl; self.out.write("\n" + c256(226, f"   ★ LEVEL UP! {lvl}/4 ★") + "\n"); self.out.flush()

    def done(self, msg="COMPLETE"):
        self.update(self.total)
        if self.tty: self.out.write("\n" + c256(46, f"   ✔ {msg}  ") + c256(220, f"{self.total} in {time.time() - self.t0:.0f}s") + "\n")
        self.out.flush()
