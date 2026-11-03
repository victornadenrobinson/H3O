"""Non-blocking keyboard and mouse input for terminal viewers (standard library only). Enables xterm mouse reporting (button + drag + wheel, SGR format) and cbreak mode; always restores the terminal.
  with Controls() as ctl:
      for ev in ctl.poll(): ...        # ('rot', dx_chars, dy_chars) left-button drag, ('zoom', +1/-1) wheel, ('key', 'q'), ('arrow', 'left'|'right'|'up'|'down')
Does nothing (poll() returns []) when stdin is not a terminal. Mouse reporting needs a terminal that supports it (xterm, Windows Terminal, kitty, WezTerm, tmux with mouse on ...);
the keyboard events work everywhere."""
import os, re, select, sys

SGR = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


class Controls:
    def __init__(self, mouse=True): self.mouse = mouse; self.on = False; self.buf = ""; self.last = None

    def __enter__(self):
        if not sys.stdin.isatty(): return self
        import termios, tty
        self.fd = sys.stdin.fileno(); self.old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd); self.on = True
        if self.mouse: sys.stdout.write("\033[?1000h\033[?1002h\033[?1006h"); sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        if not self.on: return False
        import termios
        if self.mouse: sys.stdout.write("\033[?1006l\033[?1002l\033[?1000l"); sys.stdout.flush()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old); self.on = False; return False

    def poll(self):
        if not self.on: return []
        while select.select([self.fd], [], [], 0)[0]:
            chunk = os.read(self.fd, 4096)
            if not chunk: break
            self.buf += chunk.decode("utf8", "ignore")
        ev = []; b = self.buf; i = 0
        while i < len(b):
            m = SGR.match(b, i)
            if m:
                code, x, y, kind = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4); i = m.end()
                if code in (64, 65) and kind == "M": ev.append(("zoom", 1 if code == 64 else -1))
                elif code == 0 and kind == "M": self.last = (x, y)
                elif code == 32 and kind == "M" and self.last is not None: ev.append(("rot", x - self.last[0], y - self.last[1])); self.last = (x, y)
                elif kind == "m": self.last = None
                continue
            if b[i] == "\x1b":
                m = re.match(r"\x1b[\[O]([ABCD])", b[i:])
                if m: ev.append(("arrow", ARROWS[m.group(1)])); i += m.end(); continue
                if re.match(r"\x1b(\[(<[\d;]*)?)?$", b[i:]): break                  # incomplete escape sequence: wait for the rest
                i += 1; continue
            ev.append(("key", b[i])); i += 1
        self.buf = b[i:]; return ev
