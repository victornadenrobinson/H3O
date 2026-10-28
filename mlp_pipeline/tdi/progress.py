"""Progress of MD runs for ETAs: an observer attached to an ASE dynamics object (it only reads the step counter and the clock, so the trajectory is unchanged).
  StepProgress(dyn, total, base, path=<tag>_node<k>_progress.json)   the lambda runs: a small json rewritten every 30 s (steps done of this lambda, its total, recent speed);
                                                                      tdi/eta.py adds them up to a whole-job ETA (young_tdi_par.sh prints it)
  StepProgress(dyn, total, label='lambda=1 equilibration', print_s=60)  stage 0 / prep: a progress line every minute instead of minutes of silence
"""
import json, os, time
from collections import deque


def fmt_eta(rem_s):
    """'ETA 13:42 (in 25 min)' / '(in 3.2 h)'"""
    if not rem_s == rem_s or rem_s < 0:   # nan
        return "ETA ?"
    when = time.strftime("%H:%M", time.localtime(time.time() + rem_s)) + ("" if rem_s < 20 * 3600 else time.strftime(" %a", time.localtime(time.time() + rem_s)))
    return f"ETA {when} (in {rem_s / 60:.0f} min)" if rem_s < 5400 else f"ETA {when} (in {rem_s / 3600:.1f} h)"


class StepProgress:
    """call it after every MD step (dyn.attach(p, interval=1)). steps done = base + (dyn.nsteps - nsteps at attach time); speed over the last `window_s` seconds.
    path: json {k, lam, done, total, rate (steps/s), t, t_start, pid, state} written atomically every `write_s`; label + print_s: a progress line on stdout every print_s."""

    def __init__(self, dyn, total, base=0, path=None, info=None, label=None, print_s=None, write_s=30.0, window_s=300.0):
        self.dyn, self.total, self.base, self.n0 = dyn, int(total), int(base), dyn.nsteps
        self.path, self.info, self.label, self.print_s, self.write_s, self.window_s = path, dict(info or {}), label, print_s, write_s, window_s
        self.t_start = time.time(); self.hist = deque([(self.t_start, self.done())]); self.t_write = self.t_print = self.t_start
        if path:
            self.write("running")

    def done(self):
        return self.base + self.dyn.nsteps - self.n0

    def rate(self):
        """steps/s over the window (the first step after start-up includes the model compilation: the window drops it after window_s)"""
        t1, n1 = self.hist[-1]; t0, n0 = self.hist[0]
        return (n1 - n0) / (t1 - t0) if t1 > t0 and n1 > n0 else float("nan")

    def __call__(self):
        now = time.time()
        if now - self.hist[-1][0] < 1.0:   # bookkeeping at most once a second
            return
        self.hist.append((now, self.done()))
        while len(self.hist) > 2 and now - self.hist[1][0] > self.window_s:
            self.hist.popleft()
        if self.path and (now - self.t_write >= self.write_s or (self.t_write == self.t_start and now - self.t_start >= 5)):   # first update after 5 s (a speed early)
            self.write("running"); self.t_write = now
        if self.print_s and now - self.t_print >= self.print_s:
            self.t_print = now; print(self.line(), flush=True)

    def line(self):
        d, r = self.done(), self.rate()
        return f"  {self.label}: step {d} / {self.total} ({100 * d / max(self.total, 1):.0f}%) | {1000 / r if r == r and r > 0 else float('nan'):.0f} ms/step | {fmt_eta((self.total - d) / r if r == r and r > 0 else float('nan'))}"

    def write(self, state):
        d = dict(self.info, done=self.done(), total=self.total, rate=self.rate(), t=time.time(), t_start=self.t_start, pid=os.getpid(), state=state)
        try:
            with open(self.path + ".tmp", "w") as fh:
                json.dump(d, fh)
            os.replace(self.path + ".tmp", self.path)
        except OSError:
            pass
