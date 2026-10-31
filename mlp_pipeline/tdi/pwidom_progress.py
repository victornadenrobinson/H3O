"""Optional progress sidecar for the Perlmutter launcher; no model/MD changes."""
import json
import os
from pathlib import Path
import time


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.stage = None
        self.started = self.last = 0.0

    def update(self, stage, done=0, total=0, unit="items", force=False):
        if self.path is None:
            return
        now = time.monotonic()
        changed = stage != self.stage
        if changed:
            self.stage, self.started, self.last = stage, now, -float("inf")
        if not force and now - self.last < 10 and done != total:
            return
        self.last = now
        elapsed = now - self.started
        eta = elapsed / done * max(0, total - done) if done > 0 and elapsed > 0 else None
        fraction = done / total if total else None
        record = dict(stage=stage, done=done, total=total, unit=unit,
                      fraction=fraction, elapsed_seconds=elapsed,
                      eta_seconds=eta, updated=time.time())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record) + "\n")
        tmp.replace(self.path)
        eta_text = "estimating" if eta is None else f"{eta / 60:.1f} min"
        detail = f"{done}/{total} {unit}, {100 * fraction:.0f}%" if fraction is not None else "initializing"
        left = "estimating" if eta is None else duration(eta)
        print(f"PROGRESS {stage}: {detail}; elapsed {duration(elapsed)} / remaining {left} (stage); stage ETA {eta_text}", flush=True)


def from_environment():
    return Progress(os.environ.get("PW_PROGRESS_FILE"))
