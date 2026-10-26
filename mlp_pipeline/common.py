"""Shared helpers: composition parsing from run paths."""
import os
import re
from pathlib import Path

ROOT_PREFIXES = ("quartz-work/tdi/h3o", "tdi/h3o", "quartz-work")
KB_TO_GPA = 0.1


def strip_root(path):
    p = path.replace("\\", "/").lower()
    for pre in ROOT_PREFIXES:
        i = p.find(pre)
        if i >= 0:
            p = p[i + len(pre):]
            break
    return p.strip("/")


def parse_composition(path):
    """Return H-per-O ratio as a float (pure H2 -> inf), or None if unknown.

    Matches the deepest path component first, after removing the project root
    (which is itself called tdi/h3o and would otherwise label everything H3O).
    """
    comps = strip_root(path).split("/")[:-1]  # drop filename
    for c in reversed(comps):
        m = re.search(r"(?<![a-z0-9.])h(\d+(?:\.\d+)?)o(?![a-z])", c)
        if m:
            return float(m.group(1))
        if re.search(r"(^|[-_])(pure[-_])?h2?($|[-_])", c) and "o" not in c.replace("pure", ""):
            return float("inf")
    return None


def comp_label(r):
    if r is None or r != r:
        return "unknown"
    return "h" if r == float("inf") else f"h{r:g}o"


# hull-CSV column suffix -> H per O
HULL_COMPS = {"h2o": 2.0, "h2.25": 2.25, "h3o": 3.0, "h5o": 5.0, "h7o": 7.0, "h": float("inf")}


def progress(it, **kw):
    """tqdm progress bar if available, else plain iterator."""
    try:
        from tqdm import tqdm
        return tqdm(it, **kw)
    except ImportError:
        return it


def phase_from_path(path):
    """Starting-structure tag from folder names (deepest component first). Not the dynamical phase: see 09_label_phase.py."""
    for comp in reversed(strip_root(path).split("/")[:-1]):
        for key, lab in (("superionic", "superionic"), ("plastic", "plastic"), ("solid", "solid"),
                         ("glass", "glass"), ("liquid", "liquid"), ("melt", "liquid"), ("fluid", "liquid")):
            if key in comp:
                return lab
    return "unknown"


# All data products go to <repo>/runs (gitignored) unless H3O_RUNS is set.
RUNS = Path(os.environ.get("H3O_RUNS", Path(__file__).resolve().parent.parent / "runs"))


def out(name):
    """Path of an output file inside the runs folder (bare names only; paths with a directory are kept)."""
    if os.path.dirname(name):
        return os.path.expanduser(name)
    RUNS.mkdir(parents=True, exist_ok=True)
    return str(RUNS / name)


def inp(path):
    """Input file. A bare name resolves to runs/<name> first (so stale copies in the cwd can't shadow current outputs);
    otherwise the path as given."""
    p = os.path.expanduser(path)
    if not os.path.dirname(p) and (RUNS / p).exists():
        return str(RUNS / p)
    return p


PHASE_WORDS = ["fcc", "bcc", "sc", "pbcm", "p21c", "liquid", "solid", "superionic", "melt"]


def path_phase_tag(path):
    """'+'-joined phase words found in the run path (starting structure / intent, not the dynamical phase), or 'untagged'."""
    import re
    p = str(path).lower(); hit = [t for t in PHASE_WORDS if re.search(rf"(^|[^a-z0-9]){t}([^a-z]|$)", p)]
    return "+".join(hit) if hit else "untagged"


def phase_bucket(path):
    """Coarse selection bucket from the path tag: liquid > fcc > bcc > other (a 'bcc-...-liquid' run is a liquid)."""
    t = path_phase_tag(path)
    return "liquid" if ("liquid" in t or "melt" in t) else "fcc" if "fcc" in t else "bcc" if "bcc" in t else "other"


def run_key(path):
    """Run directory: the folder above out<N>/ (restart segments out1, out2, ... of one MD run share it); the folder itself if the OUTCAR is not in an out<N>."""
    import re
    parts = str(path).split("/")[:-1]
    if parts and re.fullmatch(r"out\d*", parts[-1]):
        parts = parts[:-1]
    return "/".join(parts)


def seg_no(path):
    """Restart segment number from .../out<N>/OUTCAR (0 if there is none)."""
    import re
    m = re.search(r"/out(\d*)/[^/]*$", str(path))
    return int(m.group(1)) if m and m.group(1) else 0
