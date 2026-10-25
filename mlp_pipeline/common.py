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
    """Input file: as given if it exists, else looked up in the runs folder."""
    p = os.path.expanduser(path)
    return p if os.path.exists(p) or not (RUNS / path).exists() else str(RUNS / path)
