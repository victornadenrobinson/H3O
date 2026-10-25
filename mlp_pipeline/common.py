"""Shared helpers: composition parsing from run paths."""
import re

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
