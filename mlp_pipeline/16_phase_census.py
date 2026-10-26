"""Count how much fcc / bcc / other-phase H2O data we have, from words in the run path, and how much of it is in the dataset tiers.
usage: python 16_phase_census.py [inventory_binned.csv] [--tiers train_m valid_m]
Path words only (fcc, bcc, sc, pbcm, p21c, liquid, solid, superionic, melt); runs without such a word are 'untagged' and need
09_label_phase.py (MSD: solid/superionic/liquid) plus a look at the O sublattice (fcc vs bcc) before they can be called anything.
"""
import argparse, os, re
import pandas as pd
from common import inp, out

TAGS = ["fcc", "bcc", "sc", "pbcm", "p21c", "liquid", "solid", "superionic", "melt"]

def tag(path):
    p = path.lower(); hit = [t for t in TAGS if re.search(rf"(^|[^a-z0-9]){t}([^a-z]|$)", p)]
    return "+".join(hit) if hit else "untagged"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("inv", nargs="?", default="inventory_binned.csv")
    ap.add_argument("--tiers", nargs="*", default=["train_m", "valid_m"])
    a = ap.parse_args()
    df = pd.read_csv(inp(a.inv)); df = df[df.comp_snap == "h2o"].copy()
    df["tag"] = df.Path.map(tag); df["P100"] = (df.True_P_GPa // 100 * 100).astype(int)
    print(f"{len(df)} h2o runs in the inventory")
    print(df.tag.value_counts().to_string())
    for t in ["fcc", "bcc"]:
        d = df[df.tag.str.contains(t)]
        if len(d):
            print(f"\n{t}: runs by T bin x 100 GPa bin"); print(d.pivot_table(index="T_bin", columns="P100", values="Path", aggfunc="count", fill_value=0).to_string())
    print("\nexample paths per tag:")
    for g, d in df.groupby("tag"):
        print(f"  {g}: {d.Path.iloc[0]}")
    try:
        from ase.io import read
        src = {p: tag(p) for p in df.Path}
        for t in a.tiers:
            fn = inp(f"{t}.extxyz")
            if not os.path.exists(fn): continue
            fr = read(fn, index=":"); rows = [(f.info.get("config_type", ""), src.get(f.info.get("source"), tag(str(f.info.get("source", "")))))
                                               for f in fr if str(f.info.get("config_type", "")).startswith("h2o_")]
            c = pd.DataFrame(rows, columns=["config", "tag"])
            print(f"\n{t}: {len(c)} h2o frames"); print(c.groupby(["tag", "config"]).size().unstack(0, fill_value=0).to_string())
    except ImportError:
        print("(ase not available: skipping tier counts)")
