"""Survey the simulation cells in a snapshot file: sizes, shapes and the effective box width, by composition and atom count.

usage: python 15_cell_survey.py snapshots_l.extxyz [--min-edge 8.5]
For every (composition, number of atoms): frames, cell edge V^(1/3), the MINIMUM PERPENDICULAR WIDTH (the true finite-size length for a
non-cubic cell: smallest distance between opposite faces, V / |a_j x a_k|), cell shape class, pressure and temperature range.
V^(1/3) overstates the size of a sheared cell, so both are shown.
"""
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from ase.io import read
from common import inp, out


def shape(cell, tol=0.02):
    a, b, c, al, be, ga = cell.cellpar()
    L = np.array([a, b, c])
    ang = np.array([al, be, ga])
    if np.ptp(L) / L.mean() < tol and np.allclose(ang, 90, atol=1):
        return "cubic"
    if np.allclose(ang, 90, atol=1):
        return "orthorhombic"
    if np.ptp(L) / L.mean() < tol and np.allclose(ang, 60, atol=1):
        return "fcc-primitive"
    if np.ptp(L) / L.mean() < tol and np.allclose(ang, 109.47, atol=1):
        return "bcc-primitive"
    return "triclinic"


def widths(cell):
    m = np.array(cell); V = abs(np.linalg.det(m))
    return min(V / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3))


def main(a):
    rows = []
    for f in read(inp(a.data), index=":"):
        V = f.get_volume()
        rows.append(dict(comp=f.info.get("config_type", "?").rsplit("_", 1)[0], T=f.info.get("T_K"), P=f.info.get("P_GPa"), N=len(f),
                         edge=V ** (1 / 3), width=widths(f.cell), shape=shape(f.cell), source=f.info.get("source", "")))
    df = pd.DataFrame(rows); df.to_csv(out("cell_survey.csv"), index=False)
    print(f"{len(df)} frames, {df.source.nunique()} runs. cell shapes: {df['shape'].value_counts().to_dict()}")
    g = df.groupby(["comp", "N"]).agg(frames=("N", "size"), runs=("source", "nunique"), edge=("edge", lambda x: f"{x.min():.2f}-{x.max():.2f}"),
                                      width=("width", lambda x: f"{x.min():.2f}-{x.max():.2f}"), shape=("shape", lambda x: ",".join(sorted(set(x)))),
                                      P=("P", lambda x: f"{x.min():.0f}-{x.max():.0f}"), T=("T", lambda x: f"{x.min():.0f}-{x.max():.0f}"))
    pd.set_option("display.width", 220)
    print(g.to_string())
    print("\nframes by minimum perpendicular width (A):")
    bins = pd.cut(df.width, [0, 6, 7, 8, 8.5, 9, 9.5, 10, 11, 12, 100])
    print(pd.crosstab(bins, df["shape"]).to_string())
    if a.min_edge:
        keep = df.width >= a.min_edge
        print(f"\nwidth >= {a.min_edge} A keeps {int(keep.sum())} of {len(df)} frames; "
              f"edge (V^1/3) >= {a.min_edge} would keep {int((df.edge >= a.min_edge).sum())}")
        print("lost by composition:", df[~keep].comp.value_counts().head(8).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--min-edge", type=float, default=8.5)
    main(ap.parse_args())
