#!/usr/bin/env python3
"""PWidom host RDFs, saved-extra-H environments and data inspection."""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms
from ase.neighborlist import neighbor_list


def inspect(path):
    path = Path(path)
    print(f"\n=== {path} ===")
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                a = data[key]
                print(f"{key:30s} {str(a.shape):20s} {a.dtype}")
                if a.size <= 5:
                    print("   ", a)
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        print(json.dumps(data, indent=2, default=str)[:12000])


def get_cells(data, nframes, assume_cubic):
    if "cell" in data:
        cells = np.asarray(data["cell"], dtype=float)
        if cells.shape == (3, 3):
            cells = np.broadcast_to(cells, (nframes, 3, 3))
        elif cells.shape == (3,):
            cells = np.broadcast_to(np.diag(cells), (nframes, 3, 3))
        elif cells.shape == (nframes, 3):
            cells = np.array([np.diag(v) for v in cells])
        elif cells.shape != (nframes, 3, 3):
            raise ValueError(f"Unsupported cell shape {cells.shape}")
        return cells

    if not assume_cubic:
        raise ValueError(
            "Missing cell array. Use --assume-cubic only if "
            "the simulation cell really was cubic."
        )

    v = np.asarray(data["V"], dtype=float)
    if v.ndim == 0:
        v = np.repeat(float(v), nframes)
    if v.shape != (nframes,) or np.any(v <= 0):
        raise ValueError("Invalid volume array")

    return np.array([np.eye(3) * x**(1/3) for x in v])


def rdf(data, args, out):
    pos = np.asarray(data["host"], dtype=float)
    numbers = np.asarray(data["numbers"], dtype=int)

    nf, na, xyz = pos.shape
    if xyz != 3 or numbers.shape != (na,):
        raise ValueError("Incompatible host and numbers arrays")

    cells = get_cells(data, nf, args.assume_cubic)

    edges = np.arange(0, args.rmax + args.dr, args.dr)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4*np.pi/3 * (edges[1:]**3 - edges[:-1]**3)

    pairs = [(8, 1, "OH"), (1, 1, "HH"), (8, 8, "OO")]
    counts = {name: np.zeros(len(r)) for _, _, name in pairs}
    norms = {name: np.zeros(len(r)) for _, _, name in pairs}

    insertion = {
        "H": np.zeros(len(r)),
        "O": np.zeros(len(r))
    }
    insertion_norm = {
        "H": np.zeros(len(r)),
        "O": np.zeros(len(r))
    }

    has_full = "full" in data
    full = np.asarray(data["full"]) if has_full else None

    if has_full and full.shape != (nf, na+1, 3):
        raise ValueError("Unexpected full array shape")

    frames = list(range(0, nf, args.stride))

    for frame in frames:
        cell = cells[frame]
        volume = abs(np.linalg.det(cell))
        heights = 1 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)

        if args.rmax >= 0.5 * min(heights):
            raise ValueError(
                f"rmax={args.rmax:g} too large for cell in frame {frame}"
            )

        atoms = Atoms(
            numbers=numbers,
            positions=pos[frame],
            cell=cell,
            pbc=True
        )

        i, j, d = neighbor_list("ijd", atoms, args.rmax)

        for za, zb, name in pairs:
            mask = ((numbers[i] == za) &
                    (numbers[j] == zb) & (i != j))

            hist, _ = np.histogram(d[mask], bins=edges)
            counts[name] += hist

            n_a = np.count_nonzero(numbers == za)
            n_b = np.count_nonzero(numbers == zb)
            possible = n_a * (n_b - int(za == zb))
            norms[name] += possible * shell / volume

        if has_full:
            # Assumes final coordinate represents the added H.
            # This is NOT automatically a weighted insertion RDF.
            extra = full[frame, -1]
            delta = full[frame, :-1] - extra
            frac = np.linalg.solve(cell.T, delta.T).T
            frac -= np.rint(frac)
            dist = np.linalg.norm(frac @ cell, axis=1)

            for z, name in [(1, "H"), (8, "O")]:
                subset = dist[numbers == z]
                hist, _ = np.histogram(subset, bins=edges)
                insertion[name] += hist
                insertion_norm[name] += (
                    np.count_nonzero(numbers == z) * shell / volume
                )

    fig, ax = plt.subplots(figsize=(8, 5))
    results = {"r_A": r}

    for _, _, name in pairs:
        g = np.divide(
            counts[name], norms[name],
            out=np.zeros_like(r),
            where=norms[name] > 0
        )
        results["g_" + name] = g
        ax.plot(r, g, label=name)

    ax.set(
        xlabel="r (Å)",
        ylabel="g(r)",
        title=f"Host RDF — {len(frames)} frames"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "host_rdf.pdf")
    plt.close(fig)

    if has_full:
        fig, ax = plt.subplots(figsize=(8, 5))

        for name in ("H", "O"):
            g = np.divide(
                insertion[name], insertion_norm[name],
                out=np.zeros_like(r),
                where=insertion_norm[name] > 0
            )
            results["saved_extraH_" + name] = g
            ax.plot(r, g, label=f"Extra H – host {name}")

        ax.set(
            xlabel="r (Å)",
            ylabel="Relative neighbour density",
            title="Saved extra-H configurations (unweighted)"
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "saved_extraH_environment.pdf")
        plt.close(fig)

    np.savez_compressed(out / "rdf_data.npz", **results)
    print(f"Analysed {len(frames)} frames")
    print(f"Figures saved in {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", nargs="+")
    p.add_argument("--frames")
    p.add_argument("--assume-cubic", action="store_true")
    p.add_argument("--rmax", type=float, default=4.0)
    p.add_argument("--dr", type=float, default=0.05)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", default="pwidom_diagnostics")
    args = p.parse_args()

    if args.inspect:
        for path in args.inspect:
            inspect(path)
        return

    if not args.frames:
        p.error("Specify --frames or --inspect")
    if args.dr <= 0 or args.rmax <= 0 or args.stride < 1:
        p.error("Invalid RDF settings")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(args.frames, allow_pickle=False) as data:
        rdf(data, args, out)


if __name__ == "__main__":
    main()
