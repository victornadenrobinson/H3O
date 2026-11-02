#!/usr/bin/env python3
"""Live text-mode movie of a toy-model MD in your terminal (ToyIce pair potential, Langevin). Needs ase + numpy; run from the repo root with a python that has ase ($PY on Perlmutter).
  python mlp_pipeline/toy_live.py                       # 54 H2O in bcc (162 atoms), 5000 K, 2.12 A^3/atom
  python mlp_pipeline/toy_live.py --T 3000 --vpa 2.4 --fps 12 --slab 2.0 --phase bcc --nrep 3
bcc with --nrep 3 has 3*3*3*2 = 54 oxygens; fcc with --nrep 3 has 108. Ctrl-C stops. Make the terminal at least 100 columns x 40 rows (--width / --height shrink the picture).
Toy energies are only a fast structure sampler (see build_toy_cells.py); this is for looking at proton hopping / disorder, not for numbers."""
import argparse, os, sys, time
from types import SimpleNamespace
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tdi"))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("--phase", default="bcc"); ap.add_argument("--nrep", type=int, default=3)
ap.add_argument("--vpa", type=float, default=2.12); ap.add_argument("--T", type=float, default=5000.0); ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--every", type=int, default=2, help="MD steps per drawn frame")
ap.add_argument("--slab", type=float, default=2.2); ap.add_argument("--width", type=int, default=96); ap.add_argument("--height", type=int, default=30); ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 = until Ctrl-C)")
ap.add_argument("--view3d", action="store_true", help="orbiting 3D view with perspective, bonds and the box wireframe"); ap.add_argument("--spin", type=float, default=2.5, help="degrees of rotation per drawn frame (3D)")
ap.add_argument("--ascii", action="store_true", help="plain characters instead of 24-bit colour (3D)"); ap.add_argument("--nobox", action="store_true"); ap.add_argument("--delta", action="store_true", help="redraw only the cells that changed (less for the terminal to paint; try it if the plain mode stutters)"); ap.add_argument("--atom-scale", type=int, default=0, help="2D: pixels per atom unit (0 = auto from the width)"); ap.add_argument("--braille", action="store_true", help="braille characters: 2x4 dots per cell = 4x the pixels (one colour per cell); needs a font with braille glyphs"); ap.add_argument("--view", default="", help="fixed crystallographic view: 001, 101, 111 ... (look along that direction)"); ap.add_argument("--ortho", action="store_true", help="no perspective (best for lattice views)"); ap.add_argument("--hi", type=float, default=1.0, help="keep only atoms nearer than this depth fraction (0.4 = a thin front slab)")
ap.add_argument("--zoom", type=float, default=2.0); ap.add_argument("--clip", type=float, default=0.0, help="hide the farthest fraction of atoms in 3D (0.3 = back 30 percent)"); ap.add_argument("--nobonds", action="store_true"); ap.add_argument("--hper", type=float, default=2.0, help="H per O"); ap.add_argument("--seed", type=int, default=1); a = ap.parse_args()
import build_toy_cells as b
from toy_ice import ToyIce
from mdview import render, render3d, DiffPrinter
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

b.HPER["custom"] = a.hper
at = b.import_module("17_eos_md").build(a.phase, a.vpa, a.nrep, a.seed, a.hper) if hasattr(b, "import_module") else None
if at is None:
    from importlib import import_module; at = import_module("17_eos_md").build(a.phase, a.vpa, a.nrep, a.seed, a.hper)
at.calc = ToyIce(); rng = np.random.default_rng(a.seed); MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=rng); Stationary(at)
dyn = Langevin(at, 1.0 * units.fs, temperature_K=a.T, friction=1 / (100 * units.fs), rng=rng)
dp = DiffPrinter(); nO = int((at.numbers == 8).sum()); sys.stdout.write("\033[2J\033[?25l")
try:
    t_prev = time.time(); n = 0
    while not a.steps or dyn.nsteps < a.steps:
        dyn.run(a.every); Tk = 2 / 3 * at.get_kinetic_energy() / (len(at) * units.kB); at.info["T_inst"] = Tk
        now = time.time(); sps = a.every / max(now - t_prev, 1e-9); t_prev = now
        note = f"{nO} H2O {a.phase}  target {a.T:.0f} K  {sps:.0f} steps/s"
        if a.view3d or a.view: n += 1; txt = render3d(at, yaw=n * a.spin, step=dyn.nsteps, width=a.width, height=a.height, ascii_only=a.ascii, note=note, bonds=not (a.nobonds or a.ascii), box=not a.nobox, zoom=a.zoom, clip=a.clip, hi=a.hi, view_dir=([float(c) for c in a.view] if a.view else None), ortho=a.ortho or bool(a.view), braille=a.braille)
        else: txt = render(at, step=dyn.nsteps, width=a.width, height=a.height, slab=a.slab, note=note, atom_scale=a.atom_scale or None)
        if a.delta: dp.draw()
        else: print(txt, flush=True)
        time.sleep(max(0.0, 1.0 / a.fps - (time.time() - now)))
except KeyboardInterrupt: pass
finally: sys.stdout.write("\033[?25h\n")
