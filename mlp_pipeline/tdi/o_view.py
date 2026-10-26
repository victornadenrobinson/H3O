"""O-only, wrapped copy of a TDI trajectory with the displacement of every O from its lattice site as an extra column, for looking at the lattice in OVITO / VMD / ase gui, plus a printout of which O moved most.
usage (any python with numpy, no ase needed):  python3 mlp_pipeline/tdi/o_view.py runs/tdi/<tag>_node<k>_traj.extxyz [-o out.extxyz] [--hop 1.0]
Lattice sites = the mean position of each O over the first fifth of the frames (minimal image; centre of mass removed from the displacements). Output columns: species, x, y, z (wrapped), disp (A). Colour by 'disp' in OVITO: a few bright atoms mean individual O have left their sites (defects / hops), a uniformly bright cloud means the lattice is melting.
Assumes an orthorhombic cell (the TDI cells are cubic)."""
import argparse, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("traj"); ap.add_argument("-o", default=None); ap.add_argument("--hop", type=float, default=1.0, help="report O whose largest displacement exceeds this (A)"); a = ap.parse_args()
frames = []; lines = open(a.traj).read().splitlines(); i = 0
while i < len(lines):
    n = int(lines[i]); com = lines[i + 1]; body = [l.split() for l in lines[i + 2:i + 2 + n]]; i += 2 + n
    lat = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', com).group(1).split()]).reshape(3, 3)
    t = re.search(r't_ps=([0-9.eE+-]+)', com); frames.append((float(t.group(1)) if t else len(frames), lat, np.array([b[0] for b in body]), np.array([[float(x) for x in b[1:4]] for b in body])))
iO = np.where(frames[0][2] == "O")[0]; L = np.diag(frames[0][1]); T = np.array([f[0] for f in frames]); P = np.array([f[3][iO] for f in frames])
mic = lambda d: d - np.round(d / L) * L
nref = max(3, len(T) // 5); ref = P[0] + mic(P[:nref] - P[0]).mean(0)    # lattice sites from the first fifth of the frames, so a later hop shows its full size
d = mic(P - ref); d -= d.mean(1, keepdims=True)           # displacement from the lattice site, centre of mass removed
disp = np.linalg.norm(d, axis=2)                           # frames x O
print(f"{a.traj}: {len(frames)} frames, t = {T[0]:.2f}-{T[-1]:.2f} ps, {len(iO)} O, cell {L.round(3)}")
print(f"u_rms per frame (first, middle, last): {np.sqrt((disp ** 2).mean(1))[[0, len(T) // 2, -1]].round(3)} A;   largest single displacement {disp.max():.2f} A")
worst = np.argsort(disp.max(0))[::-1][:5]
print("five most displaced O (index in the O list, max displacement A, frame time of the max):")
for k in worst: print(f"   O {k:4d}   {disp[:, k].max():.2f} A   at t = {T[np.argmax(disp[:, k])]:.2f} ps")
hop = np.where(disp.max(0) > a.hop)[0]; print(f"O with displacement > {a.hop} A: {len(hop)} of {len(iO)}")
if len(hop):
    k = worst[0]; ts = disp[:, k]; step = np.abs(np.diff(ts)).max()
    print(f"worst O {k}: displacement over time (every {max(len(T) // 12, 1)} frames): {ts[::max(len(T) // 12, 1)].round(2)};  largest frame-to-frame jump {step:.2f} A ({'sudden: a hop or a defect forming' if step > 0.6 else 'gradual drift'})")
out = a.o or a.traj.replace(".extxyz", "_Oonly.extxyz")
with open(out, "w") as f:
    for fi, (t, lat, sp, pos) in enumerate(frames):
        w = (P[fi] / L - np.floor(P[fi] / L)) * L
        f.write(f"{len(iO)}\nLattice=\"{' '.join(f'{x:.6f}' for x in lat.flatten())}\" Properties=species:S:1:pos:R:3:disp:R:1 pbc=\"T T T\" t_ps={t}\n")
        for j in range(len(iO)): f.write(f"O {w[j, 0]:.5f} {w[j, 1]:.5f} {w[j, 2]:.5f} {disp[fi, j]:.4f}\n")
print("wrote", out)
