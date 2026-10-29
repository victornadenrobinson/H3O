"""Are all training frames computed with the same DFT settings? Reads each source OUTCAR header (ENCUT, smearing, k-points, PREC,
POTCARs, VASP build ...), groups runs by identical settings and reports the groups against composition / T / frames.

usage: python 12_check_settings.py snapshots_l.extxyz [--root ~/work] [--workers 32]      (run where the OUTCARs are: Perlmutter)
Writes runs/dft_settings.csv (one row per run: all parsed settings + settings_id).
"""
import argparse, os, re, subprocess
from collections import Counter
from functools import partial
from multiprocessing import Pool
import pandas as pd
from ase.io import read
from common import out, inp, progress

HEAD_N = 6000
PAT = dict(
    ENCUT=r"ENCUT\s*=\s*([\d.]+)\s*eV", PREC=r"PREC\s*=\s*(\w+)", ISPIN=r"ISPIN\s*=\s*(\d)", LASPH=r"LASPH\s*=\s*(\w)",
    LREAL=r"LREAL\s*=\s*(\w+)", EDIFF=r"EDIFF\s*=\s*([\d.eE+-]+)", NELECT=r"NELECT\s*=\s*([\d.]+)", NKPTS=r"NKPTS\s*=\s*(\d+)",
    ALGO=r"ALGO\s*=\s*(\w+)", ISYM=r"ISYM\s*=\s*(-?\d)", NBANDS=r"NBANDS\s*=\s*(\d+)",
)


def head(path):
    cmd = ["gzip", "-dc", path] if path.endswith(".gz") else ["cat", path]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(["head", "-n", str(HEAD_N)], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p1.stdout.close(); t = p2.communicate()[0].decode(errors="replace"); p1.wait()
    return t


def parse(txt):
    d = {}
    for k, p in PAT.items():
        m = re.search(p, txt); d[k] = m.group(1) if m else None
    m = re.search(r"ISMEAR\s*=\s*(-?\d+);\s*SIGMA\s*=\s*([\d.]+)", txt); d["ISMEAR"], d["SIGMA"] = (m.group(1), m.group(2)) if m else (None, None)
    m = re.search(r"generate k-points for:\s+(\d+)\s+(\d+)\s+(\d+)", txt); d["kmesh"] = "x".join(m.groups()) if m else None
    d["gamma_only_build"] = "gamma-only" in txt[:300]
    m = re.match(r"\s*(vasp\.[\w.\-]+)", txt); d["vasp"] = m.group(1) if m else None
    pots = list(dict.fromkeys(re.findall(r"POTCAR:\s+(PAW_PBE\s+\S+\s+\S+|PAW\S*\s+\S+\s+\S+)", txt)))
    d["potcars"] = "; ".join(pots) if pots else None
    return d


def one(path, root):
    try:
        d = parse(head(os.path.join(root, path)))
    except Exception as e:
        d = {"error": str(e)}
    d["source"] = path
    return d


def main(a):
    frames = read(inp(a.data), index=":")
    meta = {}
    for f in frames:
        s = f.info["source"]; m = meta.setdefault(s, dict(frames=0, comp=f.info.get("config_type", "?").rsplit("_", 1)[0], T=int(f.info.get("T_K", 0)), atoms=len(f), box=f.get_volume() ** (1 / 3)))
        m["frames"] += 1
    root = os.path.abspath(os.path.expanduser(a.root))
    with Pool(a.workers) as p:
        rows = list(progress(p.imap(partial(one, root=root), list(meta), chunksize=4), total=len(meta), desc="headers"))
    df = pd.DataFrame(rows)
    for k in ("frames", "comp", "T", "atoms", "box"):
        df[k] = df.source.map(lambda s: meta[s][k])
    key = ["ENCUT", "ISMEAR", "SIGMA", "PREC", "kmesh", "NKPTS", "gamma_only_build", "LASPH", "LREAL", "ISPIN", "potcars"]
    df[key] = df[key].astype(object).where(df[key].notna(), "-")
    df["settings"] = df[key].astype(str).agg(" | ".join, axis=1)
    # effective sampling length: box edge (V^1/3) x largest k-mesh dimension (gamma-only -> 1): ~ size of the equivalent Gamma-point supercell
    df["eff_L"] = [b * (max(int(x) for x in str(k).split("x")) if k not in ("-", None) and "x" in str(k) else 1) for b, k in zip(df.box, df.kmesh)]
    ids = {s: i for i, s in enumerate(df.settings.value_counts().index)}
    df["settings_id"] = df.settings.map(ids)
    df.to_csv(out("dft_settings.csv"), index=False)
    print(f"{len(df)} runs, {df.frames.sum()} frames, {len(ids)} distinct setting groups ({int(df.get('error', pd.Series()).notna().sum())} unreadable)\n")
    g = df.groupby("settings_id").agg(runs=("source", "size"), frames=("frames", "sum"), comps=("comp", lambda x: ",".join(sorted(set(x))[:6])),
                                      T=("T", lambda x: f"{min(x)}-{max(x)}"), atoms=("atoms", lambda x: f"{min(x)}-{max(x)}"),
                                      box_A=("box", lambda x: f"{min(x):.1f}-{max(x):.1f}"), effL_A=("eff_L", lambda x: f"{min(x):.1f}-{max(x):.1f}"))
    g["settings"] = [df[df.settings_id == i].settings.iloc[0] for i in g.index]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(g.to_string())
    print("\nsettings fields that vary across the dataset:")
    for k in key:
        v = df[k].astype(str).value_counts()
        if len(v) > 1:
            print(f"  {k}: " + ", ".join(f"{i} ({n})" for i, n in v.head(8).items()))
    out_rule = df[(df.eff_L < 9) | (df.eff_L > 12.5)]
    print(f"\nruns with effective sampling length outside 9-12.5 A: {len(out_rule)} of {len(df)} ({out_rule.frames.sum()} frames)")
    if len(out_rule):
        print(out_rule.groupby(["comp", "T"]).agg(runs=("source", "size"), L=("eff_L", lambda x: f"{x.min():.1f}-{x.max():.1f}")).to_string())
    print("\nexample sources of each minority group (id>0):")
    for i in sorted(ids.values())[1:6]:
        print(f"  [{i}] {df[df.settings_id == i].source.iloc[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--root", default="~/work"); ap.add_argument("--workers", type=int, default=32)
    main(ap.parse_args())
