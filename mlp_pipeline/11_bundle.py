"""Pack everything needed on another machine (Young) into one tar.gz with a manifest + checksums.

usage: python 11_bundle.py [--out H3O_bundle.tar.gz] [--include-plots] [--tiers s m l]
Contains (from runs/): snapshots_<tier>.extxyz, train_/valid_<tier>.extxyz if present, inventory_binned.csv,
hull_target_coverage.csv, crosscheck_*.csv, mined_h3o.csv if present, + MANIFEST.json (git commit, frame counts, sha256).
Code comes from git (git clone / pull on the target); the repo's data-h3o-convex-hull.csv is already tracked.
"""
import argparse, hashlib, json, os, subprocess, tarfile, time
from common import RUNS


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nframes(p):
    n = 0
    with open(p) as f:
        while True:
            l = f.readline()
            if not l:
                return n
            if l.strip():
                n += 1
                for _ in range(int(l) + 1):
                    f.readline()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--tiers", nargs="+", default=["s", "m", "l"])
    ap.add_argument("--include-plots", action="store_true")
    a = ap.parse_args()
    names = [f"{k}_{t}.extxyz" for t in a.tiers for k in ("snapshots", "train", "valid")]
    names += ["inventory_binned.csv", "hull_target_coverage.csv", "crosscheck_eos.csv", "crosscheck_eos_dat.csv", "mined_h3o.csv", "phase.csv"]
    files = [n for n in names if (RUNS / n).exists()]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RUNS.parent).stdout.strip()
    except Exception:
        commit = "unknown"
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), git_commit=commit, files={})
    for n in files:
        p = RUNS / n
        man["files"][n] = dict(bytes=p.stat().st_size, sha256=sha(p), **({"frames": nframes(p)} if n.endswith(".extxyz") else {}))
    out = a.out or f"H3O_bundle_{time.strftime('%Y%m%d')}.tar.gz"
    out = out if os.path.dirname(out) else str(RUNS / out)
    mp = RUNS / "MANIFEST.json"
    mp.write_text(json.dumps(man, indent=1))
    with tarfile.open(out, "w:gz") as tf:
        for n in files + ["MANIFEST.json"]:
            tf.add(RUNS / n, arcname=f"runs/{n}")
        if a.include_plots and (RUNS / "plots").exists():
            tf.add(RUNS / "plots", arcname="runs/plots")
    print(json.dumps({k: v.get("frames", v["bytes"]) for k, v in man["files"].items()}, indent=1))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB), sha256 {sha(out)}")
    print("on the target: cd <repo> && tar xzf", os.path.basename(out), "  (unpacks into runs/); verify with sha256sum -c-style comparison against MANIFEST.json")
