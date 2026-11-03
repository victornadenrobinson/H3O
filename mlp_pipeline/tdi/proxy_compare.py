"""Plot DFT-trained and insertion-fitted pair proxies; never launch model work."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex

import numpy as np

from toy_ice import pair_energy


def replay_plan(run, old):
    """Prepare commands only; identical cached ensembles, different proxy sampling."""
    m = json.loads((run / "manifest.json").read_text())
    if m["status"] != "complete":
        raise ValueError("Baseline replay needs a complete run; no MD will be scheduled")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(str(Path.cwd())),
             "# Single points only on cached frames. No MD expected; allow up to 10 min per host.",
             "pids=()"]
    for job in m["jobs"]:
        cmd = list(job["command"])
        tag = cmd[cmd.index("--tag") + 1]
        frames = run / f"{tag}_frames.npz"
        if not frames.is_file():
            raise ValueError(f"Missing cached frames: {frames}")
        with np.load(frames) as z:
            if not all(k in z.files for k in ("host", "full", "model_fp")):
                raise ValueError(f"Incomplete/unverified cached ensemble: {frames}")
        cmd[cmd.index("--proxy") + 1] = str(old)
        cmd[cmd.index("--label") + 1] = "old_dft"
        result = run / f"{tag}_old_dft_K{cmd[cmd.index('--K') + 1]}.json"
        if result.exists():
            lines.append("# Already present: " + str(result)); continue
        log = run / f"{job['name']}_old_dft.log"
        # Recheck at execution time. Existence checks also protect against duplicate launches.
        line = ("test -s " + shlex.quote(str(frames)) + "\n"
                + "test ! -e " + shlex.quote(str(result)) + "\n"
                + "test ! -e " + shlex.quote(str(log)) + "\n"
                + "(set -o noclobber; env -u PW_GPU_LOCK CUDA_VISIBLE_DEVICES=" + shlex.quote(job["gpu"])
                + " TF_FORCE_GPU_ALLOW_GROWTH=true OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "
                + "timeout --signal=TERM --kill-after=10s 10m " + shlex.join(cmd)
                + " > " + shlex.quote(str(log)) + " 2>&1) &\npids+=(\"$!\")")
        lines.append(line)
    lines += ["failed=0", 'for pid in "${pids[@]}"; do wait "$pid" || failed=1; done',
              'exit "$failed"']
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", default="docs/data/toyfit_t4.json")
    ap.add_argument("--ice", default="runs/pwidom_fit/proxy_ice.json")
    ap.add_argument("--H", default="runs/pwidom_fit/proxy_H.json")
    ap.add_argument("--out", required=True, help="fresh output directory")
    ap.add_argument("--run-dir", help="also write a baseline replay script; do not execute it")
    a = ap.parse_args()
    paths = [Path(p).resolve() for p in (a.old, a.ice, a.H)]
    fits = [json.loads(p.read_text()) for p in paths]
    plan = replay_plan(Path(a.run_dir), paths[0]) if a.run_dir else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    (out / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    labels = ["old: DFT force/energy/stress fit", "new: GRACE insertion fit (H/O)",
              "new: GRACE insertion fit (pure H)"]
    colors = ["#555555", "#d55181", "#2878b5"]
    radii = np.linspace(.25, max(f["rc"] for f in fits) + .1, 700)
    with (out / "pair_curves.csv").open("w") as file:
        writer = csv.writer(file); writer.writerow(["proxy", "pair", "r_A", "energy_eV", "force_eV_A"])
        for kind, pair in enumerate(("OO", "OH", "HH")):
            for i, (fit, label, color) in enumerate(zip(fits, labels, colors)):
                if i == 2 and pair != "HH":
                    continue  # O-containing terms are unidentifiable in a pure-H-only fit.
                energy, derivative = pair_energy(kind, radii, fit=fit)
                axes[0, kind].plot(radii, energy, color=color, ls="--" if i == 0 else "-", label=label)
                axes[1, kind].plot(radii, -derivative, color=color, ls="--" if i == 0 else "-")
                for ax in axes[:, kind]:
                    ax.axvline(fit["pairs"][pair]["r_lo"], color=color, alpha=.3, lw=.8)
                writer.writerows((label, pair, r, e, -d) for r, e, d in zip(radii, energy, derivative))
            axes[0, kind].set_title(pair)
            for ax in axes[:, kind]:
                ax.set_yscale("symlog", linthresh=1); ax.axhline(0, color="k", lw=.5)
                ax.grid(alpha=.2); ax.set_xlabel("r (A)")
            axes[0, kind].set_ylabel("pair energy (eV; symlog)")
            axes[1, kind].set_ylabel("radial force -dphi/dr (eV/A; symlog)")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Old DFT fit vs GRACE insertion-fitted proxies; vertical lines: spline/core boundaries")
    fig.tight_layout(); fig.savefig(out / "proxy_pairs.png", dpi=160); plt.close(fig)
    if plan is not None:
        (out / "replay_old.sh").write_text(plan)
        print(f"Unexecuted cached-frame baseline replay plan: {out / 'replay_old.sh'}")
    print(f"Wrote {out / 'proxy_pairs.png'}, pair_curves.csv, inputs.json; no GPU calls")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
