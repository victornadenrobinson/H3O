"""Bounded, unlabeled geometry queue for later DFT review; never run VASP.

Proxy residuals and importance weights diagnose sampling, not GRACE accuracy.
Each insertion/deletion trial retains BOTH geometries needed for its energy
difference. Predicted energies are metadata, never ASE reference energy labels.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def geometry_id(at):
    h = hashlib.sha256()
    for arr in (at.numbers, np.round(at.positions, 8), np.round(at.cell.array, 8), at.pbc):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:24]


def enqueue(out, structures, metadata):
    """Serialize writers, deduplicate exact geometries and preserve request links."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ids = [geometry_id(at) for at in structures]
    request = dict(metadata, structures=[f"{i}.extxyz" for i in ids], reference_status="unlabeled")
    encoded = json.dumps(request, sort_keys=True, allow_nan=False)
    key = hashlib.sha256(encoded.encode()).hexdigest()[:24]
    with (out / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / f"request_{key}.json").exists():
            return False
        for at, identity in zip(structures, ids):
            path = out / f"{identity}.extxyz"
            if not path.exists():
                at = at.copy(); at.calc = None; at.info.clear()
                at.info.update(reference_status="unlabeled", geometry_id=identity)
                write(path, at, format="extxyz")
        tmp = out / f"request_{key}.tmp"
        tmp.write_text(encoded + "\n"); tmp.replace(out / f"request_{key}.json")
        with (out / "queue.jsonl").open("a") as index:
            index.write(json.dumps(dict(request=f"request_{key}.json", **request), allow_nan=False) + "\n")
    return True


def export_trials(out, trials, limit, provenance, policy="top", seed=0):
    """Choose trials for DFT, bounded per host run. policy 'top': leverage / residual / representative (diagnostic). policy 'weighted': limit//2 draws per direction WITH replacement with
    probability proportional to the estimator weight exp(log_weight) (importance resampling of the model's own Widom / deletion ensemble); a trial drawn k times is exported once with draw_count = k.
    The mean of exp(-beta dD) over the draws (dD = DFT - model difference of the pair) is then the DFT-over-model correction factor of the estimator (see reweight_dft.py)."""
    if limit <= 0 or not trials:
        return 0
    if policy == "weighted":
        rng = np.random.default_rng(seed + 4242); selected = {}
        for direction in ("insertion", "deletion"):
            group = [i for i, t in enumerate(trials) if t["direction"] == direction]
            if not group: continue
            lw = np.array([trials[i]["log_weight"] for i in group]); pr = np.exp(lw - lw.max()); pr /= pr.sum()
            for j in rng.choice(len(group), size=max(1, limit // 2), p=pr, replace=True): selected.setdefault(group[j], {"n": 0})["n"] += 1
        count = 0
        for i, d in selected.items():
            trial = dict(trials[i]); structures = trial.pop("structures")
            metadata = dict(provenance, **trial, reasons=["weighted_draw"], draw_count=d["n"], policy="weighted",
                            diagnosis="importance-resampled trial for the DFT-over-model correction; not a measured GRACE error",
                            geometry_order=["host_without_selected_H", "host_with_selected_H"])
            count += enqueue(out, structures, metadata)
        return count
    selected = {}
    # Disagreement is centered within each direction to remove an arbitrary proxy offset.
    for direction in ("insertion", "deletion"):
        group = [i for i, t in enumerate(trials) if t["direction"] == direction]
        if not group:
            continue
        median = np.median([trials[i]["gap_kT"] for i in group])
        ranked = [
            ("dominant_estimator_weight", sorted(group, key=lambda i: trials[i]["log_weight"], reverse=True)),
            ("large_centered_proxy_residual", sorted(group, key=lambda i: abs(trials[i]["gap_kT"] - median), reverse=True)),
            ("representative", [group[len(group)//2]]),
        ]
        for rank in range(2):
            for reason, indices in ranked:
                if rank < len(indices):
                    i = indices[rank]
                    if len(selected) < limit or i in selected:
                        selected.setdefault(i, set()).add(reason)
    count = 0
    for i, reasons in selected.items():
        trial = dict(trials[i]); structures = trial.pop("structures")
        metadata = dict(provenance, **trial, reasons=sorted(reasons),
                        diagnosis="sampling/proxy diagnostic; not a measured GRACE error",
                        geometry_order=["host_without_selected_H", "host_with_selected_H"])
        count += enqueue(out, structures, metadata)
    return count


def cached_frames(a):
    """Representative MD snapshots only: older archives do not contain trial sites."""
    count = 0
    for file in a.frames:
        with np.load(file) as z:
            numbers = z["numbers"]
            cell = z["cell"] if "cell" in z.files else np.eye(3) * float(z["V"]) ** (1/3)
            for kind in ("host", "full"):
                positions = z[kind]
                for i in np.unique(np.linspace(0, len(positions)-1, min(a.per_source, len(positions))).astype(int)):
                    at = Atoms(numbers=numbers if kind == "host" else np.r_[numbers, 1],
                               positions=positions[i], cell=cell, pbc=True)
                    meta = dict(source=str(Path(file).resolve()), kind=kind, frame=int(i),
                                T_K=float(z["T"]), reasons=["representative_cached_MD"],
                                model=str(z["model"]) if "model" in z.files else "unknown",
                                model_fp=str(z["model_fp"]) if "model_fp" in z.files else "unknown",
                                diagnosis="representative MD geometry; no trial-level diagnostic preserved")
                    count += enqueue(a.out, [at], meta)
    print(f"Queued {count} new requests in {a.out}; no model/DFT calls. Review before labeling/training.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="+", help="existing pwidom *_frames.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=3, help="representative frames per ensemble")
    args = ap.parse_args()
    if args.per_source < 1:
        ap.error("--per-source must be positive")
    cached_frames(args)
