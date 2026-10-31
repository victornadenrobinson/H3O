"""CPU-only retrospective running estimates from saved frame-grouped works.

This is versus cumulative production MD sampled, not wall time. Each prefix uses
both host and host+H ensembles and the completed, fixed pure-H reference. No new
MD or model calls. Widom has no acceptance/rejection; counts are evaluated trials
and Kish effective weight counts, not independent samples or accepted moves.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import pwidom as pw
import pwidom_scan as scan
from perlmutter_pwidom import shared_reference


def prefixes(result, command, reference=None, points=30):
    workpath = Path(result["work_file"])
    with np.load(workpath) as work:
        forward = work["forward_work_eV"].copy()
        reverse = work["reverse_work_eV"].copy()
        host_ids = work["host_frame_ids"].copy()
        full_ids = work["full_frame_ids"].copy()
        source = Path(str(work["source_frames"]))
        interval = float(work["frame_interval_fs"]) if "frame_interval_fs" in work else (
            float(command[command.index("--every-fs")+1]) if "--every-fs" in command else 20.)
        if str(work["model_fp"]) != result["model_fp"]:
            raise ValueError("Work/result model fingerprint mismatch")
    if reference is not None and (reference["T"] != result["T"] or
        reference["P_target"] != result["P_target"] or reference["model_fp"] != result["model_fp"]):
        raise ValueError("Reference P/T/model mismatch")
    with np.load(source) as frames:
        pressures = frames["P_host"].copy()
    if (forward.ndim != 2 or reverse.ndim != 2 or len(forward) < 5 or
        len(reverse) < 5 or len(host_ids) != len(forward) or len(full_ids) != len(reverse)):
        raise ValueError("Need at least five grouped forward/reverse frames with matching IDs")
    kt = pw.KB*result["T"]
    rows = []
    for count in np.unique(np.linspace(5,len(forward),min(points,len(forward)-4),dtype=int)):
        reverse_count = max(1,round(count*len(reverse)/len(forward)))
        f = forward[:count].ravel(); r = reverse[:reverse_count].ravel()
        mw = -kt*(logsumexp(-f/kt)-np.log(len(f)))
        md = kt*(logsumexp(-r/kt)-np.log(len(r)))
        mb = pw.bar(f,r,kt)
        prefix = dict(result, mu_fixedV_meV=mb*1000+result["ideal_meV"])
        limit = len(pressures) if count == len(forward) else int(host_ids[count-1])+1
        mean, error = pw.frame_mean_P(pressures[:limit])
        prefix.update(P_host_GPa=mean,P_host_err_GPa=error)
        mu = scan.mu_P(prefix)[0]
        weights = np.exp(-(f-f.min())/kt)
        effective = weights.sum()**2/(weights**2).sum()
        rows.append(dict(production_ps=limit*interval/1000,
            frames=int(count), insertion_trials=len(f), deletion_trials=len(r),
            candidate_sites=int(count)*result["K"], kish_fraction=effective/len(f),
            kish_effective_count=effective, mu_widom_meV=mu+(mw-mb)*1000,
            mu_delete_meV=mu+(md-mb)*1000, mu_bar_meV=mu,
            dg_meV=scan.dg(prefix,reference)[0] if reference is not None else float("nan")))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--out",required=True,help="fresh PNG/PDF/CSV output prefix")
    ap.add_argument("--points",type=int,default=30)
    a = ap.parse_args()
    if a.points < 2: raise ValueError("Need at least two requested points")
    root = Path(a.run); manifest = json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "complete": raise ValueError("Use a completed run")
    results = {job["name"]:json.loads(Path(job["result"]).read_text()) for job in manifest["jobs"]}
    reference = shared_reference(manifest) if manifest.get("reference") else results.get("pureH")
    if reference is None: raise ValueError("No completed pure-H reference")
    jobs = [j for j in manifest["jobs"] if j["name"] != "pureH"]
    data = {j["name"]:prefixes(results[j["name"]],j["command"],reference,a.points) for j in jobs}
    out = Path(a.out)
    for suffix in (".png",".pdf",".csv"):
        if out.with_suffix(suffix).exists(): raise ValueError("Choose a fresh output prefix")
    out.parent.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,len(jobs),squeeze=False,sharex="col",figsize=(3.6*len(jobs),8))
    for col,job in enumerate(jobs):
        rows = data[job["name"]]; t = [r["production_ps"] for r in rows]
        for key,label,color in (("mu_widom_meV","Widom","#0072b2"),
            ("mu_delete_meV","Deletion","#e69f00"),("mu_bar_meV","BAR","#cc79a7")):
            axes[0,col].plot(t,[r[key] for r in rows],label=label,color=color)
        axes[0,col].set_title(job["name"]); axes[0,col].legend(fontsize=8)
        axes[1,col].plot(t,[r["dg_meV"] for r in rows],color="#cc79a7")
        axes[1,col].axhline(0,color="0.4",lw=.7)
        for key,label in (("insertion_trials","Evaluated insertion trials"),
                          ("kish_effective_count","Kish effective weight count")):
            axes[2,col].plot(t,[r[key] for r in rows],label=label)
        axes[2,col].legend(fontsize=7)
        axes[2,col].set_xlabel("Host production sampled (ps)")
        for ax in axes[:,col]: ax.grid(alpha=.2)
    for ax,label in zip(axes[:,0],("Running μH (meV/H, robust P)","Running ΔμH (meV/H)","Cumulative count")):
        ax.set_ylabel(label)
    fig.suptitle("Retrospective prefix estimates — no new model calls")
    fig.text(.5,.015,"Pure-H reference held at its completed estimate; prefixes overlap and are correlated.\n"
             "Time excludes equilibration; curves omit uncertainty bands. Kish count is not independent ESS. No accepted/rejected moves.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.95))
    for suffix in (".png",".pdf"): fig.savefig(out.with_suffix(suffix),dpi=170)
    plt.close(fig)
    with out.with_suffix(".csv").open("w") as file:
        writer = csv.DictWriter(file,fieldnames=["host",*next(iter(data.values()))[0]])
        writer.writeheader()
        for name,rows in data.items(): writer.writerows(dict(host=name,**row) for row in rows)
    print(f"Wrote {out}.png/.pdf/.csv; cumulative production time, not wall time")


if __name__ == "__main__": main()
