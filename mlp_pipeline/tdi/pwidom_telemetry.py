"""Plot GPU memory/utilization and stage progress from telemetry or a compact log.

Input: a run directory (telemetry.jsonl), a telemetry JSONL file, an exported
CSV, or a saved compact terminal transcript. Stage boundaries are sampled, not
exact transition times. Transcripts may contain only part of the run.
"""
import argparse
import csv
import json
from pathlib import Path
import re

FIELDS = ["elapsed_seconds", "name", "gpu", "stage", "fraction", "utilization",
          "memory_MiB", "total_MiB", "rss_MiB"]
NUMERIC = set(FIELDS)-{"name", "gpu", "stage"}


def seconds(text):
    value = 0.
    for part in text.split(":"):
        value = value*60+float(part)
    return value


def load_samples(source):
    source = Path(source)
    if source.is_dir(): source = source/"telemetry.jsonl"
    rows = []
    if source.suffix == ".jsonl":
        for line in source.read_text().splitlines():
            sample = json.loads(line)
            rows.extend(dict(elapsed_seconds=sample["elapsed_seconds"], **host)
                        for host in sample["hosts"])
    elif source.suffix == ".csv":
        with source.open() as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            for key in NUMERIC:
                row[key] = float(row[key]) if row.get(key) else None
    else:
        elapsed = None
        pattern = re.compile(r"^(\S+)\s+(.*?)\s+[█░#-]+\s+(\d+)%\s+.*?GPU(\d+)\s+([\d.]+)%\s+([\d.]+)/([\d.]+) GiB")
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", source.read_text())
        for line in text.splitlines():
            clock = re.match(r"Elapsed (\d+(?::\d+){1,2})", line)
            if clock: elapsed = seconds(clock[1])
            match = pattern.match(line)
            if match and elapsed is not None:
                name, stage, fraction, gpu, util, used, total = match.groups()
                rows.append(dict(elapsed_seconds=elapsed, name=name, stage=stage.strip(),
                    fraction=float(fraction)/100, gpu=gpu, utilization=float(util),
                    memory_MiB=float(used)*1024, total_MiB=float(total)*1024, rss_MiB=None))
    if not rows: raise ValueError("No telemetry samples found; older runs need a saved compact terminal transcript")
    return sorted(rows, key=lambda r: (r["elapsed_seconds"], r["name"]))


def category(stage):
    s = stage.lower()
    if "complete" in s or s == "done": return "Finished"
    if "failed" in s: return "Failed"
    if "single points" in s: return "GRACE probes"
    if "proxy" in s: return "CPU proxy selection"
    if "host+" in s or "host +" in s: return "Host+H MD"
    if "host" in s: return "Host MD"
    if "analysis" in s: return "CPU analysis"
    return "Initialization / other"


COLORS = {"Host MD":"#56b4e9", "Host+H MD":"#009e73", "GRACE probes":"#cc79a7",
          "CPU proxy selection":"#e69f00", "Finished":"#999999", "Failed":"#d55e00",
          "CPU analysis":"#f0e442", "Initialization / other":"#cccccc"}


def plot(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    names = sorted(set(r["name"] for r in rows),key=lambda name: next(str(r["gpu"]) for r in rows if r["name"]==name))
    fig, axes = plt.subplots(3, len(names), squeeze=False, sharex=True,
                             figsize=(3.5*len(names),8))
    seen = set()
    for col, name in enumerate(names):
        samples = [r for r in rows if r["name"] == name]
        t = [r["elapsed_seconds"]/60 for r in samples]
        for left, right in zip(samples, samples[1:]):
            phase = category(left["stage"]); seen.add(phase)
            for ax in axes[:,col]:
                ax.axvspan(left["elapsed_seconds"]/60, right["elapsed_seconds"]/60,
                           color=COLORS[phase], alpha=.17, lw=0)
        for row, field, scale in ((0,"memory_MiB",1/1024),(1,"utilization",1),(2,"fraction",100)):
            valid = [r for r in samples if r.get(field) is not None]
            axes[row,col].plot([r["elapsed_seconds"]/60 for r in valid],
                               [r[field]*scale for r in valid], "o-", color="#25465f", ms=4)
            axes[row,col].grid(alpha=.2)
        capacities = [r["total_MiB"]/1024 for r in samples if r.get("total_MiB")]
        if capacities:
            capacity = max(capacities)
            axes[0,col].axhline(capacity, color="0.4", ls=":", label=f"Capacity {capacity:g} GiB")
            axes[0,col].set_ylim(0,capacity*1.08)
            axes[0,col].legend(fontsize=8, loc="upper left")
        axes[0,col].set_title(f"{name} · GPU {samples[0]['gpu']}")
        axes[1,col].set_ylim(-3,103); axes[2,col].set_ylim(-3,103)
        axes[2,col].set_xlabel("Elapsed run time (min)")
    for ax, label in zip(axes[:,0], ("GPU memory used (GiB)", "GPU utilization (%)", "Current-stage progress (%)")):
        ax.set_ylabel(label)
    fig.suptitle("pwidom resource use — observed samples only", fontsize=14)
    fig.legend(handles=[Patch(facecolor=color,alpha=.4,label=name) for name,color in COLORS.items() if name in seen],
               loc="lower center", ncol=min(4,len(seen)), bbox_to_anchor=(.5,.04),fontsize=9)
    fig.text(.5,.012,"Shading uses stage at each sample until the next; transitions may occur between samples.\n"
             "GPU utilization is instantaneous; progress resets at stage changes. No unrecorded history is inferred.",
             ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.16,1,.95))
    for suffix in (".png", ".pdf"): fig.savefig(out.with_suffix(suffix), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--out",required=True,help="fresh output prefix for PNG/PDF/CSV")
    args = parser.parse_args()
    out = Path(args.out)
    for suffix in (".png", ".pdf", ".csv"):
        if out.with_suffix(suffix).exists(): raise ValueError(f"Already exists: {out.with_suffix(suffix)}")
    rows = load_samples(args.source)
    out.parent.mkdir(parents=True,exist_ok=True)
    plot(rows,out)
    with out.with_suffix(".csv").open("w") as file:
        writer = csv.DictWriter(file,fieldnames=FIELDS,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {out}.png/.pdf/.csv from {len(rows)} observations")


if __name__ == "__main__": main()
