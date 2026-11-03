"""Plot separate MD/probe throughput and sampled memory from benchmark reports.

Accepts benchmark directories (results.jsonl + summary.json), exported CSVs,
or saved terminal transcripts. Never combines MD-step and structure rates.
"""
import argparse
import csv
import json
from pathlib import Path
import re

FIELDS = ["run", "kind", "phase", "gpu", "mode", "size", "rate", "unit", "peak_MiB"]


def kind(phase):
    if phase == "memory_calibration": return "calibration"
    if "node_validation" in phase: return "validation"
    return "screen"


def load(source):
    source = Path(source)
    rows = []
    if source.suffix == ".csv":
        with source.open() as file: rows = list(csv.DictReader(file))
        for row in rows:
            row["size"] = int(row["size"]); row["rate"] = float(row["rate"])
            row["peak_MiB"] = float(row["peak_MiB"]) if row["peak_MiB"] else None
    elif source.is_dir() or source.suffix == ".jsonl":
        root = source if source.is_dir() else source.parent
        path = root/"results.jsonl" if source.is_dir() else source
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["status"] != "complete": continue
            rows.append(dict(run=root.name,kind=kind(item["phase"]),phase=item["phase"],
                gpu=str(item["gpu"]),mode=item["mode"],size=item["size"],
                rate=item["work_units_s"],unit=item["unit"],peak_MiB=item["sampled_peak_MiB"]))
        summary = json.loads((root/"summary.json").read_text())
        for mode, item in summary.get("whole_node_by_mode",{}).items():
            best = summary["best_by_mode"][mode]
            rows.append(dict(run=root.name,kind="node_total",phase="whole_node",gpu="all",
                mode=best["mode"],size=best["size"],rate=item["work_units_s"],
                unit=item["unit"],peak_MiB=None))
    else:
        run = None; phase = None
        for line in source.read_text().splitlines():
            match = re.search(r"outputs runs/(bench_\d+_\d+)",line)
            if match: run=match[1]
            match = re.match(r"== (\w+):",line)
            if match: phase=match[1]
            match = re.match(r"RESULT GPU (\S+) (\w+) (\d+): ([\d.]+) (.*?)/s; sampled peak ([\d.]+) MiB",line)
            if match and run and phase:
                gpu,mode,size,rate,unit,peak=match.groups()
                rows.append(dict(run=run,kind=kind(phase),phase=phase,gpu=gpu,
                    mode=mode,size=int(size),rate=float(rate),unit=unit,peak_MiB=float(peak)))
            match = re.match(r"MEASURED FOUR-GPU (\w+): ([\d.]+) (.*?)/s; size (\d+)/GPU",line)
            if match and run:
                mode,rate,unit,size=match.groups()
                rows.append(dict(run=run,kind="node_total",phase="whole_node",gpu="all",
                    mode=mode,size=int(size),rate=float(rate),unit=unit,peak_MiB=None))
    if not rows: raise ValueError(f"No completed benchmark records in {source}")
    return rows


def plot(rows,out,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    runs = sorted(set(r["run"] for r in rows))
    colors = ["#a8b3c2", "#6293b8", "#ce477f"]
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    labels=[]
    for index,run in enumerate(runs):
        latest=index==len(runs)-1
        label=run.replace("bench_20261031_","")
        label=":".join(label[k:k+2] for k in range(0,len(label),2)) if len(label)==6 else label
        labels.append(label)
        for col,mode in enumerate(("pwmd","pwsp")):
            points=sorted((r for r in rows if r["run"]==run and r["kind"]=="screen" and r["mode"]==mode),key=lambda r:r["size"])
            if not points: continue
            sizes=[r["size"] for r in points]
            for ax,field,scale in ((axes[0,col],"rate",1),(axes[1,col],"peak_MiB",1/1024)):
                recorded=[r for r in points if r[field] is not None]
                xvalues=[r["size"] for r in recorded]; values=[r[field]*scale for r in recorded]
                ax.plot(xvalues,values,"o-",color=colors[min(index,len(colors)-1)],
                    lw=2.3 if latest else 1.3,alpha=1 if latest else .75,
                    label=label+(" (latest)" if latest else ""))
                ax.set_xscale("log",base=2); ax.set_xticks(sizes)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.grid(alpha=.2); ax.set_ylim(bottom=0)
                if latest:
                    for x,y in zip(xvalues,values):
                        ax.annotate(f"{y:.2f}" if ax is axes[0,col] else f"{y:.1f}",
                            (x,y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
            axes[0,col].legend(fontsize=8,loc="lower right")
    axes[0,0].set_title("MD: more independent workers")
    axes[0,1].set_title("Probes: more structures per batch")
    axes[0,0].set_ylabel("Aggregate MD steps/s per GPU")
    axes[0,1].set_ylabel("GRACE structures/s per GPU")
    for ax in axes[1]: ax.set_ylabel("Sampled peak GPU memory (GiB)")
    for ax in axes[:,0]: ax.set_xlabel("Independent MD workers per GPU")
    for ax in axes[:,1]: ax.set_xlabel("Structures per GRACE probe batch")
    for col,mode in enumerate(("pwmd","pwsp")):
        points=[r for r in rows if r["kind"]=="screen" and r["mode"]==mode]
        for row,field,scale in ((0,"rate",1),(1,"peak_MiB",1/1024)):
            values=[r[field]*scale for r in points if r[field] is not None]
            if values: axes[row,col].set_ylim(0,max(values)*1.18)
    fig.suptitle(title,fontsize=13)
    notes=[]
    for run,label in zip(runs,labels):
        totals={r["mode"]:r for r in rows if r["run"]==run and r["kind"]=="node_total"}
        if "pwmd" in totals and "pwsp" in totals:
            md,sp=totals["pwmd"],totals["pwsp"]
            notes.append(f"{label}  Whole node: MD {md['size']}/GPU → {md['rate']:.2f} steps/s; "
                         f"probe batch {sp['size']} → {sp['rate']:.2f} structures/s")
    fig.text(.5,.08,"\n".join(notes),ha="center",fontsize=9)
    fig.text(.5,.015,"Screen layouts run on different GPUs while sharing node CPUs. Lines connect discrete tested choices.\n"
             "Between-run variation is not a confidence interval; memory values are sampled peaks, not guaranteed maxima.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.91))
    for suffix in (".png",".pdf"): fig.savefig(out.with_suffix(suffix),dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources",nargs="+")
    ap.add_argument("--out",required=True,help="fresh PNG/PDF/CSV output prefix")
    ap.add_argument("--title",default="pwidom throughput and GPU memory — short screens")
    a=ap.parse_args(); records={}
    for source in a.sources:
        for row in load(source): records[tuple(row[k] for k in ("run","kind","phase","gpu","mode","size"))]=row
    rows=sorted(records.values(),key=lambda r:(r["run"],r["phase"],r["mode"],r["size"],r["gpu"]))
    if not any(r["kind"]=="screen" and r["mode"] in ("pwmd","pwsp") for r in rows):
        raise ValueError("Need completed pwidom MD/probe screens")
    out=Path(a.out)
    for suffix in (".png",".pdf",".csv"):
        if out.with_suffix(suffix).exists(): raise ValueError("Choose a fresh output prefix")
    out.parent.mkdir(parents=True,exist_ok=True)
    plot(rows,out,a.title)
    with out.with_suffix(".csv").open("w") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    print(f"Wrote {out}.png/.pdf/.csv; {len(rows)} completed records")


if __name__=="__main__":main()
