#!/usr/bin/env python3
"""Read-only PWidom collection and cumulative/time-coloured hull diagnostics."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    tasks=p.add_subparsers(dest='task',required=True)
    collect=tasks.add_parser('collect',help='all finished results, mu/dG tables and provisional hulls')
    collect.add_argument('--root',type=Path,default=Path('runs'))
    collect.add_argument('--out',type=Path,required=True)
    collect.add_argument('--pressure-mode',choices=['robust','measured'],default='robust')
    collect.add_argument('--include-flagged',action='store_true')
    time=tasks.add_parser('time',help='cumulative mu/G/dG and hulls coloured by sampling time')
    time.add_argument('--runs',nargs='+',type=Path,required=True,help='completed reference and extra-H directories; one state/model only')
    time.add_argument('--out',type=Path,required=True)
    time.add_argument('--step-ps',type=float,default=.1)
    time.add_argument('--block-ps',type=float,default=.1,help='contiguous MD block duration for sampling SE')
    time.add_argument('--min-blocks',type=int,default=3,help='minimum complete blocks before drawing uncertainty')
    time.add_argument('--reference-mode',choices=['cumulative','fixed'],default='cumulative')
    replot=tasks.add_parser('replot-time',help='redraw time figures from an existing prefix_input.json; no work/cache reads')
    replot.add_argument('--input',type=Path,required=True)
    replot.add_argument('--out',type=Path,required=True)
    replot.add_argument('--step-ps',type=float,default=.1)
    replot.add_argument('--reference-mode',choices=['cumulative','fixed'],default='cumulative')
    a=p.parse_args()
    if a.out.exists():p.error(f'Output exists: {a.out}; choose a fresh report directory')
    if a.task=='collect':
        from collect_pwidom_results import inventory,report
        records,errors=inventory(a.root)
        report(records,a.out,a.pressure_mode,a.include_flagged,errors)
    elif a.task=='replot-time':
        from plot_pwidom_time_hulls import analyse
        analyse(json.loads(a.input.read_text()),a.out,a.step_ps,a.reference_mode)
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from export_pwidom_prefixes import export_runs
        from plot_pwidom_time_hulls import analyse
        records=export_runs(a.runs,a.block_ps,a.min_blocks)
        analyse(records,a.out,a.step_ps,a.reference_mode)


if __name__=='__main__':main()
