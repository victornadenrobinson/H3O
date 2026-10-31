#!/usr/bin/env python3
"""Read-only PWidom collection and cumulative/time-coloured hull diagnostics."""
import argparse
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
    time.add_argument('--reference-mode',choices=['cumulative','fixed'],default='cumulative')
    a=p.parse_args()
    if a.out.exists():p.error(f'Output exists: {a.out}; choose a fresh report directory')
    if a.task=='collect':
        from collect_pwidom_results import inventory,report
        records,errors=inventory(a.root)
        report(records,a.out,a.pressure_mode,a.include_flagged,errors)
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from export_pwidom_prefixes import export_runs
        from plot_pwidom_time_hulls import analyse
        records=export_runs(a.runs)
        analyse(records,a.out,a.step_ps,a.reference_mode)


if __name__=='__main__':main()
