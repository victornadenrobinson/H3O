#!/usr/bin/env python3
"""Reconstruct production BAR from saved work, keeping trials grouped by frame."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit, logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tdi.pwidom import bar, KB


def diagnose(forward, reverse, temperature):
    """reverse is already sign-adjusted as stored; never negate it again."""
    f, r = np.asarray(forward).ravel(), np.asarray(reverse).ravel()
    kt = KB * temperature
    if (not np.isfinite(kt) or kt <= 0 or not len(f) or not len(r)
            or not np.isfinite(f).all() or not np.isfinite(r).all()):
        raise ValueError('Need finite nonempty works and positive finite T')
    mu = bar(f, r, kt)
    offset = kt * np.log(len(f) / len(r))
    af = expit((mu - offset - f) / kt)
    ar = expit((offset - mu - r) / kt)
    q = np.exp(-(f - f.min()) / kt)
    return dict(mu_ex_bar_meV=1000 * mu,
                forward_exp_meV=-1000 * kt * (logsumexp(-f/kt)-np.log(len(f))),
                reverse_exp_meV=1000 * kt * (logsumexp(-r/kt)-np.log(len(r))),
                overlap=float(af.mean()+ar.mean()),
                forward_acceptance=float(af.mean()), reverse_acceptance=float(ar.mean()),
                kish_fraction=float(q.sum()**2 / np.dot(q, q) / len(q)),
                n_forward=len(f), n_reverse=len(r))


def analyse(path, out, fractions=10, blocks=5, result=None):
    with np.load(path, allow_pickle=False) as z:
        f, r = z['forward_work_eV'], z['reverse_work_eV']
        temperature = float(z['T'])
    if f.ndim != 2 or r.ndim != 2:
        raise ValueError('Expected frame x trial arrays; cannot infer frame grouping from flat work')
    if min(len(f), len(r)) < 1:
        raise ValueError('Empty frame arrays')
    rows = []
    for fraction in np.linspace(1/fractions, 1, fractions):
        nf, nr = max(1, int(len(f)*fraction)), max(1, int(len(r)*fraction))
        rows.append(dict(kind='cumulative', fraction=float(fraction),
                         host_frames=nf, full_frames=nr,
                         **diagnose(f[:nf], r[:nr], temperature)))
    nb = min(blocks, len(f), len(r))
    for i, (bf, br) in enumerate(zip(np.array_split(f, nb), np.array_split(r, nb))):
        rows.append(dict(kind='block', fraction=(i+1)/nb,
                         host_frames=len(bf), full_frames=len(br),
                         **diagnose(bf, br, temperature)))
    result = Path(result) if result else Path(str(path).replace('_work.npz', '.json'))
    if result.exists():
        expected = float(json.loads(result.read_text())['mu_ex_bar_meV'])
        if not np.isfinite(expected) or not np.isclose(rows[fractions-1]['mu_ex_bar_meV'], expected, atol=1e-5, rtol=1e-8):
            raise ValueError(f'{result}: saved mu_ex_bar_meV disagrees with work reconstruction')
        print(f'Validated final BAR against {result}')
    out.mkdir(parents=True, exist_ok=False)
    with (out/'diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    cumulative = rows[:fractions]
    axes[0,0].plot([x['fraction'] for x in cumulative], [x['mu_ex_bar_meV'] for x in cumulative], 'o-')
    axes[0,0].set(xlabel='Sampling fraction (not physical time)', ylabel='BAR excess chemical potential (meV)')
    axes[0,1].hist(f.ravel(), bins=50, density=True, alpha=.5, label='Forward')
    axes[0,1].hist(-r.ravel(), bins=50, density=True, alpha=.5, label='Negative stored reverse')
    axes[0,1].set(xlabel='Insertion-oriented work (eV)', ylabel='Density'); axes[0,1].legend()
    for key in ('overlap', 'kish_fraction'):
        axes[1,0].plot([x['fraction'] for x in cumulative], [x[key] for x in cumulative], label=key)
    axes[1,0].set(xlabel='Sampling fraction', ylabel='Diagnostic fraction'); axes[1,0].legend()
    axes[1,1].plot(range(1, nb+1), [x['mu_ex_bar_meV'] for x in rows[fractions:]], 'o')
    axes[1,1].set(xlabel='Contiguous frame block', ylabel='BAR (meV); no independence assumed')
    fig.tight_layout(); fig.savefig(out/'bar_convergence.pdf'); plt.close(fig)
    (out/'README.txt').write_text('Trials share frames and proxy candidate pools. Kish is a weight concentration diagnostic, not an independent sample count.\nBlock scatter is not a confidence interval; blocks must exceed correlation times. Host/full fractions refer to independent trajectories, not aligned physical time.\nOverlap = mean forward Fermi acceptance + mean reverse Fermi acceptance (0 to 1); low overlap warns of poor support.\nDo not pool overlapping parent and extension caches.\n')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('work', nargs='+', type=Path)
    p.add_argument('--out', type=Path, default=Path('bar_convergence'))
    p.add_argument('--fractions', type=int, default=10)
    p.add_argument('--blocks', type=int, default=5)
    p.add_argument('--result', type=Path, help='Explicit result JSON (single work file only)')
    a = p.parse_args()
    if a.fractions < 1 or a.blocks < 1 or (a.result and len(a.work) != 1):
        p.error('Positive fractions/blocks required; --result requires one work file')
    for path in a.work:
        analyse(path, a.out/path.stem, a.fractions, a.blocks, a.result)


if __name__ == '__main__':
    main()
