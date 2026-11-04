#!/usr/bin/env python3
"""Read-only collection of finished PWidom results, tables and provisional hulls.

No MD/checkpoints/trajectories are read. All output goes to a fresh directory.
"""
import argparse
import csv
import hashlib
import json
from math import lgamma, log
from pathlib import Path

import numpy as np

from tdi import pwidom_scan as scan
from tdi.pwidom_formation import integrate, statistical_error_bound


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Results root unavailable: {root}')
    payload, errors = {}, []
    for directory in sorted(root.glob('pwidom*')):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob('*.json')):
            if {'_history', '_resume_history'} & set(path.relative_to(directory).parts):
                continue
            try:
                data = json.loads(path.read_text())
                if isinstance(data, dict) and ('mu_ex_bar_meV' in data or path.name == 'manifest.json'):
                    payload[str(path.resolve())] = data
            except (OSError, ValueError) as exc:
                errors.append(dict(source=str(path), reason=str(exc)))
    return payload, errors


def compatible(a, b):
    if a['T'] != b['T'] or a['P_target'] != b['P_target'] or b['phase'] != 'H':
        return False
    return (a.get('model_fp') == b.get('model_fp') if a.get('model_fp') and b.get('model_fp')
            else a.get('model') == b.get('model'))


def suffix(path, data):
    return Path(path).name.removeprefix(data.get('tag', '')+'_')


def reference(path, data, payload, manifest):
    if manifest:
        if manifest.get('reference'):
            ref = manifest['reference']
            snapshot = ref.get('snapshot')
            if isinstance(snapshot, dict) and compatible(data, snapshot):
                return snapshot, ref.get('result', 'manifest reference snapshot')
            target = payload.get(str(ref.get('result')))
            if target and compatible(data, target):
                return target, ref['result']
        jobs = [j for j in manifest.get('jobs', []) if j.get('name') == 'pureH']
        if len(jobs) == 1:
            # Prefer local filename on moved/exported results.
            targetpath = str(Path(path).parent/Path(jobs[0]['result']).name)
            target = payload.get(targetpath) or payload.get(jobs[0]['result'])
            if target and compatible(data, target):
                return target, targetpath
    candidates = [(p, r) for p, r in payload.items()
                  if Path(p).parent == Path(path).parent and 'mu_ex_bar_meV' in r
                  and compatible(data, r)]
    matched = [(p, r) for p, r in candidates if suffix(path, data) == suffix(p, r)]
    if len(matched) == 1:
        return matched[0][1], matched[0][0]
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]
    return None, ''


def rows_from(payload, mode='measured'):
    rows, errors = [], []
    manifests = {str(Path(p).parent): v for p, v in payload.items() if Path(p).name == 'manifest.json'}
    for path, result in sorted(payload.items()):
        if 'mu_ex_bar_meV' not in result:
            continue
        try:
            w = result
            phase = w['phase']
            mu, error = scan.mu_P(w, mode)
            required = [w['T'], w['P_target'], mu, error, w['mu_ex_bar_meV']]
            if not np.isfinite(required).all() or min(w['T'], w['P_target']) <= 0 or error < 0:
                raise ValueError('Nonfinite/invalid result or uncertainty')
            manifest = manifests.get(str(Path(path).parent))
            flags = []
            if not w.get('model_fp'):
                flags.append('unverified_model')
            if w.get('melted') or scan.o_growing(w):
                flags.append('melted')
            if not w.get('pressure_corrected', bool(w.get('B_GPa'))):
                flags.append('no_pressure_correction')
            if w.get('kish_frac_ins', 0) < .05:
                flags.append('low_kish')
            if abs(w.get('mu_ex_delete_meV', np.inf)-w.get('mu_ex_widom_meV', -np.inf)) > max(200, 3*error):
                flags.append('forward_reverse_disagreement')
            nh, nat = int(w['N_H']), int(w['natoms'])
            n = nh/(nat-nh)-2 if phase != 'H' and nat > nh else None
            row = dict(P_GPa=w['P_target'], T_K=w['T'], phase=phase, n=n,
                       x_H=(2+n)/(3+n) if n is not None else 1.,
                       N_H=nh, natoms=nat, mu_ex_bar_meV=w['mu_ex_bar_meV'],
                       mu_ex_bar_err_meV=w.get('mu_ex_bar_err_meV', ''),
                       mu_H_meV=mu, mu_H_err_meV=error,
                       mu_fixedV_meV=w.get('mu_fixedV_meV', ''),
                       mu_P_measured_meV=w['mu_P_meV'], mu_P_measured_err_meV=w['mu_P_err_meV'],
                       dg_meV='', dg_err_meV='', dg_measured_meV='', dg_measured_err_meV='',
                       dg_robust_meV='', dg_robust_err_meV='', reservoir_G_H_meV='', reservoir_err_meV='',
                       reference='', pressure_mode=mode, kish=w.get('kish_frac_ins', ''),
                       n_frames_host=w.get('n_frames_host', 0), n_frames_full=w.get('n_frames_full', 0),
                       model=w.get('model', ''), model_fp=w.get('model_fp', ''),
                       run=str(Path(path).parent), source=path,
                       primary_result=bool(manifest and any(Path(j.get('result','')).name == Path(path).name for j in manifest.get('jobs',[]))),
                       launch_status=manifest.get('status', 'no_manifest') if manifest else 'no_manifest',
                       flags='', selected_for_integral=False)
            if phase == 'H':
                fs = scan.KB*w['T']*1000*(log(nh+1)-lgamma(nh+1)/nh-1)
                row.update(reservoir_G_H_meV=mu-fs, reservoir_err_meV=error)
            else:
                h, source = reference(path, w, payload, manifest)
                if h is None:
                    flags.append('missing_or_ambiguous_reference')
                else:
                    value, uncertainty = scan.dg(w, h, mode)
                    if not np.isfinite([value, uncertainty]).all() or uncertainty < 0:
                        raise ValueError('Invalid reference/dG')
                    mh, eh = scan.mu_P(h, mode)
                    for variant in ('measured','robust'):
                        try:
                            value_variant, error_variant = scan.dg(w,h,variant)
                            if np.isfinite([value_variant,error_variant]).all():
                                row[f'dg_{variant}_meV'] = value_variant
                                row[f'dg_{variant}_err_meV'] = error_variant
                        except (KeyError,ValueError,TypeError,ZeroDivisionError):
                            pass
                    row.update(dg_meV=value, dg_err_meV=uncertainty,
                               reservoir_G_H_meV=mu-value, reservoir_err_meV=eh, reference=source)
                    if not h.get('model_fp'):
                        flags.append('unverified_reference')
                    if h.get('melted') or not h.get('pressure_corrected', bool(h.get('B_GPa'))):
                        flags.append('invalid_reference')
                    if h.get('kish_frac_ins', 0) < .05:
                        flags.append('reference_low_kish')
                    if abs(h.get('mu_ex_delete_meV', np.inf)-h.get('mu_ex_widom_meV', -np.inf)) > max(200,3*eh):
                        flags.append('reference_forward_reverse_disagreement')
            row['flags'] = ';'.join(flags) or 'ok'
            rows.append(row)
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            errors.append(dict(source=path, reason=str(exc)))
    return rows, errors


def lower_hull(x, y):
    """Indices of the lower convex envelope of sorted, distinct x."""
    if np.any(np.diff(x) <= 0):
        raise ValueError('Hull compositions must be strictly increasing')
    hull = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2:]
            cross = (x[b]-x[a])*(y[i]-y[a])-(y[b]-y[a])*(x[i]-x[a])
            if cross > 0:
                break
            hull.pop()
        hull.append(i)
    return hull


def selected_groups(rows, include_flagged=False):
    groups = {}
    for row in rows:
        # Parent/extension/replay rows are listed but never pooled. Select the
        # largest frame count per composition within each verified Hamiltonian.
        if row['phase'] != 'fcc' or row['n'] is None or row['n'] < -1e-10 or row['dg_meV'] == '':
            continue
        hard = {'melted', 'unverified_model', 'unverified_reference', 'invalid_reference', 'no_pressure_correction'}
        flags = set(row['flags'].split(';'))-{'ok'}
        if flags & hard or (flags and not include_flagged):
            continue
        key = (row['P_GPa'], row['T_K'], row['model_fp'])
        group = groups.setdefault(key, {})
        n = round(row['n'], 10)
        previous = group.get(n)
        rank = lambda r: (min(r['n_frames_host'], r['n_frames_full']), r['n_frames_host']+r['n_frames_full'], r['primary_result'], r['source'])
        if previous is None or rank(row) > rank(previous):
            group[n] = row
    return groups


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(payload, out, mode='measured', include_flagged=False, errors=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, problems = rows_from(payload, mode)
    if not rows:
        raise ValueError('No valid finished PWidom result JSONs found')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out/'results_bundle.json').write_text(json.dumps(dict(schema=1, records=payload), indent=1))
    problems = (errors or [])+problems
    if problems:
        write_csv(out/'skipped.csv', problems, ['source', 'reason'])
    # Raw chemical-potential/transfer plots show all finished results, even if
    # none meet the stricter requirements for an integrated formation curve.
    states = {}
    for row in rows:
        if row['n'] is not None:
            key = (row['P_GPa'],row['T_K'],row['model_fp'] or 'unverified:'+row['model'])
            states.setdefault(key,[]).append(row)
    for (pressure, temperature, fingerprint), points in sorted(states.items()):
        fig, axes = plt.subplots(1,2,figsize=(11,4.8))
        runs = sorted(set(r['run'] for r in points))
        for i, run in enumerate(runs):
            series=sorted((r for r in points if r['run']==run),key=lambda r:r['n'])
            for axis,field,error in ((axes[0],'mu_H_meV','mu_H_err_meV'),(axes[1],'dg_meV','dg_err_meV')):
                labelled=False
                for row in series:
                    if row[field]=='': continue
                    axis.errorbar(row['n'],row[field],yerr=row[error],fmt='o',capsize=3,
                                  color=f'C{i%10}',mfc=f'C{i%10}' if row['flags']=='ok' else 'white',
                                  label=Path(run).name if not labelled else None)
                    labelled=True
        axes[0].set(ylabel='Hydrogen chemical potential (meV/H)')
        axes[1].set(ylabel='Transfer dG (meV/H)'); axes[1].axhline(0,color='0.5',lw=.8)
        for axis in axes: axis.set_xlabel('Extra H per O, n'); axis.grid(alpha=.2)
        if len(runs)<=12: axes[0].legend(fontsize=6)
        fig.suptitle(f'{pressure:g} GPa / {temperature:g} K — all finished results; {mode} correction\nOpen points carry quality flags; datasets are not pooled')
        fig.tight_layout()
        tag=hashlib.sha256(fingerprint.encode()).hexdigest()[:10]
        for extension in ('png','pdf'): fig.savefig(out/f'all_{pressure:g}GPa_{temperature:g}K_{tag}.{extension}',dpi=180)
        plt.close(fig)
    curves = []
    groups = selected_groups(rows, include_flagged)
    for (pressure, temperature, fingerprint), group in sorted(groups.items()):
        ordered = [group[n] for n in sorted(group)]
        if len(ordered) < 2 or not np.isclose(ordered[0]['n'], 0):
            continue
        n = np.array([r['n'] for r in ordered]); n[0] = 0
        dg = np.array([r['dg_meV'] for r in ordered])
        err = np.array([r['dg_err_meV'] for r in ordered])
        values, _, _ = integrate(n, dg)
        y = np.r_[values/(3+n), 0.]
        x = np.r_[(2+n)/(3+n), 1.]
        bound = np.r_[statistical_error_bound(n, err), 0.]
        hull = lower_hull(x, y)
        above = y-np.interp(x, x[hull], y[hull])
        for i, row in enumerate(ordered):
            row['selected_for_integral'] = True
            curves.append(dict(P_GPa=pressure, T_K=temperature, model_fp=fingerprint,
                               n=n[i], x_H=x[i], formation_meV_atom=y[i],
                               stat_sigma_upper_bound_meV_atom=bound[i], on_lower_hull=i in hull,
                               above_hull_meV_atom=max(0.,above[i]), flags=row['flags'], source=row['source']))
        label = f'{pressure:g}GPa_{temperature:g}K_{hashlib.sha256(fingerprint.encode()).hexdigest()[:10]}'
        fig, axes = plt.subplots(1, 3, figsize=(15,4.6))
        for i, row in enumerate(ordered):
            marker = 'o' if row['flags']=='ok' else 's'
            for axis, field, error in ((axes[0], 'mu_H_meV', 'mu_H_err_meV'), (axes[1], 'dg_meV', 'dg_err_meV')):
                axis.errorbar(n[i], row[field], yerr=row[error], fmt=marker, capsize=3,
                              color='C0', mfc='C0' if row['flags']=='ok' else 'white')
        axes[0].set(xlabel='Extra H per O, n', ylabel='Hydrogen chemical potential (meV/H)')
        axes[1].set(xlabel='Extra H per O, n', ylabel='Transfer dG (meV/H)')
        axes[1].axhline(0, color='0.5', lw=.8)
        axes[2].errorbar(x[:-1], y[:-1], yerr=bound[:-1], fmt='o', capsize=3, label='Integrated dG; statistical sigma bound')
        axes[2].plot(x[hull], y[hull], '--', color='C3', label='Lower envelope (provisional)')
        axes[2].plot(1,0,'D',color='0.4',label='Pure-H reference')
        axes[2].set(xlabel='Atomic H fraction', ylabel='Formation free energy (meV/atom)')
        axes[2].legend(fontsize=7)
        for axis in axes: axis.grid(alpha=.2)
        fig.suptitle(f'{pressure:g} GPa / {temperature:g} K — {mode} pressure correction\nProvisional H2O + H hull; open squares flag sampling issues')
        fig.tight_layout()
        for extension in ('png','pdf'): fig.savefig(out/f'{label}.{extension}',dpi=180)
        plt.close(fig)
    # Every finished host and reservoir remains in the table, including rejected
    # hull inputs and results from launches with unfinished sibling jobs.
    write_csv(out/'finished_results.csv', rows)
    if curves: write_csv(out/'formation_hull.csv', curves)
    with (out/'summary.md').open('w') as stream:
        stream.write('# Finished PWidom results\n\nAll mu/dG values are meV per H; errors are reported statistical sigma.\n\n')
        stream.write('| P (GPa) | T (K) | n | mu H ± sigma | transfer dG ± sigma | flags | run |\n|---:|---:|---:|---:|---:|---|---|\n')
        for row in rows:
            n='reservoir' if row['n'] is None else f"{row['n']:.6g}"
            value='—' if row['dg_meV']=='' else f"{row['dg_meV']:.1f} ± {row['dg_err_meV']:.1f}"
            stream.write(f"| {row['P_GPa']:g} | {row['T_K']:g} | {n} | {row['mu_H_meV']:.1f} ± {row['mu_H_err_meV']:.1f} | {value} | {row['flags']} | {Path(row['run']).name} |\n")
        stream.write('\nSee finished_results.csv for provenance, source launch status, fingerprint and reservoir.\n')
        stream.write(f'\n{len(rows)} finished results; {len(curves)} integrated composition points. Pressure mode: {mode}.\n')
    (out/'README.txt').write_text(
        'Read-only collection; no simulations modified. A finished result can belong to an incomplete launch.\n'
        'Archives _history/_resume_history are excluded. All live results are listed, but never averaged or pooled.\n'
        'For each P/T/verified model, the largest frame count at each composition is selected for integration.\n'
        'Equal frame counts prefer manifest-listed production results, then lexical source-path order; this is a deterministic tie-break, not scientific preference.\n'
        'Inspect selected_for_integral and source columns. Replays/parent extensions are correlated.\n'
        'Without --include-flagged, flagged results are excluded from integration; melted/unverified inputs are always excluded.\n'
        'Formation = integral_0^n transfer_dG(n) dn / (3+n), relative to H2O plus pure H.\n'
        'This trapezoidal reconstruction is provisional, not an independently calculated equilibrium H/O convex hull.\n'
        'The hull covers the H2O+H composition domain only; statistical error bounds omit model and sampling bias.\n'
        'Shared-reference correlations are not assumed independent; propagated sigma uses a maximal-correlation bound.\n'
        'Pure-H structural collapse cannot be checked without trajectories. These JSON-only flags do not certify convergence.\n'
        'Measured mode reproduces stored mu_P values. Robust mode uses pwidom_scan.mu_P partial-volume assumptions.\n'
        'No eligible n=0 plus another composition means no integrated hull for that group.\n')
    print(f'{len(rows)} finished results; {len(curves)} integrated points; {len(problems)} skipped inputs; output: {out}')
    return rows, curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs'))
    p.add_argument('--from-bundle', type=Path, help='replot exported JSON-only results without Perlmutter files')
    p.add_argument('--out', type=Path, required=True, help='fresh output directory')
    p.add_argument('--pressure-mode', choices=['measured','robust'], default='robust', help='robust (default, as pwidom_scan / the launcher) or measured (noisy in short runs)')
    p.add_argument('--include-flagged', action='store_true', help='include sampling-flagged points in explicitly provisional hulls; still exclude melted/unverified')
    a = p.parse_args()
    if a.from_bundle:
        bundle = json.loads(a.from_bundle.read_text())
        if bundle.get('schema') != 1: p.error('Unsupported bundle schema')
        payload, errors = bundle['records'], []
    else:
        payload, errors = inventory(a.root)
    report(payload,a.out,a.pressure_mode,a.include_flagged,errors)


if __name__ == '__main__':
    main()
