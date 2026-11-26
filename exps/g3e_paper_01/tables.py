"""G3E-PAPER-01 tables (read-only over cell artifacts): RESULTS.json + markdown snippets.

Sources: NL03-MULTIRATE-01 §17 benchmark cells (U/E 8 MiB seed 2711, hash-recorded in REUSE.json) and the new
G3E-PAPER-01 cells. Paired CIs use the registered bootstrap weights (depend only on the category's image count and its
position in the order), so replicates of different cells of the same order are paired.

    python -B -m exps.g3e_paper_01.tables
"""
from __future__ import annotations

import numpy as np

from exps.g3e_paper_01 import run as R
from exps.nl03_multirate_01 import final as FN

MR = FN.BENCH_OUT


def _src(ds, ok, seed, arm):
    """(score json, metrics json, arm key in that cell) for (ds, order, seed, arm) or None."""
    cands = []
    if seed == 2711 and arm in ('U', 'E@8') and (MR / f'SCORE_{ds}_{ok}2711.json').exists():
        cands.append((MR / f'SCORE_{ds}_{ok}2711.json', MR / f'METRICS_{ds}_{ok}2711.json', 'E' if arm == 'E@8' else 'U'))
    for p in sorted((R.OUT / 'cells').glob(f'SCORE_*_dinov2_{ds}_{ok}{seed}.json')):
        if p.name.startswith('SCORE_T1_'):
            continue
        sc = R.rj(p)
        if arm in sc['arms']:
            cands.append((p, p.with_name(p.name.replace('SCORE_', 'METRICS_')), arm))
    for s, m, a in cands:
        if m.exists():
            return s, m, a
    return None


def row(ds, ok, seed, arm, src=None):
    src = src or _src(ds, ok, seed, arm)
    if src is None:
        return None
    sp, mp_, a = src
    sc, me = R.rj(sp), R.rj(mp_)
    order, T = sc['order'], len(sc['order'])
    ap = {t: {int(l): v['ap'] for l, v in me['cells'][t][a].items()} for t in order}
    au = {t: {int(l): v['i_auroc'] for l, v in me['cells'][t][a].items()} for t in order}
    out = {'pixel_ap': float(np.mean([ap[t][T - 1] for t in order])),
           'image_auroc': float(np.mean([au[t][T - 1] for t in order])),
           'fm': float(np.mean([max(ap[t][l] for l in range(j, T - 1)) - ap[t][T - 1] for j, t in enumerate(order[:-1])])),
           'prequential_ap': float(np.mean([np.mean([ap[order[j]][l] for j in range(l + 1)]) for l in range(T)])),
           'acq_ap': float(np.mean([ap[t][j] for j, t in enumerate(order)])),
           'bytes_max': max(sc['bytes'][a]), 'update_s': sc['update_s'][a],
           'latency_ms_per_img': sc['latency_ms_per_img_p50'][a],
           'per_class': {t: {'final_ap': ap[t][T - 1], 'final_auroc': au[t][T - 1], 'acq_ap': ap[t][j],
                             'fm_ap': (max(ap[t][l] for l in range(j, T - 1)) - ap[t][T - 1]) if j < T - 1 else None}
                         for j, t in enumerate(order)},
           'source': str(sp.relative_to(R.ROOT)), 'arm_key': a}
    bf = np.load(me['boot_file']) if 'boot_file' in me else None
    if bf is not None and f'{order[0]}||{a}||ap' in bf.files:      # MULTIRATE orders b/c carry an empty boot npz
        out['_boot'] = {m: np.mean([bf[f'{t}||{a}||{m}'] for t in order], axis=0) for m in ('ap', 'auroc')}
        out['pixel_ap_ci95'] = [float(x) for x in np.nanpercentile(out['_boot']['ap'], [2.5, 97.5])]
        out['image_auroc_ci95'] = [float(x) for x in np.nanpercentile(out['_boot']['auroc'], [2.5, 97.5])]
    return out


def paired(x, y, key='pixel_ap', m='ap'):
    if x is None or y is None or '_boot' not in x or '_boot' not in y:
        return None
    d = x['_boot'][m] - y['_boot'][m]
    return {'delta': x[key] - y[key], 'ci95': [float(v) for v in np.nanpercentile(d, [2.5, 97.5])]}


def ms(vals):
    v = [x for x in vals if x is not None]
    return {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v)} if v else None


METRICS = ('pixel_ap', 'image_auroc', 'fm', 'acq_ap', 'prequential_ap', 'bytes_max', 'update_s', 'latency_ms_per_img')


def strip(r):
    return None if r is None else {k: v for k, v in r.items() if k != '_boot'}


def main():
    res = {'A1': {}, 'A2': {}, 'A3': {}, 'A4': {}, 'A5': {}, 'B': {}, 'order_a_seed2711': {}}
    for ds in R.DSETS:
        # A1 / primary order-a rows
        res['order_a_seed2711'][ds] = {a: strip(row(ds, 'a', 2711, a)) for a in ('U', 'E@8')}
        res['A1'][ds] = {a: strip(row(ds, 'c', 2711, a)) for a in ('U', 'E@8')} if ds == 'mpdd' else None
        # A2: mean +- std over orders x seeds
        cells = {a: {f'{ok}|{s}': row(ds, ok, s, a) for ok in 'abc' for s in R.SEEDS} for a in ('U', 'E@8')}
        res['A2'][ds] = {a: {'n_cells': sum(v is not None for v in cells[a].values()),
                             'cells_present': sorted(k for k, v in cells[a].items() if v is not None),
                             **{m: ms([v[m] for v in cells[a].values() if v is not None]) for m in METRICS}}
                         for a in cells}
        res['A2'][ds]['per_cell'] = {a: {k: (None if v is None else {m: v[m] for m in METRICS}) for k, v in cells[a].items()}
                                     for a in cells}
        # A3: budget curve, order a, seeds
        curve = {}
        for a in ('E@4', 'E@8', 'E@16', 'E@32', 'U'):
            rs = {s: row(ds, 'a', s, a) for s in R.SEEDS}
            curve[a] = {'n_seeds': sum(v is not None for v in rs.values()),
                        **{m: ms([v[m] for v in rs.values() if v is not None]) for m in METRICS},
                        'per_seed': {s: (None if v is None else {m: v[m] for m in METRICS}) for s, v in rs.items()}}
        res['A3'][ds] = curve
        # A4: ablations vs E (order a, seed 2711), paired CIs E - X
        e = row(ds, 'a', 2711, 'E@8')
        a4 = {'E': strip(e)}
        for x in R.A4_ARMS:
            rx = row(ds, 'a', 2711, x)
            a4[x] = strip(rx)
            if rx is not None:
                a4[x]['E_minus_ap'] = paired(e, rx, 'pixel_ap', 'ap')
                a4[x]['E_minus_auroc'] = paired(e, rx, 'image_auroc', 'auroc')
        u = row(ds, 'a', 2711, 'U')
        a4['U'] = strip(u)
        a4['U_minus_E_ap'] = paired(u, e, 'pixel_ap', 'ap')
        res['A4'][ds] = a4
        # A5: per-class
        res['A5'][ds] = {a: (None if r is None else r['per_class']) for a, r in (('U', u), ('E@8', e))}
    for ds in ('mvtec', 'visa'):
        p = R.OUT / 'cells' / f'SCORE_B_vitb8_{ds}_a2711.json'
        if p.exists() and p.with_name(p.name.replace('SCORE_', 'METRICS_')).exists():
            src = lambda a: (p, p.with_name(p.name.replace('SCORE_', 'METRICS_')), a)
            rr = {a: row(ds, 'a', 2711, a, src(a)) for a in ('U', 'E@8', 'F')}
            res['B'][ds] = {a: strip(v) for a, v in rr.items()}
            res['B'][ds]['E_minus_F_ap'] = paired(rr['E@8'], rr['F'])
            res['B'][ds]['U_minus_E_ap'] = paired(rr['U'], rr['E@8'])
            res['B'][ds]['label'] = 'MATCHED_BACKBONE_ABLATION'
        else:
            res['B'][ds] = 'NOT_RUN'
    import time
    res['plan_status'] = R.rj(R.OUT / 'PLAN_STATUS.json') if (R.OUT / 'PLAN_STATUS.json').exists() else None
    res['time'] = time.strftime('%FT%T')
    R.wj(R.OUT / 'RESULTS.json', res)
    R.log('TABLES written')


if __name__ == '__main__':
    main()
