"""NL03-MEMORY-PRESSURE-00 summary: registered tables, gates and verdict (registration §5-§8). Pure post-processing of
METRICS_*.json, boot npz, cells/*.json, cover/*.json, BUDGET_LEDGER.jsonl. Writes RESULTS.json and VERDICT.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exps.nanm_eval_01 import common as U
from exps.nl03_pressure_00 import core as C

OUT = C.OUT
G02, G03, SEC = 0.020, 0.010, 0.005
CONTRASTS = (('U', 'E'), ('Q', 'E'), ('U', 'Q'), ('E', 'E4'), ('E', 'F'), ('E', 'G'), ('E', 'FIFO'))


class Res:
    def __init__(self, stream):
        self.stream = stream
        self.res, self.boot = {}, {}
        for tag in ('t1', 't2'):
            p = OUT / f'METRICS_{stream}_{tag}.json'
            if p.exists():
                d = U.rj(p)
                for t, jj in d['cells'].items():
                    self.res.setdefault(t, {}).update(jj)
                z = np.load(d['boot_file'])
                self.boot.update({k: z[k] for k in z.files})

    def has(self, jid, order):
        return all(jid in self.res.get(t, {}) for t in order)

    def v(self, jid, t, l, met='ap'):
        return self.res[t][jid][str(l)][met]


def summarize(r, jid, order, stages, met='ap'):
    T = len(order); last = T - 1
    a = lambda t, l: r.v(jid, t, l, met)
    acq = [a(t, min(l for l in stages if l >= j)) for j, t in enumerate(order)]
    f = []
    for j, t in enumerate(order[:-1]):
        L = [l for l in stages if j <= l <= T - 2]
        if L:
            f.append(max(a(t, l) for l in L) - a(t, last))
    traj = [float(np.mean([a(t, l) for t in order[:l + 1]])) for l in stages]
    return {'final_all': float(np.mean([a(t, last) for t in order])),
            'final_old': float(np.mean([a(t, last) for t in order[:-1]])),
            'acq_macro': float(np.mean(acq)), 'fm': float(np.mean(f)) if f else None, 'fm_n_tasks': len(f),
            'prequential_mean': float(np.mean(traj)), 'prequential_traj': traj,
            'per_task_final': {t: a(t, last) for t in order}, 'oldest_task_retention': [a(order[0], l) for l in stages]}


def rep(r, jid, tasks, met):
    keys = [f'{t}||{jid}||{met}' for t in tasks]
    if not all(k in r.boot for k in keys):
        return None
    return np.nanmean(np.stack([r.boot[k] for k in keys]), 0)


def contrast(r, x, y, order, met='ap', bmet='ap'):
    T = len(order); old = order[:-1]
    px = float(np.mean([r.v(x, t, T - 1, met) for t in old])); py = float(np.mean([r.v(y, t, T - 1, met) for t in old]))
    out = {'x': x, 'y': y, 'delta': px - py, 'x_value': px, 'y_value': py}
    rx, ry = rep(r, x, old, bmet), rep(r, y, old, bmet)
    if rx is not None and ry is not None:
        d = rx - ry
        out['ci95'] = [float(np.nanquantile(d, .025)), float(np.nanquantile(d, .975))]
    pt = {t: r.v(x, t, T - 1, met) - r.v(y, t, T - 1, met) for t in old}
    out['worst_old_task'] = min(pt.items(), key=lambda kv: kv[1])
    out['n_old_tasks_positive'] = sum(v > 0 for v in pt.values())
    return out


def name(arm, b):
    return 'U' if arm == 'U' else f'{arm}@{b}'


def tier1(stream):
    r = Res(stream)
    order = C.orders(stream)['a']; T = len(order); stages = C.logged_stages(stream, T)
    cell = 'a|2711'
    meta = U.rj(OUT / 'cells' / f'{stream}_a_2711.json')
    cover = U.rj(OUT / 'cover' / f'{stream}_a_2711.json')
    tab, con, cov, ev = {}, {}, {}, {}
    for b in C.BUDGETS_MIB:
        tab[b], con[b], cov[b], ev[b] = {}, {}, {}, {}
        for arm in C.ARMS:
            n = name(arm, b); jid = f'{cell}|{n}'
            if not r.has(jid, order):
                tab[b][arm] = {'status': 'MISSING'}; continue
            tab[b][arm] = {'ap': summarize(r, jid, order, stages, 'ap'), 'auroc': summarize(r, jid, order, stages, 'i_auroc'),
                           'n_keys_final': meta['n_keys_final'][n], 'bytes_final': meta['bytes_final'][n],
                           'budget_bytes': None if arm == 'U' else b * C.MiB}
            tab[b][arm]['auroc'].pop('per_task_final', None)
            # coverage at the final stage, old tasks, vs U
            med = {t: cover[f'{n}|{t}|{T - 1}']['median'] for t in order[:-1]}
            p95 = {t: cover[f'{n}|{t}|{T - 1}']['p95'] for t in order[:-1]}
            umed = {t: cover[f'U|{t}|{T - 1}']['median'] for t in order[:-1]}
            up95 = {t: cover[f'U|{t}|{T - 1}']['p95'] for t in order[:-1]}
            cov[b][arm] = {'median': med, 'p95': p95, 'median_ratio_to_U': {t: med[t] / umed[t] for t in med},
                           'p95_ratio_to_U': {t: p95[t] / up95[t] for t in p95},
                           'macro_median': float(np.mean(list(med.values()))), 'macro_p95': float(np.mean(list(p95.values()))),
                           'macro_median_ratio_to_U': float(np.mean([med[t] / umed[t] for t in med])),
                           'oldest_task_median_by_stage': {l: cover[f'{n}|{order[0]}|{l}']['median'] for l in stages}}
            e = meta['evict'][n]
            ev[b][arm] = {'evicted_total': int(sum(x['n_evicted'] for x in e)) if arm not in ('U', 'Q') else None,
                          'dropped_final_vs_union': e[-1]['n_evicted'] if arm == 'Q' else None,
                          'survivors_final_per_task': dict(zip(order, e[-1]['survivors_per_task'])),
                          'survivor_trajectory_task0': [x['survivors_per_task'][0] for x in e],
                          'n_keys_by_stage': [x['n_keys'] for x in e], 'bytes_by_stage': [x['bytes'] for x in e],
                          'evicted_age_mean_by_stage': [x['evicted_age_mean'] for x in e],
                          'invariants_ok': all(x['invariant_ok'] for x in e)}
        for x, y in CONTRASTS:
            jx, jy = f'{cell}|{name(x, b)}', f'{cell}|{name(y, b)}'
            if r.has(jx, order) and r.has(jy, order):
                con[b][f'{x}-{y}'] = {'ap': contrast(r, jx, jy, order, 'ap', 'ap'),
                                      'auroc': contrast(r, jx, jy, order, 'i_auroc', 'auroc')}
    m3 = None
    if stream == 'visa':
        diffs = {}
        for n in ('U', 'Q@8', 'E@8', 'E4@8'):
            jid = f'{cell}|{n}'
            for t in order:
                c = r.res[t][jid][str(T - 1)]
                if 'ap_exact' in c:
                    diffs[f'{n}|{t}'] = abs(c['ap'] - c['ap_exact'])
        m3 = {'max_abs': max(diffs.values()) if diffs else None, 'n': len(diffs),
              'pass': bool(diffs) and max(diffs.values()) <= 1e-3}
    return {'order': order, 'stages': stages, 'table': tab, 'contrasts': con, 'coverage': cov, 'eviction': ev,
            'M2_parity_eval01_U': meta.get('parity_eval01_U'), 'M3_binned_vs_exact': m3, 'cond': meta['cond']}


def gates(t1):
    c8 = t1['contrasts'][C.PRIMARY_MIB]
    e = t1['eviction'][C.PRIMARY_MIB]['E']
    g1 = {'evicted_total': e['evicted_total'], 'stages_logged': len(e['n_keys_by_stage']),
          'pass': bool(e['evicted_total'] and e['evicted_total'] > 0 and len(e['n_keys_by_stage']) == len(t1['order'])
                       and e['invariants_ok'])}
    ue, qe, ee4 = c8['U-E']['ap'], c8['Q-E']['ap'], c8['E-E4']['ap']
    g2 = {'delta': ue['delta'], 'ci95': ue.get('ci95'), 'pass': ue['delta'] >= G02 and ue['ci95'][0] > 0}
    g3 = {'delta': qe['delta'], 'ci95': qe.get('ci95'), 'pass': qe['delta'] >= G03 and qe['ci95'][0] > 0}
    if g1['pass'] and g2['pass'] and g3['pass']:
        v = 'PRESSURE_CONFIRMED'
    elif g1['pass'] and g2['pass']:
        v = 'COMPRESSION_ONLY'
    else:
        v = 'NO_FAILURE_MODE'
    sec = {'delta': ee4['delta'], 'ci95': ee4.get('ci95'),
           'label': ('SLOW_ACCUMULATION_SIGNAL_UNDER_PRESSURE' if ee4['delta'] >= SEC and ee4['ci95'][0] > 0
                     else 'NO_SLOW_ACCUMULATION_SIGNAL')}
    return {'G0-1': g1, 'G0-2': g2, 'G0-3': g3, 'verdict': v, 'secondary': sec}


def eval01_U(cell, order):
    """VisA tier-2 U reference: EVAL-01 measured N0 metrics of the same (order, seed) cell (same canonical grids)."""
    d = U.rj(U.OUT / 'METRICS_visa_t1.json')['cells']
    T = len(order)
    try:
        return float(np.mean([d[t.split('/')[1]][f'{cell}|N0'][str(T - 1)]['ap_native_binned'] for t in order[:-1]]))
    except KeyError:
        return None


def tier2(stream):
    r = Res(stream)
    T = len(C.orders(stream)['a'])
    rows = {}
    for o in ('a', 'b', 'c'):
        order = C.orders(stream)[o]
        for s in C.SEEDS:
            cell = f'{o}|{s}'
            vals = {}
            for arm in C.T2_ARMS:
                jid = f'{cell}|{arm}@{C.PRIMARY_MIB}'
                if r.has(jid, order):
                    vals[arm] = float(np.mean([r.v(jid, t, T - 1) for t in order[:-1]]))
            if stream == 'visa':
                if cell == 'a|2711' and r.has(f'{cell}|U', order):
                    vals['U'] = float(np.mean([r.v(f'{cell}|U', t, T - 1) for t in order[:-1]]))
                else:
                    vals['U_eval01'] = eval01_U(cell, order)
            if vals:
                rows[cell] = vals
    spread = {}
    for x, y in (('Q', 'E'), ('E', 'E4'), ('E', 'F'), ('U', 'E')):
        ds = []
        for cell, v in rows.items():
            xv = v.get(x, v.get('U_eval01') if x == 'U' else None)
            if xv is not None and v.get(y) is not None:
                ds.append(xv - v[y])
        if ds:
            spread[f'{x}-{y}'] = {'mean': float(np.mean(ds)), 'min': float(min(ds)), 'max': float(max(ds)),
                                  'n_positive': int(sum(d > 0 for d in ds)), 'n': len(ds)}
    e_vals = [v['E'] for v in rows.values() if 'E' in v]
    order_spread = {}
    for s in C.SEEDS:
        ev = [rows[f'{o}|{s}']['E'] for o in 'abc' if f'{o}|{s}' in rows and 'E' in rows[f'{o}|{s}']]
        if ev:
            order_spread[s] = {'E_old_final_by_order': ev, 'range': float(max(ev) - min(ev))}
    return {'cells': rows, 'spread': spread, 'E_order_spread_by_seed': order_spread,
            'E_range_all_cells': float(max(e_vals) - min(e_vals)) if e_vals else None}


def budget():
    p = OUT / 'BUDGET_LEDGER.jsonl'
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return {'gpu_wall_s': sum(r['seconds'] for r in rows if r['gpu']), 'cpu_wall_s': sum(r['seconds'] for r in rows if not r['gpu']),
            'by_phase': rows, 'ceiling_gpu_s': 6 * 3600, 'budget_stop': (OUT / 'BUDGET_STOP.json').exists()}


def main():
    res = {}
    for st in C.STREAMS:
        if (OUT / f'METRICS_{st}_t1.json').exists():
            res[st] = {'tier1': tier1(st)}
            res[st]['gates_quantities'] = gates(res[st]['tier1'])
            if (OUT / f'METRICS_{st}_t2.json').exists():
                res[st]['tier2'] = tier2(st)
    if (OUT / 'METRICS_visa_t1.json').exists():
        res['visa']['tier2'] = res['visa'].get('tier2') or tier2('visa')
    bud = budget()
    U.wj(OUT / 'RESULTS.json', {**res, 'budget': bud})
    g = res['visa']['gates_quantities']
    t1 = res['visa']['tier1']
    mech = {'M2_U_vs_EVAL01_N0_bitwise': (all(t1['M2_parity_eval01_U'].values()) if t1['M2_parity_eval01_U'] else None),
            'M3_binned_vs_exact': t1['M3_binned_vs_exact'],
            'M4_budget_assertions': 'never fired (a violation raises BudgetViolation and aborts the cell)',
            'eviction_invariants': all(e['invariants_ok'] for b in t1['eviction'].values() for e in b.values())}
    verdict = {'verdict': g['verdict'], 'gates': {k: g[k] for k in ('G0-1', 'G0-2', 'G0-3')},
               'secondary': g['secondary'], 'primary_cell': 'VisA a|2711 8 MiB, old tasks 1..T-1, macro final native Pixel AP',
               's33_descriptive': res.get('s33', {}).get('gates_quantities'), 'mechanical': mech, 'budget': {
                   k: v for k, v in bud.items() if k != 'by_phase'},
               'exposure': 'EXPOSED-DEV; diagnostic only; no promotion'}
    U.wj(OUT / 'VERDICT.json', verdict)
    U.log('VERDICT', g['verdict'], g['secondary']['label'])
