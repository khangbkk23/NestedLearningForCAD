"""Aggregate the NANM-NL-02 forensics phases and apply the frozen Task 4 decision rules (protocol §Task 4)."""
from __future__ import annotations

import json
import math
import statistics as stt
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/nanm_nl_02_forensics'
OBJS = ('A', 'B', 'C', 'Cp')


def ld(ph):
    p = OUT / f'{ph}.json'
    return json.loads(p.read_text()) if p.exists() else None


def G(vals_by_key):
    """vals_by_key: {'n|s|t': value} -> mean over cells of mean over stages (G0a aggregation)."""
    cells = defaultdict(list)
    for k, v in vals_by_key.items():
        n, s, t = k.split('|'); cells[(n, s)].append(v)
    assert all(len(v) == 4 for v in cells.values()), 'incomplete cell'
    return sum(sum(v) / 4 for v in cells.values()) / len(cells)


def by_task(vals_by_key, task_of):
    d = defaultdict(list)
    for k, v in vals_by_key.items():
        d[task_of[k]].append(v)
    return {t: sum(v) / len(v) for t, v in sorted(d.items())}


def main():
    t1, t2z, t2s, t3, al = ld('TASK1'), ld('TASK2Z'), ld('TASK2S'), ld('TASK3'), ld('ALIGN')
    S = {}
    meta = next(x for x in (t1, t2z, t2s, t3, al) if x)['meta']
    gate = meta['gate']; S['gate'] = gate; S['U'] = meta['U']
    task_of = {k: v[2] for k, v in meta['c0_c0q'].items()}
    S['c0q_minus_c0_G'] = G({k: v[1] - v[0] for k, v in meta['c0_c0q'].items()})
    S['meta'] = {ph: {k: x['meta'][k] for k in ('protocol_sha256', 'code_sha256', 'prereg_lock_unchanged', 'wall_s', 'peak_rss_GB',
                                                 'opened_files', 'selected_accessed')}
                 for ph, x in (('task1', t1), ('task2z', t2z), ('task2s', t2s), ('task3', t3), ('align', al)) if x}
    S['storage'] = {ph: (x['meta']['storage_before'], x['meta']['storage_after'])
                    for ph, x in (('task1', t1), ('task2z', t2z), ('task2s', t2s), ('task3', t3), ('align', al)) if x}
    if t1:
        c = t1['result']['cells']
        m = lambda f: {k: sum(r[f] for r in v) / len(v) for k, v in c.items()}
        S['task1'] = {'G_cap_crossfit': G(m('gain_crossfit')), 'G_cap_vs_ref': G(m('gain_crossfit_vs_ref')),
                      'G_insample': G(m('gain_insample')), 'G_fullrank_oracle_crossfit': G(m('fullrank_oracle_gain_crossfit')),
                      'per_stage': {t: stt.mean(v for k, v in m('gain_crossfit').items() if k.endswith(f'|{t}')) for t in range(4)},
                      'per_task': by_task(m('gain_crossfit'), task_of),
                      'conv_max_abs_last10pct': max(abs(r['last10pct_change']) for v in c.values() for r in v),
                      'conv_max_final_grad': max(r['final_grad_norm'] for v in c.values() for r in v),
                      'max_cond_A': max(r['cond_A'] for v in c.values() for r in v),
                      'min_det_sign': min(r['det_sign'] for v in c.values() for r in v)}
        rob = t1['result']['robustness_a2711']
        S['task1']['init_seed_spread'] = {t: [r['gain_crossfit'] for r in v['seeds']] for t, v in rob.items()}
        S['task1']['long1600_minus_800'] = {t: [l['gain_crossfit'] - c[f'a|2711|{t}'][i]['gain_crossfit'] for i, l in enumerate(v['long1600'])]
                                            for t, v in rob.items()}
    if t2z:
        r = t2z['result']
        S['task2z'] = {o: {'G': G({k: v[o]['gain'] for k, v in r.items()}), 'G_vs_ref': G({k: v[o]['gain_vs_ref'] for k, v in r.items()}),
                           'per_task': by_task({k: v[o]['gain'] for k, v in r.items()}, task_of),
                           'cells_positive': sum(v[o]['gain'] > 0 for v in r.values()),
                           'max_abs_last10pct': max(abs(v[o]['last10pct_change']) for v in r.values()),
                           'max_cond_A': max(v[o]['cond_A'] for v in r.values()),
                           'mean_A_minus_I': stt.mean(v[o]['A_minus_I_F'] for v in r.values())} for o in OBJS}
        S['task2z']['fullrank_A_opt'] = G({k: v['fullrank_A_opt_gain'] for k, v in r.items()})
        S['task2z']['fullrank_B_opt'] = G({k: v['fullrank_B_opt_gain'] for k, v in r.items()})
    if t2s:
        r = t2s['result']; etas = list(next(iter(r.values()))['A'].keys())
        S['task2s'] = {o: {e: {'G': G({k: (v[o][e]['gain'] if v[o][e]['finite'] else float('nan')) for k, v in r.items()}),
                               'n_finite': sum(v[o][e]['finite'] for v in r.values()),
                               'rejected': sum(v[o][e]['rejected'] for v in r.values())} for e in etas} for o in OBJS}
    if t3:
        r = t3['result']; etas = ('0.0001', '0.001', '0.003')
        S['task3'] = {}
        for e in etas:
            acq = {f'{c}|{t}': st['acq_gain'] for c, v in r.items() if isinstance(v, dict) and e in v for t, st in enumerate(v[e])}
            stages = [st for c, v in r.items() if isinstance(v, dict) and e in v for st in v[e]]
            chk = [ch for st in stages for ch in st['checks']]
            S['task3'][e] = {'G_acq': G(acq), 'per_task': by_task(acq, {f'{c}|{t}': st['task'] for c, v in r.items()
                                                                       if isinstance(v, dict) and e in v for t, st in enumerate(v[e])}),
                             'cos_big_hold_mean': stt.mean(ch['cos_big_hold'] for ch in chk),
                             'cos_big_hold_at_step0': stt.mean(st['checks'][0]['cos_big_hold'] for st in stages),
                             'cos_mb_hold_mean': stt.mean(ch['cos_mb_hold'] for ch in chk),
                             'cos_mb_big_mean': stt.mean(ch['cos_mb_big'] for ch in chk),
                             'loss_first20_mean': stt.mean(stt.mean(st['loss'][:20]) for st in stages),
                             'loss_last20_mean': stt.mean(stt.mean(st['loss'][-20:]) for st in stages),
                             'max_A_minus_I': max(st['final']['A_minus_I_F'] for st in stages),
                             'max_cond_A': max(st['final']['cond_A'] for st in stages),
                             'rejected': sum(st['rejected'] for st in stages), 'halved': sum(st['halved'] for st in stages),
                             'all_finite': all(st['final']['finite'] for st in stages)}
        S['task3']['repro_bitwise'] = {k: v for c, x in r.items() if isinstance(x, dict) for k, v in x.items() if k.endswith('repro_bitwise')}
    if al:
        r = al['result']
        S['align'] = {o: {'cos_hold_mean': stt.mean(v[o]['cos_hold'] for v in r.values()),
                          'cos_hold_min': min(v[o]['cos_hold'] for v in r.values()),
                          'cos_hold_max': max(v[o]['cos_hold'] for v in r.values())} for o in OBJS}
        S['align']['A_below_X_count'] = {o: sum(v['A']['cos_hold'] < v[o]['cos_hold'] for v in r.values()) for o in ('B', 'C', 'Cp')}
        S['align']['analytic_A_relerr_max'] = max(v['analytic_A_relerr'] for v in r.values())
        S['align']['per_task_cos'] = {o: by_task({k: v[o]['cos_hold'] for k, v in r.items()}, task_of) for o in OBJS}
    t1v2 = ld('TASK1_V2')
    if t1v2:
        c = t1v2['result']['cells']; rob = t1v2['result']['robustness_a2711']
        rin = {k: v['gain_in'] for k, v in c.items()}
        S['task1_v2'] = {'G_R_in': G(rin), 'per_stage': {t: stt.mean(v for k, v in rin.items() if k.endswith(f'|{t}')) for t in range(4)},
                         'per_task': by_task(rin, task_of), 'cells_positive': sum(v > 0 for v in rin.values()),
                         'max_abs_last10pct': max(abs(v['last10pct_gain_change']) for v in c.values()),
                         'max_cond_A': max(v['cond_A'] for v in c.values()), 'min_det_sign': min(v['det_sign'] for v in c.values()),
                         'seed_spread_a2711': {t: [c[f'a|2711|{t}']['gain_in']] + v['seeds'] for t, v in rob.items()},
                         'crossfit_oracle_stop_a2711': {t: [(f['heldout_best'], f['heldout_best_step'], f['heldout_final']) for f in v['crossfit_oracle_stop']]
                                                        for t, v in rob.items()},
                         'meta': {k: t1v2['meta'][k] for k in ('protocol_sha256', 'code_sha256', 'prereg_lock_unchanged', 'wall_s', 'peak_rss_GB', 'opened_files')}}
        S['storage']['task1_v2'] = (t1v2['meta']['storage_before'], t1v2['meta']['storage_after'])
    # ---------------- decision rules (Amendment 1: G_cap = max(G(R_in), max_X G(X)))
    if t1v2 and t2z and t3 and al:
        Gc = max(S['task1_v2']['G_R_in'], max(S['task2z'][o]['G'] for o in OBJS)); f = {o: S['task2z'][o]['G'] / Gc for o in OBJS}
        S['G_cap'] = Gc; S['f'] = f
        S['sensitivity'] = {'G_cap_for_f_A_eq_0.5': 2 * S['task2z']['A']['G'], 'G_cap_for_f_A_eq_0.25': 4 * S['task2z']['A']['G']}
        sel_c1 = S['task3']['0.003']['G_acq']
        if Gc < gate:
            v = 'NL02_CAPACITY_LIMITED'
        elif f['A'] >= 0.5 and sel_c1 < 0.25 * Gc:
            v = 'NL02_OPTIMIZATION_LIMITED'
        elif f['A'] < 0.25 and any(f[x] >= 0.5 and S['align']['A_below_X_count'][x] >= 27 for x in ('B', 'C', 'Cp')):
            v = 'NL02_OBJECTIVE_MISMATCH_SUPPORTED'
        else:
            v = 'NL02_INCONCLUSIVE'
        S['verdict'] = v; S['objective_not_to_be_modified'] = f['A'] >= 0.8
    (OUT / 'SUMMARY.json').write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps({k: S[k] for k in S if k not in ('meta', 'storage')}, indent=1, default=str))


if __name__ == '__main__':
    main()
