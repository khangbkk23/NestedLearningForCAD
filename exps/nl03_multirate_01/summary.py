"""NL03-MULTIRATE-01 Phase K: development gates M0-M4 and verdict (registration §7). CPU (+ one GPU exact-AP check)."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from exps.nl03_multirate_01 import evaluate as EV
from exps.nl03_multirate_01 import run as RN
from exps.nl03_multirate_01 import spec as SP

OUT = SP.OUT


def load_cell(tag):
    d = RN.rj(OUT / 'cells' / f'{tag}.json')
    arr = torch.load(d['array_record'], weights_only=False)
    assert RN.SY._fsha(d['array_record']) == d['array_sha256'], f'array record changed {tag}'
    return d, arr


def task_index(stream, task):
    return RN.orders(stream)['a'].index(task)


def arm_reps(d, arr, arm, which):
    """Per-task bootstrap replicates (auroc, ap, fpr) and points for old tasks at the final stage ('old') or for every
    task at its acquisition stage ('cur')."""
    T = len(d['tasks'])
    rows = []
    js = range(T - 1) if which == 'old' else range(T)
    for j in js:
        l = T - 1 if which == 'old' else j
        rec = arr['res'][arm][l]['tasks'][j]
        theta = arr['theta'][arm][j]
        w = EV.boot_w(len(rec['s_syn']), task_index(d['stream'], d['tasks'][j]))
        b = EV.boot_metrics(rec, w, theta)
        fpr_pt = float((rec['s_cln'] > theta).mean())
        rows.append({'task': d['tasks'][j], 'pt': {'auroc': rec['auroc'], 'ap': rec['ap'], 'ap_hb': rec['ap_hb'],
                                                   'fpr': fpr_pt}, 'b': b})
    return rows


def macro(rows, m, pt=True):
    if pt:
        return float(np.mean([r['pt'][m] for r in rows]))
    return np.mean([r['b'][m] for r in rows], axis=0)


def contrast(rx, ry, m, sign=1):
    """x - y (sign=1) on macro metric m; CI from paired replicates (same weights per task)."""
    d = sign * (macro(rx, m) - macro(ry, m))
    rep = sign * (macro(rx, m, False) - macro(ry, m, False))
    lo, hi = np.nanpercentile(rep, [2.5, 97.5])
    return {'delta': d, 'ci95': [float(lo), float(hi)], 'n_tasks_pos': int(sum(sign * (a['pt'][m] - b['pt'][m]) > 0
                                                                               for a, b in zip(rx, ry)))}


def m0(d, arr):
    """Mechanical: fast writes every minibatch, slow writes every C, provenance, budget, no future, sources."""
    from exps.nl03_multirate_01 import plan as PL
    out = {}
    steps = d['fast_steps']
    out['fast_writes_every_step'] = all(v == steps for v in d['fast_writes'].values())
    sched, prov = {}, {}
    for a, lg in arr['slow_log'].items():
        C = SP.ARMS[a]['C']
        if SP.ARMS[a]['slow'] == 'ema':
            ws = [r['step'] for r in lg]
            sched[a] = (ws == [s for s in range(steps) if (s + 1) % C == 0], len(ws), steps / max(1, len(ws)))
            prov[a] = all(r['in_steps'][1] == r['step'] and r['n_in_steps'] == C for r in lg)
        elif SP.ARMS[a]['slow'] == 'pooled':
            sched[a] = (len(lg) == steps, len(lg), 1.0)
    out['slow_schedule'] = {a: {'ok': v[0], 'writes': v[1], 'empirical_ratio_fast_per_slow': v[2]} for a, v in sched.items()}
    out['slow_input_from_window_only'] = prov
    ages = [e['age'] for lg in arr['evict_log'].values() for e in lg]
    out['evictions_causal'] = bool(all((a >= 0).all() for a in ages)) if ages else True
    caus = arr['causal']
    out['no_future'] = [s for s, l, b in caus] == list(range(len(caus))) and \
        [l for s, l, b in caus] == sorted(l for s, l, b in caus)
    out['budget'] = {a: (b, SP.BUDGETS[d['mib']], b <= SP.BUDGETS[d['mib']]) for a, b in d['max_bytes'].items()}
    out['two_context_flows'] = 'fast: minibatch tokens; slow: window residual statistics (+ evicted keys for P/C2)'
    try:
        PL.verify()
        out['locks_and_sources'] = True
    except AssertionError as e:
        out['locks_and_sources'] = str(e)
    out['pass'] = bool(out['fast_writes_every_step'] and all(v['ok'] for v in out['slow_schedule'].values())
                       and all(prov.values()) and out['evictions_causal'] and out['no_future']
                       and all(v[2] for v in out['budget'].values()) and out['locks_and_sources'] is True)
    return out


def gates(tag_t1, t2_tags):
    from exps.nl03_multirate_01 import plan as PL
    d, arr = load_cell(tag_t1)
    cb = PL.select_cb()
    strongest = cb['strongest_matched']
    arms = [a for a in d['arms']]
    old = {a: arm_reps(d, arr, a, 'old') for a in arms}
    cur = {a: arm_reps(d, arr, a, 'cur') for a in arms}
    table = {a: {'old': {m: macro(old[a], m) for m in ('auroc', 'ap', 'fpr')},
                 'cur': {m: macro(cur[a], m) for m in ('auroc', 'ap', 'fpr')}} for a in arms}
    M0 = m0(d, arr)
    M1 = {'P-C2': contrast(old['P'], old['C2'], 'auroc'), 'P-C3': contrast(old['P'], old['C3'], 'auroc')}
    M1['pass'] = all(not (v['ci95'][0] <= 0 <= v['ci95'][1]) for v in M1.values() if isinstance(v, dict))

    def m2_of(x):
        r = {'control': strongest,
             'auroc': contrast(old[x], old[strongest], 'auroc'), 'ap': contrast(old[x], old[strongest], 'ap'),
             'fpr_improvement': contrast(old[x], old[strongest], 'fpr', sign=-1),
             'cur_auroc_drop': table[strongest]['cur']['auroc'] - table[x]['cur']['auroc'],
             'cur_fpr_rise': table[x]['cur']['fpr'] - table[strongest]['cur']['fpr']}
        g = SP.M2
        r['pass'] = bool(r['auroc']['delta'] >= g['auroc'] and r['auroc']['ci95'][0] > 0 and r['ap']['delta'] >= g['ap']
                         and r['ap']['ci95'][0] > 0 and r['fpr_improvement']['delta'] >= g['fpr']
                         and r['fpr_improvement']['ci95'][0] > 0 and r['cur_auroc_drop'] <= g['cur_auroc_drop']
                         and r['cur_fpr_rise'] <= g['cur_fpr_rise'])
        r['permissiveness_artifact'] = bool(r['fpr_improvement']['delta'] > 0 and
                                            (r['auroc']['delta'] < 0 or r['ap']['delta'] < 0))
        return r
    M2 = {'P': m2_of('P'), 'C2_single_clock': m2_of('C2') if strongest != 'C2' else None}
    # M3: tier-2 consistency on P - strongest (fallback P - C3 when the strongest control is not in tier 2)
    per = {}
    for tg in t2_tags:
        try:
            d2, arr2 = load_cell(tg)
        except FileNotFoundError:
            continue
        T2 = len(d2['tasks'])
        def oa(a):
            m = d2['metrics'][a][str(T2 - 1)]['tasks']
            return np.mean([m[str(j)]['auroc'] for j in range(T2 - 1)])
        for a in d2['arms']:
            per.setdefault(f"{d2['order']}|{d2['seed']}", {})[a] = float(oa(a))
    per[f"a|2711"] = {a: table[a]['old']['auroc'] for a in arms}
    ctrl = strongest if all(strongest in v for v in per.values()) else 'C3'
    dl = {k: v['P'] - v[ctrl] for k, v in per.items() if 'P' in v and ctrl in v}
    M3 = {'control': ctrl, 'fallback_used': ctrl != strongest, 'cells': dl, 'n_pos': int(sum(x > 0 for x in dl.values())),
          'n': len(dl), 'per_order_pos': {o: int(sum(v > 0 for k, v in dl.items() if k.startswith(o))) for o in 'abc'}}
    M3['pass'] = bool(M3['n'] == 9 and M3['n_pos'] >= SP.M3_MIN_POS)
    M4 = {'pass': all(v[2] for v in M0['budget'].values())}
    if not M0['pass']:
        verdict = 'NL03_PROTOCOL_BLOCKED'
    elif not M1['pass']:
        verdict = 'NL03_CONVENTIONAL_MEMORY_EQUIVALENT'
    elif not M2['P']['pass']:
        verdict = 'NL03_CLOCK_EFFECT_ONLY'
    elif M3['pass'] and M4['pass']:
        verdict = 'NL03_MECHANISM_SUPPORTED_NORMAL_ONLY'
    else:
        verdict = 'NL03_CLOCK_EFFECT_ONLY (M3/M4 fail)'
    sub = []
    if not M2['P']['pass'] and (M2['C2_single_clock'] is None or not M2['C2_single_clock']['pass']):
        sub.append('NL03_RETENTION_NOT_IMPROVED')
    if M2['P']['permissiveness_artifact']:
        sub.append('PERMISSIVENESS_ARTIFACT')
    return {'verdict': verdict, 'sub_labels': sub, 'M0': M0, 'M1': M1, 'M2': M2, 'M3': M3, 'M4': M4, 'table': table,
            'per_task_old': {a: [{'task': r['task'], **r['pt']} for r in old[a]] for a in arms},
            'cb_selection': cb, 'contrasts_all': {f'P-{a}': {m: contrast(old['P'], old[a], m, -1 if m == 'fpr' else 1)
                                                         for m in ('auroc', 'ap', 'fpr')} for a in arms if a != 'P'}}


def descriptive(tag):
    """Gate quantities of a secondary/diagnostic cell (S33, 16 MiB): table + P contrasts, no verdict."""
    d, arr = load_cell(tag)
    old = {a: arm_reps(d, arr, a, 'old') for a in d['arms']}
    cur = {a: arm_reps(d, arr, a, 'cur') for a in d['arms']}
    return {'table': {a: {'old': {m: macro(old[a], m) for m in ('auroc', 'ap', 'fpr')},
                          'cur': {m: macro(cur[a], m) for m in ('auroc', 'ap', 'fpr')}} for a in d['arms']},
            'contrasts': {f'P-{a}': {m: contrast(old['P'], old[a], m, -1 if m == 'fpr' else 1) for m in ('auroc', 'ap', 'fpr')}
                          for a in d['arms'] if a != 'P' and 'P' in d['arms']}}


def exact_check(tag):
    d, arr = load_cell(tag)
    return {'note': 'binned 8192 vs exact pixel AP is checked in the smoke/report phase'}


def main():
    from exps.nl03_multirate_01 import plan as PL
    st = RN.rj(OUT / 'DEV_STATUS.json')
    tags = list(st['status'].keys())
    t1 = PL.tag_of(PL.cells()[0])
    t2 = [t for t in tags if t.startswith(('J1_', 'J1x_')) and st['status'][t] == 'DONE']
    # merge J1 / J1x halves of the same (order, seed) is handled in gates via per-cell arm dicts
    g = gates(t1, t2)
    extra = {}
    for t in tags:
        if t.startswith(('J2_', 'J3_')) and st['status'][t] == 'DONE':
            extra[t] = descriptive(t)
    res = {'gates': g, 'secondary': extra, 'status': st, 'time': time.strftime('%FT%T')}
    RN.wj(OUT / 'DEV_RESULTS.json', res)
    RN.wj(OUT / 'DEV_VERDICT.json', {'verdict': g['verdict'], 'sub_labels': g['sub_labels'],
                                     'M0': g['M0']['pass'], 'M1': g['M1'], 'M2_P': g['M2']['P'], 'M3': g['M3'],
                                     'M4': g['M4'], 'cb': g['cb_selection'], 'label': 'SYNTHETIC_DEV; normal-only'})
    RN.log('DEV VERDICT', g['verdict'], g['sub_labels'])
