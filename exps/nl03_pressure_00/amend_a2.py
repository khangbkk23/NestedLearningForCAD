"""NL03-MEMORY-PRESSURE-00 amendment A2 (agents/tasks/task_nl03_pressure_00_amendment_a2.md): MVTec AD stream, run
first. Rebinds module attributes of the locked core / run / summary; the A1 ceiling (10 GPU-h) applies.

    python -B -m exps.nl03_pressure_00.amend_a2 lock
    python -B -m exps.nl03_pressure_00.amend_a2 project
    python -B -m exps.nl03_pressure_00.amend_a2 score|metrics <tier> [--only mvtec]
    python -B -m exps.nl03_pressure_00.amend_a2 summary_mvtec | summary
"""
from __future__ import annotations

import shutil
import sys
import time

import numpy as np
import torch

from exps.nanm_eval_01 import common as U
from exps.nl03_pressure_00 import amend_a1 as A1
from exps.nl03_pressure_00 import core as C
from exps.nl03_pressure_00 import run as R

AMEND = U.ROOT / 'agents/tasks/task_nl03_pressure_00_amendment_a2.md'
CODE = ['exps/nl03_pressure_00/amend_a2.py']
LOCK = R.OUT / 'AMENDMENT_A2_LOCK.json'
STREAMS = ('mvtec', 'visa', 's33')
_ORIG = {'base_order': C.base_order, 'orders': C.orders, 'plan': R.plan}
ACTIVE = set(STREAMS)


def base_order(stream):
    if stream == 'mvtec':
        from exps.calm02 import data as CD
        return [f'mvtec/{c}' for c in CD.MVTEC_ORDER]
    return _ORIG['base_order'](stream)


def orders(stream):
    if stream == 'mvtec':                                # exactly EVAL-01 A1's U.orders on the bare task names
        from exps.calm02 import data as CD
        od = U.orders(CD.MVTEC_ORDER)
        return {k: [f'mvtec/{c}' for c in v] for k, v in od.items()}
    return _ORIG['orders'](stream)


def plan():
    return [cd for cd in _ORIG['plan']() if cd['stream'] in ACTIVE]


def bind(only=None):
    A1.bind()                                            # verifies A1 and sets the 10 h ceiling
    lk = U.rj(LOCK)
    bad = [f for f, h in {**lk['amendment'], **lk['code']}.items() if U.sha(U.ROOT / f) != h]
    assert not bad and U.sha(A1.LOCK) == lk['a1_lock_sha256'], f'A2 LOCK_FAIL {bad}'
    C.STREAMS = STREAMS
    C.base_order, C.orders = base_order, orders
    R.plan = plan
    ACTIVE.clear(); ACTIVE.update([only] if only else STREAMS)


def lock():
    assert not LOCK.exists(), 'A2 lock exists'
    R.verify_lock()
    assert not (C.CACHE / 'grids' / 'mvtec').exists(), 'mvtec grids exist before the A2 lock'
    U.wj(LOCK, {'amendment': {str(AMEND.relative_to(U.ROOT)): U.sha(AMEND)}, 'code': {f: U.sha(U.ROOT / f) for f in CODE},
                'base_lock_sha256': U.sha(R.OUT / 'PREREG_LOCK.json'), 'a1_lock_sha256': U.sha(A1.LOCK),
                'owner_decision': '2026-11-24 "Ưu tiên chạy mvtec ad trước ... report chi tiết trước 1 lần"',
                'time': time.strftime('%FT%T')}, exclusive=True)
    U.log('A2 LOCK', U.sha(LOCK)[:16])


def project():
    keep = R.OUT / 'PROJECTION_A1.json'
    if not keep.exists():
        shutil.copy2(R.OUT / 'PROJECTION.json', keep)
    R.project()
    p = U.rj(R.OUT / 'PROJECTION.json')
    p['fits_full_plan'] = p['fits']
    p['fits'] = True
    p['a2_note'] = ('A2: tail cells run in priority order while the locked per-cell auto-stop (10 h, A1) allows; '
                    '`fits` set true only to disable the base whole-stream S33 tier-2 skip')
    U.wj(R.OUT / 'PROJECTION.json', p)
    U.log('A2 PROJECT full=%.2f h fits_full_plan=%s' % (p['gpu_projected_s']['total'] / 3600, p['fits_full_plan']))


# ------------------------------------------------------------------ MVTec interim summary
def m2_mvtec():
    from exps.nanm_eval_01 import mvtec as MV
    out = {}
    for t in orders('mvtec')['a']:
        c = t.split('/')[1]
        mine = torch.load(R.grid_path('mvtec', 'a|2711', 'U', t), weights_only=True)
        ref = torch.load(MV.MV_CACHE / 'grids' / 'mvtec' / 'a_2711' / 'N0' / f'{c}.pt', weights_only=True)
        out[t] = bool(mine['rows'] == ref['rows'] and torch.equal(mine['grid'], ref['grid']))
    return out


def u_ref_mvtec(t2):
    """Tier-2 U reference: EVAL-01 A1 measured N0 metrics of the same (order, seed) cell (same canonical grids)."""
    from exps.nanm_eval_01 import mvtec as MV
    d = U.rj(MV.MV_OUT / 'METRICS_mvtec_t1.json')['cells']
    T = 15
    for cell, vals in t2['cells'].items():
        order = orders('mvtec')[cell.split('|')[0]]
        try:
            vals['U_eval01_a1'] = float(np.mean([d[t.split('/')[1]][f'{cell}|N0'][str(T - 1)]['ap_native_binned']
                                                 for t in order[:-1]]))
        except KeyError:
            vals['U_eval01_a1'] = None
    ds = [v['U_eval01_a1'] - v['E'] for v in t2['cells'].values() if v.get('U_eval01_a1') is not None and 'E' in v]
    if ds:
        t2['spread']['U_eval01_a1-E'] = {'mean': float(np.mean(ds)), 'min': float(min(ds)), 'max': float(max(ds)),
                                         'n_positive': int(sum(d > 0 for d in ds)), 'n': len(ds)}
    return t2


def summary_mvtec():
    from exps.nl03_pressure_00 import summary as S
    R.verify_lock()
    t1 = S.tier1('mvtec')
    g = S.gates(t1)
    t2 = u_ref_mvtec(S.tier2('mvtec')) if (R.OUT / 'METRICS_mvtec_t2.json').exists() else None
    m2 = m2_mvtec()
    U.wj(R.OUT / 'RESULTS_mvtec.json', {'tier1': t1, 'gates_quantities': g, 'tier2': t2, 'M2_mvtec': m2,
                                        'budget': {k: v for k, v in S.budget().items() if k != 'by_phase'}})
    U.wj(R.OUT / 'VERDICT_mvtec.json', {'stream': 'mvtec (A2 secondary; primary verdict = VisA)',
                                        'verdict_label_by_registered_rules': g['verdict'] if all(m2.values()) else
                                        'NL03_PROTOCOL_BLOCKED (M2-mvtec)',
                                        'gates': {k: g[k] for k in ('G0-1', 'G0-2', 'G0-3')}, 'secondary': g['secondary'],
                                        'M2_mvtec_all_equal': all(m2.values()), 'exposure': 'EXPOSED-DEV; diagnostic only'})
    U.log('MVTEC VERDICT', g['verdict'], g['secondary']['label'], 'M2', all(m2.values()))


def summary():
    from exps.nl03_pressure_00 import summary as S
    A1.summary()                                         # full summary with the A1 budget record (streams incl. mvtec)
    v = U.rj(R.OUT / 'VERDICT.json')
    res = U.rj(R.OUT / 'RESULTS.json')
    v['mvtec_descriptive'] = res.get('mvtec', {}).get('gates_quantities')
    v['amendments'] = ['A1: GPU ceiling 6 -> 10 h', 'A2: MVTec AD stream added and run first; tail rule per-cell']
    U.wj(R.OUT / 'VERDICT.json', v)


if __name__ == '__main__':
    ph, rest = sys.argv[1], sys.argv[2:]
    only = None
    if '--only' in rest:
        i = rest.index('--only'); only = rest[i + 1]; rest = rest[:i] + rest[i + 2:]
    if ph == 'lock':
        lock()
    else:
        bind(only)
        {'project': project, 'summary_mvtec': summary_mvtec, 'summary': summary,
         'score': R.score, 'metrics': R.metrics}[ph](*rest)
