"""NL03-MEMORY-PRESSURE-00 amendment A1 (agents/tasks/task_nl03_pressure_00_amendment_a1.md): owner raised the GPU
wall ceiling 6 -> 10 h. Rebinds run.BUDGET and calls the LOCKED phases unchanged.

    python -B -m exps.nl03_pressure_00.amend_a1 lock | project | score <tier> | metrics <tier> | summary
"""
from __future__ import annotations

import shutil
import sys
import time

from exps.nanm_eval_01 import common as U
from exps.nl03_pressure_00 import run as R

AMEND = U.ROOT / 'agents/tasks/task_nl03_pressure_00_amendment_a1.md'
CODE = ['exps/nl03_pressure_00/amend_a1.py']
CEILING_H = 10.0
LOCK = R.OUT / 'AMENDMENT_A1_LOCK.json'


def lock():
    assert not LOCK.exists(), 'amendment lock exists'
    base = R.verify_lock()
    old = R.OUT / 'PROJECTION.json'
    keep = R.OUT / 'PROJECTION_A0_ceiling6h.json'
    if not keep.exists():
        shutil.copy2(old, keep)
    U.wj(LOCK, {'amendment': {str(AMEND.relative_to(U.ROOT)): U.sha(AMEND)},
                'code': {f: U.sha(U.ROOT / f) for f in CODE},
                'base_lock_sha256': U.sha(R.OUT / 'PREREG_LOCK.json'), 'base_budget': base['budget'],
                'new_gpu_wall_hours': CEILING_H, 'projection_a0_sha256': U.sha(keep),
                'owner_decision': '2026-11-24 "Nâng trần lên 10 GPU-h"', 'time': time.strftime('%FT%T')},
         exclusive=True)
    U.log('A1 LOCK', U.sha(LOCK)[:16])


def bind():
    lk = U.rj(LOCK)
    bad = [f for f, h in {**lk['amendment'], **lk['code']}.items() if U.sha(U.ROOT / f) != h]
    assert not bad and U.sha(R.OUT / 'PREREG_LOCK.json') == lk['base_lock_sha256'], f'A1 LOCK_FAIL {bad}'
    R.BUDGET['gpu_wall_hours'] = CEILING_H


def summary():
    from exps.nl03_pressure_00 import summary as S
    R.verify_lock()
    orig = S.budget

    def budget():
        b = orig()
        b.update(ceiling_gpu_s=CEILING_H * 3600, base_ceiling_gpu_s=6 * 3600,
                 amendment='A1: owner raised the GPU wall ceiling 6 -> 10 h (2026-11-24)',
                 budget_stop_note=('BUDGET_STOP.json = locked 6 h checkpoint; resumed under A1'
                                   if (R.OUT / 'BUDGET_STOP.json').exists() else None))
        return b
    S.budget = budget
    S.main()


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'lock':
        lock()
    else:
        bind()
        if ph == 'summary':
            summary()
        else:
            {'project': R.project, 'score': R.score, 'metrics': R.metrics}[ph](*sys.argv[2:])
