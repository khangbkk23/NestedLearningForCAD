"""NL03-MEMORY-PRESSURE-00 amendment A3 (agents/tasks/task_nl03_pressure_00_amendment_a3.md): engineering only.
Applies the A2 bind (A1 ceiling 10 h, MVTec stream, priority plan), then run.POOL = 2 (low-memory restart).

    python -B -m exps.nl03_pressure_00.amend_a3 lock | score <tier> | metrics <tier> | summary
"""
from __future__ import annotations

import sys
import time

from exps.nanm_eval_01 import common as U
from exps.nl03_pressure_00 import amend_a2 as A2
from exps.nl03_pressure_00 import run as R

AMEND = U.ROOT / 'agents/tasks/task_nl03_pressure_00_amendment_a3.md'
CODE = ['exps/nl03_pressure_00/amend_a3.py']
LOCK = R.OUT / 'AMENDMENT_A3_LOCK.json'
POOL = 2


def lock():
    assert not LOCK.exists(), 'A3 lock exists'
    R.verify_lock()
    U.wj(LOCK, {'amendment': {str(AMEND.relative_to(U.ROOT)): U.sha(AMEND)}, 'code': {f: U.sha(U.ROOT / f) for f in CODE},
                'a2_lock_sha256': U.sha(A2.LOCK), 'pool': POOL,
                'owner_decision': '2026-11-24 "Chạy lại, 2 worker"', 'time': time.strftime('%FT%T')}, exclusive=True)
    U.log('A3 LOCK', U.sha(LOCK)[:16])


def bind():
    A2.bind()
    lk = U.rj(LOCK)
    bad = [f for f, h in {**lk['amendment'], **lk['code']}.items() if U.sha(U.ROOT / f) != h]
    assert not bad and U.sha(A2.LOCK) == lk['a2_lock_sha256'], f'A3 LOCK_FAIL {bad}'
    R.POOL = POOL


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'lock':
        lock()
    else:
        bind()
        {'summary': A2.summary, 'score': R.score, 'metrics': R.metrics}[ph](*sys.argv[2:])
