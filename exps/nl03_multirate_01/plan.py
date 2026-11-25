"""NL03-MULTIRATE-01 locks, tests, projection and the registered development plan (Phases D, F, H, I, J, K).

    python -B -m exps.nl03_multirate_01.run prereg_lock | tests | impl_lock | project | dev | summary
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time

from exps.nl03_multirate_01 import run as RN
from exps.nl03_multirate_01 import spec as SP

OUT = SP.OUT
ROOT = SP.ROOT
PRE = ['agents/tasks/task_nl03_multirate_01.md', 'agents/tasks/task_nl03_multirate_01_registration.md',
       'exps/nl03_multirate_01/spec.py', 'exps/nl03_multirate_01/audit.py', 'exps/nl03_multirate_01/synth.py',
       'exps/nl03_multirate_01/ad2_monitor.py', 'results/hope_cad/nl03_multirate_01/AUDIT.json']
IMPL = ['exps/nl03_multirate_01/core.py', 'exps/nl03_multirate_01/evaluate.py', 'exps/nl03_multirate_01/run.py',
        'exps/nl03_multirate_01/plan.py', 'exps/nl03_multirate_01/summary.py', 'exps/nl03_multirate_01/test_core.py',
        'exps/nl03_multirate_01/__init__.py']


def sha(p):
    return RN.SY._fsha(p)


def _hashes(files):
    return {f: sha(ROOT / f) for f in files}


def _synth_meta():
    return {p.name: json.loads(p.read_text())['sha256'] for p in sorted((SP.CACHE / 'synth').glob('*.json'))}


def prereg_lock():
    p = OUT / 'PREREG_LOCK.json'
    assert not p.exists(), 'PREREG_LOCK exists'
    a = RN.rj(OUT / 'AUDIT.json')
    assert a['gate_4']['pass'] and not a['canonical_sources_changed'], 'audit gate'
    rec = {'files': _hashes(PRE), 'canonical_sources': a['canonical_sources_now'], 'pressure00_locks': a['pressure00_locks'],
           'synth_caches': _synth_meta(), 'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                                     text=True, cwd=ROOT).stdout.strip(),
           'time': time.strftime('%FT%T'), 'note': 'locked before any development scoring'}
    with open(p, 'x') as f:
        json.dump(rec, f, indent=1)
    RN.log('PREREG_LOCK', sha(p)[:16])


def verify(impl=True):
    lk = RN.rj(OUT / 'PREREG_LOCK.json')
    bad = [f for f, h in lk['files'].items() if sha(ROOT / f) != h]
    bad += [f for f, h in lk['canonical_sources'].items() if sha(ROOT / f) != h]
    bad += [n for n, h in lk['synth_caches'].items() if _synth_meta().get(n) != h]
    if impl:
        il = RN.rj(OUT / 'IMPL_LOCK.json')
        bad += [f for f, h in il['files'].items() if sha(ROOT / f) != h]
    assert not bad, f'LOCK_FAIL {bad}'
    return True


def tests():
    t0 = time.time()
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider',
                        'exps/nl03_multirate_01/test_core.py'], capture_output=True, text=True, cwd=ROOT)
    tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
    RN.wj(OUT / 'TESTS.json', {'rc': r.returncode, 'tail': tail, 'seconds': time.time() - t0, 'time': time.strftime('%FT%T')})
    RN.ledger_add({'phase': 'dev_tests', 'seconds': time.time() - t0, 'gpu': True})
    RN.log('TESTS rc', r.returncode, tail[-1] if tail else '')
    assert r.returncode == 0, 'tests failed'


def impl_lock():
    p = OUT / 'IMPL_LOCK.json'
    assert not p.exists(), 'IMPL_LOCK exists'
    verify(impl=False)
    assert RN.rj(OUT / 'TESTS.json')['rc'] == 0
    rec = {'files': _hashes(IMPL), 'tests_sha256': sha(OUT / 'TESTS.json'), 'smoke_sha256': sha(OUT / 'SMOKE.json'),
           'prereg_sha256': sha(OUT / 'PREREG_LOCK.json'), 'time': time.strftime('%FT%T')}
    with open(p, 'x') as f:
        json.dump(rec, f, indent=1)
    RN.log('IMPL_LOCK', sha(p)[:16])


# ------------------------------------------------------------------ registered plan
def steps_of(stream):
    return sum(math.ceil(n / SP.MB_IMAGES) for n in RN.fit_dev_counts(stream))


def cells(cb=None, reduced=False):
    """Registered priority list: (name, stream, order, seed, MiB, arms). Arms are independent given the minibatch
    stream (C1/C4 share one fast trajectory by construction), so splitting an arm set across cells is exact."""
    allarms = RN.ALL_ARMS
    out = [('I', 'visa', 'a', 2711, 8, allarms)]
    full = tuple(a for a in ('P', 'C2', 'C3', cb or 'CB') if a)
    t2 = ('P', 'C3') if reduced else full
    t2 = tuple(dict.fromkeys(t2))
    j1 = [(ok, s) for ok in 'abc' for s in SP.SEEDS if (ok, s) != ('a', 2711)]
    out += [('J1', 'visa', ok, s, 8, t2) for ok, s in j1]
    if reduced:
        out += [('J1x', 'visa', ok, s, 8, tuple(a for a in full if a not in t2)) for ok, s in j1]
    out.append(('J2', 's33', 'a', 2711, 8, allarms))
    out.append(('J3', 'visa', 'a', 2711, 16, allarms))
    return out


def est(c, rates):
    name, stream, ok, seed, mib, arms = c
    arms = [a for a in arms if a in SP.ARMS]
    st = steps_of(stream)
    T = len(RN.orders(stream)['a'])
    return (st * (rates['per_fast_level_step_s'] * n_fast_levels(arms) + rates['per_step_cand_slow_s']
                  + (rates['c0_per_step_s'] if 'C0' in arms else 0))
            + rates['per_eval_s'] * len(arms) * T * (T + 1) / 2 + rates['overhead_task_s'] * T) * 1.5


def tag_of(c):
    name, stream, ok, seed, mib, arms = c
    return f'{name}_{stream}_{ok}{seed}_{mib}mib'


def n_fast_levels(arms):
    return len({(SP.total_cap(a, 8 * SP.MiB) - (SP.ARMS[a]['C'] * 63 if SP.ARMS[a]['pending'] else 0),
                 SP.ARMS[a]['counts']) for a in arms if a != 'C0'})


def project():
    sm = RN.rj(OUT / 'SMOKE.json')
    tm, steps = sm['timing'], sm['fast_steps']
    fl = len(tm['fast_s'])
    per_fl_step = sum(tm['fast_s'].values()) / fl / steps
    per_step = tm['cand_s'] / steps + tm['slow_s'] / steps
    n_eval = 9 * 3                                            # 9 arms x (1 + 2) task-evaluations in the 2-task smoke
    per_eval = tm['eval_s'] / n_eval
    led = [json.loads(x) for x in (OUT / 'BUDGET_LEDGER.jsonl').read_text().splitlines()]
    smoke_wall = sum(r['seconds'] for r in led if r['phase'] == 'dev_smoke')
    overhead_task = max(0.0, smoke_wall - sum(tm['fast_s'].values()) - tm['cand_s'] - tm['slow_s'] - tm['eval_s']
                        - tm['c0_s']) / 2
    c0_per_step = tm['c0_s'] / steps
    rates = {'per_fast_level_step_s': per_fl_step, 'per_step_cand_slow_s': per_step, 'per_eval_s': per_eval,
             'overhead_task_s': overhead_task, 'c0_per_step_s': c0_per_step}
    used = RN.gpu_used(('dev',))
    full = [{'cell': tag_of(c), 'seconds': est(c, rates)} for c in cells(cb='C1')]
    red = [{'cell': tag_of(c), 'seconds': est(c, rates)} for c in cells(cb='C1', reduced=True)]
    mand = sum(p['seconds'] for p in red if p['cell'].startswith(('I_', 'J1_')))
    total = sum(p['seconds'] for p in full)
    rec = {'rates': rates, 'safety': 1.5, 'cells_full': full, 'cells_reduced': red, 'used_before_s': used,
           'total_projected_s': total, 'ceiling_s': SP.DEV_GPU_CEILING_S,
           'full_fits': used + total <= SP.DEV_GPU_CEILING_S, 'reduced_mandatory_s': mand,
           'time': time.strftime('%FT%T')}
    rec['plan'] = 'full' if rec['full_fits'] else 'reduced (I + J1 {P, C3} mandatory; rest in priority order)'
    RN.wj(OUT / 'PROJECTION.json', rec)
    RN.log('PROJECT total %.2f h used %.2f h plan %s' % (total / 3600, used / 3600, rec['plan']))


def select_cb():
    d = RN.rj(OUT / 'cells' / f"{tag_of(cells()[0])}.json")
    T = len(d['tasks'])
    def old_auroc(a):
        m = d['metrics'][a][str(T - 1)]['tasks']
        return sum(m[str(j)]['auroc'] for j in range(T - 1)) / (T - 1)
    conv = {a: old_auroc(a) for a in SP.CONVENTIONAL}
    matched = {a: old_auroc(a) for a in SP.MATCHED}
    cb = max(conv, key=conv.get)
    strongest = max(matched, key=matched.get)
    rec = {'CB': cb, 'conventional_old_auroc': conv, 'strongest_matched': strongest, 'matched_old_auroc': matched,
           'rule': 'argmax tier-1 VisA a|2711 8 MiB final old-task synthetic image AUROC (point)', 'time': time.strftime('%FT%T')}
    p = OUT / 'CB_SELECTION.json'
    if not p.exists():
        RN.wj(p, rec)
    return RN.rj(p)


def dev():
    verify()
    proj = RN.rj(OUT / 'PROJECTION.json')
    reduced = not proj['full_fits']
    status = {}
    plan = cells(reduced=reduced)
    i = 0
    while i < len(plan):
        if i == 1:
            plan = cells(cb=select_cb()['CB'], reduced=reduced)
        c = plan[i]; i += 1
        name, stream, ok, seed, mib, arms = c
        if not arms:
            continue
        tag = tag_of(c)
        e = est(c, proj['rates'])
        used = RN.gpu_used(('dev',))
        if used + e > SP.DEV_GPU_CEILING_S and not (OUT / 'cells' / f'{tag}.json').exists():
            status[tag] = 'NOT_RUN (budget: projected)'
            RN.log('NOT_RUN', tag, 'used %.2f h est %.2f h' % (used / 3600, e / 3600))
            continue
        try:
            RN.run_cell(stream, ok, seed, mib, arms, tag, ceiling=SP.DEV_GPU_CEILING_S)
            status[tag] = 'DONE'
        except RN.BudgetStop as ex:
            status[tag] = f'NOT_RUN (budget: stopped in cell) {ex}'
            RN.log('BUDGET_STOP', tag)
            for c2 in plan[i:]:
                status[tag_of(c2)] = 'NOT_RUN (budget: after stop)'
            break
        finally:
            RN.wj(OUT / 'DEV_STATUS.json', {'status': status, 'reduced': reduced, 'arms': {tag_of(x): x[5] for x in plan},
                                            'time': time.strftime('%FT%T')})
    RN.log('DEV DONE', status)


def summary():
    from exps.nl03_multirate_01 import summary as SU
    SU.main()
