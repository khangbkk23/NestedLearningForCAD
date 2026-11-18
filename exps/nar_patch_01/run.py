"""NAR-PATCH-01: warm-start + mean patchwise cosine loss; the only change from the sealed NAR-WARM-01 arm.

    python -B -m exps.nar_patch_01.run lock
    python -B -m exps.nar_patch_01.run execute --owner-authorized

Reuses exps.nar_warm_01.run (budget, caps, scoring, cost, metrics) and exps.nar_pilot_00.full.fit_task unchanged.
The loss binding full.loss is redirected at run time to ``patch_loss``; no source file is edited.
Comparator: sealed NAR-WARM-01 (global cosine), read only.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from exps.nar_pilot_00 import core as K, full as R
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/nar_patch_01'
WARM = R.ROOT / 'results/hope_cad/nar_warm_01'
WARM_RESULT_SHA = '8ff1d4a3f76a7ad2435998985bbb286b758743b1ad95d03a9aaa3c3252ecd1db'
SOURCES = ('exps/nar_patch_01/run.py', 'exps/nar_patch_01/test_run.py', 'exps/nar_warm_01/run.py',
           'exps/nar_pilot_00/cost.py', 'exps/nl_sgm_01a/am03_eval.py', 'agents/tasks/task_nar_patch_01.md',
           'agents/reports/nar_acq_01.md')


def patch_loss(model, x):
    """Mean over images and patches of 1 - cos(x_p, y_p); encoder side detached (as in global_cosine)."""
    en, de = model(x)
    return (1 - F.cosine_similarity(en[0].detach(), de[0], dim=1)).mean()


def bind():
    W.OUT = OUT
    W.SOURCES = SOURCES
    W.verify_training = verify_training


def verify_training():
    m = R.read(OUT / 'TRAIN_COMPLETE.json')
    K.require(m['lock_sha256'] == R.sha(OUT / 'LOCK.json') and len(m['checkpoints']) == 15, 'training seal incomplete')
    for key in ('checkpoints', 'records'):
        for p, h in m[key].items():
            K.require(R.sha(OUT / p) == h, 'training seal mismatch')
    return m


def train(lk, budget):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    fixed = K.init_state(loras, R.SEED)
    lam = torch.zeros(len(loras), dtype=torch.float64)
    archive, records = [], []
    for stage, c in enumerate(R.ORDER):
        data = W.load_cache(lk, c, 'train')
        roles = R.assert_cache(data, c)
        init = W.initial_state(fixed, archive)
        K.set_state(loras, init)
        K.require(R.state_equal(K.get_state(loras), init), 'warm initialization mismatch')
        if stage == 0:
            ref = R.read(WARM / 'stage01.json')
            K.require(W.digest(init) == ref['initial_digest'], 'task1 initial state differs from NAR-WARM-01')
            R.log('TASK1_INIT_MATCHES_WARM')
        before = R.normal_eval(model, data['features'], roles['EVAL'])
        lam, trace, wall = R.fit_task(model, loras, data['features'], roles, stage, lam, False)
        endpoint = K.get_state(loras)
        after = R.normal_eval(model, data['features'], roles['EVAL'])
        K.require(torch.equal(lam, torch.zeros_like(lam)), 'nonzero lambda')
        K.require(len(trace['fit']) == 300 and len(trace['credit']) == 30, 'gradient budget')
        archive.append(endpoint)
        target = OUT / f'stage{stage + 1:02}.pt'
        W.bounded_save(target, {'archive': archive, 'phi': fixed, 'lam': lam, 'cfg': R.CFG.dict(), 'stage': stage + 1,
                                'arm': 'warm_patch', 'base_sha256': lk['base_sha256']},
                       sum(K.tensor_state_bytes(a) for a in archive) + K.slow_bytes(lam, fixed), budget)
        rec = {'stage': stage + 1, 'category': c, 'loss': 'mean_patch_cosine', 'before': before, 'after': after,
               'seconds': wall, 'trace': trace, 'roles': roles, 'initial_digest': W.digest(init),
               'endpoint_digest': W.digest(endpoint), 'state_sha256': R.sha(target)}
        R.write_json(target.with_suffix('.json'), rec)
        records.append(rec)
        R.log('TRAIN', stage + 1, c, round(before, 6), round(after, 6), 'seconds', round(wall, 1))
        del data
        budget.check()
    R.write_json(OUT / 'TRAIN_COMPLETE.json', {
        'lock_sha256': R.sha(OUT / 'LOCK.json'), 'base_sha256': lk['base_sha256'],
        'checkpoints': {f'stage{i:02}.pt': R.sha(OUT / f'stage{i:02}.pt') for i in range(1, 16)},
        'records': {f'stage{i:02}.json': R.sha(OUT / f'stage{i:02}.json') for i in range(1, 16)},
        'update_seconds': sum(r['seconds'] for r in records), 'resource_at_seal': budget.json()})
    del model, loras, base, archive
    gc.collect()
    torch.cuda.empty_cache()


def compare(result):
    """Gate patch (this arm) vs global (sealed warm)."""
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    K.require(R.sha(WARM / 'RESULT.json') == WARM_RESULT_SHA, 'warm reference changed')
    ref = R.read(WARM / 'RESULT.json')['arms']['warm']
    new = result['arms']['warm']
    ci = bootstrap_delta([new['per_category'][c]['final']['pixel_ap'] for c in R.ORDER],
                         [ref['per_category'][c]['final']['pixel_ap'] for c in R.ORDER], seed=20261117, B=2000)
    return {'arms': {'warm_global': ref, 'warm_patch': new, 'nested': result['arms']['nested'],
                     'fixed': result['arms']['fixed']},
            'patch_minus_global': ci, 'PATCH_LOSS_GATE': W.gate(new, ref, ci),
            'exposure': 'One seed 2711, alphabetical 15, historically exposed MVTec; exploratory only'}


def lock():
    bind()
    W.lock()


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = W.verify()
    K.require(R.sha(WARM / 'RESULT.json') == WARM_RESULT_SHA, 'warm reference changed')
    R.configure()
    original_cap, original_loss = R.cap, R.loss
    R.cap, R.loss = budget.check, patch_loss
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0),
                                        'free_disk': shutil.disk_usage(R.ROOT).free})
    try:
        with budget.phase('training', gpu=True):
            train(lk, budget)
        with budget.phase('scoring', gpu=True):
            W.score(lk, budget)
        with budget.phase('inference_cost', gpu=True):
            W.cost(lk, budget)
        with budget.phase('metrics'):
            result = compare(W.metrics(budget))
        W.verify()
        verify_training()
        budget.check()
        result['resources'] = budget.json()
        result['cost'] = R.read(OUT / 'COST.json')
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()},
                                               'resources': budget.json()})
        R.log('COMPLETE', json.dumps(result['patch_minus_global']), json.dumps(result['PATCH_LOSS_GATE']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise
    finally:
        R.cap, R.loss = original_cap, original_loss


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'lock':
        lock()
    else:
        K.require(a.owner_authorized, 'explicit owner execution flag required')
        execute()
