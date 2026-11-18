"""HOPE-S-02: fixed surprise-weighting attribution control (gamma = 2.1, lambda = 0, no slow-level updates).
Same code path as HOPE-S-01 (acquire(learn=False)): CREDIT tail and gamma-direction gradients are computed and
discarded, so data access and gradient compute are matched. See agents/tasks/task_hope_s_02.md.

    python -B -m exps.hope_s_02.run lock
    python -B -m exps.hope_s_02.run execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time

import torch

from exps.hope_s_01 import run as H
from exps.nar_b14_01 import run as B
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_patch_01 import run as P
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/hope_s_02'
HOPE = H.OUT                                  # sealed HOPE-S-01 (read only)
LITE = L.OUT                                  # sealed C = NAR-LITE-01 (read only); captured before any rebinding
GAMMA_FIXED = 2.1
GAMMA_CLASS = 'posthoc trajectory-informed diagnostic comparator; not preregistered, not future-task-causal'
HOPE_RESULT_SHA = 'e8f0392284074c5ae4bea4aeccce4c299741b8837e921f66fb02d3dea58e4e82'
LITE_RESULT_SHA = 'bd66bb7568dd376d363fbcbea87cb5ccc14ac49acec6fc6af9d304c750a668d7'
CAPS = {'wall_seconds': 2700, 'gpu_seconds': 1800, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 700_000_000}
SOURCES = ('exps/hope_s_02/run.py', 'exps/hope_s_02/test_run.py', 'exps/hope_s_01/run.py', 'exps/nar_lite_01/run.py',
           'exps/nar_b14_01/run.py', 'exps/nar_patch_01/run.py', 'exps/nar_warm_01/run.py',
           'exps/nar_pilot_00/core.py', 'exps/nar_pilot_00/full.py', 'exps/calm02/native.py',
           'agents/tasks/task_hope_s_02.md')


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES
    L.OUT, L.REF = OUT, OUT


def references_intact():
    K.require(R.sha(HOPE / 'RESULT.json') == HOPE_RESULT_SHA, 'HOPE-S-01 result changed')
    K.require(R.sha(LITE / 'RESULT.json') == LITE_RESULT_SHA, 'C result changed')


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
        hope_rec = R.read(HOPE / f'stage{stage + 1:02}.json')
        if stage == 0:
            K.require(W.digest(init) == hope_rec['initial_digest'], 'task-1 init differs from HOPE-S-01')
            R.log('TASK1_INIT_MATCHES_HOPE_S')
        before = R.normal_eval(model, data['features'], roles['EVAL'])
        lam_out, gamma_out, trace, wall = H.acquire(model, loras, data['features'], roles, stage, lam, GAMMA_FIXED,
                                                    learn=False, budget=budget)
        K.require(torch.equal(lam_out, lam) and gamma_out == GAMMA_FIXED, 'fixed policy changed')
        K.require(len(trace['fit']) == 300 and len(trace['credit']) == 30 and len(trace['h_gamma']) == 30,
                  'matched fast / slow-compute budget')
        endpoint = K.get_state(loras)
        after = R.normal_eval(model, data['features'], roles['EVAL'])
        archive.append(endpoint)
        target = OUT / f'stage{stage + 1:02}.pt'
        W.bounded_save(target, {'archive': archive, 'phi': {}, 'lam': lam, 'gamma': GAMMA_FIXED, 'cfg': R.CFG.dict(),
                                'stage': stage + 1, 'arm': 'fixed_gamma', 'base_sha256': lk['base_sha256']},
                       sum(K.tensor_state_bytes(a) for a in archive) + lam.numel() * 8, budget)
        rec = {'stage': stage + 1, 'category': c, 'before': before, 'after': after, 'seconds': wall, 'trace': trace,
               'gamma': GAMMA_FIXED, 'initial_digest': W.digest(init), 'endpoint_digest': W.digest(endpoint),
               'state_sha256': R.sha(target), 'credit_identities_match_hope_s': True}
        R.write_json(target.with_suffix('.json'), rec)
        records.append(rec)
        R.log('TRAIN', stage + 1, c, 'tail', round(trace['credit'][-1], 4), 'h_gamma(discarded)',
              round(sum(trace['h_gamma']) / 30, 4), 's', round(wall, 1))
        del data
        budget.check()
    R.write_json(OUT / 'TRAIN_COMPLETE.json', {
        'lock_sha256': R.sha(OUT / 'LOCK.json'), 'base_sha256': lk['base_sha256'],
        'checkpoints': {f'stage{i:02}.pt': R.sha(OUT / f'stage{i:02}.pt') for i in range(1, 16)},
        'records': {f'stage{i:02}.json': R.sha(OUT / f'stage{i:02}.json') for i in range(1, 16)},
        'update_seconds': sum(r['seconds'] for r in records), 'resource_at_seal': budget.json()})
    del model, loras, base, archive
    gc.collect(); torch.cuda.empty_cache()


def fairness(fixed_tc, hope_tc):
    hope = [R.read(HOPE / f'stage{i:02}.json') for i in range(1, 16)]
    mine = [R.read(OUT / f'stage{i:02}.json') for i in range(1, 16)]
    return {
        'normal_training_observations': 'identical FIT index stream (same per-stage generator and draws; code path shared)',
        'fast_steps': [len(m['trace']['fit']) for m in mine] == [len(h['trace']['fit']) for h in hope],
        'credit_access': [len(m['trace']['credit']) for m in mine] == [len(h['trace']['credit']) for h in hope],
        'slow_gradient_compute': 'matched: 30 CREDIT tail backward + 30 gamma-direction backward per task, '
                                 'updates discarded',
        'adapter_memory': 'identical rank-8 x 34 sites x 15 tasks',
        'task_history': 'warm start from previous endpoint in both; task-1 init digest equal',
        'evaluation_identities': 'same rows_for(c, test), same scorer, same bf16 lite deployment, same A1 readout',
        'update_seconds': {'fixed': fixed_tc['update_seconds'], 'hope_s': hope_tc['update_seconds']},
        'unmatched': ['lambda = 0 (base learning rates) vs learned lambda (HOPE-S ~3x lower mean rate): part of the '
                      'treatment', 'gamma 2.1 constant vs learned 0 -> 3.44 trajectory: the treatment']}


def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    references_intact()
    hl = R.read(HOPE / 'LOCK.json')
    for c in R.ORDER:
        d = hl['inputs'][c]['train']
        K.require(R.sha(R.ROOT / d['path']) == d['sha256'], f'train cache {c}')
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'inputs': hl['inputs'],
        'base_sha256': R.sha(W.HIST / 'BASE.pt'), 'encoder_sha256': B.ENC_SHA, 'cfg': R.CFG.dict(), 'seed': R.SEED,
        'order': list(R.ORDER), 'gamma_fixed': GAMMA_FIXED, 'gamma_classification': GAMMA_CLASS, 'lambda': 0,
        'hope_s_result_sha256': HOPE_RESULT_SHA, 'c_result_sha256': LITE_RESULT_SHA, 'caps': CAPS,
        'diagnostic_threshold': 'HOPE-S minus F: +0.010 AP, CI lo > 0, tolerances 0.010 (diagnostic, not confirmatory)',
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    references_intact()
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'], 'base hash')
    return lk


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = verify()
    R.configure()
    original = (R.cap, R.loss)
    R.cap, R.loss = budget.check, P.patch_loss
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0)})
    try:
        with budget.phase('training', gpu=True):
            train(lk, budget)
        with budget.phase('fidelity_normal_only', gpu=True):
            fid = L.fidelity(torch.bfloat16, budget)
            R.write_json(OUT / 'FIDELITY.json', fid)
            K.require(fid['passed'], 'bf16 fidelity failed')
        with budget.phase('scoring', gpu=True):
            L.score(torch.bfloat16, budget)
        with budget.phase('inference_cost', gpu=True):
            cost = L.cost(torch.bfloat16, budget)
        with budget.phase('metrics_a1'):
            a1 = L.metrics('smooth', budget)
        with budget.phase('metrics_raw'):
            raw = L.metrics('raw', budget)
        hope, c = R.read(HOPE / 'RESULT.json'), R.read(LITE / 'RESULT.json')
        h_a1, h_raw, c_a1 = hope['arms']['hope_s_a1'], hope['arms']['hope_s_raw'], c['arms']['lite_a1']
        d_a1, d_raw = L.ci(h_a1, a1), L.ci(h_raw, raw)
        tc = R.read(OUT / 'TRAIN_COMPLETE.json')
        functional = bool(fid['passed'] and a1['final_image_auroc'] >= .90 and raw['final_image_auroc'] >= .90)
        result = {
            'arms': {'fixed_gamma_a1': a1, 'fixed_gamma_raw': raw, 'hope_s_a1': h_a1, 'hope_s_raw': h_raw,
                     'c_a1': c_a1},
            'FIXED_GAMMA_DETECTOR_FUNCTIONAL': functional,
            'FIXED_GAMMA_PIXEL_AP': a1['final_pixel_ap'],
            'HOPE_S_MINUS_FIXED_GAMMA': d_a1, 'HOPE_S_MINUS_FIXED_GAMMA_RAW': d_raw,
            'DIAGNOSTIC_THRESHOLD_HOPE_S_VS_FIXED': W.gate(h_a1, a1, d_a1),
            'FIXED_GAMMA_MINUS_C': L.ci(a1, c_a1),
            'gamma_classification': GAMMA_CLASS, 'fairness': fairness(tc, R.read(HOPE / 'TRAIN_COMPLETE.json')),
            'update_seconds': tc['update_seconds'], 'cost': cost, 'fidelity': fid,
            'exposure': 'One seed 2711, alphabetical 15, historically exposed MVTec; exploratory only'}
        verify(); P.OUT = OUT; P.verify_training()
        result['resources'] = budget.json()
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', json.dumps(d_a1), json.dumps(result['DIAGNOSTIC_THRESHOLD_HOPE_S_VS_FIXED']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise
    finally:
        R.cap, R.loss = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'lock': lock, 'execute': execute}[a.phase]()
