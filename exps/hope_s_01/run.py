"""HOPE-S-01: surprise-gated two-level nested memory on top of the current best NAR model.
See agents/tasks/task_hope_s_01.md. Fast level = warm rank-8 LoRA with surprise-weighted patch loss (gamma);
slow level = (lambda, gamma) learned every 10 fast steps from a held-out normal tail objective, carried across tasks.

    python -B -m exps.hope_s_01.run lock
    python -B -m exps.hope_s_01.run execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import signal
import time

import torch
import torch.nn.functional as F

from exps.nar_b14_01 import run as B
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_patch_01 import run as P
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/hope_s_01'
LITE = L.OUT                                       # sealed NAR-LITE-01 (current best, read only)
ETA_GAMMA, GAMMA_MAX, GAMMA0, TAIL_K, EPS = .05, 4., 0., 8, 1e-6
CAPS = {'wall_seconds': 4500, 'gpu_seconds': 3000, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 700_000_000}
SOURCES = ('exps/hope_s_01/run.py', 'exps/hope_s_01/test_run.py', 'exps/nar_lite_01/run.py',
           'exps/nar_b14_01/run.py', 'exps/nar_patch_01/run.py', 'exps/nar_warm_01/run.py',
           'exps/nar_pilot_00/core.py', 'exps/nar_pilot_00/full.py', 'exps/calm02/native.py',
           'agents/tasks/task_hope_s_01.md')


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES
    L.OUT, L.REF = OUT, OUT                        # lite deployment / scoring / metrics act on HOPE-S checkpoints


# ---------------------------------------------------------------- objectives
def patch_errors(model, x):
    en, de = model(x)
    return (1 - F.cosine_similarity(en[0].detach(), de[0], dim=1)).flatten(1)     # [B, 784]


def weights(e, gamma):
    u = (e.detach() + EPS) ** gamma
    return u / u.mean(dim=1, keepdim=True)


def fit_loss(model, x, gamma):
    """Surprise-weighted patch loss; gamma = 0 gives exactly the mean patch cosine of NAR-PATCH-01."""
    e = patch_errors(model, x)
    return e.mean() if gamma == 0 else (weights(e, gamma) * e).mean()


def gamma_direction_loss(model, x, gamma):
    """Its parameter gradient equals d/dgamma of the fit gradient (weights detached w.r.t. parameters)."""
    e = patch_errors(model, x)
    w = weights(e, gamma)
    le = torch.log(e.detach() + EPS)
    s = w * (le - (w * le).mean(dim=1, keepdim=True))
    return (s.detach() * e).mean()


def tail_loss(model, x):
    """Held-out normal tail: mean of the top-1% (8 of 784) patch errors."""
    return patch_errors(model, x).topk(TAIL_K, dim=1).values.mean()


def cosine(a, b):
    den = a.norm() * b.norm()
    return float((a @ b) / den) if float(den) > 0 else 0.


# ---------------------------------------------------------------- nested acquisition
def acquire(model, loras, feats, roles, stage, lam, gamma, cfg=R.CFG, learn=True, budget=None):
    torch.manual_seed(R.SEED + 100 * stage)
    gen = torch.Generator().manual_seed(R.SEED + 100 * stage)
    names = list(loras)
    params = [q for n in names for q in (loras[n].A, loras[n].B)]
    opt = R.O.StableAdamW([{'params': [loras[n].A, loras[n].B]} for n in names], lr=cfg.lr, betas=cfg.betas,
                          weight_decay=cfg.weight_decay, eps=cfg.eps, amsgrad=True)
    lam = lam.clone().double()
    snap = [K._flat(loras[n]).clone() for n in names]
    log = {'fit': [], 'credit': [], 'gamma': [], 'h_gamma': [], 'lam_mean': [], 'lam_std': []}
    credit_n, last = 0, None
    model.train(); torch.cuda.synchronize(); start = time.monotonic()
    for i in range(cfg.steps):
        base = K.lr_at(cfg, i)
        for g, grp in enumerate(opt.param_groups):
            grp['lr'] = base * math.exp(float(lam[g]))
        opt.zero_grad(set_to_none=True)
        last = roles['FIT'][int(torch.randperm(len(roles['FIT']), generator=gen)[0])]
        loss = fit_loss(model, feats[last:last + 1].cuda(), gamma)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.clip)
        lv = float(loss.detach())
        K.require(math.isfinite(lv) and bool(torch.isfinite(norm)), f'non-finite fast step {i}')
        opt.step()
        log['fit'].append(lv)
        if (i + 1) % cfg.m == 0:
            model.eval()
            opt.zero_grad(set_to_none=True)
            j = roles['CREDIT'][credit_n % len(roles['CREDIT'])]; credit_n += 1
            credit = tail_loss(model, feats[j:j + 1].cuda())
            credit.backward()
            gc_ = [K._flat_grad(loras[n]).double() for n in names]
            h = torch.zeros(len(names), dtype=torch.float64)
            for g, n in enumerate(names):
                h[g] = cosine((K._flat(loras[n]) - snap[g]).double(), gc_[g])
            opt.zero_grad(set_to_none=True)
            gamma_direction_loss(model, feats[last:last + 1].cuda(), gamma).backward()
            gg = torch.cat([K._flat_grad(loras[n]).double() for n in names])
            hg = cosine(torch.cat(gc_), gg)
            opt.zero_grad(set_to_none=True)
            model.train()
            K.require(bool(torch.isfinite(h).all()) and math.isfinite(hg), f'non-finite slow signal {i}')
            if learn:
                lam = (lam - cfg.eta_lam * h).clamp(-cfg.lam_max, cfg.lam_max)
                gamma = min(GAMMA_MAX, max(0., gamma + ETA_GAMMA * hg))
            snap = [K._flat(loras[n]).clone() for n in names]
            log['credit'].append(float(credit.detach())); log['h_gamma'].append(hg); log['gamma'].append(gamma)
            log['lam_mean'].append(float(lam.mean())); log['lam_std'].append(float(lam.std()))
        if budget is not None and (i + 1) % 50 == 0:
            budget.check()
    torch.cuda.synchronize()
    K.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'parameter finiteness')
    return lam, gamma, log, time.monotonic() - start


def train(lk, budget):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    fixed = K.init_state(loras, R.SEED)
    lam, gamma = torch.zeros(len(loras), dtype=torch.float64), GAMMA0
    archive, records = [], []
    for stage, c in enumerate(R.ORDER):
        data = W.load_cache(lk, c, 'train')
        roles = R.assert_cache(data, c)
        init = W.initial_state(fixed, archive)
        K.set_state(loras, init)
        before = R.normal_eval(model, data['features'], roles['EVAL'])
        lam, gamma, trace, wall = acquire(model, loras, data['features'], roles, stage, lam, gamma, budget=budget)
        if stage == 0:
            ref = R.read(B.OUT / 'stage01.json')['trace']['fit'][:R.CFG.m]
            K.require(trace['fit'][:R.CFG.m] == ref, 'pre-slow-update parity with NAR-B14-01 failed')
            R.log('TASK1_PRE_SLOW_PARITY_PASS')
        endpoint = K.get_state(loras)
        after = R.normal_eval(model, data['features'], roles['EVAL'])
        archive.append(endpoint)
        target = OUT / f'stage{stage + 1:02}.pt'
        W.bounded_save(target, {'archive': archive, 'phi': {}, 'lam': lam, 'gamma': gamma, 'cfg': R.CFG.dict(),
                                'stage': stage + 1, 'arm': 'hope_s', 'base_sha256': lk['base_sha256']},
                       sum(K.tensor_state_bytes(a) for a in archive) + lam.numel() * 8, budget)
        rec = {'stage': stage + 1, 'category': c, 'before': before, 'after': after, 'seconds': wall, 'trace': trace,
               'gamma_end': gamma, 'lam': lam.tolist(), 'initial_digest': W.digest(init),
               'endpoint_digest': W.digest(endpoint), 'state_sha256': R.sha(target)}
        R.write_json(target.with_suffix('.json'), rec)
        records.append(rec)
        R.log('TRAIN', stage + 1, c, 'gamma', round(gamma, 4), 'lam', round(float(lam.mean()), 4),
              round(float(lam.std()), 4), 'tail', round(trace['credit'][-1], 4), 's', round(wall, 1))
        del data
        budget.check()
    R.write_json(OUT / 'TRAIN_COMPLETE.json', {
        'lock_sha256': R.sha(OUT / 'LOCK.json'), 'base_sha256': lk['base_sha256'],
        'checkpoints': {f'stage{i:02}.pt': R.sha(OUT / f'stage{i:02}.pt') for i in range(1, 16)},
        'records': {f'stage{i:02}.json': R.sha(OUT / f'stage{i:02}.json') for i in range(1, 16)},
        'update_seconds': sum(r['seconds'] for r in records), 'resource_at_seal': budget.json()})
    del model, loras, base, archive
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- lock / execute
def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    b14 = R.read(B.OUT / 'LOCK.json')
    for c in R.ORDER:
        d = b14['inputs'][c]['train']
        K.require(R.sha(R.ROOT / d['path']) == d['sha256'], f'B14 train cache {c}')
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES},
        'inputs': {c: {'train': b14['inputs'][c]['train']} for c in R.ORDER},
        'base_sha256': R.sha(W.HIST / 'BASE.pt'), 'encoder_sha256': B.ENC_SHA, 'cfg': R.CFG.dict(), 'seed': R.SEED,
        'order': list(R.ORDER), 'slow': {'eta_gamma': ETA_GAMMA, 'gamma_max': GAMMA_MAX, 'gamma0': GAMMA0,
                                         'tail_k': TAIL_K, 'eta_lam': R.CFG.eta_lam, 'every': R.CFG.m},
        'deployment': 'NAR-LITE-01 (bf16 if normal-only fidelity passes, truncated encoder, A1 readout)',
        'reference_lite_result_sha256': R.sha(LITE / 'RESULT.json'), 'caps': CAPS,
        'gate': 'NL_GATE: W.gate(HOPE-S lite+A1, C lite+A1, CI) — +0.010, CI lo>0, tolerances 0.010',
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(R.sha(LITE / 'RESULT.json') == lk['reference_lite_result_sha256'], 'reference changed')
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'], 'base hash')
    return lk


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = verify()
    R.configure()
    original = (R.cap, R.loss)
    R.cap, R.loss = budget.check, P.patch_loss         # normal_eval logging identical to NAR-B14-01
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0)})
    try:
        with budget.phase('training', gpu=True):
            train(lk, budget)
        with budget.phase('fidelity_normal_only', gpu=True):
            fid = L.fidelity(torch.bfloat16, budget)
            R.write_json(OUT / 'FIDELITY.json', fid)
            K.require(fid['passed'], 'bf16 fidelity failed on HOPE-S experts')
        with budget.phase('scoring', gpu=True):
            L.score(torch.bfloat16, budget)
        with budget.phase('inference_cost', gpu=True):
            cost = L.cost(torch.bfloat16, budget)
        with budget.phase('metrics_a1'):
            a1 = L.metrics('smooth', budget)
        with budget.phase('metrics_raw'):
            raw = L.metrics('raw', budget)
        ref = R.read(LITE / 'RESULT.json')
        c_a1, c_raw = L.ci(a1, ref['arms']['lite_a1']), L.ci(raw, ref['arms']['lite_raw'])
        tc = R.read(OUT / 'TRAIN_COMPLETE.json')
        result = {'arms': {'hope_s_a1': a1, 'hope_s_raw': raw, 'current_best_a1': ref['arms']['lite_a1'],
                           'current_best_raw': ref['arms']['lite_raw']},
                  'hope_minus_best_a1': c_a1, 'NL_GATE': W.gate(a1, ref['arms']['lite_a1'], c_a1),
                  'hope_minus_best_raw': c_raw, 'RAW_GATE_SECONDARY': W.gate(raw, ref['arms']['lite_raw'], c_raw),
                  'slow_state': {f'stage{i:02}': {k: R.read(OUT / f'stage{i:02}.json')[k] for k in ('gamma_end',)}
                                 for i in range(1, 16)},
                  'update_seconds': tc['update_seconds'], 'cost': cost, 'fidelity': fid,
                  'exposure': 'One seed 2711, alphabetical 15, historically exposed MVTec; exploratory only'}
        verify(); P.OUT = OUT; P.verify_training()
        result['resources'] = budget.json()
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', json.dumps(c_a1), json.dumps(result['NL_GATE']))
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
