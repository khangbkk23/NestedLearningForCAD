"""TCS-00 E0: per-task surprise headroom gate on VisA (agents/tasks/task_tcs_00.md).

Necessary-condition test for a task-conditional surprise exponent. Constant gamma in {1,2,3} x seeds {2711,2712}
-> 6 chains of the 12 VisA tasks. Each chain is the F detector (warm rank-8 LoRA, lambda 0, surprise-weighted patch
loss via exps.hope_s_01.run.acquire(learn=False), bf16 lite + A1 readout, task-free min over experts).
Evaluation uses VisA-DEV only, after every chain is sealed. VisA-FINAL and MVTec are never opened.

    python -B -m exps.tcs_00.run extract     # VisA train normals only
    python -B -m exps.tcs_00.run lock
    python -B -m exps.tcs_00.run smoke --owner-authorized
    python -B -m exps.tcs_00.run execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import signal
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from exps.cadfinal import common as CF
from exps.calm02 import data as CD, native as N
from exps.dinomaly import encoder as E
from exps.hope_s_01 import run as H
from exps.nar_b14_01 import run as B
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_patch_01 import run as P
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/tcs_00'
FEAT = Path.home() / 'hope_cad_cache/tcs_00/features'
VISA = R.ROOT / 'data/visa'
ORDER = tuple(CD.VISA_ORDER)
GAMMAS = (1.0, 2.0, 3.0)
SEEDS = (2711, 2712)
N_TRAIN = 240                                       # per-class outcome-free subsample (SHA order)
DESCRIPTOR_IMAGES = 16
DEV_SHA = '5e54063005d6ed038439e7799dcda07e82ef4317f67d8baa6ac417fb72d80d4c'   # cad_final_v1 PROTOCOL_LOCK
GATE = {'headroom': .010, 'ci_lower': 0., 'argmax_agreement': 8, 'spearman': .5, 'effect_over_noise': 2.}
BOOT = {'seed': 20261117, 'B': 2000}
CAPS = {'wall_seconds': 9000, 'gpu_seconds': 7200, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 400_000_000}
FEATURE_CAP_BYTES = 8_000_000_000
SOURCES = ('exps/tcs_00/run.py', 'exps/tcs_00/test_run.py', 'exps/hope_s_01/run.py', 'exps/nar_lite_01/run.py',
           'exps/nar_b14_01/run.py', 'exps/nar_patch_01/run.py', 'exps/nar_warm_01/run.py',
           'exps/nar_pilot_00/core.py', 'exps/nar_pilot_00/full.py', 'exps/calm02/native.py', 'exps/calm02/data.py',
           'exps/cadfinal/common.py', 'exps/dinomaly/encoder.py', 'agents/tasks/task_tcs_00.md')


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES


def chain_name(gamma, seed):
    return f'g{gamma:.0f}_s{seed}'


# ---------------------------------------------------------------- identities (paths only)
def _h(salt, path):
    return hashlib.sha256((salt + path).encode()).hexdigest()


def train_rows(c, ids=None):
    """Outcome-free subsample of the VisA 1cls train normals: N_TRAIN smallest salted SHA, then path order."""
    ids = ids or CD.identities('visa')
    rows = next(t for t in ids['tasks'] if t['task'] == c)['train']
    K.require(len(rows) >= N_TRAIN, f'too few train normals: {c}')
    keep = sorted(rows, key=lambda r: _h('TCS-00/train/v1|', r['relative_path']))[:N_TRAIN]
    return sorted(keep, key=lambda r: r['relative_path'])


def roles_for(rows):
    """Same rule as K.normal_partition (salted SHA order, n = len // 5), for VisA path layout."""
    paths = [r['relative_path'] for r in rows]
    K.require(len(paths) == len(set(paths)) and len({p.split('/')[0] for p in paths}) == 1, 'roles: one task')
    order = sorted(range(len(paths)), key=lambda i: _h('NAR-PILOT-00/roles/v1|', paths[i]))
    n = max(1, len(paths) // 5)
    return {'FIT': order[2 * n:], 'CREDIT': order[n:2 * n], 'EVAL': order[:n]}


def dev_rows(ids=None):
    """VisA-DEV (frozen cad_final_v1 rule) with outcome-free halves S1/S2 per class and label."""
    ids = ids or CD.identities('visa')
    sp = CF.visa_split(ids)
    K.require(CF.text_sha('\n'.join(p for t in sp.values() for p in t['dev'])) == DEV_SHA, 'VisA-DEV identity')
    out = {}
    for t in ids['tasks']:
        dev = set(sp[t['task']]['dev'])
        rows = [dict(r) for r in t['test'] if r['relative_path'] in dev]
        for lab in (0, 1):
            group = sorted((r for r in rows if r['label'] == lab), key=lambda r: _h('TCS-00/half|', r['relative_path']))
            for k, r in enumerate(group):
                r['half'] = 'S1' if k < len(group) // 2 else 'S2'
        out[t['task']] = rows
    return out


# ---------------------------------------------------------------- normal-only feature extraction
def feature_bytes():
    return sum(p.stat().st_size for p in FEAT.glob('*.pt')) if FEAT.exists() else 0


@torch.no_grad()
def extract():
    R.configure()
    K.require(R.sha(B.ENC_PATH) == B.ENC_SHA, 'DINOv2 checkpoint hash')
    K.require(not (OUT / 'LOCK.json').exists(), 'locked; preserve caches')
    enc = E.load(B.ENC_PATH, 'base', 'cuda')
    ids = CD.identities('visa')
    for c in ORDER:
        target = FEAT / f'{c}_train.pt'
        if target.exists():
            continue
        rows = train_rows(c, ids)
        feats = torch.empty(len(rows), 784, 768, dtype=torch.float32)
        t0 = time.monotonic()
        for i in range(0, len(rows), 8):
            x = torch.stack([E.preprocess(VISA / r['relative_path']) for r in rows[i:i + 8]]).cuda()
            feats[i:i + len(x)] = B.tokens(enc, x).cpu()
        K.require(bool(torch.isfinite(feats).all()), 'feature finiteness')
        target.parent.mkdir(parents=True, exist_ok=True)
        R.save(target, {'rows': rows, 'features': feats})
        R.write_json(target.with_suffix('.json'), {
            'sha256': R.sha(target), 'n': len(rows), 'category': c, 'seconds': time.monotonic() - t0,
            'encoder_sha256': B.ENC_SHA, 'front_end': 'dinov2_vitb14_reg4; 392; mean of blocks 6-9; patch tokens',
            'source_images': {r['relative_path']: R.sha(VISA / r['relative_path']) for r in rows}})
        R.log('EXTRACT', c, len(rows), round(time.monotonic() - t0, 1), 's')
        K.require(feature_bytes() < FEATURE_CAP_BYTES, 'feature disk cap')


def load_train(lk, c):
    d = lk['inputs'][c]
    K.require(R.sha(Path(d['path'])) == d['sha256'], f'train cache changed: {c}')
    data = torch.load(d['path'], weights_only=True, mmap=True)
    K.require([r['relative_path'] for r in data['rows']] == [r['relative_path'] for r in train_rows(c)], 'rows')
    K.require(tuple(data['features'].shape) == (N_TRAIN, 784, 768), 'cache geometry')
    return data


# ---------------------------------------------------------------- lock
def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    inputs = {}
    for c in ORDER:
        p = FEAT / f'{c}_train.pt'
        m = R.read(p.with_suffix('.json'))
        K.require(R.sha(p) == m['sha256'] and m['encoder_sha256'] == B.ENC_SHA, f'train cache {c}')
        inputs[c] = {'path': str(p), 'sha256': m['sha256'], 'metadata_sha256': R.sha(p.with_suffix('.json'))}
    dev_rows()                                     # identity check only (CSV paths); no image or mask opened
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'inputs': inputs,
        'base_sha256': R.sha(W.HIST / 'BASE.pt'), 'encoder_sha256': B.ENC_SHA, 'cfg': R.CFG.dict(),
        'order': list(ORDER), 'gammas': list(GAMMAS), 'seeds': list(SEEDS), 'n_train': N_TRAIN,
        'dev_sha256': DEV_SHA, 'gate': GATE, 'bootstrap': BOOT, 'caps': CAPS,
        'statistic': 'cross-fitted per-class argmax gamma vs constant argmax, select on (seed, half), evaluate on '
                     '(other seed, other half); 4 pairs; 12-class bootstrap',
        'readout': 'A1 (bilinear 392, Gaussian sigma 4, native); image = top-1% mean; task-free min over experts',
        'deployment': 'bf16 lite if normal-only fidelity passes on every chain, else fp32',
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'], 'base hash')
    K.require(lk['caps'] == CAPS and lk['gammas'] == list(GAMMAS) and lk['seeds'] == list(SEEDS), 'config')
    return lk


# ---------------------------------------------------------------- training (normal only)
@torch.no_grad()
def descriptor(model, feats, fit):
    """Task-start normal-only tail shape d = mean ln(top-1% / mean) of warm-init patch errors (E1 input only)."""
    model.eval()
    vals = []
    for j in fit[:DESCRIPTOR_IMAGES]:
        e = H.patch_errors(model, feats[j:j + 1].cuda())
        vals.append(float(torch.log(e.topk(H.TAIL_K, dim=1).values.mean() / e.mean())))
    model.train()
    return float(np.mean(vals))


def train_chain(lk, gamma, seed, budget, cfg=R.CFG, tasks=ORDER, save=True):
    original = R.SEED
    R.SEED = seed                                    # LoRA init + FIT draws; frozen BASE is shared
    try:
        base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
        model, loras, _ = R.model_new(base)
        fixed = K.init_state(loras, seed)
        lam = torch.zeros(len(loras), dtype=torch.float64)
        archive, records = [], []
        for stage, c in enumerate(tasks):
            data = load_train(lk, c)
            roles = roles_for(data['rows'])
            init = W.initial_state(fixed, archive)
            K.set_state(loras, init)
            d = descriptor(model, data['features'], roles['FIT'])
            lam_out, g_out, trace, wall = H.acquire(model, loras, data['features'], roles, stage, lam, gamma, cfg=cfg,
                                                    learn=False, budget=budget)
            K.require(torch.equal(lam_out, lam) and g_out == gamma, 'constant policy changed')
            K.require(len(trace['fit']) == cfg.steps and len(trace['h_gamma']) == cfg.steps // cfg.m, 'budget')
            endpoint = K.get_state(loras)
            archive.append(endpoint)
            records.append({'stage': stage + 1, 'category': c, 'seconds': wall, 'descriptor': d, 'trace': trace,
                            'initial_digest': W.digest(init), 'endpoint_digest': W.digest(endpoint)})
            R.log('TRAIN', chain_name(gamma, seed), stage + 1, c, 'fit', round(float(np.mean(trace['fit'][-10:])), 4),
                  'tail', round(trace['credit'][-1], 4), 'd', round(d, 3), 's', round(wall, 1))
            del data
            if budget is not None:
                budget.check()
        if save:
            name = chain_name(gamma, seed)
            W.bounded_save(OUT / f'{name}.pt', {'archive': archive, 'gamma': gamma, 'seed': seed, 'lam': lam,
                                               'cfg': cfg.dict(), 'base_sha256': lk['base_sha256']},
                           sum(K.tensor_state_bytes(a) for a in archive), budget)
            R.write_json(OUT / f'{name}.json', {'gamma': gamma, 'seed': seed, 'records': records,
                                                'state_sha256': R.sha(OUT / f'{name}.pt')})
        del model, loras
        gc.collect(); torch.cuda.empty_cache()
        return records
    finally:
        R.SEED = original


def verify_training():
    tc = R.read(OUT / 'TRAIN_COMPLETE.json')
    for f, h in tc['files'].items():
        K.require(R.sha(OUT / f) == h, 'sealed chain changed: ' + f)
    K.require(len(tc['files']) == 2 * len(GAMMAS) * len(SEEDS), 'all chains sealed before DEV access')
    return tc


# ---------------------------------------------------------------- deployment precision (normal only)
def chain_detector(name, dtype):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    st = torch.load(OUT / f'{name}.pt', weights_only=True)
    return model, loras, [{n: (a.to(dtype), b.to(dtype)) for n, (a, b) in s.items()} for s in st['archive']]


@torch.no_grad()
def fidelity(lk, dtype, budget):
    """Lite bf16 pipeline from images vs fp32 decoder on the sealed fp32 train tokens: 4 EVAL normals per class."""
    enc = L.lite_encoder(dtype)
    x16 = {}
    for c in ORDER:
        data = load_train(lk, c)
        idx = roles_for(data['rows'])['EVAL'][:4]
        x = torch.stack([E.preprocess(VISA / data['rows'][i]['relative_path']) for i in idx]).cuda().to(dtype)
        x16[c] = (data['features'][idx].clone(), L.tokens(enc, x).cpu())
    del enc
    out = {}
    for g in GAMMAS:
        for s in SEEDS:
            name = chain_name(g, s)
            m32, l32, a32 = chain_detector(name, torch.float32)
            m16, l16, a16 = chain_detector(name, dtype)
            num = den = 0.
            try:
                for c in ORDER:
                    f32, f16 = x16[c]
                    g32 = L.expert_grids(m32, l32, a32, f32, torch.float32).min(0).values
                    g16 = L.expert_grids(m16, l16, a16, f16, dtype).min(0).values
                    num += float((g16 - g32).abs().sum()); den += float(g32.abs().sum())
                rel = num / den
                out[name] = {'relative_mean_abs_error': rel, 'passed': bool(np.isfinite(rel)) and rel <= L.FIDELITY_TOL}
            except RuntimeError as exc:
                out[name] = {'failure': repr(exc), 'passed': False}
            del m32, l32, a32, m16, l16, a16
            gc.collect(); torch.cuda.empty_cache(); budget.check()
    return {'dtype': str(dtype), 'chains': out, 'passed': all(v['passed'] for v in out.values())}


# ---------------------------------------------------------------- VisA-DEV scoring (after the training seal)
@torch.no_grad()
def score(dtype, budget):
    verify_training()
    dev = dev_rows()
    enc = L.lite_encoder(dtype)
    feats = {}
    for c in ORDER:
        rows = dev[c]
        f = torch.empty(len(rows), 784, 768, dtype=dtype)
        for i in range(0, len(rows), 8):
            x = torch.stack([E.preprocess(VISA / r['relative_path']) for r in rows[i:i + 8]]).cuda().to(dtype)
            f[i:i + len(x)] = L.tokens(enc, x).cpu()
        K.require(bool(torch.isfinite(f).all()), 'feature finiteness')
        feats[c] = f
    del enc
    gc.collect(); torch.cuda.empty_cache()
    files = {}
    for g in GAMMAS:
        for s in SEEDS:
            name = chain_name(g, s)
            model, loras, archive = chain_detector(name, dtype)
            grids = {c: L.expert_grids(model, loras, archive, feats[c], dtype).min(0).values for c in ORDER}
            path = OUT / f'scores_{name}.pt'
            W.bounded_save(path, {'grids': grids, 'rows': dev, 'dtype': str(dtype)}, 12 * 20 * 784 * 4, budget)
            files[path.name] = R.sha(path)
            R.log('SCORE', name)
            del model, loras, archive
            gc.collect(); torch.cuda.empty_cache(); budget.check()
    R.write_json(OUT / 'SCORES_COMPLETE.json', {'files': files, 'train_complete_sha256': R.sha(OUT / 'TRAIN_COMPLETE.json')})


def a1_metrics(sm, hws, masks, labels):
    from sklearn.metrics import roc_auc_score
    k = max(1, int(L.TOP * E.SIZE * E.SIZE))
    image = sm.flatten(1).topk(k, dim=1).values.mean(1).numpy()
    original = N.native_map
    N.native_map = L.map_native
    try:
        ap = N.native_ap(list(sm), hws, masks)['ap']
    finally:
        N.native_map = original
    return ap, float(roc_auc_score(labels, image))


def metrics(budget):
    scored = R.read(OUT / 'SCORES_COMPLETE.json')
    dev = dev_rows()
    geo = {}
    for c in ORDER:
        hws = [N.native_hw(VISA / r['relative_path']) for r in dev[c]]
        geo[c] = (hws, [N.native_mask(VISA / r['mask_path'] if r['mask_path'] else None, hw)
                        for r, hw in zip(dev[c], hws)])
    cells = {}
    for g in GAMMAS:
        for s in SEEDS:
            name = chain_name(g, s)
            path = OUT / f'scores_{name}.pt'
            K.require(R.sha(path) == scored['files'][path.name], 'score hash')
            grids = torch.load(path, weights_only=True)['grids']
            for c in ORDER:
                sm = L.smooth392(grids[c]); hws, masks = geo[c]; rows = dev[c]
                for h in ('full', 'S1', 'S2'):
                    idx = [i for i, r in enumerate(rows) if h == 'full' or r['half'] == h]
                    ap, auc = a1_metrics(sm[idx], [hws[i] for i in idx], [masks[i] for i in idx],
                                         [rows[i]['label'] for i in idx])
                    cells[f'{name}|{c}|{h}'] = {'pixel_ap': ap, 'image_auroc': auc, 'n': len(idx)}
                R.log('METRIC', name, c, round(cells[f'{name}|{c}|full']['pixel_ap'], 4))
                budget.check()
    R.write_json(OUT / 'CELLS.json', cells)
    return cells


# ---------------------------------------------------------------- headroom statistics (pure functions)
def table(cells, key='pixel_ap'):
    """AP[seed][gamma][class][half]."""
    return {s: {g: {c: {h: cells[f'{chain_name(g, s)}|{c}|{h}'][key] for h in ('full', 'S1', 'S2')}
                    for c in ORDER} for g in GAMMAS} for s in SEEDS}


def argmax_gamma(values):
    return max(GAMMAS, key=lambda g: (values[g], -g))           # ties -> smaller gamma (fixed rule)


def crossfit(ap, classes=ORDER):
    """Per class: mean over the 4 (select seed,half) -> (other seed, other half) pairs of conditional and constant AP."""
    pairs = [(s, h, s2, h2) for s in SEEDS for h in ('S1', 'S2') for s2 in SEEDS for h2 in ('S1', 'S2')
             if s2 != s and h2 != h]
    cond = {c: [] for c in classes}; const = {c: [] for c in classes}; picks = []
    for s, h, s2, h2 in pairs:
        gc_ = argmax_gamma({g: float(np.mean([ap[s][g][c][h] for c in classes])) for g in GAMMAS})
        for c in classes:
            gk = argmax_gamma({g: ap[s][g][c][h] for g in GAMMAS})
            cond[c].append(ap[s2][gk][c][h2]); const[c].append(ap[s2][gc_][c][h2])
            picks.append({'select': [s, h], 'eval': [s2, h2], 'class': c, 'gamma_class': gk, 'gamma_const': gc_})
    return ({c: float(np.mean(v)) for c, v in cond.items()}, {c: float(np.mean(v)) for c, v in const.items()},
            pairs, picks)


def headroom(ap):
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    from scipy.stats import spearmanr
    cond, const, pairs, picks = crossfit(ap)
    ci = bootstrap_delta([cond[c] for c in ORDER], [const[c] for c in ORDER], seed=BOOT['seed'], B=BOOT['B'])
    s1, s2 = SEEDS
    best = {s: {c: argmax_gamma({g: ap[s][g][c]['full'] for g in GAMMAS}) for c in ORDER} for s in SEEDS}
    agree = sum(best[s1][c] == best[s2][c] for c in ORDER)
    slope = {s: [ap[s][GAMMAS[-1]][c]['full'] - ap[s][GAMMAS[0]][c]['full'] for c in ORDER] for s in SEEDS}
    rho = float(spearmanr(slope[s1], slope[s2]).statistic)
    mid = GAMMAS[len(GAMMAS) // 2]
    noise = float(np.median([abs(ap[s1][mid][c]['full'] - ap[s2][mid][c]['full']) for c in ORDER]))
    effect = float(np.median([np.mean([max(ap[s][g][c]['full'] for g in GAMMAS) - min(ap[s][g][c]['full'] for g in GAMMAS)
                                       for s in SEEDS]) for c in ORDER]))
    guards = {'headroom': ci['point'] >= GATE['headroom'], 'ci_lower': ci['lo'] > GATE['ci_lower'],
              'argmax_agreement': agree >= GATE['argmax_agreement'], 'spearman': bool(rho >= GATE['spearman']),
              'effect_over_noise': effect > GATE['effect_over_noise'] * noise}
    return {'H': ci, 'conditional_per_class': cond, 'constant_per_class': const, 'pairs': pairs, 'picks': picks,
            'best_gamma_full': {str(s): best[s] for s in SEEDS}, 'argmax_agreement': agree,
            'spearman_slope_across_seeds': rho, 'slope_per_class': {str(s): slope[s] for s in SEEDS},
            'seed_noise_median': noise, 'gamma_effect_median': effect, 'guards': guards,
            'verdict': 'GO_TO_E1' if all(guards.values()) else 'NO_TASK_CONDITIONAL_HEADROOM'}


def macro(cells):
    return {chain_name(g, s): {k: float(np.mean([cells[f'{chain_name(g, s)}|{c}|full'][k] for c in ORDER]))
                               for k in ('pixel_ap', 'image_auroc')} for g in GAMMAS for s in SEEDS}


def online_signal(ap):
    """Descriptive: does the online normal-only gamma signal (mean h_gamma at gamma=2) track DEV preference?"""
    from scipy.stats import spearmanr
    mid = GAMMAS[len(GAMMAS) // 2]
    h = {c: [] for c in ORDER}
    for s in SEEDS:
        for rec in R.read(OUT / f'{chain_name(mid, s)}.json')['records']:
            h[rec['category']].append(float(np.mean(rec['trace']['h_gamma'])))
    pref = [np.mean([ap[s][GAMMAS[-1]][c]['full'] - ap[s][GAMMAS[0]][c]['full'] for s in SEEDS]) for c in ORDER]
    hh = [float(np.mean(h[c])) for c in ORDER]
    return {'mean_h_gamma_at_gamma2': dict(zip(ORDER, hh)), 'preference_ap3_minus_ap1': dict(zip(ORDER, map(float, pref))),
            'spearman': float(spearmanr(hh, pref).statistic)}


# ---------------------------------------------------------------- smoke / execute
def smoke():
    bind()
    lk = verify()
    K.require(not (OUT / 'SMOKE.json').exists(), 'smoke done; preserve')
    R.configure()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    recs = train_chain(lk, 2.0, SEEDS[0], None, cfg=replace(R.CFG, steps=30), tasks=ORDER[:1], save=False)
    sec = recs[0]['seconds']
    projected = sec * 10 * len(ORDER) * len(GAMMAS) * len(SEEDS)
    out = {'seconds_30_steps': sec, 'wall': time.monotonic() - t0, 'projected_training_seconds': projected,
           'cuda_peak': torch.cuda.max_memory_allocated(), 'descriptor': recs[0]['descriptor'],
           'passed': projected < .6 * CAPS['gpu_seconds'] and torch.cuda.max_memory_allocated() < CAPS['cuda_bytes']}
    R.write_json(OUT / 'SMOKE.json', out)
    R.log('SMOKE', json.dumps(out))


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    K.require(R.read(OUT / 'SMOKE.json')['passed'], 'smoke failed')
    budget = W.Budget()
    lk = verify()
    R.configure()
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0)})
    try:
        with budget.phase('training', gpu=True):
            for g in GAMMAS:
                for s in SEEDS:
                    train_chain(lk, g, s, budget)
            R.write_json(OUT / 'TRAIN_COMPLETE.json', {
                'lock_sha256': R.sha(OUT / 'LOCK.json'),
                'files': {p.name: R.sha(p) for p in sorted(OUT.glob('g*_s*.*'))},
                'resource_at_seal': budget.json()})
        with budget.phase('fidelity_normal_only', gpu=True):
            fid = fidelity(lk, torch.bfloat16, budget)
            R.write_json(OUT / 'FIDELITY.json', fid)
        dtype = torch.bfloat16 if fid['passed'] else torch.float32
        with budget.phase('dev_scoring', gpu=True):
            score(dtype, budget)
        with budget.phase('metrics'):
            cells = metrics(budget)
        ap = table(cells)
        result = {'E0': headroom(ap), 'macro_full_dev': macro(cells),
                  'image_auroc_table': table(cells, 'image_auroc'), 'pixel_ap_table': ap,
                  'online_signal_descriptive': online_signal(ap), 'deployment_dtype': str(dtype), 'fidelity': fid,
                  'exposure': 'VisA-DEV (240 images) consumed for the first time by this run; VisA-FINAL sealed; '
                              'MVTec not accessed',
                  'descriptors_recorded_for_E1': 'per stage in g*_s*.json; analysed only if E0 is GO'}
        verify(); verify_training()
        result['resources'] = budget.json()
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', result['E0']['verdict'], json.dumps(result['E0']['H']), json.dumps(result['E0']['guards']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract', 'lock', 'smoke', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase in ('smoke', 'execute'):
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'extract': extract, 'lock': lock, 'smoke': smoke, 'execute': execute}[a.phase]()
