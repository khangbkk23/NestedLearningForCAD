"""VBENCH-00: one-shot full-VisA benchmark of the fixed hard-patch detector F, paper protocol (agents/tasks/task_vbench_00.md).

12 VisA tasks (alphabetical), all 1cls train normals, all 2,162 official test images. Configuration F is frozen
before any VisA test access and independent of TCS-00: gamma 2.1, lambda 0, warm rank-8 LoRA, 300 steps per task,
seed 2711, bf16 lite (if normal-only fidelity passes), A1 readout, task-free min over experts.
Reported on the full test set (paper protocol) and on VisA-FINAL only (DEV excluded).

    python -B -m exps.vbench_00.run extract
    python -B -m exps.vbench_00.run lock
    python -B -m exps.vbench_00.run execute --owner-authorized
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import time
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
from exps.nar_pilot_00.detector import fm_eq8, prefix_grids
from exps.nar_warm_01 import run as W
from exps.tcs_00 import run as T

OUT = R.ROOT / 'results/hope_cad/vbench_00'
FEAT = Path.home() / 'hope_cad_cache/vbench_00/features'
VISA = T.VISA
ORDER = T.ORDER
GAMMA, SEED = 2.1, 2711
CAPS = {'wall_seconds': 16200, 'gpu_seconds': 5400, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 400_000_000}
FEATURE_CAP_BYTES = 24_000_000_000
SOURCES = ('exps/vbench_00/run.py', 'exps/vbench_00/test_run.py', 'exps/tcs_00/run.py', 'exps/hope_s_01/run.py',
           'exps/nar_lite_01/run.py', 'exps/nar_b14_01/run.py', 'exps/nar_warm_01/run.py', 'exps/nar_pilot_00/core.py',
           'exps/nar_pilot_00/full.py', 'exps/nar_pilot_00/detector.py', 'exps/calm02/native.py',
           'exps/calm02/data.py', 'exps/cadfinal/common.py', 'exps/dinomaly/encoder.py', 'agents/tasks/task_vbench_00.md')


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES


def train_rows(c, ids=None):
    ids = ids or CD.identities('visa')
    return next(t for t in ids['tasks'] if t['task'] == c)['train']


def test_rows(ids=None):
    ids = ids or CD.identities('visa')
    sp = CF.visa_split(ids)
    out = {}
    for t in ids['tasks']:
        dev = set(sp[t['task']]['dev'])
        out[t['task']] = [dict(r, dev=r['relative_path'] in dev) for r in t['test']]
    return out


def test_manifest_sha(rows):
    return CF.text_sha('\n'.join(r['relative_path'] for c in ORDER for r in rows[c]))


# ---------------------------------------------------------------- normal-only extraction
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
            'encoder_sha256': B.ENC_SHA, 'source_images': {r['relative_path']: R.sha(VISA / r['relative_path'])
                                                           for r in rows}})
        R.log('EXTRACT', c, len(rows), round(time.monotonic() - t0, 1), 's')
        del feats
        gc.collect()
        K.require(feature_bytes() < FEATURE_CAP_BYTES, 'feature disk cap')


def load_train(lk, c):
    d = lk['inputs'][c]
    K.require(R.sha(Path(d['path'])) == d['sha256'], f'train cache changed: {c}')
    data = torch.load(d['path'], weights_only=True, mmap=True)
    K.require([r['relative_path'] for r in data['rows']] == [r['relative_path'] for r in train_rows(c)], 'rows')
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
        inputs[c] = {'path': str(p), 'sha256': m['sha256'], 'n': m['n']}
    rows = test_rows()                               # CSV identities only; no test image or mask opened
    K.require(sum(len(v) for v in rows.values()) == 2162 and sum(r['dev'] for v in rows.values() for r in v) == 240,
              'VisA test / DEV counts')
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES}, 'inputs': inputs,
        'base_sha256': R.sha(W.HIST / 'BASE.pt'), 'encoder_sha256': B.ENC_SHA, 'cfg': R.CFG.dict(),
        'gamma': GAMMA, 'seed': SEED, 'lambda': 0, 'order': list(ORDER), 'caps': CAPS,
        'test_manifest_sha256': test_manifest_sha(rows), 'dev_sha256': T.DEV_SHA,
        'tcs_00_result_exists_at_lock': (T.OUT / 'RESULT.json').exists(),
        'configuration_provenance': 'F = HOPE-S-02 fixed-gamma detector, frozen before any VisA test access; '
                                    'not selected on VisA',
        'metrics': {'primary': 'native Pixel AP (A1 readout) and Image AUROC (top-1% mean), final after task 12',
                    'also': 'acquisition AP, signed Eq.8 FM (pixel, image); full test (paper) and VisA-FINAL only'},
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'], 'base hash')
    K.require(lk['caps'] == CAPS and lk['gamma'] == GAMMA and lk['seed'] == SEED, 'config')
    return lk


# ---------------------------------------------------------------- train (normal only)
def train(lk, budget):
    original = R.SEED
    R.SEED = SEED
    try:
        base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
        model, loras, _ = R.model_new(base)
        fixed = K.init_state(loras, SEED)
        lam = torch.zeros(len(loras), dtype=torch.float64)
        archive, records = [], []
        for stage, c in enumerate(ORDER):
            data = load_train(lk, c)
            roles = T.roles_for(data['rows'])
            K.set_state(loras, W.initial_state(fixed, archive))
            lam_out, g_out, trace, wall = H.acquire(model, loras, data['features'], roles, stage, lam, GAMMA,
                                                    learn=False, budget=budget)
            K.require(torch.equal(lam_out, lam) and g_out == GAMMA, 'fixed policy changed')
            archive.append(K.get_state(loras))
            records.append({'stage': stage + 1, 'category': c, 'seconds': wall,
                            'fit_last10': float(np.mean(trace['fit'][-10:])), 'tail_last': trace['credit'][-1]})
            R.log('TRAIN', stage + 1, c, 'fit', round(records[-1]['fit_last10'], 4), 's', round(wall, 1))
            del data
            budget.check()
        W.bounded_save(OUT / 'final.pt', {'archive': archive, 'gamma': GAMMA, 'seed': SEED, 'lam': lam, 'phi': {},
                                          'cfg': R.CFG.dict(), 'base_sha256': lk['base_sha256']},
                       sum(K.tensor_state_bytes(a) for a in archive), budget)
        R.write_json(OUT / 'TRAIN_COMPLETE.json', {
            'lock_sha256': R.sha(OUT / 'LOCK.json'), 'final_sha256': R.sha(OUT / 'final.pt'), 'records': records,
            'update_seconds': sum(r['seconds'] for r in records), 'resource_at_seal': budget.json()})
        del model, loras
        gc.collect(); torch.cuda.empty_cache()
    finally:
        R.SEED = original


def verify_training():
    tc = R.read(OUT / 'TRAIN_COMPLETE.json')
    K.require(R.sha(OUT / 'final.pt') == tc['final_sha256'], 'sealed checkpoint changed')
    return tc


def detector(dtype):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    st = torch.load(OUT / 'final.pt', weights_only=True)
    return model, loras, [{n: (a.to(dtype), b.to(dtype)) for n, (a, b) in s.items()} for s in st['archive']], st


@torch.no_grad()
def fidelity(lk, dtype, budget):
    enc = L.lite_encoder(dtype)
    m32, l32, a32, _ = detector(torch.float32)
    m16, l16, a16, _ = detector(dtype)
    num = den = 0.
    try:
        for c in ORDER:
            data = load_train(lk, c)
            idx = T.roles_for(data['rows'])['EVAL'][:4]
            x = torch.stack([E.preprocess(VISA / data['rows'][i]['relative_path']) for i in idx]).cuda().to(dtype)
            g32 = L.expert_grids(m32, l32, a32, data['features'][idx].clone(), torch.float32).min(0).values
            g16 = L.expert_grids(m16, l16, a16, L.tokens(enc, x).cpu(), dtype).min(0).values
            num += float((g16 - g32).abs().sum()); den += float(g32.abs().sum())
            budget.check()
        rel = num / den
        res = {'relative_mean_abs_error': rel, 'passed': bool(np.isfinite(rel)) and rel <= L.FIDELITY_TOL}
    except RuntimeError as exc:
        res = {'failure': repr(exc), 'passed': False}
    del enc, m32, l32, a32, m16, l16, a16
    gc.collect(); torch.cuda.empty_cache()
    return dict(res, dtype=str(dtype))


# ---------------------------------------------------------------- scoring (after the seal)
@torch.no_grad()
def score(dtype, budget):
    verify_training()
    rows = test_rows()
    enc = L.lite_encoder(dtype)
    model, loras, archive, _ = detector(dtype)
    files = {}
    for c in ORDER:
        rs = rows[c]
        f = torch.empty(len(rs), 784, 768, dtype=dtype)
        for i in range(0, len(rs), 8):
            x = torch.stack([E.preprocess(VISA / r['relative_path']) for r in rs[i:i + 8]]).cuda().to(dtype)
            f[i:i + len(x)] = L.tokens(enc, x).cpu()
        K.require(bool(torch.isfinite(f).all()), 'feature finiteness')
        prefixes = prefix_grids(L.expert_grids(model, loras, archive, f, dtype))
        path = OUT / f'scores_{c}.pt'
        W.bounded_save(path, {'prefix_grids': prefixes, 'rows': rs, 'dtype': str(dtype)}, prefixes.numel() * 4, budget)
        files[path.name] = R.sha(path)
        R.log('SCORE', c, len(rs)); budget.check()
        del f, prefixes
    R.write_json(OUT / 'SCORES_COMPLETE.json', {'files': files, 'train_complete_sha256': R.sha(OUT / 'TRAIN_COMPLETE.json')})
    del enc, model, loras
    gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def cost(dtype, budget):
    enc = L.lite_encoder(dtype)
    model, loras, archive, st = detector(dtype)
    rows = train_rows(ORDER[0])[:23]
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i, r in enumerate(rows):
        t = time.perf_counter()
        x = E.preprocess(VISA / r['relative_path'])[None].cuda().to(dtype)
        g = L.expert_grids(model, loras, archive, L.tokens(enc, x).cpu(), dtype).min(0).values
        L.smooth392(g); torch.cuda.synchronize()
        if i >= 3:
            times.append(time.perf_counter() - t)
    parts = {'encoder_blocks_0_9': sum(x.numel() * x.element_size() for x in enc.state_dict().values()),
             'shared_decoder': sum(p.numel() * p.element_size() for n, p in model.state_dict().items()
                                   if not n.endswith(('.A', '.B'))),
             'archive': sum(K.tensor_state_bytes(a) for a in archive)}
    del enc, model, loras
    gc.collect(); torch.cuda.empty_cache()
    return {'latency_mean_s': float(np.mean(times)), 'latency_p95_s': float(np.quantile(times, .95)),
            'inference_cuda_peak_bytes': torch.cuda.max_memory_allocated(), 'persistent_bytes': parts,
            'total_persistent_bytes': sum(parts.values()), 'note': 'per image, 12 experts, preprocessing + encoder + '
            'decoders + A1 smoothing; native resize excluded'}


# ---------------------------------------------------------------- metrics
def metrics(budget):
    scored = R.read(OUT / 'SCORES_COMPLETE.json')
    cells, final_only = {}, {}
    for i, c in enumerate(ORDER):
        path = OUT / f'scores_{c}.pt'
        K.require(R.sha(path) == scored['files'][path.name], 'score hash')
        data = torch.load(path, weights_only=True); rows = data['rows']; grids = data['prefix_grids']
        hws = [N.native_hw(VISA / r['relative_path']) for r in rows]
        masks = [N.native_mask(VISA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = [r['label'] for r in rows]
        for t in range(i, len(ORDER)):
            sm = L.smooth392(grids[t])
            ap, auc = T.a1_metrics(sm, hws, masks, labels)
            cells[t, c] = {'stage': t + 1, 'category': c, 'pixel_ap': ap, 'image_auroc': auc}
            if t == len(ORDER) - 1:
                idx = [k for k, r in enumerate(rows) if not r['dev']]
                fap, fauc = T.a1_metrics(sm[idx], [hws[k] for k in idx], [masks[k] for k in idx], [labels[k] for k in idx])
                final_only[c] = {'pixel_ap': fap, 'image_auroc': fauc, 'n': len(idx)}
            R.log('METRIC', t + 1, c, round(ap, 4), round(auc, 4)); budget.check()
            del sm
        del data, grids, masks
        gc.collect()
    R.write_json(OUT / 'CELLS.json', {f'{t + 1:02}_{c}': v for (t, c), v in cells.items()})
    last = len(ORDER) - 1
    per = {c: {'acquisition': cells[i, c], 'final': cells[last, c], 'final_visa_final_only': final_only[c]}
           for i, c in enumerate(ORDER)}
    mean = lambda f: float(np.mean([f(c) for c in ORDER]))
    return {'per_category': per,
            'full_test': {'final_pixel_ap': mean(lambda c: per[c]['final']['pixel_ap']),
                          'final_image_auroc': mean(lambda c: per[c]['final']['image_auroc']),
                          'acquisition_pixel_ap': mean(lambda c: per[c]['acquisition']['pixel_ap']),
                          'acquisition_image_auroc': mean(lambda c: per[c]['acquisition']['image_auroc']),
                          'pixel_fm_eq8': fm_eq8(cells, ORDER, 'pixel_ap'),
                          'image_fm_eq8': fm_eq8(cells, ORDER, 'image_auroc')},
            'visa_final_only': {'final_pixel_ap': mean(lambda c: final_only[c]['pixel_ap']),
                                'final_image_auroc': mean(lambda c: final_only[c]['image_auroc'])}}


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = verify()
    R.configure()
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0)})
    try:
        with budget.phase('training', gpu=True):
            train(lk, budget)
        with budget.phase('fidelity_normal_only', gpu=True):
            fid = fidelity(lk, torch.bfloat16, budget)
            R.write_json(OUT / 'FIDELITY.json', fid)
        dtype = torch.bfloat16 if fid['passed'] else torch.float32
        with budget.phase('scoring', gpu=True):
            score(dtype, budget)
        with budget.phase('inference_cost', gpu=True):
            cst = cost(dtype, budget)
        with budget.phase('metrics'):
            res = metrics(budget)
        verify(); verify_training()
        result = {'arm': 'F (gamma 2.1, lambda 0, warm rank-8 LoRA, seed 2711)', **res, 'cost': cst,
                  'update_seconds': R.read(OUT / 'TRAIN_COMPLETE.json')['update_seconds'], 'fidelity': fid,
                  'deployment_dtype': str(dtype),
                  'exposure': 'First access to VisA test images/masks except the 240 DEV images consumed by TCS-00. '
                              'Full-test numbers include DEV (paper protocol); VisA-FINAL-only excludes it. '
                              'One seed, one order.',
                  'resources': budget.json()}
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', json.dumps(res['full_test'], default=str), json.dumps(res['visa_final_only']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract', 'lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'extract': extract, 'lock': lock, 'execute': execute}[a.phase]()
