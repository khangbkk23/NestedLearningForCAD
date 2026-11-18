"""NAR-B14-01: the current best NAR configuration (warm-start + mean patchwise cosine, NAR-PATCH-01) with the
front end swapped from ViT-B/8 (224, block 9) to DINOv2-reg ViT-B/14 (392 full-view resize, mean of blocks 6-9,
patch tokens only, 28x28 = 784 tokens x 768). Everything downstream is unchanged: frozen random BASE, rank 8 / 34
sites, seed 2711, alphabetical 15 tasks, FIT/CREDIT/EVAL identities, 300 fast / 30 CREDIT, lambda 0, no Reptile,
canonical task-free 15-expert minimum scorer, native Pixel AP and grid-max Image AUROC.

    python -B -m exps.nar_b14_01.run extract-train      # normal train/good only (before the lock)
    python -B -m exps.nar_b14_01.run lock
    python -B -m exps.nar_b14_01.run execute --owner-authorized

Test features are extracted only after all 15 checkpoints are sealed. Comparator: sealed NAR-PATCH-01 (read only).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from exps.dinomaly import encoder as E
from exps.dinomaly.common import CKPT
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.cost import quantiles
from exps.nar_pilot_00.detector import Detector
from exps.nar_patch_01 import run as P
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/nar_b14_01'
FEAT = Path.home() / 'hope_cad_cache/nar_b14_01/features'
REF = R.ROOT / 'results/hope_cad/nar_patch_01'
ENC_PATH, ENC_SHA = CKPT['base']
GROUP = 1                        # E.GROUPS[1] = blocks (6, 7, 8, 9)
CAPS = {'wall_seconds': 3600, 'gpu_seconds': 2700, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 700_000_000}
FEATURE_CAP_BYTES = 14_000_000_000
SOURCES = ('exps/nar_b14_01/run.py', 'exps/nar_b14_01/test_run.py', 'exps/nar_patch_01/run.py',
           'exps/nar_warm_01/run.py', 'exps/nar_pilot_00/cost.py', 'exps/nl_sgm_01a/am03_eval.py',
           'exps/dinomaly/encoder.py', 'agents/tasks/task_nar_b14_01.md')


def bind():
    W.OUT = OUT
    W.CAPS = CAPS
    W.SOURCES = SOURCES
    P.OUT = OUT
    W.verify_training = P.verify_training


def tokens(enc, x):
    """[B,3,392,392] -> [B,784,768]: group-2 mean of raw block outputs, prefix (CLS + 4 registers) removed."""
    g = enc.groups(x)
    K.require(tuple(g.shape[1:]) == (2, 789, 768), 'DINOv2 group shape')
    return g[:, GROUP, enc.num_prefix:, :].float().contiguous()


@torch.no_grad()
def extract(c, split, enc):
    target = FEAT / f'{c}_{split}.pt'
    meta = target.with_suffix('.json')
    K.require(not target.exists() and not meta.exists(), f'cache exists, preserve: {target}')
    rows = R.rows_for(c, split)
    if split == 'train':
        K.normal_partition(rows)
    feats = torch.empty(len(rows), 784, 768, dtype=torch.float32)
    t0 = time.monotonic()
    for i in range(0, len(rows), 8):
        x = torch.stack([E.preprocess(R.DATA / r['relative_path']) for r in rows[i:i + 8]]).cuda()
        feats[i:i + len(x)] = tokens(enc, x).cpu()
    K.require(bool(torch.isfinite(feats).all()), 'feature finiteness')
    target.parent.mkdir(parents=True, exist_ok=True)
    R.save(target, {'rows': rows, 'features': feats})
    rec = {'sha256': R.sha(target), 'n': len(rows), 'split': split, 'category': c, 'seconds': time.monotonic() - t0,
           'encoder_sha256': ENC_SHA, 'front_end': 'dinov2_vitb14_reg4; 392 full-view bilinear; ImageNet mean/std; '
           'mean of blocks 6-9; patch tokens', 'source_images': {r['relative_path']: R.sha(R.DATA / r['relative_path'])
                                                                 for r in rows}}
    R.write_json(meta, rec)
    R.log('EXTRACT', c, split, len(rows), round(rec['seconds'], 1), 's')
    return rec


def feature_bytes():
    return sum(p.stat().st_size for p in FEAT.glob('*.pt')) if FEAT.exists() else 0


def extract_train():
    R.configure()
    K.require(R.sha(ENC_PATH) == ENC_SHA, 'DINOv2 checkpoint hash')
    K.require(not OUT.exists(), 'run directory exists; preserve')
    enc = E.load(ENC_PATH, 'base', 'cuda')
    t0 = time.monotonic()
    for c in R.ORDER:
        if not (FEAT / f'{c}_train.pt').exists():
            extract(c, 'train', enc)
        K.require(feature_bytes() < FEATURE_CAP_BYTES, 'feature disk cap')
    R.log('EXTRACT_TRAIN_DONE', round(time.monotonic() - t0, 1), 's', 'cuda', torch.cuda.max_memory_allocated())


def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    K.require(R.sha(ENC_PATH) == ENC_SHA, 'DINOv2 checkpoint hash')
    inputs = {}
    for c in R.ORDER:
        p = FEAT / f'{c}_train.pt'
        m = R.read(p.with_suffix('.json'))
        K.require(R.sha(p) == m['sha256'] and m['encoder_sha256'] == ENC_SHA, f'train cache {c}')
        inputs[c] = {'train': {'path': str(p), 'sha256': m['sha256'], 'metadata_sha256': R.sha(p.with_suffix('.json')),
                               'n': m['n']}}
    ref = R.read(REF / 'RESULT.json')
    K.require(ref['PATCH_LOSS_GATE'] is not None, 'reference result')
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES},
        'historical_sources': R.read(W.HIST / 'SOURCE_LOCK.json')['sources'],
        'base_sha256': R.sha(W.HIST / 'BASE.pt'), 'encoder': str(ENC_PATH.relative_to(R.ROOT)),
        'encoder_sha256': ENC_SHA, 'inputs': inputs, 'cfg': R.CFG.dict(), 'seed': R.SEED, 'order': R.ORDER,
        'caps': CAPS, 'feature_cap_bytes': FEATURE_CAP_BYTES, 'arm': 'warm_patch_b14',
        'reference_result_sha256': R.sha(REF / 'RESULT.json'),
        'gate': {'comparison': 'B14 warm+patch minus B8 warm+patch (NAR-PATCH-01)', 'ap_delta': .01, 'ci_lower': 0,
                 'image_tolerance': .01, 'acquisition_tolerance': .01, 'fm_tolerance': .01,
                 'bootstrap_seed': 20261117, 'bootstrap_B': 2000},
        'torch': torch.__version__, 'cuda_build': torch.version.cuda})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for key in ('sources', 'historical_sources'):
        for p, h in lk[key].items():
            K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'], 'base hash')
    K.require(R.sha(ENC_PATH) == lk['encoder_sha256'], 'encoder hash')
    K.require(lk['caps'] == CAPS and lk['seed'] == R.SEED and lk['order'] == list(R.ORDER), 'config mismatch')
    K.require(lk['cfg'] == json.loads(json.dumps(R.CFG.dict())), 'CFG mismatch')
    K.require(R.sha(REF / 'RESULT.json') == lk['reference_result_sha256'], 'reference changed')
    return lk


def extract_test(lk, budget):
    P.verify_training()                                    # test access only after the training seal
    enc = E.load(ENC_PATH, 'base', 'cuda')
    for c in R.ORDER:
        rec = extract(c, 'test', enc)
        lk['inputs'][c]['test'] = {'path': str(FEAT / f'{c}_test.pt'), 'sha256': rec['sha256'], 'n': rec['n']}
        K.require(feature_bytes() < FEATURE_CAP_BYTES, 'feature disk cap')
        budget.check()
    R.write_json(OUT / 'TEST_INPUTS.json', {c: lk['inputs'][c]['test'] for c in R.ORDER})
    del enc
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def cost(lk, budget):
    from PIL import Image
    rows = R.rows_for('bottle', 'train')[:20]
    enc = E.load(ENC_PATH, 'base', 'cuda')
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    state = torch.load(OUT / 'stage15.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    det = Detector(model, loras, state['archive'])
    pre, en, dt = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for i in range(23):
        j = max(0, i - 3)
        t = time.perf_counter()
        x = E.preprocess(R.DATA / rows[j]['relative_path'])[None]
        with Image.open(R.DATA / rows[j]['relative_path']) as im:
            hw = (im.height, im.width)
        tp = time.perf_counter() - t
        torch.cuda.synchronize(); t = time.perf_counter()
        f = tokens(enc, x.cuda())
        torch.cuda.synchronize(); te = time.perf_counter() - t
        t = time.perf_counter(); det.predict(f, hw); torch.cuda.synchronize(); td = time.perf_counter() - t
        if i >= 3:
            pre.append(tp); en.append(te); dt.append(td)
        budget.check()
    parts = {'shared_encoder': sum(x.numel() * x.element_size() for x in enc.state_dict().values()),
             'shared_decoder': sum(x.numel() * x.element_size() for x in base.values()),
             'archive': sum(K.tensor_state_bytes(a) for a in state['archive']),
             'policy': K.slow_bytes(state['lam'], state['phi'])}
    R.write_json(OUT / 'COST.json', {'preprocess': quantiles(pre), 'encoder': quantiles(en), 'detector': quantiles(dt),
                                     'end_to_end': quantiles(np.array(pre) + en + dt),
                                     'cuda_peak_bytes': torch.cuda.max_memory_allocated(),
                                     'persistent_bytes': parts, 'total_persistent_bytes': sum(parts.values()),
                                     'note': 'encoder counts blocks 0-11 parameters stored; only blocks 0-9 run'})
    del enc, model, loras, det
    gc.collect()
    torch.cuda.empty_cache()


def compare(result):
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    ref = R.read(REF / 'RESULT.json')['arms']['warm_patch']
    new = result['arms']['warm']
    ci = bootstrap_delta([new['per_category'][c]['final']['pixel_ap'] for c in R.ORDER],
                         [ref['per_category'][c]['final']['pixel_ap'] for c in R.ORDER], seed=20261117, B=2000)
    return {'arms': {'b14_warm_patch': new, 'b8_warm_patch': ref}, 'b14_minus_b8': ci,
            'FRONTEND_B14_GATE': W.gate(new, ref, ci),
            'exposure': 'One seed 2711, alphabetical 15, historically exposed MVTec; exploratory only'}


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    lk = verify()
    R.configure()
    original = (R.cap, R.loss)
    R.cap, R.loss = budget.check, P.patch_loss
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0),
                                        'free_disk': shutil.disk_usage(R.ROOT).free})
    try:
        with budget.phase('training', gpu=True):
            P.train(lk, budget)
        with budget.phase('test_extraction', gpu=True):
            extract_test(lk, budget)
        with budget.phase('scoring', gpu=True):
            W.score(lk, budget)
        with budget.phase('inference_cost', gpu=True):
            cost(lk, budget)
        with budget.phase('metrics'):
            result = compare(W.metrics(budget))
        verify()
        P.verify_training()
        result['resources'] = budget.json()
        result['feature_cache_bytes'] = feature_bytes()
        result['cost'] = R.read(OUT / 'COST.json')
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()},
                                               'resources': budget.json()})
        R.log('COMPLETE', json.dumps(result['b14_minus_b8']), json.dumps(result['FRONTEND_B14_GATE']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise
    finally:
        R.cap, R.loss = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('extract-train', 'lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'extract-train': extract_train, 'lock': lock, 'execute': execute}[a.phase]()
