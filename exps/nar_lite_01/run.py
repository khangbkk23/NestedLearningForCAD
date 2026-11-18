"""NAR-LITE-01: half-precision, truncated-encoder deployment of the sealed NAR-B14-01 model plus one preregistered
Dinomaly-default readout (Gaussian sigma 4 + top-1% image score). No training. See agents/tasks/task_nar_lite_01.md.

    python -B -m exps.nar_lite_01.run lock
    python -B -m exps.nar_lite_01.run execute --owner-authorized
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

from exps.calm02 import native as N
from exps.dinomaly import encoder as E
from exps.nar_b14_01 import run as B
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.cost import quantiles
from exps.nar_pilot_00.detector import prefix_grids, fm_eq8
from exps.nar_patch_01 import run as P
from exps.nar_warm_01 import run as W

OUT = R.ROOT / 'results/hope_cad/nar_lite_01'
REF = B.OUT                                   # sealed NAR-B14-01 (read only)
B8 = B.REF                                    # sealed NAR-PATCH-01 (read only)
KEEP_BLOCKS = 10                              # E.GROUPS[1][-1] + 1
REF_BYTES = 646_299_920
SIGMA, KSIZE, TOP = 4, 2 * int(4 * 4 + .5) + 1, .01
FIDELITY_TOL = .05
CAPS = {'wall_seconds': 3600, 'gpu_seconds': 2400, 'cuda_bytes': 5 * 2**30, 'rss_bytes': 8_500_000_000,
        'artifact_bytes': 400_000_000}
SOURCES = ('exps/nar_lite_01/run.py', 'exps/nar_lite_01/test_run.py', 'exps/nar_b14_01/run.py',
           'exps/nar_patch_01/run.py', 'exps/nar_warm_01/run.py', 'exps/nar_pilot_00/cost.py',
           'exps/nl_sgm_01a/am03_eval.py', 'exps/dinomaly/encoder.py', 'exps/calm02/native.py',
           'agents/tasks/task_nar_lite_01.md')


def bind():
    W.OUT, W.CAPS, W.SOURCES = OUT, CAPS, SOURCES


def verify_reference():
    """Sealed NAR-B14-01 training must be intact before any test access."""
    P.OUT = REF
    m = P.verify_training()
    P.OUT = OUT
    return m


# ---------------------------------------------------------------- lite models
def lite_encoder(dtype):
    enc = E.load(B.ENC_PATH, 'base', 'cuda')
    enc.blocks = enc.blocks[:KEEP_BLOCKS]
    del enc.norm
    return enc.to(dtype)


def lite_detector(dtype):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    model, loras, _ = R.model_new(base)
    model.eval().to(dtype)
    final = torch.load(REF / 'stage15.pt', weights_only=True)
    archive = [{n: (a.to(dtype), b.to(dtype)) for n, (a, b) in st.items()} for st in final['archive']]
    return model, loras, archive, final


def tokens(enc, x):
    g = enc.groups(x)
    K.require(tuple(g.shape[1:]) == (2, 789, 768), 'DINOv2 group shape')
    return g[:, B.GROUP, enc.num_prefix:, :].contiguous()


@torch.no_grad()
def expert_grids(model, loras, archive, feats, dtype):
    """[15,N,28,28], identical expert-outer / batch-four order to the sealed scorer."""
    out = torch.empty(len(archive), len(feats), 28, 28)
    for t, st in enumerate(archive):
        K.set_state(loras, st)
        for j in range(0, len(feats), 4):
            en, de = model(feats[j:j + 4].cuda().to(dtype))
            out[t, j:j + 4] = (1 - F.cosine_similarity(en[0].float(), de[0].float(), dim=1)).clamp_min(0).cpu()
    K.require(bool(torch.isfinite(out).all()), 'nonfinite expert score')
    return out


def persistent_bytes(enc, model_base_dtype, archive, final):
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    return {'shared_encoder': sum(x.numel() * x.element_size() for x in enc.state_dict().values()),
            'shared_decoder': sum(x.numel() * torch.empty((), dtype=model_base_dtype).element_size()
                                  for x in base.values()),
            'archive': sum(K.tensor_state_bytes(a) for a in archive),
            'policy': K.slow_bytes(final['lam'], final['phi'])}


# ---------------------------------------------------------------- normal-only precision choice
@torch.no_grad()
def fidelity(dtype, budget):
    """End-to-end lite grid vs sealed fp32 pipeline on 4 EVAL train/good images per category (normal only)."""
    enc32 = E.load(B.ENC_PATH, 'base', 'cuda')
    enc = lite_encoder(dtype)
    base = torch.load(W.HIST / 'BASE.pt', weights_only=True)
    m32, l32, _ = R.model_new(base); m32.eval()
    m, l, archive, final = lite_detector(dtype)
    tok_cos, num, den = [], 0., 0.
    for c in R.ORDER:
        rows = R.rows_for(c, 'train')
        idx = K.normal_partition(rows)['EVAL'][:4]
        x = torch.stack([E.preprocess(R.DATA / rows[i]['relative_path']) for i in idx]).cuda()
        f32, f16 = tokens(enc32, x).float(), tokens(enc, x.to(dtype))
        tok_cos.append(float(F.cosine_similarity(f32, f16.float(), dim=-1).mean()))
        g32 = K.task_agnostic_grid(list(expert_grids(m32, l32, final['archive'], f32, torch.float32)))[0]
        try:
            g16 = K.task_agnostic_grid(list(expert_grids(m, l, archive, f16, dtype)))[0]
        except RuntimeError as exc:                # overflow -> this precision fails the normal-only check
            del enc32, enc, m32, l32, m, l
            gc.collect(); torch.cuda.empty_cache()
            return {'dtype': str(dtype), 'failure': repr(exc), 'category': c, 'passed': False}
        num += float((g16 - g32).abs().sum()); den += float(g32.abs().sum())
        budget.check()
    rel = num / den
    del enc32, enc, m32, l32, m, l
    gc.collect(); torch.cuda.empty_cache()
    return {'dtype': str(dtype), 'relative_mean_abs_error': rel, 'token_cosine_min': min(tok_cos),
            'token_cosine_mean': float(np.mean(tok_cos)), 'images': 4 * len(R.ORDER),
            'passed': bool(np.isfinite(rel)) and rel <= FIDELITY_TOL}


# ---------------------------------------------------------------- test scoring (after reference seal check)
@torch.no_grad()
def score(dtype, budget):
    verify_reference()
    enc = lite_encoder(dtype)
    model, loras, archive, final = lite_detector(dtype)
    files, start = {}, time.monotonic()
    for c in R.ORDER:
        rows = R.rows_for(c, 'test')
        feats = torch.empty(len(rows), 784, 768, dtype=dtype)
        for i in range(0, len(rows), 8):
            x = torch.stack([E.preprocess(R.DATA / r['relative_path']) for r in rows[i:i + 8]]).cuda().to(dtype)
            feats[i:i + len(x)] = tokens(enc, x).cpu()
        K.require(bool(torch.isfinite(feats).all()), 'feature finiteness')
        prefixes = prefix_grids(expert_grids(model, loras, archive, feats, dtype))
        path = OUT / f'scores_{c}.pt'
        W.bounded_save(path, {'prefix_grids': prefixes, 'rows': rows, 'dtype': str(dtype)},
                       prefixes.numel() * 4, budget)
        files[path.name] = R.sha(path)
        R.log('SCORE', c, len(rows), round(time.monotonic() - start, 1)); budget.check()
        del feats, prefixes
    R.write_json(OUT / 'SCORES_COMPLETE.json', {'seconds': time.monotonic() - start, 'files': files,
                                                'reference_train_seal_sha256': R.sha(REF / 'TRAIN_COMPLETE.json')})
    del enc, model, loras
    gc.collect(); torch.cuda.empty_cache()


# ---------------------------------------------------------------- readouts and metrics
def smooth392(grid):
    """[N,28,28] -> [N,392,392]: bilinear to input size, Gaussian sigma 4 (Dinomaly get_gaussian_kernel)."""
    ax = torch.arange(KSIZE, dtype=torch.float32) - KSIZE // 2
    g = torch.exp(-ax ** 2 / (2 * SIGMA ** 2)); g = g / g.sum()
    up = F.interpolate(grid[:, None].float(), size=(E.SIZE, E.SIZE), mode='bilinear', align_corners=False)
    up = F.conv2d(up, g.view(1, 1, KSIZE, 1), padding=(KSIZE // 2, 0))      # separable == 2-D outer-product
    return F.conv2d(up, g.view(1, 1, 1, KSIZE), padding=(0, KSIZE // 2))[:, 0].clamp_min(0)


def map_native(d, hw):
    return F.interpolate(d.float().reshape(1, 1, *d.shape[-2:]), size=tuple(hw), mode='bilinear',
                         align_corners=False, antialias=False)[0, 0].numpy()


def cell_metrics(grid, readout, hws, masks, labels):
    from sklearn.metrics import roc_auc_score
    if readout == 'raw':
        flat = grid.reshape(len(grid), 784).numpy()
        return N.native_ap(flat, hws, masks)['ap'], float(roc_auc_score(labels, flat.max(axis=1)))
    sm = smooth392(grid)
    k = max(1, int(TOP * E.SIZE * E.SIZE))
    image = sm.flatten(1).topk(k, dim=1).values.mean(1).numpy()
    original = N.native_map
    N.native_map = map_native                  # same exact AP accumulator, 392x392 map instead of 28x28 grid
    try:
        ap = N.native_ap(list(sm), hws, masks)['ap']
    finally:
        N.native_map = original
    return ap, float(roc_auc_score(labels, image))


def metrics(readout, budget):
    scored = R.read(OUT / 'SCORES_COMPLETE.json'); cells = {}
    for i, c in enumerate(R.ORDER):
        path = OUT / f'scores_{c}.pt'
        K.require(R.sha(path) == scored['files'][path.name], 'score hash')
        data = torch.load(path, weights_only=True); rows = data['rows']; grids = data['prefix_grids']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = [r['label'] for r in rows]
        for t in range(i, 15):
            ap, auc = cell_metrics(grids[t], readout, hws, masks, labels)
            cells[t, c] = {'stage': t + 1, 'category': c, 'pixel_ap': ap, 'image_auroc': auc}
            R.log('METRIC', readout, t + 1, c, round(ap, 6), round(auc, 6)); budget.check()
        del data, grids, masks
    R.write_json(OUT / f'cells_{readout}.json', {f'{t + 1:02}_{c}': v for (t, c), v in cells.items()})
    per = {c: {'acquisition': cells[i, c], 'final': cells[14, c]} for i, c in enumerate(R.ORDER)}
    mean = lambda ph, k: float(np.mean([per[c][ph][k] for c in R.ORDER]))
    return {'per_category': per, 'final_pixel_ap': mean('final', 'pixel_ap'),
            'final_image_auroc': mean('final', 'image_auroc'), 'acquisition_pixel_ap': mean('acquisition', 'pixel_ap'),
            'pixel_fm_eq8': fm_eq8(cells, R.ORDER, 'pixel_ap'), 'image_fm_eq8': fm_eq8(cells, R.ORDER, 'image_auroc')}


def ci(a, b):
    from exps.nl_sgm_01a.am03_eval import bootstrap_delta
    return bootstrap_delta([a['per_category'][c]['final']['pixel_ap'] for c in R.ORDER],
                           [b['per_category'][c]['final']['pixel_ap'] for c in R.ORDER], seed=20261117, B=2000)


def lite_gate(lite, ref, total_bytes):
    guards = {'memory': total_bytes <= .60 * REF_BYTES,
              'pixel_ap': lite['final_pixel_ap'] >= ref['final_pixel_ap'] - .005,
              'image_auroc': lite['final_image_auroc'] >= ref['final_image_auroc'] - .005,
              'acquisition': lite['acquisition_pixel_ap'] >= ref['acquisition_pixel_ap'] - .005}
    return {'passed': all(guards.values()), 'guards': guards}


# ---------------------------------------------------------------- cost
@torch.no_grad()
def cost(dtype, budget):
    from PIL import Image
    rows = R.rows_for('bottle', 'train')[:20]
    enc = lite_encoder(dtype)
    model, loras, archive, final = lite_detector(dtype)
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
        f = tokens(enc, x.cuda().to(dtype))
        torch.cuda.synchronize(); te = time.perf_counter() - t
        t = time.perf_counter()
        grids = []
        for st in archive:
            K.set_state(loras, st)
            e_, d_ = model(f)
            grids.append((1 - F.cosine_similarity(e_[0].float(), d_[0].float(), dim=1)).clamp_min(0))
        grid = torch.stack(grids).amin(0)
        sm = smooth392(grid.cpu()).cuda()           # A1 readout included in latency
        F.interpolate(sm[:, None], size=hw, mode='bilinear', align_corners=False)
        sm.flatten(1).topk(max(1, int(TOP * E.SIZE * E.SIZE)), dim=1).values.mean(1)
        torch.cuda.synchronize(); td = time.perf_counter() - t
        if i >= 3:
            pre.append(tp); en.append(te); dt.append(td)
        budget.check()
    parts = persistent_bytes(enc, dtype, archive, final)
    rec = {'preprocess': quantiles(pre), 'encoder': quantiles(en), 'detector_and_readout': quantiles(dt),
           'end_to_end': quantiles(np.array(pre) + en + dt), 'cuda_peak_bytes': torch.cuda.max_memory_allocated(),
           'persistent_bytes': parts, 'total_persistent_bytes': sum(parts.values()), 'dtype': str(dtype),
           'note': 'encoder blocks 0-9 only; decoder/encoder/adapters half precision; policy fp32'}
    R.write_json(OUT / 'COST.json', rec)
    del enc, model, loras
    gc.collect(); torch.cuda.empty_cache()
    return rec


# ---------------------------------------------------------------- lock / execute
def lock():
    bind()
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    verify_reference()
    OUT.mkdir(parents=True, exist_ok=True)
    R.write_json(OUT / 'LOCK.json', {
        'sources': {s: R.sha(R.ROOT / s) for s in SOURCES},
        'reference_result_sha256': R.sha(REF / 'RESULT.json'), 'reference_stage15_sha256': R.sha(REF / 'stage15.pt'),
        'reference_train_seal_sha256': R.sha(REF / 'TRAIN_COMPLETE.json'),
        'b8_result_sha256': R.sha(B8 / 'RESULT.json'), 'base_sha256': R.sha(W.HIST / 'BASE.pt'),
        'encoder_sha256': B.ENC_SHA, 'keep_blocks': KEEP_BLOCKS, 'precision_rule': 'fp16 else bf16 if '
        f'normal-only relative grid error <= {FIDELITY_TOL}', 'readout_a1': {'sigma': SIGMA, 'kernel': KSIZE,
        'padding': 'zero', 'map_size': E.SIZE, 'image_score': 'mean top 1%'}, 'caps': CAPS, 'ref_bytes': REF_BYTES,
        'gates': {'LITE_GATE': 'bytes<=0.60*ref; AP, AUROC, acquisition AP >= ref-0.005',
                  'SMOOTH_GATE': 'W.gate(L+A1, L raw, CI): +0.010, CI lo>0, tolerances 0.010'},
        'torch': torch.__version__})
    R.log('LOCK', R.sha(OUT / 'LOCK.json'))


def verify():
    lk = R.read(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(R.sha(R.ROOT / p) == h, 'locked source changed: ' + p)
    K.require(R.sha(REF / 'RESULT.json') == lk['reference_result_sha256'], 'reference changed')
    K.require(R.sha(B8 / 'RESULT.json') == lk['b8_result_sha256'], 'B8 reference changed')
    K.require(R.sha(W.HIST / 'BASE.pt') == lk['base_sha256'] and R.sha(B.ENC_PATH) == lk['encoder_sha256'], 'weights')
    return lk


def execute():
    bind()
    K.require(not (OUT / 'STARTED.json').exists(), 'no implicit restart or resume')
    budget = W.Budget()
    verify()
    R.configure()
    original = R.cap
    R.cap = budget.check
    R.write_json(OUT / 'STARTED.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'), 'started_unix': time.time(),
                                        'device': torch.cuda.get_device_name(0)})
    try:
        with budget.phase('fidelity_normal_only', gpu=True):
            fid = [fidelity(torch.float16, budget)]
            R.log('FIDELITY', json.dumps(fid[-1]))
            if not fid[-1]['passed']:
                fid.append(fidelity(torch.bfloat16, budget)); R.log('FIDELITY', json.dumps(fid[-1]))
            K.require(fid[-1]['passed'], 'no half precision passes the normal-only fidelity check')
            dtype = torch.float16 if len(fid) == 1 else torch.bfloat16
            R.write_json(OUT / 'FIDELITY.json', {'checks': fid, 'chosen': str(dtype)})
        with budget.phase('scoring', gpu=True):
            score(dtype, budget)
        with budget.phase('inference_cost', gpu=True):
            c = cost(dtype, budget)
        with budget.phase('metrics_raw'):
            raw = metrics('raw', budget)
        with budget.phase('metrics_a1'):
            a1 = metrics('smooth', budget)
        ref = R.read(REF / 'RESULT.json')['arms']['b14_warm_patch']
        b8 = R.read(B8 / 'RESULT.json')['arms']['warm_patch']
        c_a1 = ci(a1, raw)
        result = {'arms': {'lite_raw': raw, 'lite_a1': a1, 'b14_fp32_sealed': ref, 'b8_fp32_sealed': b8},
                  'LITE_GATE': lite_gate(raw, ref, c['total_persistent_bytes']),
                  'a1_minus_lite_raw': c_a1, 'SMOOTH_GATE': W.gate(a1, raw, c_a1),
                  'lite_raw_minus_b14': ci(raw, ref), 'lite_a1_minus_b14': ci(a1, ref), 'lite_a1_minus_b8': ci(a1, b8),
                  'fidelity': R.read(OUT / 'FIDELITY.json'), 'cost': c,
                  'exposure': 'One seed 2711, alphabetical 15, historically exposed MVTec; exploratory only'}
        verify(); verify_reference()
        result['resources'] = budget.json()
        R.write_json(OUT / 'RESULT.json', result)
        R.write_json(OUT / 'FINAL_SEAL.json', {'lock_sha256': R.sha(OUT / 'LOCK.json'),
                                               'files': {p.name: R.sha(p) for p in OUT.iterdir() if p.is_file()}})
        R.log('COMPLETE', json.dumps(result['LITE_GATE']), json.dumps(result['SMOOTH_GATE']))
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        R.write_json(OUT / 'STOPPED.json', {'reason': repr(exc), 'resources': budget.json()})
        raise
    finally:
        R.cap = original


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'execute'))
    p.add_argument('--owner-authorized', action='store_true')
    a = p.parse_args()
    if a.phase == 'execute':
        K.require(a.owner_authorized, 'explicit owner execution flag required')
    {'lock': lock, 'execute': execute}[a.phase]()
