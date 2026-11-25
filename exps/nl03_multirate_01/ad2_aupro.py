"""NL03-MULTIRATE-01 PART 1.3 (owner task 2026-11-25): local AU-PRO and SegF1 on MVTec AD 2 test_public, arms U and E,
from the locked grids (computed once).

AU-PRO follows the official MVTec AD evaluation code (/mnt/k/mvtec_ad_evaluation_code, pro_curve_util.compute_pro +
generic_util.trapezoid, imported unchanged). compute_pro materialises ~24 B per pixel (>14 GB per AD 2 scenario at
native resolution > 11 GB RAM), so the curve is built EXACTLY by histogramming over the 65,536 float16 bit patterns:
the maps scored are the float16 native maps that are submitted to the server (non-negative, so the uint16 bit pattern
is monotone in value); every unique score is a bin, which reproduces compute_pro's tie-merging. Equivalence to the
official function is tested on a downsampled subset (validate()).

    python -B -m exps.nl03_multirate_01.ad2_aupro validate | run
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nl03_multirate_01 import final as FN
from exps.nl03_multirate_01 import run as RN

EVAL_DIR = '/mnt/k/mvtec_ad_evaluation_code/mvtec_ad_evaluation'
ARMS = ('U', 'E')
LIMITS = (0.05, 0.30)
STRUCT = np.ones((3, 3), dtype=int)


def _official():
    import glob
    import os
    tf = sorted(glob.glob(os.path.expanduser('~/miniconda3/envs/pytorch_env/lib/python3*/site-packages')))[0]
    if tf not in sys.path:
        sys.path.append(tf)                           # tifffile for generic_util's top-level import (no download)
    if EVAL_DIR not in sys.path:
        sys.path.insert(0, EVAL_DIR)
    import generic_util as GU
    import pro_curve_util as PU
    return GU, PU


def gt_regions(mask):
    """(flat indices of GT pixels, 1/region_size weight per GT pixel, number of regions) with the official 3x3
    connectivity."""
    from scipy.ndimage import label
    lab, n = label(mask, STRUCT)
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    if n == 0:
        return idx.astype(np.int64), np.zeros(0), 0
    sizes = np.bincount(flat, minlength=n + 1).astype(np.float64)
    return idx.astype(np.int64), 1.0 / sizes[flat[idx]], n


def hist_add(acc, m16, reg):
    bits = m16.view(np.uint16).ravel()
    assert not (bits & 0x8000).any(), 'negative score'
    acc['tot'] += np.bincount(bits, minlength=65536)
    idx, w, n = reg
    if len(idx):
        b = bits[idx]
        acc['gt'] += np.bincount(b, minlength=65536)
        acc['pro'] += np.bincount(b, weights=w, minlength=65536)
    acc['nreg'] += n


def new_acc():
    return {'tot': np.zeros(65536, np.int64), 'gt': np.zeros(65536, np.int64), 'pro': np.zeros(65536), 'nreg': 0}


def curve(acc):
    """(fprs, pros) exactly as compute_pro: thresholds at every unique score, descending, start (0,0), end (1,1)."""
    ok = acc['tot'] - acc['gt']
    keep = np.flatnonzero(acc['tot'] > 0)[::-1]                    # descending unique values
    fpr = np.cumsum(ok[keep]).astype(np.float64) / ok.sum()
    pro = np.cumsum(acc['pro'][keep]) / acc['nreg']
    fpr = np.clip(fpr.astype(np.float32), None, 1.); pro = np.clip(pro, None, 1.)
    return np.concatenate(([0.], fpr, [1.])), np.concatenate(([0.], pro, [1.]))


def au_pro(acc, limit):
    GU, _ = _official()
    f, p = curve(acc)
    return float(GU.trapezoid(f, p, x_max=limit) / limit)


def load_masks(root, rows, hws):
    from PIL import Image
    out = []
    for r, hw in zip(rows, hws):
        mp_ = FN.ad2_mask_of(root, r)
        m = np.zeros(hw, np.uint8) if mp_ is None else (np.asarray(Image.open(mp_).convert('L')) > 0).astype(np.uint8)
        assert m.shape == tuple(hw), (r, m.shape, hw)
        out.append(m)
    return out


def validate():
    """Histogram AU-PRO vs the official compute_pro on a downsampled subset (float16 maps): must agree to 1e-6."""
    import cv2
    from exps.nanm_r2 import run as R2
    _, PU = _official()
    root, _ = FN.ad2_root()
    res = {}
    for s in ('fabric', 'walnuts'):
        d = torch.load(FN.AD2_CACHE / 'grids' / 'E' / f'{s}__test_public.pt', weights_only=False)
        sel = list(range(0, len(d['rows']), 6))[:40]
        rows = [d['rows'][i] for i in sel]; hws = [d['hws'][i] for i in sel]
        maps = R2.maps_of(d['grid'][-1][sel])
        masks = load_masks(root, rows, hws)
        A16, G = [], []
        for m, hw, gt in zip(maps, hws, masks):
            h, w = hw[0] // 8, hw[1] // 8
            nat = FN._native(m, (h, w)).astype(np.float16)
            A16.append(nat); G.append(cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST))
        f, p = PU.compute_pro([a.astype(np.float32) for a in A16], G)
        GU, _ = _official()
        acc = new_acc()
        for a, g in zip(A16, G):
            hist_add(acc, a, gt_regions(g))
        res[s] = {lim: {'official': float(GU.trapezoid(f, p, x_max=lim) / lim), 'hist': au_pro(acc, lim)} for lim in LIMITS}
        for lim in LIMITS:
            r = res[s][lim]; r['abs_diff'] = abs(r['official'] - r['hist'])
    ok = all(v['abs_diff'] <= 1e-6 for r in res.values() for v in r.values())
    RN.wj(FN.AD2_OUT / 'AD2_AUPRO_VALIDATION.json', {'ok': ok, 'res': {s: {str(k): v for k, v in r.items()} for s, r in res.items()},
                                                    'note': 'subset: every 6th test_public image (<=40), maps and masks downsampled 8x, E final stage'})
    FN.log('AUPRO VALIDATE', ok, res)
    assert ok, 'STOP: histogram AU-PRO differs from official compute_pro'


def scen_job(args):
    s, j, T = args
    torch.set_num_threads(2)
    from exps.nanm_r2 import run as R2
    root, _ = FN.ad2_root()
    sc = RN.rj(FN.AD2_OUT / 'AD2_SCORE.json')
    d0 = torch.load(FN.AD2_CACHE / 'grids' / ARMS[0] / f'{s}__test_public.pt', weights_only=False)
    rows, hws = d0['rows'], d0['hws']
    masks = load_masks(root, rows, hws)
    regs = [gt_regions(m) for m in masks]
    out = {}
    t0 = time.time()
    for a in ARMS:
        d = torch.load(FN.AD2_CACHE / 'grids' / a / f'{s}__test_public.pt', weights_only=False)
        assert d['rows'] == rows and d['stages'] == list(range(j, T))
        per = {}
        for i, l in enumerate(d['stages']):
            maps = R2.maps_of(d['grid'][i])
            acc = new_acc()
            tp = fp = fn = 0
            for m, hw, gt, reg in zip(maps, hws, masks, regs):
                nat = FN._native(m, hw)
                hist_add(acc, nat.astype(np.float16), reg)
                if l == T - 1:
                    b = nat > sc['theta_dthr'][a][s]; g = gt.astype(bool)
                    tp += int((b & g).sum()); fp += int((b & ~g).sum()); fn += int((~b & g).sum())
            cell = {f'au_pro_{lim}': au_pro(acc, lim) for lim in LIMITS}
            cell['n_regions'] = acc['nreg']
            if l == T - 1:
                cell['seg_f1'] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
                cell['seg_tp_fp_fn'] = [tp, fp, fn]
                cell['theta_dthr'] = sc['theta_dthr'][a][s]
            per[int(l)] = cell
        out[a] = per
    return s, out, time.time() - t0


def run():
    FN.verify_final()
    assert RN.rj(FN.AD2_OUT / 'AD2_AUPRO_VALIDATION.json')['ok']
    p = FN.AD2_OUT / 'AD2_AUPRO.json'
    assert not p.exists(), 'computed once'
    sc = RN.rj(FN.AD2_OUT / 'AD2_SCORE.json')
    scens = sc['scenarios']; T = len(scens)
    res, secs = {}, {}
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for s, o, dt in pool.imap_unordered(scen_job, [(s, j, T) for j, s in enumerate(scens)]):
            res[s] = o; secs[s] = dt
            FN.log('AUPRO', s, {a: {k: round(v, 4) for k, v in o[a][T - 1].items() if isinstance(v, float)} for a in ARMS})
    tab = {}
    for a in ARMS:
        fin = {s: res[s][a][T - 1] for s in scens}
        fm = [max(res[s][a][l]['au_pro_0.05'] for l in range(j, T - 1)) - res[s][a][T - 1]['au_pro_0.05']
              for j, s in enumerate(scens[:-1])]
        tab[a] = {'au_pro_0.05': float(np.mean([fin[s]['au_pro_0.05'] for s in scens])),
                  'au_pro_0.30': float(np.mean([fin[s]['au_pro_0.3'] for s in scens])),
                  'seg_f1': float(np.mean([fin[s]['seg_f1'] for s in scens])),
                  'fm_au_pro_0.05': float(np.mean(fm)), 'per_scenario_final': fin}
    RN.wj(p, {'label': 'HELD_OUT_CONFIRMATION (local test_public)', 'arms': list(ARMS), 'table': tab, 'per_stage': res,
              'method': 'official compute_pro/trapezoid semantics via exact float16-bin histogram (validated: '
                        'AD2_AUPRO_VALIDATION.json); maps = submitted float16 native maps; SegF1 = pixel F1 aggregated '
                        'over all test_public pixels of a scenario, binary = float32 native map > theta_dthr',
              'eval_code': EVAL_DIR, 'seconds': time.time() - t0, 'per_scenario_seconds': secs})
    RN.ledger_add({'phase': 'cpu_ad2_aupro', 'seconds': time.time() - t0, 'gpu': False})


if __name__ == '__main__':
    {'validate': validate, 'run': run}[sys.argv[1]]()
