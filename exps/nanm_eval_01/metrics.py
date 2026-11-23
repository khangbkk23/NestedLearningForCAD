"""NANM-EVAL-01 evaluator: the BENCH-G3 metrics body (exps/bench_g3/metrics_split.py) applied to new grids.

Per category: rows / native masks / 224 masks / labels are loaded ONCE (as in the oracle) and every (cell, arm) grid of
that category is evaluated. Primitives are the oracle's own: R2.maps_of, A.image_scores, A.image_hists, A.binned_ap,
A.exact_ap, G.maps224, AUD.mask224, MX.exact_ap, sklearn roc_auc_score.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from exps.nanm_eval_01 import common as U

BOOT_B, BOOT_SEED = 2000, 20261120


def load_category(G, ds, c, rows_order):
    from exps.bench_aud_00 import run as AUD
    from exps.calm02 import native as N
    ids = {r['relative_path']: r for r in G.test_rows(ds, c)}
    rows = [ids[r] for r in rows_order]
    root = G.data_root(ds)
    hws = [N.native_hw(root / r['relative_path']) for r in rows]
    masks = [N.native_mask(root / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
    m224 = [AUD.mask224(m) for m in masks]
    y = np.array([r['label'] for r in rows])
    return {'rows': rows, 'hws': hws, 'masks': masks, 'm224': m224, 'y': y,
            'n_pixels': int(sum(h * w for h, w in hws)), 'n_pos_pixels': int(sum(m.sum() for m in masks))}


def boot_weights(n, cat_index, B=BOOT_B):
    """Paired image-resampling weights of one category, shared by every arm/cell (registered stream per category)."""
    rng = np.random.default_rng([BOOT_SEED, cat_index])
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float32)


def binned_ap_rows(P, Tt):
    """Oracle A.binned_ap applied to each row of (pos, tot) [B, bins] (same dtype and arithmetic as BENCH-G3 §6)."""
    from exps.nanm_r0 import audit as A
    return np.array([A.binned_ap(P[i], Tt[i]) for i in range(len(P))])


def eval_grid(G, cat, g, pos, T, exact_final, boot_w):
    """g: [T - pos, n, 28, 28] fp32 grid. Returns per-stage metrics and (optional) bootstrap replicates at the final stage."""
    from sklearn.metrics import roc_auc_score
    from exps.cadfinal import metrics as MX
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    y = cat['y']
    hi = float(g.max()) + 1e-12
    out, boot = {}, None
    for i, l in enumerate(range(pos, T)):
        maps = R2.maps_of(g[i])
        img = A.image_scores(maps)
        s224 = G.maps224(maps).numpy()
        h = A.image_hists(maps, cat['hws'], cat['masks'], 0., hi)
        cell = {'i_auroc': float(roc_auc_score(y, img)),
                'i_auroc_mapmax': float(roc_auc_score(y, maps.flatten(1).amax(1).numpy())),
                'i_auroc_gridmax': float(roc_auc_score(y, g[i].flatten(1).amax(1).numpy())),
                'ap224': float(MX.exact_ap(list(s224), cat['m224'])['ap']),
                'ap_native_binned': A.binned_ap(h[0].sum(0), h[1].sum(0))}
        if l == T - 1:
            if exact_final:
                cell['ap_native'] = float(A.exact_ap(maps, cat['hws'], cat['masks']))
            if boot_w is not None:
                ap = binned_ap_rows(boot_w @ h[0], boot_w @ h[1])
                au = np.array([roc_auc_score(y, img, sample_weight=w) if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0
                               else np.nan for w in boot_w])
                boot = {'ap': ap.astype(np.float32), 'auroc': au.astype(np.float32)}
        out[l] = cell
    return out, boot


def run_category(args):
    """Worker: (ds, c, cat_index, T, jobs, threads) with jobs = [(job_id, grid_path, pos, exact, boot)]."""
    ds, c, cat_index, T, jobs, threads = args
    torch.set_num_threads(threads)
    G = U.bind()
    t0 = time.time()
    first = torch.load(jobs[0][1], weights_only=True)
    cat = load_category(G, ds, c, first['rows'])
    bw = boot_weights(len(cat['y']), cat_index) if any(j[4] for j in jobs) else None
    res, boots = {}, {}
    for jid, path, pos, exact, boot in jobs:
        d = torch.load(path, weights_only=True)
        assert d['rows'] == first['rows'], f'row order differs {path}'
        res[jid], b = eval_grid(G, cat, d['grid'], pos, T, exact, bw if boot else None)
        if b is not None:
            boots[jid] = b
    info = {'n_images': len(cat['y']), 'n_anomalous': int(cat['y'].sum()), 'n_pixels': cat['n_pixels'],
            'n_pos_pixels': cat['n_pos_pixels'], 'seconds': time.time() - t0}
    return ds, c, res, boots, info
