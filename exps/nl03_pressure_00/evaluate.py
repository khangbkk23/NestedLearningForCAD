"""NL03-MEMORY-PRESSURE-00 evaluator: the EVAL-01 / BENCH-G3 metric primitives restricted to the registered metrics
(native Pixel AP binned, Image AUROC A1 readout; exact native AP where registered). Formulas are identical to
exps.nanm_eval_01.metrics.eval_grid for these fields (test T5 checks this on a synthetic category)."""
from __future__ import annotations

import time

import numpy as np
import torch


def eval_lite(cat, g, stages, boot_w=None, exact_final=False):
    """g: [len(stages), n, 28, 28] fp32. Returns ({stage: {'ap', 'i_auroc'[, 'ap_exact']}}, boot or None)."""
    from sklearn.metrics import roc_auc_score
    from exps.nanm_eval_01 import metrics as M
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    y = cat['y']
    hi = float(g.max()) + 1e-12
    out, boot = {}, None
    for i, l in enumerate(stages):
        maps = R2.maps_of(g[i])
        img = A.image_scores(maps)
        h = A.image_hists(maps, cat['hws'], cat['masks'], 0., hi)
        cell = {'ap': A.binned_ap(h[0].sum(0), h[1].sum(0)), 'i_auroc': float(roc_auc_score(y, img))}
        if i == len(stages) - 1:
            if exact_final:
                cell['ap_exact'] = float(A.exact_ap(maps, cat['hws'], cat['masks']))
            if boot_w is not None:
                ap = M.binned_ap_rows(boot_w @ h[0], boot_w @ h[1])
                au = np.array([roc_auc_score(y, img, sample_weight=w) if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0
                               else np.nan for w in boot_w])
                boot = {'ap': ap.astype(np.float32), 'auroc': au.astype(np.float32)}
        out[int(l)] = cell
    return out, boot


def run_task(args):
    """Spawn-pool worker: (task, boot_index, jobs, threads); jobs = [(jid, grid_path, boot, exact)]."""
    task, bidx, jobs, threads = args
    torch.set_num_threads(threads)
    from exps.nanm_eval_01 import metrics as M
    from exps.nl03_pressure_00 import core as C
    G = C.bind()
    ds, c = task.split('/')
    C.use(G, ds)
    t0 = time.time()
    first = torch.load(jobs[0][1], weights_only=True)
    cat = M.load_category(G, ds, c, first['rows'])
    bw = M.boot_weights(len(cat['y']), bidx) if any(j[2] for j in jobs) else None
    res, boots = {}, {}
    for jid, path, boot, exact in jobs:
        d = torch.load(path, weights_only=True)
        assert d['rows'] == first['rows'], f'row order differs {path}'
        res[jid], b = eval_lite(cat, d['grid'], d['stages'], bw if boot else None, exact)
        if b is not None:
            boots[jid] = b
    info = {'n_images': len(cat['y']), 'n_anomalous': int(cat['y'].sum()), 'n_pixels': int(sum(h * w for h, w in cat['hws'])),
            'seconds': time.time() - t0, 'n_jobs': len(jobs)}
    return task, res, boots, info
