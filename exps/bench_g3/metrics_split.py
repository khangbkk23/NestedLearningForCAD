"""BENCH-G3 amendment a2: run the locked metrics phase one dataset at a time (owner request, 2026-11-20).

    python -B -m exps.bench_g3.metrics_split mpdd
    python -B -m exps.bench_g3.metrics_split visa      # may run on another machine (CPU only; see handoff)

This calls the locked `run.py` functions unchanged (`verify_lock`, `R2.maps_of`, `A.exact_ap`, `summarize`, ...). The
only difference from `run.metrics` is the dataset subset and the output names: RESULT_<ds>.json, CELLS_<ds>.json and
SEAL_<ds>.json. Each dataset is one-shot.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from exps.bench_g3 import run as G
from exps.calm02 import native as N
from exps.nanm_r0 import audit as A
from exps.nanm_r2 import run as R2
from exps.nar_pilot_00 import core as K


def main(ds):
    from sklearn.metrics import roc_auc_score
    from exps.bench_aud_00 import run as AUD
    from exps.cadfinal import metrics as MX
    K.require(ds in G.DSETS, 'dataset')
    G.verify_lock()
    out = G.OUT
    K.require(not (out / 'RESULT.json').exists() and not (out / f'RESULT_{ds}.json').exists(), 'one shot')
    sc = A.rj(out / 'SCORES_COMPLETE.json')
    torch.set_num_threads(8)
    t_all = time.monotonic()
    cells, hist, ihist, labels_all, dev_all = {}, {}, {}, {}, {}
    boot_arms = {'G3|s2711', 'A0b|s2711', 'G1b|s2711', 'G3@C|s2711', 'A0@C|s2711', 'E-full'}
    ords, Tn = G.order(ds), len(G.order(ds))
    for j, c in enumerate(ords):                                   # identical body to run.metrics
        p = G.SCORE / f'{ds}_{c}.pt'
        K.require(A.sha(p) == sc['files'][p.name], 'score file changed ' + p.name)
        d = torch.load(p, weights_only=True)
        ids = {r['relative_path']: r for r in G.test_rows(ds, c)}
        rows = [ids[r] for r in d['rows']]
        root = G.data_root(ds)
        hws = [N.native_hw(root / r['relative_path']) for r in rows]
        masks = [N.native_mask(root / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        m224 = [AUD.mask224(m) for m in masks]
        y = np.array([r['label'] for r in rows])
        labels_all[ds, c] = y
        dev_all[ds, c] = np.array([bool(r.get('dev', False)) for r in rows])
        for key, g in d['grids'].items():
            stages = range(j, Tn) if G.full_stage(key) else [Tn - 1]
            hi = float(g.max()) + 1e-12
            for l in stages:
                maps = R2.maps_of(g[l - j])
                img = A.image_scores(maps)
                s224 = G.maps224(maps).numpy()
                cell = {'i_auroc': float(roc_auc_score(y, img)),
                        'i_auroc_mapmax': float(roc_auc_score(y, maps.flatten(1).amax(1).numpy())),
                        'ap224': float(MX.exact_ap(list(s224), m224)['ap'])}
                if G.full_stage(key):
                    h = A.image_hists(maps, hws, masks, 0., hi)
                    cell['ap_native_binned'] = A.binned_ap(h[0].sum(0), h[1].sum(0))
                    if l == Tn - 1 and key in boot_arms:
                        hist[ds, key, c] = h
                        ihist[ds, key, c] = (G.hist224(s224, m224, 0., hi), img)
                if l == Tn - 1:
                    cell['ap_native'] = float(A.exact_ap(maps, hws, masks))
                cells[ds, key, l, c] = cell
            A.log('METRIC', ds, c, key, round(cells[ds, key, Tn - 1, c]['ap_native'], 4),
                  round(cells[ds, key, Tn - 1, c]['i_auroc'], 4), round(time.monotonic() - t_all))
        del d, masks, m224; A.caps()
    A.wj(out / f'CELLS_{ds}.json', {f'{ds}|{k}|{l}|{c}': v for (_, k, l, c), v in cells.items()})
    G.DSETS = (ds,)                                                # summarize loops over G.DSETS
    res = G.summarize(cells, hist, ihist, labels_all, dev_all)
    res.update({'dataset': ds, 'seconds_metrics': time.monotonic() - t_all, 'lock_sha256': A.sha(out / 'LOCK.json'),
                'amendment': 'a2: per-dataset metrics split (agents/tasks/task_bench_g3_amendment_a2.md)'})
    A.wj(out / f'RESULT_{ds}.json', res)
    A.wj(out / f'SEAL_{ds}.json', {'lock_sha256': A.sha(out / 'LOCK.json'),
                                   'files': {n: A.sha(out / n) for n in (f'RESULT_{ds}.json', f'CELLS_{ds}.json',
                                                                         'SCORES_COMPLETE.json', 'LOCK.json')}})
    A.log('DONE', ds, res['decisions'])


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    main(sys.argv[1])
