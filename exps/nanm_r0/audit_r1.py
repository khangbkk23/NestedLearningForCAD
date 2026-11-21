"""NANM-R0 amendment r1: identical metrics with the missing task-prefix cummin applied (task-free union memory).

See agents/tasks/task_nanm_r0_amendment_r1.md. Original audit.py, RESULT.json and knn files are not modified.

    python -B -m exps.nanm_r0.audit_r1 lock
    python -B -m exps.nanm_r0.audit_r1 metrics
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from exps.calm02 import native as N
from exps.nanm_r0 import audit as A
from exps.nar_pilot_00 import core as K, full as R

OUT = A.OUT
SOURCES = ('exps/nanm_r0/audit_r1.py', 'exps/nanm_r0/test_audit_r1.py', 'agents/tasks/task_nanm_r0_amendment_r1.md')


def prefix(d):
    """Per-task distances [15,N,28,28] -> union-memory distances over tasks <= s (the registered definition)."""
    return {k: v.cummin(dim=0).values for k, v in d.items()}


def lock():
    K.require(not (OUT / 'LOCK_r1.json').exists(), 'LOCK_r1 exists')
    A.verify_lock()
    A.wj(OUT / 'LOCK_r1.json', {'sources': {s: A.sha(A.ROOT / s) for s in SOURCES},
                                'original_lock_sha256': A.sha(OUT / 'LOCK.json'),
                                'original_result_sha256': A.sha(OUT / 'RESULT.json'),
                                'original_final_seal_sha256': A.sha(OUT / 'FINAL_SEAL.json'),
                                'knn_complete_sha256': A.sha(OUT / 'KNN_COMPLETE.json')})
    A.log('LOCK_r1', A.sha(OUT / 'LOCK_r1.json'))


def metrics():
    A.verify_lock()
    lk = A.rj(OUT / 'LOCK_r1.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(A.ROOT / p) == h, 'r1 source changed: ' + p)
    K.require(A.sha(OUT / 'RESULT.json') == lk['original_result_sha256'], 'original result must be preserved')
    K.require(not (OUT / 'RESULT_r1.json').exists(), 'one-shot: RESULT_r1 exists')
    nr = A.rj(OUT / 'NORMAL.json'); kc = A.rj(OUT / 'KNN_COMPLETE.json')
    for name, h in kc['files'].items():
        K.require(A.sha(OUT / name) == h, 'knn file changed')
    kn = {(sp, c): prefix(torch.load(OUT / f'knn_{sp}_{c}.pt', weights_only=True))
          for sp in ('test', 'eval') for c in A.ORDER}
    sealed = A.rj(A.LITE / 'cells_smooth.json')
    feval = {c: A.load_norm(c, nr, 'eval')['f_eval_prefix'] for c in A.ORDER}
    stats = {s: A.stage_stats(feval, kn, s) for s in range(15)}
    A.log('STATS done')
    from sklearn.metrics import roc_auc_score
    arms = ('A0', 'A1', 'A1cs', 'A2', 'A2cs', 'A2min', 'A2max')
    cells, hists, extra, parity = {}, {}, {}, {}
    for i, c in enumerate(A.ORDER):
        d = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)
        fgrid, rows = d['prefix_grids'], d['rows']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        for phase, s in (('acquisition', i), ('final', 14)):
            st = stats[s]
            m0 = A.smooth(fgrid[s]); m1 = A.smooth(kn['test', c]['full'][s])
            m1c = A.smooth(kn['test', c]['coreset'][s])
            z0 = (m0 - st['F'][0]) / st['F'][1]; z1 = (m1 - st['K'][0]) / st['K'][1]
            z1c = (m1c - st['Kcs'][0]) / st['Kcs'][1]
            maps = {'A0': m0, 'A1': m1, 'A1cs': m1c, 'A2': z0 + z1, 'A2cs': z0 + z1c,
                    'A2min': torch.minimum(z0, z1), 'A2max': torch.maximum(z0, z1)}
            for a, m in maps.items():
                m = m - min(0., float(m.min()))
                cells[a, phase, c] = {'pixel_ap': A.exact_ap(m, hws, masks),
                                      'image_auroc': float(roc_auc_score(labels, A.image_scores(m)))}
                if phase == 'final' and a in ('A0', 'A1', 'A1cs', 'A2', 'A2cs'):
                    hists[a, c] = A.image_hists(m, hws, masks, float(m.min()), float(m.max()) + 1e-12)
                    cells[a, phase, c]['binned_ap'] = A.binned_ap(hists[a, c][0].sum(0), hists[a, c][1].sum(0))
            parity[phase, c] = cells['A0', phase, c]['pixel_ap'] - sealed[f'{s + 1:02}_{c}']['pixel_ap']
            K.require(abs(parity[phase, c]) <= 1e-6, f'A0 parity {c} {phase}')
            A.log('CELL', c, phase, {a: round(cells[a, phase, c]['pixel_ap'], 4) for a in arms})
        extra[c] = A.diagnostics(c, fgrid[14], kn, feval, stats[14], masks, labels)
        del masks; A.caps()
    result = A.summarise(cells, hists, extra, parity, arms)
    result['latency'] = A.rj(OUT / 'RESULT.json')['latency']      # kNN cost does not depend on the error
    result['normal'] = nr['per_task']
    result['stats_final'] = stats[14]
    result['amendment'] = 'r1: prefix cummin over memory tasks applied (task-free union memory, as registered)'
    A.wj(OUT / 'RESULT_r1.json', result)
    A.wj(OUT / 'FINAL_SEAL_r1.json', {'lock_r1_sha256': A.sha(OUT / 'LOCK_r1.json'),
                                      'files': {p.name: A.sha(p) for p in OUT.iterdir() if p.is_file()}})
    A.log('DECISION_r1', json.dumps(result['decision']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'metrics'))
    {'lock': lock, 'metrics': metrics}[p.parse_args().phase]()
