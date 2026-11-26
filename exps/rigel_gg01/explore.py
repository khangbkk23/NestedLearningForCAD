"""GG-01 exploratory diagnostic (owner request 2026-11-26, after the registered DEV stream): where is GG-01 weak on
MVTec AD and VisA? NOT part of the registered gate (PREREG.md); it cannot change the DEV verdict.

Order a, seed 2711, arms V1 and GG01; final stage only; scored on the existing development cohorts:
  VisA-DEV  240 images (exps.cadfinal.common.visa_split, 10 normal + 10 anomalous per class; a subset of the official
            VisA test set, which is why it was not used as the gating DEV set)
  MVTec P1  the 60 consumed P1/CALM-00A development images (exps.calm02.data.p1_development_paths)
Training-side diagnostics (test-free): keys retained per task and mean normal-token energy of a held-aside training
sample per task, under each arm's final state.

    python -B -m exps.rigel_gg01.explore
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_gg01 import gg
from exps.rigel_gg01.dev import guard, wj

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/rigel_gg01/explore'
CACHE = Path('/mnt/k/NLCAD/cache/rigel_gg01')
SEED = 2711
ARMS = ['V1', 'GG01']


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'explore.log', 'a') as f:
        f.write(s + '\n')


def dev_rows(ds):
    if ds == 'visa':
        from exps.cadfinal import common as CF
        sp = CF.visa_split()
        return {k.split('/')[-1]: set(v['dev']) for k, v in sp.items()}
    from exps.calm02 import data as CD
    out = {}
    for r in CD.p1_development_paths():
        out.setdefault(r.split('/')[0], set()).add(r)
    return out


@torch.no_grad()
def run(ds):
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    from exps.nl03_pressure_00 import core as PC
    from exps.nanm_eval_01 import metrics as EM
    from rigel_cad import metrics as M
    guard()
    G = PC.bind()
    PC.use(G, ds)
    order = FN.bench_order(ds, 'a')
    T = len(order)
    st = gg.Stream(ARMS, SEED)
    samples = []
    for l, t in enumerate(order):
        c = t.split('/')[1]
        x16, _ = RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))
        g = torch.Generator().manual_seed(20261126 + l)
        samples.append(x16[torch.randperm(len(x16), generator=g)[:4096].to(x16.device)].cpu())
        p = CACHE / 'fit' / ds / f'{c}_s{SEED}.pt'
        if p.exists():
            d = torch.load(p, weights_only=True)
            f, dt = gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16']), d['fit_s']
        else:
            t1 = time.time()
            f = gg.fit_task(x16, SEED)
            torch.cuda.synchronize()
            dt = time.time() - t1
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'kc': f.kc, 'S': f.S, 'n': f.n, 'n_tokens': f.n_tokens, 'keys16': f.keys16, 'fit_s': dt}, p)
        st.task(l, x16, f, dt)
        del x16
        torch.cuda.empty_cache()
        log(ds, l, c, st.arms['GG01'].log[-1]['metric'])
    cover = {}
    for a in ARMS:
        K, W = st.state(a, T - 1)
        kk = K.cuda().float()
        cover[a] = [float(gg.ops.energy_whitened(q, kk, W.cuda()).mean()) for q in samples]
    dev = dev_rows(ds)
    ex = PC.extract_rec(G, ds)
    per, pooled = {}, {a: [0, 0] for a in ARMS}
    pos_tot = {a: None for a in ARMS}
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = G.load_test(ds, c, ex)
        idx = [i for i, r in enumerate(rows) if r in dev.get(c, set())]
        if not idx:
            continue
        qs = q.reshape(len(rows), 784, gg.D)[idx].reshape(-1, gg.D)
        sub = [rows[i] for i in idx]
        cat = EM.load_category(G, ds, c, sub)

        class Cat:
            pass
        cc = Cat()
        cc.rows, cc.hws, cc.masks, cc.y = sub, cat['hws'], cat['masks'], cat['y']
        per[c] = {'n': len(sub), 'n_anom': int(cat['y'].sum())}
        for a in ARMS:
            grid = st.energy(a, T - 1, qs).reshape(1, len(sub), 28, 28)
            r, _ = M.evaluate_category(cc, grid, [T - 1]) if 0 < cat['y'].sum() < len(sub) else (None, None)
            if r is None:                                   # single-label cohort: AP only
                from rigel_cad import readout
                mp_ = readout.anomaly_maps(grid[0])
                h = M.image_hists(mp_, cc.hws, cc.masks, 0., float(grid.max()) + 1e-12)
                per[c][a] = {'ap': M.binned_ap(h[0].sum(0), h[1].sum(0)), 'i_auroc': None}
            else:
                per[c][a] = r[T - 1]
        log(ds, 'DEV', c, {a: round(per[c][a]['ap'], 4) if per[c][a]['ap'] == per[c][a]['ap'] else None for a in ARMS})
        del q
    k = st.k_per_task
    ret = {a: st.arms[a].log[-1]['retained_per_task'] for a in ARMS}
    res = {'dataset': ds, 'order': order, 'cohort': 'VisA-DEV (240)' if ds == 'visa' else 'MVTec P1-dev (60)',
           'per_category': per,
           'macro_ap': {a: float(np.nanmean([per[c][a]['ap'] for c in per])) for a in ARMS},
           'retained_per_task': ret, 'k_per_task': k,
           'retained_frac': {a: [r / kk for r, kk in zip(ret[a], k)] for a in ARMS},
           'coverage_final': cover, 'drift': st.drift(), 'label': 'EXPLORATORY_NOT_GATING'}
    wj(OUT / f'EXPLORE_{ds}.json', res)
    log(ds, 'DONE', res['macro_ap'])


if __name__ == '__main__':
    for ds in ('mvtec', 'visa'):
        run(ds)
