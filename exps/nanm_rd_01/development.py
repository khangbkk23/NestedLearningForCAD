"""NANM-RD-01B: controlled readout ablation on DTD-Synthetic (registration agents/tasks/task_nanm_rd_01.md §4).

Canonical G3 = nanm (unchanged). Control A0b = byte-matched Euclidean (G.a0b_sizes, same k-centre primitive, G.euclid).
Maps computed once per (method, seed, category) and cached; every readout consumes the identical tensor.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_rd_01.development extract     # nanm encoder tokens (train/good + test images)
    python -B -m exps.nanm_rd_01.development fit SEED    # G3 + A0b continual memories, acquisition + final maps
    python -B -m exps.nanm_rd_01.development analyze     # labels opened here only (DTD-Synthetic, authorised)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nanm_sys import final as F
from exps.nanm_sys import regen as RG

DATA = Path(__file__).resolve().parents[2] / 'data/dtd_synthetic'
CACHE = RG.CACHE.parent / 'nanm_rd_01'
SEEDS = (2711, 2712, 2713)
READOUTS = ('R0', 'R1', 'R2', 'R3')
BOOT_B, BOOT_SEED, ALPHA = 2000, 20261120, 0.05
MIN_EFFECT, GUARD_HARD, GUARD_SOFT, GUARD_SOFT_N = 0.010, -0.020, -0.005, 2


def out():
    o = Path(__file__).resolve().parents[2] / 'results/hope_cad/nanm_rd_01/rd01b'
    o.mkdir(parents=True, exist_ok=True)
    return o


def cats():
    return json.loads((out() / 'DATA_AUDIT.json').read_text())['order']


def rows(c):
    from exps.calm02 import data as CD
    return CD._mvtec(DATA, c)                         # (train rows, test rows with label/mask from directory names)


def extract():
    from exps.nar_pilot_00 import full as R
    from nanm import FrozenPatchEncoder
    dev = F.guard(); G, _ = RG.setup()
    R.configure()
    B14 = G.B14()
    enc = FrozenPatchEncoder.load(B14.ENC_PATH, B14.ENC_SHA, 'cuda')
    man, t0 = {}, time.monotonic()
    (CACHE / 'features').mkdir(parents=True, exist_ok=True)
    for c in cats():
        tr, te = rows(c)
        for split, rs in (('train', tr), ('test', te)):
            p = CACHE / 'features' / f'{c}_{split}.pt'
            tok = enc.tokens([DATA / r['relative_path'] for r in rs], batch=8)
            torch.save({'rows': [r['relative_path'] for r in rs], 'features': tok}, p)
            man[p.name] = G.A.sha(p)
        G.A.log('EXTRACT', c)
    G.A.wj(out() / 'FEATURES.json', {'files': man, 'seconds': time.monotonic() - t0, 'device': dev})


def _load(G, c, split):
    p = CACHE / 'features' / f'{c}_{split}.pt'
    G.K.require(G.A.sha(p) == G.A.rj(out() / 'FEATURES.json')['files'][p.name], 'feature cache')
    return torch.load(p, weights_only=True)


@torch.no_grad()
def fit(seed):
    from nanm import G3Config, NANMSystem
    from nanm.memory import greedy_kcenter
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    C = cats()
    trs = {c: _load(G, c, 'train') for c in C}
    fit_idx = {c: G.K.normal_partition([{'relative_path': r} for r in trs[c]['rows']])['FIT'] for c in C}
    n_fit = [784 * len(fit_idx[c]) for c in C]
    a0b_ratio, a0b = G.a0b_sizes(n_fit)
    g3k = [G.R1.n_of(0.01, n) for n in n_fit]
    te = {c: _load(G, c, 'test')['features'] for c in C}
    nm = NANMSystem.create(G3Config(seed=seed), 'cuda')
    a0b_keys, acq, t0 = [], {'G3': {}, 'A0b': {}}, time.monotonic()
    for t, c in enumerate(C):
        tok = trs[c]['features'][torch.as_tensor(fit_idx[c])]
        rep = nm.fit_task(tok, task_label=c)                               # canonical G3 boundary
        G.K.require(rep['k'] == g3k[t], 'G3 quota')
        fit16 = nm.writer.prepare(tok)
        a0b_keys.append(fit16[greedy_kcenter(fit16, a0b[t], seed, 128).cuda()].float())   # same primitive as oracle
        acq['G3'][c] = nm.predict(te[c])['map392']
        acq['A0b'][c] = _a0b_maps(G, te[c], a0b_keys)
    final = {'G3': {c: nm.predict(te[c])['map392'] for c in C}, 'A0b': {c: _a0b_maps(G, te[c], a0b_keys) for c in C}}
    d = CACHE / 'maps'; d.mkdir(parents=True, exist_ok=True)
    man = {}
    for stage, M in (('final', final), ('acq', acq)):
        for meth in M:
            for c in C:
                p = d / f's{seed}_{stage}_{meth}_{c}.pt'
                torch.save(M[meth][c], p)
                man[p.name] = G.A.sha(p)
    led = nm.state.ledger()
    G.A.wj(out() / f'MAPS_s{seed}.json', {'files': man, 'seconds': time.monotonic() - t0, 'device': dev,
                                          'g3_keys': g3k, 'a0b_keys': a0b, 'a0b_ratio': a0b_ratio,
                                          'g3_registered_ledger': led['historical_registered_ledger'],
                                          'a0b_bytes': 1536 * sum(a0b), 'g3_actual_bytes': led['actual_persistent_tensor_bytes'],
                                          'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
    G.A.log('FIT', seed, round(time.monotonic() - t0, 1))


def _a0b_maps(G, tok, keys):
    q16 = G.A.unit(tok.reshape(-1, G.D)).half()
    e = torch.stack([G.euclid(q16, k) for k in keys]).amin(0).reshape(len(tok), 28, 28)
    return G.R2.maps_of(e)


def _readouts(m):
    from exps.nanm_rd_01.diagnostic import readouts
    return {k: v.double().numpy() for k, v in readouts(m)[0].items()}


def analyze():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from exps.calm02 import native as N
    torch.set_num_threads(8)
    G, _ = RG.setup()
    C = cats()
    lab, hws, masks = {}, {}, {}
    for c in C:                                                           # labels/masks opened here only
        _, te = rows(c)
        G.K.require([r['relative_path'] for r in te] == torch.load(CACHE / 'features' / f'{c}_test.pt',
                                                                    weights_only=True, mmap=True)['rows'], 'test order')
        lab[c] = np.array([r['label'] for r in te])
        hws[c] = [N.native_hw(DATA / r['relative_path']) for r in te]
        masks[c] = [N.native_mask(DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(te, hws[c])]
    S, pix, maphash = {}, {}, {}
    for seed in SEEDS:
        man = G.A.rj(out() / f'MAPS_s{seed}.json')['files']
        for stage in ('final', 'acq'):
            for meth in ('G3', 'A0b'):
                for c in C:
                    p = CACHE / 'maps' / f's{seed}_{stage}_{meth}_{c}.pt'
                    G.K.require(G.A.sha(p) == man[p.name], 'map hash')
                    m = torch.load(p, weights_only=True)
                    h0 = G.A.sha(p)
                    for k, v in _readouts(m).items():
                        S[seed, stage, meth, k, c] = v
                    maphash[f's{seed}|{stage}|{meth}|{c}'] = h0               # one tensor for all readouts
                    if stage == 'final':
                        pix[seed, meth, c] = float(G.A.exact_ap(m, hws[c], masks[c]))
    au = lambda seed, stage, meth, k, c, w=None: roc_auc_score(lab[c], S[seed, stage, meth, k, c], sample_weight=w)
    cells = {}
    for seed in SEEDS:
        for stage in ('final', 'acq'):
            for meth in ('G3', 'A0b'):
                for k in READOUTS:
                    per = {c: {'i_auroc': float(au(seed, stage, meth, k, c)),
                               'i_ap': float(average_precision_score(lab[c], S[seed, stage, meth, k, c]))} for c in C}
                    cells[f's{seed}|{stage}|{meth}|{k}'] = {
                        'macro_i_auroc': float(np.mean([v['i_auroc'] for v in per.values()])),
                        'macro_i_ap': float(np.mean([v['i_ap'] for v in per.values()])), 'per_category': per}
        for meth in ('G3', 'A0b'):
            cells[f's{seed}|final|{meth}|pixel_ap'] = {'macro': float(np.mean([pix[seed, meth, c] for c in C])),
                                                      'per_category': {c: pix[seed, meth, c] for c in C}}
    # paired category-stratified bootstrap (seed 2711, final stage); one shared resample per replicate
    rng = np.random.default_rng(BOOT_SEED)
    arms = [(m, k) for m in ('G3', 'A0b') for k in READOUTS]
    reps = {a: np.empty(BOOT_B) for a in arms}
    for b in range(BOOT_B):
        per = {a: [] for a in arms}
        for c in C:
            n = len(lab[c]); w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            ok = (w * lab[c]).sum() > 0 and (w * (1 - lab[c])).sum() > 0
            for a in arms:
                per[a].append(au(2711, 'final', a[0], a[1], c, w) if ok else np.nan)
        for a in arms:
            reps[a][b] = np.nanmean(per[a])
    pt = lambda m, k, s=2711: cells[f's{s}|final|{m}|{k}']['macro_i_auroc']
    bp = lambda d: float(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean())))
    contrasts = {k: reps['G3', k] - reps['G3', 'R0'] for k in ('R1', 'R2', 'R3')}
    hol = G.holm({k: bp(v) for k, v in contrasts.items()})
    order_p = sorted(contrasts, key=lambda k: bp(contrasts[k]))
    dec = {}
    for k, d in contrasts.items():
        a_k = ALPHA / (len(contrasts) - order_p.index(k))
        pcat = {c: cells[f's2711|final|G3|{k}']['per_category'][c]['i_auroc'] -
                cells['s2711|final|G3|R0']['per_category'][c]['i_auroc'] for c in C}
        guard = min(pcat.values()) >= GUARD_HARD and sum(v < GUARD_SOFT for v in pcat.values()) <= GUARD_SOFT_N
        by_seed = {s: pt('G3', k, s) - pt('G3', 'R0', s) for s in SEEDS}
        delta = pt('G3', k) - pt('G3', 'R0')
        lo_adj = float(np.quantile(d, a_k / 2))
        qual = (delta >= MIN_EFFECT and hol[k][1] and lo_adj > 0 and guard and all(v > 0 for v in by_seed.values()))
        dec[k] = {'delta_macro_i_auroc': delta, 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                  'p_boot': bp(d), 'p_holm': hol[k][0], 'holm_rejected': hol[k][1], 'holm_adjusted_lower': lo_adj,
                  'per_category_delta': pcat, 'guard_ok': guard, 'delta_by_seed': by_seed, 'qualifies': bool(qual)}
    qual = [k for k in ('R1', 'R2', 'R3') if dec[k]['qualifies']]
    chosen = max(qual, key=lambda k: (dec[k]['delta_macro_i_auroc'], -int(k[1]))) if qual else None
    ctrl = {k: {'A0b_delta': pt('A0b', k) - pt('A0b', 'R0'),
                'A0b_ci95': [float(np.quantile(reps['A0b', k] - reps['A0b', 'R0'], q)) for q in (.025, .975)],
                'interaction': (pt('G3', k) - pt('G3', 'R0')) - (pt('A0b', k) - pt('A0b', 'R0')),
                'interaction_ci95': [float(np.quantile((reps['G3', k] - reps['G3', 'R0']) -
                                                       (reps['A0b', k] - reps['A0b', 'R0']), q)) for q in (.025, .975)]}
            for k in ('R1', 'R2', 'R3')}
    geo = {k: {'G3_minus_A0b': pt('G3', k) - pt('A0b', k),
               'ci95': [float(np.quantile(reps['G3', k] - reps['A0b', k], q)) for q in (.025, .975)]} for k in READOUTS}
    verdict = 'RD1_DEV_PROMISING' if chosen else 'RD1_DEV_NOT_SUPPORTED'
    res = {'verdict': verdict, 'chosen': chosen, 'decisions': dec, 'A0b_controls_and_interaction': ctrl,
           'geometry_effect_per_readout': geo, 'cells': cells, 'map_sha256_shared_by_all_readouts': maphash,
           'pixel_ap_invariance': 'one map tensor per (seed, stage, method, category); readouts never modify it',
           'n_test_images': {c: int(len(lab[c])) for c in C}, 'n_anomalous': {c: int(lab[c].sum()) for c in C}}
    G.A.wj(out() / 'RD01B_RESULT.json', res)
    print(verdict, 'chosen', chosen)
    for k in READOUTS:
        print(k, {m: round(pt(m, k), 4) for m in ('G3', 'A0b')}, 'ap', {m: round(cells[f's2711|final|{m}|{k}']['macro_i_ap'], 4)
                                                                       for m in ('G3', 'A0b')})
    for k, v in dec.items():
        print(k, {x: (round(y, 4) if isinstance(y, float) else y) for x, y in v.items() if x != 'per_category_delta'})
    print('controls', json.dumps(ctrl)[:600]); print('geo', json.dumps(geo)[:500])
    print('pixel AP', {f'{m}': round(cells[f's2711|final|{m}|pixel_ap']['macro'], 4) for m in ('G3', 'A0b')})


if __name__ == '__main__':
    ph, *a = sys.argv[1:]
    {'extract': extract, 'fit': lambda: fit(int(a[0])), 'analyze': analyze}[ph]()
