"""NANM-R2b: robustness of the two-level residual-geometry memory (G3). No training.

Pre-registration: agents/tasks/task_nanm_r2b_robustness.md.

    python -B -m exps.nanm_r2b.run lock
    python -B -m exps.nanm_r2b.run stats     # normal-only residual statistics (2 %, full LOO-image), per task
    python -B -m exps.nanm_r2b.run mvtec     # MVTec grids: 2 %, full, two extra orders
    python -B -m exps.nanm_r2b.run visa      # VisA-DEV grids at 1 %
    python -B -m exps.nanm_r2b.run metrics   # one-shot decision
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from exps.calm02 import native as N
from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nar_pilot_00 import core as K, full as R

ROOT = A.ROOT
OUT = ROOT / 'results/hope_cad/nanm_r2b'
CACHE = Path.home() / 'hope_cad_cache/nanm_r2b'
D, ALPHA, EXTRA = 768, R2.SHRINK, R2.ACC_BYTES
QC = 512                                  # amendment a1: was 1024 (CUDA cap guard)
LOO_CHUNK = 512                           # amendment a1: was 2048 for the full-memory LOO pass
SOURCES = ('exps/nanm_r2b/run.py', 'exps/nanm_r2b/test_run.py', 'agents/tasks/task_nanm_r2b_robustness.md',
           'exps/nanm_r2/run.py', 'exps/nanm_r1/run.py', 'exps/nanm_r0/audit.py',
           'agents/tasks/task_nanm_r3_nested_geometry.md', 'agents/tasks/task_nanm_r2b_amendment_a1.md')


def orders():
    perm = torch.randperm(15, generator=torch.Generator().manual_seed(2711)).tolist()
    return {'reverse': tuple(reversed(A.ORDER)), 'perm2711': tuple(A.ORDER[i] for i in perm)}


def control_sizes(n_fit, extra_bytes, base):
    """Smallest common k-centre ratio whose total patches >= sum ceil(base*N) + extra_bytes/1536."""
    need = sum(R1.n_of(base, n) for n in n_fit) + math.ceil(extra_bytes / (2 * D))
    lo, hi = base, 0.04
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sum(R1.n_of(mid, n) for n in n_fit) >= need else (mid, hi)
    return hi, [R1.n_of(hi, n) for n in n_fit]


def whiten(Rsum, n):
    return R2.inv_sqrt_psd(R2.shrink(Rsum / n, ALPHA))


@torch.no_grad()
def residual_stats(fit, mem_idx=None, img_of=None, chunk=2048):
    """Uncentred residual second moment. mem_idx: memory rows (members excluded from queries);
    mem_idx=None: full memory, leave-one-image-out (img_of = image id per row)."""
    n = len(fit)
    S = torch.zeros(D, D, dtype=torch.float64, device=fit.device); cnt = 0
    if mem_idx is not None:
        member = torch.zeros(n, dtype=torch.bool, device=fit.device); member[mem_idx] = True
        mem = fit[mem_idx]
    for i in range(0, n, chunk):
        x = fit[i:i + chunk]
        if mem_idx is not None:
            x = x[~member[i:i + chunk]]
            j = R2.sqdist(x, mem).argmin(1); r = x - mem[j]
        else:
            d = R2.sqdist(x, fit)
            same = img_of[i:i + chunk][:, None] == img_of[None]
            d.masked_fill_(same, float('inf'))
            r = x - fit[d.argmin(1)]
        S += (r.T @ r).double(); cnt += len(r)
    return S.cpu(), cnt


# ------------------------------------------------------------------ lock
def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    r2 = A.rj(R2.OUT / 'RESULT.json')
    K.require(r2['decision'] == 'GEOMETRY_HEADROOM' and r2['verdicts']['G3']['pass'], 'R2 decision of record')
    cs = A.rj(R1.OUT / 'CORESETS.json')
    n_fit = [cs[c]['n_fit_patches'] for c in A.ORDER]
    ctrl2 = control_sizes(n_fit, EXTRA, 0.02)
    vb = R1.VB()
    A.wj(OUT / 'LOCK.json', {
        'sources': {s: A.sha(ROOT / s) for s in SOURCES},
        'r2': {n: A.sha(R2.OUT / n) for n in ('LOCK.json', 'NORMAL.json', 'RESULT.json', 'FINAL_SEAL.json')},
        'r2_stage_metrics_sha256': A.sha(R2.CACHE / 'stage_metrics.pt'),
        'r1': {n: A.sha(R1.OUT / n) for n in ('LOCK.json', 'CORESETS.json', 'VISA_COMPLETE.json', 'RESULT.json')},
        'orders': orders(), 'ctrl2': {'ratio': ctrl2[0], 'sizes': ctrl2[1], 'bytes': 2 * D * sum(ctrl2[1])},
        'visa_order': vb.ORDER, 'alpha': ALPHA, 'extra_bytes': EXTRA, 'boot': [R2.BOOT_B, A.BOOT_SEED, A.BINS],
        'torch': torch.__version__})
    A.log('LOCK', A.sha(OUT / 'LOCK.json'))


def verify_lock():
    lk = A.rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(ROOT / p) == h, 'locked source changed: ' + p)
    for n, h in lk['r2'].items():
        K.require(A.sha(R2.OUT / n) == h, 'R2 changed: ' + n)
    return lk


# ------------------------------------------------------------------ normal-only statistics (MVTec)
@torch.no_grad()
def stats():
    lk = verify_lock()
    K.require(not (OUT / 'STATS.json').exists(), 'no recompute')
    nr = A.rj(A.OUT / 'NORMAL.json')
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {}
    for ti, c in enumerate(A.ORDER):
        t0 = time.monotonic()
        fit = A.load_norm(c, nr)['fit'].cuda().float(); n = len(fit)
        kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx']
        K.require(A.sha(R1.CACHE / f'mv_{c}.pt') == A.rj(R1.OUT / 'CORESETS.json')[c]['sha256'], 'coreset cache')
        S2, n2 = residual_stats(fit, kc[:R1.n_of(0.02, n)].cuda())
        S_full, n_full = residual_stats(fit, None, torch.arange(n, device='cuda') // 784, chunk=LOO_CHUNK)
        torch.save({'R2pct': S2, 'n2pct': n2, 'Rfull': S_full, 'nfull': n_full}, CACHE / f'res_{c}.pt')
        rec[c] = {'n_fit': n, 'n_res_2pct': n2, 'n_res_full': n_full, 'sha256': A.sha(CACHE / f'res_{c}.pt'),
                  'seconds': time.monotonic() - t0}
        A.log('STATS', c, rec[c]); del fit; torch.cuda.empty_cache(); A.caps()
    A.wj(OUT / 'STATS.json', rec)


def load_res(c, rec):
    K.require(A.sha(CACHE / f'res_{c}.pt') == rec[c]['sha256'], 'res cache changed')
    return torch.load(CACHE / f'res_{c}.pt', weights_only=True)


# ------------------------------------------------------------------ MVTec grids
@torch.no_grad()
def mvtec():
    lk = verify_lock()
    K.require(not (OUT / 'MVTEC_COMPLETE.json').exists(), 'no rescoring')
    rec = A.rj(OUT / 'STATS.json'); nr = A.rj(A.OUT / 'NORMAL.json'); r2n = A.rj(R2.OUT / 'NORMAL.json')
    K.require(A.sha(R2.CACHE / 'stage_metrics.pt') == lk['r2_stage_metrics_sha256'], 'R2 stage metrics')
    r2w = torch.load(R2.CACHE / 'stage_metrics.pt', weights_only=True)
    res = {c: load_res(c, rec) for c in A.ORDER}
    r2st = {c: R2.load_stat(c, r2n) for c in A.ORDER}
    pos = {c: i for i, c in enumerate(A.ORDER)}
    # metrics (all from normal-only statistics of the tasks seen at that stage)
    W = {}
    for s in range(15):
        seen = A.ORDER[:s + 1]
        W['G3_2', s] = whiten(sum(res[c]['R2pct'] for c in seen), sum(res[c]['n2pct'] for c in seen))[0].float()
    W['G3_full', 14] = whiten(sum(res[c]['Rfull'] for c in A.ORDER), sum(res[c]['nfull'] for c in A.ORDER))[0].float()
    W['G3r1', 14] = r2w[14]['G3']
    ords = lk['orders']
    for o, seq in ords.items():
        for p in range(15):
            seen = seq[:p + 1]
            W[o, p] = whiten(sum(r2st[c]['R'] for c in seen), sum(r2n[c]['n_res'] for c in seen))[0].float()
    conds = {f'{k[0]}|{k[1]}': None for k in W}
    # jobs per query category: (arm, W key, set of memory tasks, memory kind)
    def jobs(c):
        i = pos[c]
        out = [('G3_2|acq', ('G3_2', i), set(A.ORDER[:i + 1]), '2'), ('G3_2|final', ('G3_2', 14), set(A.ORDER), '2'),
               ('G3r1_2|final', ('G3r1', 14), set(A.ORDER), '2'),
               ('G3_full|final', ('G3_full', 14), set(A.ORDER), 'full'),
               ('G3r1_full|final', ('G3r1', 14), set(A.ORDER), 'full')]
        for o, seq in ords.items():
            p = seq.index(c)
            out.append((f'G3_1|{o}|acq', (o, p), set(seq[:p + 1]), '1'))
        return out
    queries, rows_n = {}, {}
    for c in A.ORDER:
        d = torch.load(A.FEAT / f'{c}_test.pt', weights_only=True)
        K.require(A.sha(A.FEAT / f'{c}_test.pt') == A.rj(A.FEAT / f'{c}_test.json')['sha256'], 'test cache')
        queries[c] = A.unit(d['features'].reshape(-1, D)).half(); rows_n[c] = len(d['rows']); del d
    eu_kinds = {'A0_2': '2', 'A0c_2': 'c2', 'A0_full': 'full', 'A0_1': '1', 'A0c_1': 'c1'}
    per = {c: {a: torch.empty(15, len(queries[c])) for a in eu_kinds} for c in A.ORDER}
    best = {c: {j[0]: torch.full((len(queries[c]),), float('inf')) for j in jobs(c)} for c in A.ORDER}
    r2lock = A.rj(R2.OUT / 'LOCK.json')
    t0 = time.monotonic()
    for t, m in enumerate(A.ORDER):
        fit = A.load_norm(m, nr)['fit']
        kc = torch.load(R1.CACHE / f'mv_{m}.pt', weights_only=True)['kc_idx']
        n = len(fit)
        mems = {'1': fit[kc[:R1.n_of(0.01, n)]], '2': fit[kc[:R1.n_of(0.02, n)]],
                'c1': fit[kc[:r2lock['controls']['A0p236']['sizes'][t]]], 'c2': fit[kc[:lk['ctrl2']['sizes'][t]]],
                'full': fit}
        mems = {k: v.float().cuda() for k, v in mems.items()}
        for c in A.ORDER:
            q = queries[c]
            for a, kind in eu_kinds.items():
                v = torch.empty(len(q))
                for i in range(0, len(q), QC):
                    v[i:i + QC] = (0.5 * R2.sqdist(q[i:i + QC].cuda().float(), mems[kind]).min(1).values).cpu()
                per[c][a][t] = v
            for name, wkey, tasks, kind in jobs(c):
                if m not in tasks:
                    continue
                Wg = W[wkey].cuda(); mw = mems[kind] @ Wg
                v = torch.empty(len(q))
                for i in range(0, len(q), QC):
                    v[i:i + QC] = (0.5 * R2.sqdist(q[i:i + QC].cuda().float() @ Wg, mw).min(1).values).cpu()
                best[c][name] = torch.minimum(best[c][name], v)
                del mw
        A.log('MV memory task', t + 1, m, round(time.monotonic() - t0, 1)); A.caps()
        del mems; torch.cuda.empty_cache()
    files, par = {}, 0.
    for c in A.ORDER:
        n = rows_n[c]
        r1k = torch.load(R1.OUT / f'mvknn_test_{c}.pt', weights_only=True)
        par = max(par, float((per[c]['A0_2'].reshape(15, n, 28, 28) - r1k['kc2']).abs().max()),
                  float((per[c]['A0_full'].reshape(15, n, 28, 28) - r1k['full']).abs().max()),
                  float((per[c]['A0_1'].reshape(15, n, 28, 28) - r1k['kc1']).abs().max()))
        K.require(all(bool(torch.isfinite(v).all()) for v in best[c].values()), 'nonfinite')
        torch.save({'per_task': {a: v.reshape(15, n, 28, 28) for a, v in per[c].items()},
                    'g3': {k: v.reshape(n, 28, 28) for k, v in best[c].items()}}, OUT / f'mv_{c}.pt')
        files[f'mv_{c}.pt'] = A.sha(OUT / f'mv_{c}.pt')
    K.require(par <= 2e-3, f'R1 distance parity {par}')
    A.wj(OUT / 'MVTEC_COMPLETE.json', {'files': files, 'r1_distance_parity_max_abs': par,
                                       'seconds': time.monotonic() - t0})


# ------------------------------------------------------------------ VisA-DEV
@torch.no_grad()
def visa():
    lk = verify_lock()
    K.require(not (OUT / 'VISA_COMPLETE.json').exists(), 'no rescoring')
    vb, tt = R1.VB(), R1.T()
    vlock = A.rj(vb.OUT / 'LOCK.json')
    r1lk = A.rj(R1.OUT / 'LOCK.json')
    order = vb.ORDER
    dev = {}
    for c in order:
        p = R1.DEVR / f'devr_{c}.pt'
        K.require(A.sha(p) == r1lk['visa_devr'][c], 'devr hash')
        d = torch.load(p, weights_only=False)
        dev[c] = {'q': A.unit(d['features'].float().reshape(-1, D)).half(), 'rows': d['rows']}
    n_fit = [784 * len(tt.roles_for(vb.train_rows(m))['FIT']) for m in order]      # sizes from rows only
    sizes = control_sizes(n_fit, EXTRA, 0.01)[1]
    store = []
    for t, m in enumerate(order):                       # one pass per task: memory + residual statistics (normal only)
        data = vb.load_train(vlock, m)
        fit_idx = torch.as_tensor(tt.roles_for(data['rows'])['FIT'])
        fit = torch.cat([A.unit(data['features'][fit_idx[i:i + 64]].reshape(-1, D).cuda()).half()
                         for i in range(0, len(fit_idx), 64)])
        K.require(len(fit) == n_fit[t], 'FIT size')
        kc = A.greedy_coreset(fit, sizes[t])
        n1 = R1.n_of(0.01, len(fit))
        f32 = fit.float()
        S, cnt = residual_stats(f32, kc[:n1].cuda())
        store.append({'m1': f32[kc[:n1].cuda()].cpu(), 'mc': f32[kc.cuda()].cpu(), 'R': S, 'n': cnt})
        A.log('VISA task', m, n1, sizes[t], cnt); A.caps()
        del data, fit, f32; torch.cuda.empty_cache()
    Ws = {s: whiten(sum(x['R'] for x in store[:s + 1]), sum(x['n'] for x in store[:s + 1]))[0].float().cuda()
          for s in range(12)}
    files, par = {}, 0.
    for i, c in enumerate(order):
        q = dev[c]['q'].cuda().float(); n = len(dev[c]['rows'])
        per = {'A0': torch.empty(12, len(q)), 'A0c': torch.empty(12, len(q))}
        g3 = {}
        for t in range(12):
            per['A0'][t] = (0.5 * R2.sqdist(q, store[t]['m1'].cuda()).min(1).values).cpu()
            per['A0c'][t] = (0.5 * R2.sqdist(q, store[t]['mc'].cuda()).min(1).values).cpu()
        for key, s in (('acq', i), ('final', 11)):
            mw = torch.cat([x['m1'] for x in store[:s + 1]]).cuda() @ Ws[s]
            g3[key] = (0.5 * R2.sqdist(q @ Ws[s], mw).min(1).values).cpu().reshape(n, 28, 28)
        r1v = torch.load(R1.OUT / f'visa_{c}.pt', weights_only=False)['dev']['kc1']
        par = max(par, float((per['A0'].reshape(12, n, 28, 28) - r1v).abs().max()))
        torch.save({'per_task': {a: v.reshape(12, n, 28, 28) for a, v in per.items()}, 'g3': g3,
                    'rows': dev[c]['rows']}, OUT / f'visa_{c}.pt')
        files[f'visa_{c}.pt'] = A.sha(OUT / f'visa_{c}.pt')
    K.require(par <= 2e-3, f'VisA R1 parity {par}')
    A.wj(OUT / 'VISA_COMPLETE.json', {'files': files, 'r1_distance_parity_max_abs': par, 'control_sizes': sizes,
                                      'control_bytes': 2 * D * sum(sizes),
                                      'memory_bytes': 2 * D * sum(R1.n_of(0.01, n) for n in n_fit)})


# ------------------------------------------------------------------ metrics
def ap_cells(maps, hws, masks, labels, hist):
    from sklearn.metrics import roc_auc_score
    out, h = {}, {}
    for a, g in maps.items():
        m = R2.maps_of(g)
        out[a] = {'pixel_ap': A.exact_ap(m, hws, masks), 'image_auroc': float(roc_auc_score(labels, A.image_scores(m)))}
        if a in hist:
            h[a] = A.image_hists(m, hws, masks, float(m.min()), float(m.max()) + 1e-12)
    return out, h


def metrics():
    lk = verify_lock()
    K.require(not (OUT / 'RESULT.json').exists(), 'one-shot')
    for f in ('MVTEC_COMPLETE.json', 'VISA_COMPLETE.json'):
        for n, h in A.rj(OUT / f)['files'].items():
            K.require(A.sha(OUT / n) == h, 'artifact changed: ' + n)
    ords = lk['orders']
    cells, hists = {}, {}
    for i, c in enumerate(A.ORDER):
        d = torch.load(OUT / f'mv_{c}.pt', weights_only=True)
        rows = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)['rows']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        pt, g3 = d['per_task'], d['g3']
        pre = lambda a, tasks: pt[a][[A.ORDER.index(x) for x in tasks]].amin(0)
        maps = {'A0_2|final': pre('A0_2', A.ORDER), 'A0c_2|final': pre('A0c_2', A.ORDER),
                'A0_2|acq': pre('A0_2', A.ORDER[:i + 1]), 'A0c_2|acq': pre('A0c_2', A.ORDER[:i + 1]),
                'A0_full|final': pre('A0_full', A.ORDER), **g3}
        for o, seq in ords.items():
            p = seq.index(c)
            maps[f'A0_1|{o}|acq'] = pre('A0_1', seq[:p + 1]); maps[f'A0c_1|{o}|acq'] = pre('A0c_1', seq[:p + 1])
        out, h = ap_cells(maps, hws, masks, labels, {'A0c_2|final', 'G3_2|final', 'A0_full|final', 'G3_full|final'})
        for a, v in out.items():
            cells['mvtec', a, c] = v
        for a, v in h.items():
            hists[a, c] = v
        A.log('MV', c, {a: round(v['pixel_ap'], 4) for a, v in out.items()})
        del masks; A.caps()
    vb = R1.VB(); vh = {}
    for c in vb.ORDER:
        d = torch.load(OUT / f'visa_{c}.pt', weights_only=False)
        rows = d['rows']; i = vb.ORDER.index(c)
        hws = [N.native_hw(vb.VISA / r['relative_path']) for r in rows]
        masks = [N.native_mask(vb.VISA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        pt = d['per_task']
        maps = {'A0|final': pt['A0'].amin(0), 'A0c|final': pt['A0c'].amin(0), 'G3|final': d['g3']['final'],
                'A0|acq': pt['A0'][:i + 1].amin(0), 'A0c|acq': pt['A0c'][:i + 1].amin(0), 'G3|acq': d['g3']['acq']}
        out, h = ap_cells(maps, hws, masks, labels, {'A0c|final', 'G3|final', 'A0|final'})
        for a, v in out.items():
            cells['visa', a, c] = v
        for a, v in h.items():
            vh[a, c] = v
        A.log('VISA', c, {a: round(v['pixel_ap'], 4) for a, v in out.items()})
    mac = lambda ds, a, k='pixel_ap': float(np.mean([cells[ds, a, c][k] for c in (A.ORDER if ds == 'mvtec' else vb.ORDER)
                                                     if (ds, a, c) in cells]))
    reps = R1.boot_reps(hists, ['A0c_2|final', 'G3_2|final', 'A0_full|final', 'G3_full|final'], A.ORDER, B=R2.BOOT_B)
    vreps = R1.boot_reps(vh, ['A0c|final', 'G3|final', 'A0|final'], vb.ORDER, B=R2.BOOT_B)
    r2 = A.rj(R2.OUT / 'RESULT.json')['verdicts']['G3']
    crit = {}
    d2 = mac('mvtec', 'G3_2|final') - mac('mvtec', 'A0c_2|final'); ci2 = R1.ci(reps, 'G3_2|final', 'A0c_2|final')
    crit['1pct (R2)'] = {'delta': r2['delta'], 'lo': r2['bonferroni']['lo'], 'pass': r2['pass']}
    crit['2pct'] = {'delta': d2, 'ci': ci2, 'pass': d2 >= .005 and ci2['lo'] > 0}
    df = mac('mvtec', 'G3_full|final') - mac('mvtec', 'A0_full|final')
    crit['full'] = {'delta': df, 'ci': R1.ci(reps, 'G3_full|final', 'A0_full|final'), 'pass': df >= 0}
    for o in ords:
        do = mac('mvtec', f'G3_1|{o}|acq') - mac('mvtec', f'A0c_1|{o}|acq')
        crit['order_' + o] = {'delta_acq': do, 'G3_acq': mac('mvtec', f'G3_1|{o}|acq'),
                              'A0c_acq': mac('mvtec', f'A0c_1|{o}|acq'), 'A0_acq': mac('mvtec', f'A0_1|{o}|acq'),
                              'pass': do > 0}
    dv = mac('visa', 'G3|final') - mac('visa', 'A0c|final')
    crit['visa_dev'] = {'delta_vs_A0c': dv, 'ci_vs_A0c': R1.ci(vreps, 'G3|final', 'A0c|final'),
                        'delta_vs_A0': mac('visa', 'G3|final') - mac('visa', 'A0|final'),
                        'ci_vs_A0': R1.ci(vreps, 'G3|final', 'A0|final'), 'pass': dv >= 0}
    dec = 'G3_ROBUST' if all(v['pass'] for v in crit.values()) else 'G3_NOT_ROBUST'
    macro = {f'{ds}|{a}': {'pixel_ap': mac(ds, a), 'image_auroc': mac(ds, a, 'image_auroc')}
             for ds, a in sorted({(k[0], k[1]) for k in cells})}
    res = {'decision': dec, 'failing': [k for k, v in crit.items() if not v['pass']], 'criteria': crit,
           'macro': macro, 'per_category': {f'{ds}|{a}|{c}': v for (ds, a, c), v in cells.items()},
           'mvtec_parity': A.rj(OUT / 'MVTEC_COMPLETE.json')['r1_distance_parity_max_abs'],
           'visa': {k: v for k, v in A.rj(OUT / 'VISA_COMPLETE.json').items() if k != 'files'},
           'orders': ords, 'stats': {k: v for k, v in A.rj(OUT / 'STATS.json').items()}}
    A.wj(OUT / 'RESULT.json', res)
    A.wj(OUT / 'FINAL_SEAL.json', {'lock_sha256': A.sha(OUT / 'LOCK.json'),
                                   'files': {p.name: A.sha(p) for p in OUT.iterdir() if p.is_file()}})
    A.log('DECISION', dec, json.dumps({k: v['pass'] for k, v in crit.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'stats', 'mvtec', 'visa', 'metrics'))
    torch.backends.cuda.matmul.allow_tf32 = False
    {'lock': lock, 'stats': stats, 'mvtec': mvtec, 'visa': visa, 'metrics': metrics}[p.parse_args().phase]()
