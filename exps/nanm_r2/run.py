"""NANM-R2: normality-geometry feasibility on the R1 1 % task-free memory (no training).

Pre-registration: agents/tasks/task_nanm_r2_geometry.md.

    python -B -m exps.nanm_r2.run lock
    python -B -m exps.nanm_r2.run normal    # normal-only statistics per task (FIT role, while the task is present)
    python -B -m exps.nanm_r2.run score     # per-arm grids on MVTec test + EVAL normals
    python -B -m exps.nanm_r2.run metrics   # one-shot decision
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exps.calm02 import native as N
from exps.dinomaly import encoder as E
from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R

ROOT = A.ROOT
OUT = ROOT / 'results/hope_cad/nanm_r2'
CACHE = Path.home() / 'hope_cad_cache/nanm_r2'
D = 768
SHRINK, G2_RANK, L2_RANK, KAPPA, VFLOOR, WIN = 0.5, 64, 32, 20.0, 0.1, 3
ACC_BYTES = D * D * 4 + D * 4 + 8                        # pooled covariance accumulator (fp32)
L2_BYTES = 15 * (D * L2_RANK + L2_RANK + 1) * 4
CONTROLS = {'A0p236': ACC_BYTES, 'A0p148': L2_BYTES}
GEOM = ('G1', 'G2', 'G3', 'L1nll', 'L1ratio', 'L2nll', 'L2nolog', 'S1')
MATCH = {'G1': 'A0p236', 'G2': 'A0p236', 'G3': 'A0p236', 'L2nll': 'A0p148', 'L2nolog': 'A0p148',
         'L1nll': 'A0', 'L1ratio': 'A0', 'S1': 'A0'}
TASK_ARMS = ('A0', 'A0p236', 'A0p148', 'L1nll', 'L1ratio', 'L2nll', 'L2nolog', 'S1')
STAGE_ARMS = ('G1', 'G2', 'G3')
BOOT_B = 2000
Q_LO, Q_HI = 0.05 / 8 / 2, 1 - 0.05 / 8 / 2
SOURCES = ('exps/nanm_r2/run.py', 'exps/nanm_r2/test_run.py', 'agents/tasks/task_nanm_r2_geometry.md',
           'exps/nanm_r0/audit.py', 'exps/nanm_r1/run.py')
MEM_BYTES_A0 = 26_348_544


# ------------------------------------------------------------------ pure functions (unit-tested)
def sqdist(q, m):
    """fp32 squared Euclidean distances [n, k], clamped >= 0."""
    return (q.pow(2).sum(1, keepdim=True) + m.pow(2).sum(1)[None] - 2 * q @ m.T).clamp_min(0)


def inv_sqrt_psd(S, floor=1e-12):
    w, V = torch.linalg.eigh(S.double())
    K.require(float(w.min()) > floor, 'metric not positive definite')
    return (V * w.rsqrt()) @ V.T, float(w.max() / w.min())


def shrink(C, alpha=SHRINK):
    return (1 - alpha) * C + alpha * (torch.trace(C) / C.shape[0]) * torch.eye(C.shape[0], dtype=C.dtype)


def ppca(C, r):
    """Top-r eigenpairs and isotropic remainder sigma^2 = mean of discarded eigenvalues."""
    w, V = torch.linalg.eigh(C.double())
    w, V = w.flip(0), V.flip(1)
    s2 = float(w[r:].mean())
    K.require(s2 > 0, 'PPCA remainder not positive')
    return V[:, :r], w[:r], s2


def ppca_inv_sqrt(V, e, s2):
    """Sigma^{-1/2} for Sigma = V diag(e) V^T + s2 (I - V V^T); also log|Sigma| and condition number."""
    d = V.shape[0]
    W = (V * e.rsqrt()) @ V.T + (torch.eye(d, dtype=V.dtype) - V @ V.T) / math.sqrt(s2)
    return W, float(e.log().sum() + (d - len(e)) * math.log(s2)), float(e.max() / s2)


def cheb_table(win=WIN):
    p = torch.arange(784)
    r, c = p // 28, p % 28
    return ((r[:, None] - r[None]).abs() <= win) & ((c[:, None] - c[None]).abs() <= win)


def smooth_nc(grid):
    """NAR-LITE A1 readout without the final clamp (identical to L.smooth392 on non-negative input)."""
    ks, sg = L.KSIZE, L.SIGMA
    ax = torch.arange(ks, dtype=torch.float32) - ks // 2
    g = torch.exp(-ax ** 2 / (2 * sg ** 2)); g = g / g.sum()
    up = F.interpolate(grid[:, None].float(), size=(E.SIZE, E.SIZE), mode='bilinear', align_corners=False)
    up = F.conv2d(up, g.view(1, 1, ks, 1), padding=(ks // 2, 0))
    return F.conv2d(up, g.view(1, 1, 1, ks), padding=(0, ks // 2))[:, 0]


def control_sizes(n_fit, extra_bytes):
    """Smallest common ratio r' (k-centre prefix) whose total patches >= R1 1 % total + extra_bytes / 1536."""
    base = sum(R1.n_of(0.01, n) for n in n_fit)
    need = base + math.ceil(extra_bytes / (2 * D))
    lo, hi = 0.01, 0.04
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(R1.n_of(mid, n) for n in n_fit) >= need:
            hi = mid
        else:
            lo = mid
    return hi, [R1.n_of(hi, n) for n in n_fit]


# ------------------------------------------------------------------ lock / verify
def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    r1 = R1.OUT
    cs = A.rj(r1 / 'CORESETS.json')
    n_fit = [cs[c]['n_fit_patches'] for c in A.ORDER]
    ctrl = {k: control_sizes(n_fit, b) for k, b in CONTROLS.items()}
    A.wj(OUT / 'LOCK.json', {
        'sources': {s: A.sha(ROOT / s) for s in SOURCES},
        'r1': {n: A.sha(r1 / n) for n in ('LOCK.json', 'CORESETS.json', 'SCORE_COMPLETE.json', 'RESULT.json')},
        'r0': {n: A.sha(A.OUT / n) for n in ('NORMAL.json', 'RESULT_r1.json')},
        'r1_cache': {c: cs[c]['sha256'] for c in A.ORDER},
        'test_inputs': {c: A.rj(A.FEAT / f'{c}_test.json')['sha256'] for c in A.ORDER},
        'controls': {k: {'ratio': v[0], 'sizes': v[1], 'patches': sum(v[1]), 'bytes': 2 * D * sum(v[1])}
                     for k, v in ctrl.items()},
        'hyper': {'shrink': SHRINK, 'g2_rank': G2_RANK, 'l2_rank': L2_RANK, 'kappa': KAPPA, 'vfloor': VFLOOR,
                  'window': WIN}, 'aux_bytes': aux_bytes(), 'boot': [BOOT_B, A.BOOT_SEED, A.BINS, Q_LO, Q_HI],
        'decision': 'see task file; GEOM arms vs MATCH baselines', 'match': MATCH, 'torch': torch.__version__})
    A.log('LOCK', A.sha(OUT / 'LOCK.json'))


def aux_bytes():
    return {'A0': 0, 'G1': ACC_BYTES, 'G2': ACC_BYTES, 'G3': ACC_BYTES, 'L1nll': 17154 * 4 + 15 * 4,
            'L1ratio': 17154 * 4 + 15 * 4, 'L2nll': L2_BYTES, 'L2nolog': L2_BYTES, 'S1': 17154 * 2}


def verify_lock():
    lk = A.rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(ROOT / p) == h, 'locked source changed: ' + p)
    for n, h in lk['r1'].items():
        K.require(A.sha(R1.OUT / n) == h, 'R1 changed: ' + n)
    return lk


# ------------------------------------------------------------------ normal-only statistics
@torch.no_grad()
def normal():
    lk = verify_lock()
    K.require(not (OUT / 'NORMAL.json').exists(), 'no recompute')
    nr = A.rj(A.OUT / 'NORMAL.json')
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {}
    for ti, c in enumerate(A.ORDER):
        t0 = time.monotonic()
        nm = A.load_norm(c, nr)
        p = R1.CACHE / f'mv_{c}.pt'
        K.require(A.sha(p) == lk['r1_cache'][c], 'R1 coreset cache')
        kc = torch.load(p, weights_only=True)['kc_idx']
        fit = nm['fit'].cuda().float(); n = len(fit)
        n1 = R1.n_of(0.01, n)
        K.require(torch.equal(kc[:n1], nm['coreset_idx']), 'memory index parity')
        cs = kc[:n1].cuda()
        S = torch.zeros(D, D, dtype=torch.float64, device='cuda'); s1 = torch.zeros(D, dtype=torch.float64,
                                                                                         device='cuda')
        Rm = torch.zeros(D, D, dtype=torch.float64, device='cuda')
        member = torch.zeros(n, dtype=torch.bool, device='cuda'); member[cs] = True
        cnt = torch.zeros(n1, dtype=torch.float64, device='cuda'); ss = torch.zeros(n1, dtype=torch.float64,
                                                                                     device='cuda')
        mem = fit[cs]
        nres = 0
        for i in range(0, n, 8192):
            x = fit[i:i + 8192]
            S += (x.T @ x).double(); s1 += x.sum(0).double()
            keep = ~member[i:i + 8192]
            xr = x[keep]
            j = sqdist(xr, mem).argmin(1)
            r = xr - mem[j]
            Rm += (r.T @ r).double(); nres += len(r)
            cnt.index_add_(0, j, torch.ones(len(j), dtype=torch.float64, device='cuda'))
            ss.index_add_(0, j, r.pow(2).sum(1).double())
        vbar = float(ss.sum() / (D * nres))
        vhat = torch.where(cnt > 0, ss / (D * cnt.clamp_min(1)), torch.full_like(ss, vbar))
        v = ((cnt * vhat + KAPPA * vbar) / (cnt + KAPPA)).clamp_min(VFLOOR * vbar)
        V, e, s2 = ppca((Rm / nres).cpu(), L2_RANK)
        W2, logdet, cond2 = ppca_inv_sqrt(V, e, s2)
        torch.save({'S': S.cpu(), 's1': s1.cpu(), 'n': n, 'R': Rm.cpu(), 'n_res': nres, 'v': v.float().cpu(),
                    'vbar': vbar, 'l2_V': V.float(), 'l2_e': e.float(), 'l2_s2': s2, 'l2_logdet': logdet,
                    'l2_cond': cond2, 'pos': (kc[:n1] % 784).to(torch.int16), 'kc_idx': kc},
                   CACHE / f'stat_{c}.pt')
        rec[c] = {'n_fit': n, 'n_mem': n1, 'n_res': nres, 'vbar': vbar, 'v_min_max': [float(v.min()), float(v.max())],
                  'empty_cells': int((cnt == 0).sum()), 'l2_cond': cond2, 'l2_logdet': logdet,
                  'sha256': A.sha(CACHE / f'stat_{c}.pt'), 'seconds': time.monotonic() - t0}
        A.log('NORMAL', c, json.dumps({k: v for k, v in rec[c].items() if k != 'sha256'}))
        del fit, mem, nm; torch.cuda.empty_cache(); A.caps()
    stage = {}
    allst = {c: load_stat(c, rec) for c in A.ORDER}
    for s in range(15):                                   # pooled stage-s global metrics (tasks <= s only)
        seen = A.ORDER[:s + 1]
        Ss = sum(allst[c]['S'] for c in seen); s1 = sum(allst[c]['s1'] for c in seen)
        n = sum(rec[c]['n_fit'] for c in seen)
        Rs = sum(allst[c]['R'] for c in seen); nr_ = sum(rec[c]['n_res'] for c in seen)
        mu = s1 / n
        C = Ss / n - torch.outer(mu, mu)
        W1, c1 = inv_sqrt_psd(shrink(C))
        V, e, s2 = ppca(C, G2_RANK); W2, _, c2 = ppca_inv_sqrt(V, e, s2)
        W3, c3 = inv_sqrt_psd(shrink(Rs / nr_))
        stage[s] = {'G1': W1.float(), 'G2': W2.float(), 'G3': W3.float()}
        rec.setdefault('_stage', {})[s] = {'cond_G1': c1, 'cond_G2': c2, 'cond_G3': c3}
    torch.save(stage, CACHE / 'stage_metrics.pt')
    rec['_stage_sha256'] = A.sha(CACHE / 'stage_metrics.pt')
    A.wj(OUT / 'NORMAL.json', rec)


def load_stat(c, rec):
    p = CACHE / f'stat_{c}.pt'
    K.require(A.sha(p) == rec[c]['sha256'], 'stat cache changed')
    return torch.load(p, weights_only=True)


# ------------------------------------------------------------------ memories and per-task kernels
def memories(lk, rec, nr):
    """All per-task memory structures on GPU."""
    out = []
    cheb = cheb_table().cuda()
    for ti, c in enumerate(A.ORDER):
        st = load_stat(c, rec)
        fit = A.load_norm(c, nr)['fit']
        kc = st['kc_idx']; n1 = rec[c]['n_mem']
        W2, logdet, _ = ppca_inv_sqrt(st['l2_V'].double(), st['l2_e'].double(), st['l2_s2'])
        m = fit[kc[:n1]].float().cuda()
        out.append({'A0': m, 'A0p236': fit[kc[:lk['controls']['A0p236']['sizes'][ti]]].float().cuda(),
                    'A0p148': fit[kc[:lk['controls']['A0p148']['sizes'][ti]]].float().cuda(),
                    'v': st['v'].cuda(), 'W2': W2.float().cuda(), 'logdet': logdet,
                    'pos': st['pos'].long().cuda(), 'm_w2': m @ W2.float().cuda()})
        del fit
    return out, cheb


def task_energy(q, qpos, mt, cheb, arm, qw2=None):
    """Min energy of queries q [n,768] against one task memory under `arm` (fp32)."""
    if arm in ('A0', 'A0p236', 'A0p148'):
        return 0.5 * sqdist(q, mt[arm]).min(1).values
    if arm == 'S1':
        d = sqdist(q, mt['A0'])
        d = torch.where(cheb[qpos][:, mt['pos']], d, torch.full_like(d, float('inf')))
        return 0.5 * d.min(1).values
    if arm in ('L1nll', 'L1ratio'):
        d = sqdist(q, mt['A0']) / (2 * mt['v'][None])
        if arm == 'L1nll':
            d = d + 0.5 * D * mt['v'].log()[None]
        return d.min(1).values
    if arm in ('L2nll', 'L2nolog'):
        d = 0.5 * sqdist(qw2, mt['m_w2']).min(1).values
        return d + 0.5 * mt['logdet'] if arm == 'L2nll' else d
    raise ValueError(arm)


@torch.no_grad()
def grids_for(q, mems, cheb, stage_w, stages, chunk=4096):
    """Per arm: {stage: energy [nq]}, plus nearest-memory task at the last requested stage."""
    nq = len(q)
    qpos = torch.arange(nq, device='cuda') % 784
    per = {a: torch.empty(15, nq, device='cuda') for a in TASK_ARMS}
    for t, mt in enumerate(mems):
        for i in range(0, nq, chunk):
            qq, pp = q[i:i + chunk], qpos[i:i + chunk]
            qw2 = qq @ mt['W2']
            for a in TASK_ARMS:
                per[a][t, i:i + chunk] = task_energy(qq, pp, mt, cheb, a, qw2)
    out, nn = {}, {}
    for a in TASK_ARMS:
        pre = per[a].cummin(0)
        out[a] = {s: pre.values[s].clone() for s in stages}
        nn[a] = per[a].argmin(0)                          # first index of the minimum = nearest task (all 15)
    a0 = out['A0']
    fallback = {}
    for s in stages:                                       # S1: no in-window item in union -> unrestricted
        bad = ~torch.isfinite(out['S1'][s])
        fallback[s] = int(bad.sum())
        out['S1'][s][bad] = a0[s][bad]
    for a in STAGE_ARMS:
        out[a] = {}
        for s in stages:
            W = stage_w[s][a].cuda()
            mem = torch.cat([m['A0'] for m in mems[:s + 1]]) @ W
            owner = torch.cat([torch.full((len(m['A0']),), t, device='cuda') for t, m in enumerate(mems[:s + 1])])
            e = torch.empty(nq, device='cuda'); j = torch.empty(nq, dtype=torch.long, device='cuda')
            for i in range(0, nq, chunk):
                v, idx = sqdist(q[i:i + chunk] @ W, mem).min(1)
                e[i:i + chunk] = 0.5 * v; j[i:i + chunk] = owner[idx]
            out[a][s] = e
            if s == stages[-1]:
                nn[a] = j
    return out, nn, fallback


# ------------------------------------------------------------------ score
@torch.no_grad()
def score():
    lk = verify_lock()
    K.require(not (OUT / 'SCORE_COMPLETE.json').exists(), 'no rescoring')
    rec = A.rj(OUT / 'NORMAL.json'); nr = A.rj(A.OUT / 'NORMAL.json')
    K.require(A.sha(CACHE / 'stage_metrics.pt') == rec['_stage_sha256'], 'stage metrics changed')
    stage_w = torch.load(CACHE / 'stage_metrics.pt', weights_only=True)
    mems, cheb = memories(lk, rec, nr)
    files, par, fb = {}, 0., {}
    t0 = time.monotonic()
    for i, c in enumerate(A.ORDER):
        p = A.FEAT / f'{c}_test.pt'
        K.require(A.sha(p) == lk['test_inputs'][c], 'test cache hash')
        d = torch.load(p, weights_only=True)
        lite_rows = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)['rows']
        K.require([r['relative_path'] for r in d['rows']] == [r['relative_path'] for r in lite_rows], 'rows')
        n = len(d['rows'])
        q = A.unit(d['features'].reshape(-1, D).cuda())
        del d
        g, nn, fallback = grids_for(q, mems, cheb, stage_w, (i, 14))
        r1 = torch.load(R1.OUT / f'mvknn_test_{c}.pt', weights_only=True)['kc1'].cummin(0).values
        par = max(par, float((g['A0'][14].cpu().reshape(n, 28, 28) - r1[14]).abs().max()),
                  float((g['A0'][i].cpu().reshape(n, 28, 28) - r1[i]).abs().max()))
        aware = torch.load(R1.OUT / f'mvknn_test_{c}.pt', weights_only=True)['kc1'][i]   # own-task memory (diag.)
        out = {a: {'acq': g[a][i].cpu().reshape(n, 28, 28), 'final': g[a][14].cpu().reshape(n, 28, 28)}
               for a in g}
        K.require(all(bool(torch.isfinite(v).all()) for x in out.values() for v in x.values()), f'nonfinite {c}')
        torch.save({'grids': out, 'nn_final': {a: v.cpu().to(torch.int8).reshape(n, 28, 28) for a, v in nn.items()},
                    'taskaware_A0': aware}, OUT / f'grid_test_{c}.pt')
        files[f'grid_test_{c}.pt'] = A.sha(OUT / f'grid_test_{c}.pt')
        fb[c] = {str(k): v for k, v in fallback.items()}
        ev = A.load_norm(c, nr, 'eval')['eval_unit'].float().cuda()
        ge, _, _ = grids_for(ev, mems, cheb, stage_w, (14,))
        torch.save({a: v[14].cpu().reshape(-1, 28, 28) for a, v in ge.items()}, OUT / f'grid_eval_{c}.pt')
        files[f'grid_eval_{c}.pt'] = A.sha(OUT / f'grid_eval_{c}.pt')
        K.require(par <= 2e-3, f'A0 distance parity vs R1 {par}')
        A.log('SCORE', c, n, 'parity', round(par, 6), 'S1 fallback', fb[c], round(time.monotonic() - t0, 1))
        del q, ev; torch.cuda.empty_cache(); A.caps()
    lat = latency(mems, cheb, stage_w, nr)
    A.wj(OUT / 'SCORE_COMPLETE.json', {'files': files, 'a0_distance_parity_max_abs': par, 's1_fallback': fb,
                                       'latency': lat, 'seconds': time.monotonic() - t0})


@torch.no_grad()
def latency(mems, cheb, stage_w, nr):
    q = A.load_norm('bottle', nr, 'eval')['eval_unit'][:784 * 20].float().reshape(20, 784, D)
    out = {}
    one = lambda a: (lambda x: grids_one(x, a, mems, cheb, stage_w))
    for a in TASK_ARMS + STAGE_ARMS:
        torch.cuda.reset_peak_memory_stats(); ts = []
        for k in range(23):
            x = q[max(0, k - 3)].cuda(); torch.cuda.synchronize(); t = time.perf_counter()
            one(a)(x); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        ts = np.array(ts[3:]) * 1e3
        out[a] = {'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, .95)),
                  'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
    return out


def grids_one(x, arm, mems, cheb, stage_w):
    """One image, final-stage memory (all 15 tasks); used for latency only."""
    pos = torch.arange(784, device='cuda')
    if arm in STAGE_ARMS:
        W = stage_w[14][arm].cuda()
        mem = torch.cat([m['A0'] for m in mems]) @ W
        return 0.5 * sqdist(x @ W, mem).min(1).values
    e = torch.stack([task_energy(x, pos, mt, cheb, arm, x @ mt['W2'] if arm.startswith('L2') else None)
                     for mt in mems])
    return e.min(0).values


# ------------------------------------------------------------------ metrics
def boot_q(reps, a, b, lo=Q_LO, hi=Q_HI):
    d = reps[a] - reps[b]
    return {'lo': float(np.quantile(d, lo)), 'hi': float(np.quantile(d, hi)), 'B': len(d), 'unit': 'image',
            'adjust': 'Bonferroni 8 arms' if lo < .01 else 'none'}


def maps_of(grid):
    m = smooth_nc(grid)
    return m - min(0., float(m.min()))


def metrics():
    from sklearn.metrics import roc_auc_score
    lk = verify_lock()
    K.require(not (OUT / 'RESULT.json').exists(), 'one-shot')
    sc = A.rj(OUT / 'SCORE_COMPLETE.json')
    for n, h in sc['files'].items():
        K.require(A.sha(OUT / n) == h, 'artifact changed: ' + n)
    r1 = A.rj(R1.OUT / 'RESULT.json')['mvtec']['per_category']
    arms = ('A0',) + tuple(CONTROLS) + GEOM
    cells, hists, diag, parity = {}, {}, {}, {}
    for i, c in enumerate(A.ORDER):
        d = torch.load(OUT / f'grid_test_{c}.pt', weights_only=True)
        ev = torch.load(OUT / f'grid_eval_{c}.pt', weights_only=True)
        rows = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)['rows']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        for phase, key in (('acquisition', 'acq'), ('final', 'final')):
            for a in arms:
                m = maps_of(d['grids'][a][key])
                cells[a, phase, c] = {'pixel_ap': A.exact_ap(m, hws, masks),
                                      'image_auroc': float(roc_auc_score(labels, A.image_scores(m)))}
                if phase == 'final':
                    hists[a, c] = A.image_hists(m, hws, masks, float(m.min()), float(m.max()) + 1e-12)
        m = maps_of(d['taskaware_A0'])
        cells['A0_taskaware', 'final', c] = {'pixel_ap': A.exact_ap(m, hws, masks),
                                             'image_auroc': float(roc_auc_score(labels, A.image_scores(m)))}
        parity[c] = cells['A0', 'final', c]['pixel_ap'] - r1[f'kc1|final|{c}']['pixel_ap']
        diag[c] = diagnostics(c, i, d, ev, masks, labels)
        A.log('CELL', c, {a: round(cells[a, 'final', c]['pixel_ap'], 4) for a in arms})
        del masks; A.caps()
    K.require(max(abs(v) for v in parity.values()) <= 0.003, f'A0 category AP parity {parity}')
    macro = {a: {ph: {k: float(np.mean([cells[a, ph, c][k] for c in A.ORDER])) for k in ('pixel_ap', 'image_auroc')}
                 for ph in ('acquisition', 'final') if (a, ph, A.ORDER[0]) in cells}
             for a in arms + ('A0_taskaware',)}
    macro_par = macro['A0']['final']['pixel_ap'] - A.rj(R1.OUT / 'RESULT.json')['mvtec']['macro']['kc1']['final']['pixel_ap']
    K.require(abs(macro_par) <= 0.001, f'A0 macro parity {macro_par}')
    reps = R1.boot_reps(hists, list(arms), A.ORDER, B=BOOT_B)
    lat = sc['latency']
    per = lambda a: {c: cells[a, 'final', c]['pixel_ap'] for c in A.ORDER}
    verdicts = {}
    for x in GEOM:
        b = MATCH[x]
        dc = {c: per(x)[c] - per(b)[c] for c in A.ORDER}
        delta = macro[x]['final']['pixel_ap'] - macro[b]['final']['pixel_ap']
        top = max(dc, key=dc.get)
        ex = float(np.mean([v for c, v in dc.items() if c != top]))
        ci = boot_q(reps, x, b)
        checks = {'delta>=0.005': delta >= .005, 'bonf_lo>0': ci['lo'] > 0, 'no_cat<-0.03': min(dc.values()) >= -.03,
                  'not_one_category': ex > 0, 'p95<=50ms': lat[x]['p95_ms'] <= 50}
        verdicts[x] = {'baseline': b, 'delta': delta, 'bonferroni': ci, 'unadjusted': boot_q(reps, x, b, .025, .975),
                       'per_category': dc, 'delta_excl_top': ex, 'top_category': top, 'checks': checks,
                       'pass': all(checks.values()), 'aux_bytes': aux_bytes()[x],
                       'total_bytes': MEM_BYTES_A0 + aux_bytes()[x],
                       'baseline_total_bytes': MEM_BYTES_A0 if b == 'A0' else lk['controls'][b]['bytes']}
    if any(v['pass'] for v in verdicts.values()):
        dec = 'GEOMETRY_HEADROOM'
    elif all(v['delta'] < .005 for v in verdicts.values()):
        dec = 'NO_GEOMETRY_HEADROOM'
    else:
        dec = 'INCONCLUSIVE'
    ctrl = {k: {'delta_vs_A0': macro[k]['final']['pixel_ap'] - macro['A0']['final']['pixel_ap'],
                'ci': boot_q(reps, k, 'A0', .025, .975)} for k in CONTROLS}
    result = {'decision': dec, 'verdicts': verdicts, 'controls': ctrl, 'macro': macro,
              'per_category': {f'{a}|{ph}|{c}': v for (a, ph, c), v in cells.items()},
              'diagnostics': diag, 'latency': lat, 'parity': {'category_ap': parity, 'macro_ap': macro_par,
                                                              'distance_max_abs': sc['a0_distance_parity_max_abs']},
              'normal': {k: v for k, v in A.rj(OUT / 'NORMAL.json').items() if not k.endswith('sha256')},
              'bytes': {'memory_A0': MEM_BYTES_A0, 'aux': aux_bytes(),
                        'controls': {k: v['bytes'] for k, v in lk['controls'].items()}},
              'exposure': 'MVTec official test, historically exposed; development cohort; F not involved'}
    A.wj(OUT / 'RESULT.json', result)
    A.wj(OUT / 'FINAL_SEAL.json', {'lock_sha256': A.sha(OUT / 'LOCK.json'),
                                   'files': {p.name: A.sha(p) for p in OUT.iterdir() if p.is_file()}})
    A.log('DECISION', dec, json.dumps({x: [round(v['delta'], 4), v['pass']] for x, v in verdicts.items()}))


def patch_auroc(sc, pos_mask, neg_mask):
    from sklearn.metrics import roc_auc_score
    p, n = sc[pos_mask].numpy(), sc[neg_mask].numpy()
    if len(p) < 5:
        return None
    n = n[::max(1, len(n) // 200000)]
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))


def diagnostics(c, ci, d, ev, masks, labels):
    frac = A.patch_labels(masks)
    lab = torch.as_tensor(labels, dtype=torch.bool)
    normal_p = (~lab)[:, None, None].expand_as(frac)
    defect_p = frac >= .5
    out = {}
    for a, g in d['grids'].items():
        f = g['final']
        tau = float(torch.quantile(ev[a].flatten()[::3], .99))
        nn = d['nn_final'][a].long()
        out[a] = {'patch_auroc': patch_auroc(f, defect_p, normal_p),
                  'tpr_at_eval_q99': float((f[defect_p] > tau).float().mean()) if defect_p.any() else None,
                  'fpr_at_eval_q99': float((f[normal_p] > tau).float().mean()),
                  'normal_q99': float(torch.quantile(f[normal_p][::7], .99)),
                  'defect_median': float(f[defect_p].median()) if defect_p.any() else None,
                  'acq_minus_final_mean_normal': float((g['acq'] - f)[normal_p].mean()),
                  'nn_same_task_normal': float((nn[normal_p] == ci).float().mean()),
                  'nn_same_task_defect': float((nn[defect_p] == ci).float().mean()) if defect_p.any() else None}
    aw, fr = d['taskaware_A0'], d['grids']['A0']['final']
    out['A4'] = {'normal_score_drop_free_vs_aware': float((aw - fr)[normal_p].mean()),
                 'defect_score_drop_free_vs_aware': float((aw - fr)[defect_p].mean()) if defect_p.any() else None,
                 'normal_mean_aware': float(aw[normal_p].mean()),
                 'defect_mean_aware': float(aw[defect_p].mean()) if defect_p.any() else None,
                 'frac_defect_patches_cross_task_nn': float((d['nn_final']['A0'].long()[defect_p] != ci).float()
                                                            .mean()) if defect_p.any() else None,
                 'frac_normal_patches_cross_task_nn': float((d['nn_final']['A0'].long()[normal_p] != ci).float()
                                                            .mean())}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'normal', 'score', 'metrics'))
    torch.backends.cuda.matmul.allow_tf32 = False
    {'lock': lock, 'normal': normal, 'score': score, 'metrics': metrics}[p.parse_args().phase]()
