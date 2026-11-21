"""NANM-WRITE-01: slow-to-fast transfer on memory writing — normal-only gate. No training, no defect labels.

Pre-registration: agents/tasks/task_nanm_write_01.md. G3 (exps/nanm_r0..r2b) is imported, never modified.

    python -B -m exps.nanm_write_01.run lock
    python -B -m exps.nanm_write_01.run write     # FIT tokens only
    python -B -m exps.nanm_write_01.run eval      # EVAL / CREDIT train-good normals only
    python -B -m exps.nanm_write_01.run metrics   # one-shot decision
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B
from exps.nar_pilot_00 import core as K, full as R

ROOT = A.ROOT
OUT = ROOT / 'results/hope_cad/nanm_write_01'
CACHE = Path.home() / 'hope_cad_cache/nanm_write_01'
TASK = 'agents/tasks/task_nanm_write_01.md'
SOURCES = ('exps/nanm_write_01/run.py', 'exps/nanm_write_01/test_run.py', TASK, 'exps/nanm_r0/audit.py',
           'exps/nanm_r1/run.py', 'exps/nanm_r2/run.py', 'exps/nanm_r2b/run.py',
           'agents/tasks/task_nanm_write_01_amendment_a1.md')
D, RATIO, ALPHA = 768, 0.01, 0.5
ARMS = ('B0', 'B5', 'B2', 'B6', 'B4', 'B3')            # run order inside a config
READER_ARMS = ('B0', 'B2', 'B4', 'B3')                  # arms whose own per-stage reader is stored (B5, B6 read W^B0)
CONFIGS = {'c1': ('alpha', 2711), 'c2': ('reverse', 2711), 'c3': ('perm2711', 2711),
           'c4': ('alpha', 2712), 'c5': ('alpha', 2713)}
BOOT_B, BOOT_SEED = 2000, 20261119
G_PASS, G_FAIL, G_ID, CAT_MAX, OVERLAP_MAX = 0.25, 0.10, 0.10, math.log(1.05), 0.95
I2_TOL = 1e-4                                           # amendment a1 (was 1e-5)
S_BYTES = D * D * 4 + 8                                 # one slow statistic: S (fp32) + n (int64)
GPU_H_CAP = 1.0
QC = 4096


def order_of(name):
    return tuple(A.ORDER) if name == 'alpha' else tuple(B.orders()[name])


# ------------------------------------------------------------------ pure kernels (unit-tested)
@torch.no_grad()
def kcentre(x, n, seed, H=None, chosen=None):
    """Greedy farthest-first on z = x H P with exactly A.greedy_coreset's generator calls (P, first centre).
    H=None -> bit-identical to A.greedy_coreset. chosen: centres already committed (distances recomputed under the
    current metric with the same per-centre expression); the selection then continues to n."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x.shape[1], A.PROJ_DIM, generator=g) / math.sqrt(A.PROJ_DIM)).to(x.device, torch.float32)
    first = int(torch.randint(len(x), (1,), generator=g))
    z = x.float() @ (P if H is None else H.to(x.device, torch.float32) @ P)
    mind = torch.full((len(z),), float('inf'), device=x.device)
    idx = []
    for c in ([] if chosen is None else chosen.tolist()):
        idx.append(c)
        mind = torch.minimum(mind, ((z - z[c]) ** 2).sum(1))
    cur = first if not idx else int(mind.argmax())
    while len(idx) < n:
        idx.append(cur)
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        cur = int(mind.argmax())
    return torch.tensor(idx, dtype=torch.long)


@torch.no_grad()
def feat_metric(x, chunk=8192):
    """B2: shrink(centred feature covariance of the current task)^{-1/2} (fp64)."""
    S = torch.zeros(D, D, dtype=torch.float64, device=x.device); s1 = torch.zeros(D, dtype=torch.float64,
                                                                                  device=x.device)
    for i in range(0, len(x), chunk):
        v = x[i:i + chunk].double()
        S += v.T @ v; s1 += v.sum(0)
    mu = s1 / len(x)
    return R2.inv_sqrt_psd(R2.shrink((S / len(x) - torch.outer(mu, mu)).cpu(), ALPHA))


def res_metric(S, n):
    return B.whiten(S, n)                                # canonical G3 slow level: (W fp64, cond)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def write_task(x, seed, arm, state):
    """One task boundary for one arm. x: current-task FIT tokens (fp32). state: {'S','n'} = the arm's slow statistic
    (B6: S^ref). Returns (indices, slow-statistic increment for the boundary, timings, info). Uses only x and state."""
    k = R1.n_of(RATIO, len(x)); tm = {}; info = {'k': k}
    t0 = sync()
    if arm == 'B0':
        idx = A.greedy_coreset(x, k, seed); tm['event1_s'] = sync() - t0
        H1 = None
    else:
        if arm == 'B5':
            H1 = None
        elif arm == 'B2':
            H1, info['cond_T_writer'] = feat_metric(x)
        else:                                             # B4, B3 (own S), B6 (S^ref): previous boundary only
            H1 = None
            if state['n'] > 0:
                H1, info['cond_T_writer'] = res_metric(state['S'], state['n'])
        t1 = sync(); tm['metric_s'] = t1 - t0
        h = math.ceil(k / 2)
        idx1 = kcentre(x, h, seed, H1)
        t2 = sync(); tm['event1_s'] = t2 - t1
        Sp, npv = B.residual_stats(x, idx1.to(x.device))  # provisional fit: computed in every arm, installed in B3 only
        t3 = sync(); tm['provisional_s'] = t3 - t2
        H2 = H1
        if arm == 'B3':
            H2, info['cond_T_provisional'] = res_metric(state['S'] + Sp, state['n'] + npv)
        idx = kcentre(x, k, seed, H2, chosen=idx1)
        tm['event2_s'] = sync() - t3
        info['H1'] = H1; info['H2'] = H2
    t4 = sync()
    src = idx
    if arm == 'B6':                                        # B6's slow context comes from an independent Euclidean bank
        tr = sync(); ref_idx = A.greedy_coreset(x, k, seed); tm['ref_kcentre_s'] = sync() - tr
        src = ref_idx; info['ref_idx'] = ref_idx
    Sb, nb = B.residual_stats(x, src.to(x.device))
    tm['boundary_stat_s'] = sync() - t4
    return idx, (Sb, nb), tm, info


@torch.no_grad()
def run_stream(get_task, n_tasks, arm, seed, keep_reader=False, log=None):
    """Sequential stream. get_task(t) is called once per task in increasing t and is the only data access."""
    state = {'S': torch.zeros(D, D, dtype=torch.float64), 'n': 0}
    keys, out = [], []
    for t in range(n_tasks):
        x = get_task(t)
        idx, (Sb, nb), tm, info = write_task(x, seed, arm, state)
        state['S'] = state['S'] + Sb; state['n'] += nb
        keys.append(x[idx.to(x.device)].half())
        t0 = sync()                                        # reader rebuild: eigh + transform of the union bank
        W, cond = res_metric(state['S'], state['n'])
        _ = torch.cat(keys).float() @ W.float().to(x.device)
        tm['reader_rebuild_s'] = sync() - t0
        rec = {'idx': idx, 'tm': tm, 'k': info['k'], 'cond_T_reader': cond,
               'cond_T_writer': info.get('cond_T_writer'), 'cond_T_provisional': info.get('cond_T_provisional'),
               'H1': info.get('H1'), 'ref_idx': info.get('ref_idx')}
        if keep_reader:
            rec['W'] = W.float()
        out.append(rec)
        if log:
            log(arm, t, info['k'], {k: round(v, 3) for k, v in tm.items()})
        del x, _
    return out


def p_stats(e, n_img):
    """Per-image P95, P99 of patch energies e [n_img*784]."""
    g = e.reshape(n_img, 784)
    return torch.quantile(g, 0.95, dim=1), torch.quantile(g, 0.99, dim=1)


@torch.no_grad()
def energies(q, keys, W):
    """Task-free energy 0.5 min ||(q-m) W||^2 over union keys (fp32, exhaustive)."""
    if W is not None:
        Wg = W.to(q.device, torch.float32); keys = keys @ Wg
    out = torch.empty(len(q), device=q.device)
    for i in range(0, len(q), QC):
        qq = q[i:i + QC] if W is None else q[i:i + QC] @ Wg
        out[i:i + QC] = 0.5 * R2.sqdist(qq, keys).min(1).values
    return out


def boot_index(n_imgs, B=BOOT_B, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return {c: rng.integers(0, n, size=(B, n)) for c, n in n_imgs.items()}


def macro_log(per_img, cats, bidx=None):
    """(1/|cats|) sum_c log(mean_i v_ci); with bidx returns the [B] bootstrap replicates."""
    if bidx is None:
        return float(np.mean([math.log(float(np.mean(per_img[c]))) for c in cats]))
    return np.mean([np.log(per_img[c][bidx[c]].mean(1)) for c in cats], axis=0)


def ci(x, level=0.95):
    a = (1 - level) / 2
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


def decide(d_pool, lo95, d_cfg, d99, cat_worst, ret_diff, D_anchor):
    """Primary gate of §5 (B4 vs B0). Returns (verdict, per-criterion booleans)."""
    if not D_anchor > 0:
        return 'INCONCLUSIVE', {'anchor_positive': False}
    c = {'P1': d_pool >= G_PASS * D_anchor, 'P2': lo95 > 0, 'P3': all(v > 0 for v in d_cfg),
         'P4': d99 >= 0, 'P5': cat_worst <= CAT_MAX, 'P6': ret_diff <= G_PASS * D_anchor}
    if all(c.values()):
        return 'PASS', c
    if d_pool < G_FAIL * D_anchor or not (c['P4'] and c['P5'] and c['P6']):
        return 'FAIL', c
    return 'INCONCLUSIVE', c


def contrast_supported(c_pool, lo_bonf, c_cfg, D_anchor):
    return bool(c_pool >= G_ID * D_anchor and lo_bonf > 0 and all(v > 0 for v in c_cfg))


# ------------------------------------------------------------------ lock
def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    nr = A.rj(A.OUT / 'NORMAL.json')
    A.wj(OUT / 'LOCK.json', {
        'sources': {s: A.sha(ROOT / s) for s in SOURCES},
        'inputs': {'r0_NORMAL': A.sha(A.OUT / 'NORMAL.json'), 'r1_CORESETS': A.sha(R1.OUT / 'CORESETS.json'),
                   'r2_NORMAL': A.sha(R2.OUT / 'NORMAL.json'), 'r2_stage_metrics': A.sha(R2.CACHE / 'stage_metrics.pt'),
                   'norm_eval': {c: [nr['per_task'][c]['sha256'], nr['per_task'][c]['eval_sha256']] for c in A.ORDER},
                   'train_features': {c: A.rj((A.FEAT / f'{c}_train.pt').with_suffix('.json'))['sha256']
                                      for c in A.ORDER}},
        'configs': {k: [v[0], v[1], list(order_of(v[0]))] for k, v in CONFIGS.items()},
        'arms': ARMS, 'ratio': RATIO, 'alpha': ALPHA, 'proj_dim': A.PROJ_DIM, 'boot': [BOOT_B, BOOT_SEED],
        'gates': {'pass': G_PASS, 'fail': G_FAIL, 'id': G_ID, 'cat_max_log': CAT_MAX, 'overlap_max': OVERLAP_MAX,
                  'i2_tol': I2_TOL},
        'persistent_bytes_per_arm': None, 's_bytes': S_BYTES, 'gpu_h_cap': GPU_H_CAP, 'torch': torch.__version__,
        'note': 'normal-only; no test image, mask or Pixel AP is opened by any phase',
        'amendment_a1': {'supersedes_lock_sha256': '922ccbf47f271e15a3c562abc39e19b16b36226c4a5a7efa0631267f39ae2bb5',
                         'reused_writes_sha256': A.sha(OUT / 'WRITES.json') if (OUT / 'WRITES.json').exists() else None}})
    A.log('LOCK', A.sha(OUT / 'LOCK.json'))


def verify_lock():
    lk = A.rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(ROOT / p) == h, 'locked source changed: ' + p)
    for n, p in (('r0_NORMAL', A.OUT / 'NORMAL.json'), ('r1_CORESETS', R1.OUT / 'CORESETS.json'),
                 ('r2_NORMAL', R2.OUT / 'NORMAL.json'), ('r2_stage_metrics', R2.CACHE / 'stage_metrics.pt')):
        K.require(A.sha(p) == lk['inputs'][n], 'input changed: ' + n)
    return lk


# ------------------------------------------------------------------ write (FIT only)
@torch.no_grad()
def write():
    lk = verify_lock()
    K.require(not (OUT / 'WRITES.json').exists(), 'no rewrite')
    CACHE.mkdir(parents=True, exist_ok=True)
    nr = A.rj(A.OUT / 'NORMAL.json')
    cs = A.rj(R1.OUT / 'CORESETS.json')
    t_start = time.monotonic()
    rec = {}
    for cfg, (oname, seed) in CONFIGS.items():
        order = order_of(oname)
        calls = []

        def get_task(t, order=order, calls=calls):
            calls.append(t)
            K.require(calls == sorted(calls) and len(set(calls)) == len(calls), 'non-causal task access')
            return A.load_norm(order[t], nr)['fit'].cuda().float()

        runs, crec = {}, {}
        for arm in ARMS:
            calls.clear(); torch.cuda.reset_peak_memory_stats()
            runs[arm] = run_stream(get_task, 15, arm, seed, keep_reader=arm in READER_ARMS,
                                   log=lambda *a, cfg=cfg: A.log('WRITE', cfg, *a))
            crec[arm] = {'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                         'tm': [r['tm'] for r in runs[arm]], 'k': [r['k'] for r in runs[arm]],
                         'cond_T_reader': [r['cond_T_reader'] for r in runs[arm]],
                         'cond_T_writer': [r['cond_T_writer'] for r in runs[arm]],
                         'cond_T_provisional': [r['cond_T_provisional'] for r in runs[arm]]}
            A.caps()
            K.require((time.monotonic() - t_start) / 3600 < GPU_H_CAP, 'GPU-h cap')
        # ---- integrity I1 / I3
        for t, c in enumerate(order):
            b0 = runs['B0'][t]['idx']
            K.require(torch.equal(runs['B5'][t]['idx'], b0), f'I1 B5==B0 {cfg} {c}')
            K.require(torch.equal(runs['B6'][t]['ref_idx'], b0), f'I1 ref==B0 {cfg} {c}')
            if seed == A.SEED:
                kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx']
                K.require(A.sha(R1.CACHE / f'mv_{c}.pt') == cs[c]['sha256'], 'R1 cache')
                K.require(torch.equal(b0, kc[:len(b0)]), f'I1 B0==R1 {cfg} {c}')
            K.require(len({runs[a][t]['k'] for a in ARMS}) == 1 and all(len(runs[a][t]['idx']) == runs[a][t]['k']
                                                                     for a in ARMS), 'I3 key count')
            K.require(all(len(set(runs[a][t]['idx'].tolist())) == runs[a][t]['k'] for a in ARMS), 'duplicate keys')
        # ---- structural: overlaps and writer-metric cosines
        def ov(a, b):
            return [len(set(runs[a][t]['idx'].tolist()) & set(runs[b][t]['idx'].tolist())) / runs[a][t]['k']
                    for t in range(15)]

        def hcos(a, b):
            out = []
            for t in range(15):
                Ha, Hb = runs[a][t]['H1'], runs[b][t]['H1']
                Ha = torch.eye(D, dtype=torch.float64) if Ha is None else Ha
                Hb = torch.eye(D, dtype=torch.float64) if Hb is None else Hb
                out.append(float((Ha * Hb).sum() / (Ha.norm() * Hb.norm())))
            return out
        crec['overlap'] = {f'{a}-{b}': ov(a, b) for a, b in (('B4', 'B6'), ('B4', 'B2'), ('B4', 'B0'), ('B6', 'B0'),
                                                              ('B2', 'B0'), ('B3', 'B4'), ('B3', 'B0'))}
        crec['writer_H_cosine'] = {f'{a}-{b}': hcos(a, b) for a, b in (('B4', 'B6'), ('B4', 'B2'), ('B4', 'B5'),
                                                                        ('B2', 'B5'))}
        for t in (0, 1):                                    # §2: B4 and B6 coincide at tasks 1 and 2
            K.require(torch.equal(runs['B4'][t]['idx'], runs['B6'][t]['idx']), f'§2 B4==B6 at t={t + 1} {cfg}')
        if cfg == 'c1':                                     # I6 determinism
            calls.clear()
            again = run_stream(get_task, 15, 'B4', seed)
            K.require(all(torch.equal(a['idx'], b['idx']) for a, b in zip(again, runs['B4'])), 'I6 determinism')
            crec['I6'] = True
        torch.save({'order': order, 'seed': seed,
                    'idx': {a: [r['idx'] for r in runs[a]] for a in ARMS},
                    'W': {a: [r['W'] for r in runs[a]] for a in READER_ARMS}}, CACHE / f'banks_{cfg}.pt')
        crec['sha256'] = A.sha(CACHE / f'banks_{cfg}.pt')
        rec[cfg] = crec
        A.log('CONFIG DONE', cfg, round((time.monotonic() - t_start) / 60, 1), 'min')
        del runs; torch.cuda.empty_cache()
    rec['_gpu_hours'] = (time.monotonic() - t_start) / 3600
    A.wj(OUT / 'WRITES.json', rec)


# ------------------------------------------------------------------ eval (EVAL / CREDIT normals only)
def load_banks(cfg, wr):
    p = CACHE / f'banks_{cfg}.pt'
    K.require(A.sha(p) == wr[cfg]['sha256'], 'bank cache changed')
    return torch.load(p, weights_only=True)


@torch.no_grad()
def evaluate():
    lk = verify_lock()
    K.require((OUT / 'WRITES.json').exists() and not (OUT / 'EVAL.json').exists(), 'phase order / no re-eval')
    wr = A.rj(OUT / 'WRITES.json')
    nr = A.rj(A.OUT / 'NORMAL.json')
    t_start = time.monotonic()
    banks = {cfg: load_banks(cfg, wr) for cfg in CONFIGS}
    keys = {}                                              # (cfg, arm, task position) -> fp16 keys (CPU)
    evalq, credq, n_img = {}, {}, {}
    for c in A.ORDER:
        fit = A.load_norm(c, nr)['fit']
        for cfg, bk in banks.items():
            pos = bk['order'].index(c)
            for arm in ARMS:
                keys[cfg, arm, pos] = fit[bk['idx'][arm][pos]]
        kc = torch.load(R1.CACHE / f'mv_{c}.pt', weights_only=True)['kc_idx']
        keys['anchor2', c] = fit[kc[:R1.n_of(0.02, len(fit))]]
        del fit
        ev = A.load_norm(c, nr, 'eval')
        evalq[c] = ev['eval_unit']; n_img[c] = len(ev['eval_images'])
        K.require(A.rj((A.FEAT / f'{c}_train.pt').with_suffix('.json'))['sha256'] == lk['inputs']['train_features'][c],
                  'train feature meta changed')
        d = A.load_feat(c, 'train')                          # file hash verified inside
        roles = K.normal_partition(d['rows'])
        K.require([d['rows'][i]['relative_path'] for i in roles['EVAL']] == ev['eval_images'], 'EVAL identity')
        credq[c] = A.unit(d['features'][roles['CREDIT']].reshape(-1, D)).half()
        K.require(not set(roles['CREDIT']) & (set(roles['EVAL']) | set(roles['FIT'])), 'CREDIT disjoint')
        del d
        A.log('LOADED', c, n_img[c], len(credq[c]) // 784)
    # I2: common reader parity with R2 (c1 = canonical order and seed)
    st = torch.load(R2.CACHE / 'stage_metrics.pt', weights_only=True)
    i2 = [float((banks['c1']['W']['B0'][s] - st[s]['G3']).norm() / st[s]['G3'].norm()) for s in range(15)]
    K.require(max(i2) <= I2_TOL, f'I2 reader parity {max(i2)}')
    res = {'I2_rel_err': i2, 'n_img': n_img}
    per = {}                                               # (cfg, arm, reader, stage, cat) -> dict of np arrays

    def union(cfg, arm, s):
        return torch.cat([keys[cfg, arm, t] for t in range(s + 1)]).cuda().float()

    for cfg, bk in banks.items():
        order = bk['order']
        Wc = bk['W']['B0']
        tau = {}
        kfin = union(cfg, 'B0', 14)
        for c in A.ORDER:                                  # coverage thresholds from CREDIT, B0, common, final
            e = energies(credq[c].cuda().float(), kfin, Wc[14])
            K.require(bool(torch.isfinite(e).all()), 'I5 credit')
            tau[c] = float(torch.quantile(e, 0.99))
        res.setdefault('tau', {})[cfg] = tau
        for arm in ARMS:
            if arm == 'B5':                                # identical indices (I1) -> identical energies
                for kk in [k for k in per if k[0] == cfg and k[1] == 'B0']:
                    per[(cfg, 'B5') + kk[2:]] = per[kk]
                continue
            readers = {'common': Wc, 'eucl': [None] * 15}
            if arm in ('B2', 'B4', 'B3'):
                readers['own'] = bk['W'][arm]
            for rname, Ws in readers.items():
                for s in range(15):
                    kk = union(cfg, arm, s)
                    todo = [(order[s], 'acq')] + ([(c, 'fin') for c in A.ORDER] if s == 14 else [])
                    for c, stg in todo:
                        if stg == 'fin' and c == order[14]:   # same bank and reader as its acquisition stage
                            per[cfg, arm, rname, 'fin', c] = per[cfg, arm, rname, 'acq', c]
                            continue
                        e = energies(evalq[c].cuda().float(), kk, Ws[s])
                        K.require(bool(torch.isfinite(e).all()), 'I5 energies')
                        p95, p99 = p_stats(e, n_img[c])
                        cov = (e.reshape(n_img[c], 784) <= tau[c]).float().mean(1)
                        per[cfg, arm, rname, stg, c] = {'p95': p95.cpu().numpy(), 'p99': p99.cpu().numpy(),
                                                        'cov': cov.cpu().numpy()}
                    del kk
            A.log('EVAL', cfg, arm, round((time.monotonic() - t_start) / 60, 1), 'min'); A.caps()
            K.require((time.monotonic() - t_start) / 3600 + wr['_gpu_hours'] < GPU_H_CAP, 'GPU-h cap')
        torch.cuda.empty_cache()
    # anchor D: B0 @ 2 % (R1 prefix) under the c1 common final reader
    k2 = torch.cat([keys['anchor2', c] for c in A.ORDER]).cuda().float()
    for c in A.ORDER:
        e = energies(evalq[c].cuda().float(), k2, banks['c1']['W']['B0'][14])
        p95, p99 = p_stats(e, n_img[c])
        per['c1', 'B0@2%', 'common', 'fin', c] = {'p95': p95.cpu().numpy(), 'p99': p99.cpu().numpy(),
                                                  'cov': np.zeros(n_img[c], np.float32)}
    res['anchor_keys'] = len(k2)
    torch.save({'per': per}, CACHE / 'per_image.pt')
    res['per_image_sha256'] = A.sha(CACHE / 'per_image.pt')
    res['gpu_hours'] = (time.monotonic() - t_start) / 3600
    A.wj(OUT / 'EVAL.json', res)
    A.log('EVAL written', round(res['gpu_hours'] * 60, 1), 'min')


# ------------------------------------------------------------------ metrics (one shot)
def metrics():
    lk = verify_lock()
    K.require((OUT / 'EVAL.json').exists() and not (OUT / 'RESULT.json').exists(), 'phase order / one shot')
    ev = A.rj(OUT / 'EVAL.json'); wr = A.rj(OUT / 'WRITES.json')
    K.require(A.sha(CACHE / 'per_image.pt') == ev['per_image_sha256'], 'per-image cache changed')
    per = torch.load(CACHE / 'per_image.pt', weights_only=False)['per']
    cats = list(A.ORDER); cfgs = list(CONFIGS)
    bidx = boot_index(ev['n_img'])

    def vals(cfg, arm, reader, stg, stat):
        return {c: per[cfg, arm, reader, stg, c][stat] for c in cats}

    def L(cfg, arm, reader='common', stg='fin', stat='p95', boot=False):
        return macro_log(vals(cfg, arm, reader, stg, stat), cats, bidx if boot else None)

    D_anchor = L('c1', 'B0') - L('c1', 'B0@2%')
    out = {'anchor_D': D_anchor, 'anchor_keys': ev['anchor_keys']}
    table = {}
    for reader in ('common', 'own', 'eucl'):
        for arm in ARMS:
            r = 'common' if (reader == 'own' and arm in ('B0', 'B5', 'B6')) else reader
            row = {}
            for stat in ('p95', 'p99', 'cov'):
                if stat == 'cov' and r != 'common':
                    continue
                row[f'L{stat}_fin'] = {cfg: L(cfg, arm, r, 'fin', stat) for cfg in cfgs}
                row[f'L{stat}_acq'] = {cfg: L(cfg, arm, r, 'acq', stat) for cfg in cfgs}
            table[reader, arm] = row
    out['table'] = {f'{r}|{a}': v for (r, a), v in table.items()}

    def delta(a, b, reader='common', stat='p95', stg='fin'):
        """L(a) - L(b) per config, pooled, bootstrap replicates (positive = b better)."""
        ra = 'common' if reader == 'own' and a in ('B0', 'B5', 'B6') else reader
        rb = 'common' if reader == 'own' and b in ('B0', 'B5', 'B6') else reader
        per_cfg = [L(cfg, a, ra, stg, stat) - L(cfg, b, rb, stg, stat) for cfg in cfgs]
        boot = np.mean([L(cfg, a, ra, stg, stat, True) - L(cfg, b, rb, stg, stat, True) for cfg in cfgs], axis=0)
        return per_cfg, float(np.mean(per_cfg)), boot

    # primary
    d_cfg, d_pool, d_boot = delta('B0', 'B4')
    lo95, hi95 = ci(d_boot)
    _, d99, _ = delta('B0', 'B4', stat='p99')
    cat_lr = {c: float(np.mean([math.log(per[cfg, 'B4', 'common', 'fin', c]['p95'].mean()
                                         / per[cfg, 'B0', 'common', 'fin', c]['p95'].mean()) for cfg in cfgs]))
              for c in cats}

    def ret(arm):
        return [float(np.mean([math.log(per[cfg, arm, 'common', 'fin', c]['p95'].mean())
                               - math.log(per[cfg, arm, 'common', 'acq', c]['p95'].mean()) for c in cats]))
                for cfg in cfgs]
    ret_b4, ret_b0 = ret('B4'), ret('B0')
    ret_diff = float(np.mean(ret_b4) - np.mean(ret_b0))
    verdict, crit = decide(d_pool, lo95, d_cfg, d99, max(cat_lr.values()), ret_diff, D_anchor)
    out['primary'] = {'delta_B0_minus_B4_per_cfg': dict(zip(cfgs, d_cfg)), 'delta_pool': d_pool,
                      'ci95': [lo95, hi95], 'delta_p99_pool': d99, 'cat_log_ratio_B4_over_B0': cat_lr,
                      'worst_cat': max(cat_lr, key=cat_lr.get), 'retention': {'B4': ret_b4, 'B0': ret_b0,
                                                                              'diff_pool': ret_diff},
                      'criteria': crit, 'thresholds': {'P1': G_PASS * D_anchor, 'fail_below': G_FAIL * D_anchor},
                      'verdict': verdict}
    # identifiability contrasts (Bonferroni 3)
    ident = {}
    for X in ('B2', 'B6', 'B3'):
        c_cfg, c_pool, c_boot = delta(X, 'B4')
        lo, hi = ci(c_boot, 1 - 0.05 / 3)
        ident[X] = {'per_cfg': dict(zip(cfgs, c_cfg)), 'pool': c_pool, 'ci_bonf': [lo, hi],
                    'supported': contrast_supported(c_pool, lo, c_cfg, D_anchor)}
    ov = {p: [float(np.mean(wr[cfg]['overlap'][p][2:])) for cfg in cfgs] for p in ('B4-B6', 'B4-B2', 'B4-B0')}
    structural = {'B6': all(v >= OVERLAP_MAX for v in ov['B4-B6']), 'B2': all(v >= OVERLAP_MAX for v in ov['B4-B2'])}
    not_isolated = all(v >= 1.0 for v in ov['B4-B0'])
    if verdict == 'PASS':
        if structural['B2'] or structural['B6'] or not (ident['B2']['supported'] and ident['B6']['supported']):
            nl = 'NON_IDENTIFIABLE'
        else:
            nl = 'NL_IDENTIFIED' + ('' if ident['B3']['supported'] else ' (feedback only; timescale not supported)')
    else:
        nl = 'NOT_EVALUATED (primary ' + verdict + ')'
        if structural['B2'] or structural['B6']:
            nl += '; structural NON_IDENTIFIABLE'
    out['identifiability'] = {'contrasts': ident, 'overlap_tasks3plus_mean': ov, 'structural_rule': structural,
                              'not_isolated': not_isolated, 'verdict': nl}
    # secondary end-to-end (own reader) and diagnostic Euclidean reader — descriptive only
    out['secondary_own_reader'] = {X: dict(zip(['per_cfg', 'pool'], delta('B0', X, 'own')[:2])) for X in ARMS}
    out['diagnostic_eucl_reader'] = {X: dict(zip(['per_cfg', 'pool'], delta('B0', X, 'eucl')[:2])) for X in ARMS}
    out['coverage_common_fin'] = {X: {cfg: L(cfg, X, 'common', 'fin', 'cov') for cfg in cfgs} for X in ARMS}
    # accounting
    k_tot = sum(wr['c1']['B0']['k'])
    out['bytes'] = {'keys': k_tot * D * 2, 'slow_statistic': S_BYTES, 'total_per_arm': k_tot * D * 2 + S_BYTES}

    def cost(cfg, arm):
        tm = wr[cfg][arm]['tm']
        return {k: float(sum(t.get(k, 0.0) for t in tm)) for k in sorted({k for t in tm for k in t})}
    out['update_cost_s_15_tasks'] = {arm: {cfg: cost(cfg, arm) for cfg in cfgs} for arm in ARMS}
    out['peak_cuda_bytes'] = {arm: max(wr[cfg][arm]['peak_cuda_bytes'] for cfg in cfgs) for arm in ARMS}
    conds = [x for cfg in cfgs for arm in ARMS for key in ('cond_T_reader', 'cond_T_writer', 'cond_T_provisional')
             for x in wr[cfg][arm][key] if x is not None]
    out['stability'] = {'max_cond_T': max(conds), 'I2_max_rel_err': max(ev['I2_rel_err']), 'I6': wr['c1'].get('I6'),
                        'gpu_hours_total': wr['_gpu_hours'] + ev['gpu_hours']}
    out['lock_sha256'] = A.sha(OUT / 'LOCK.json')
    A.wj(OUT / 'RESULT.json', out)
    A.wj(OUT / 'FINAL_SEAL.json', {n: A.sha(OUT / n) for n in ('LOCK.json', 'WRITES.json', 'EVAL.json', 'RESULT.json')}
         | {'per_image.pt': ev['per_image_sha256']})
    A.log('RESULT', verdict, nl, 'D', round(D_anchor, 5), 'delta', round(d_pool, 5))


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=('lock', 'write', 'eval', 'metrics'))
    {'lock': lock, 'write': write, 'eval': evaluate, 'metrics': metrics}[ap.parse_args().phase]()
