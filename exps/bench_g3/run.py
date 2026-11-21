"""BENCH-G3: one-shot confirmation benchmark of the frozen G3 (agents/tasks/task_bench_g3.md). No training.

PREP (normal-only; no test image, mask or label of a confirmation cohort is opened):
    python -B -m exps.bench_g3.run custody        # MPDD manifest: paths, file hashes, layout, duplicates, counts
    python -B -m exps.bench_g3.run extract_train  # MPDD train/good tokens (fp32 encoder, identical to VBENCH-00)
    python -B -m exps.bench_g3.run normal         # banks + statistics for every dataset x seed
    python -B -m exps.bench_g3.run parity         # VisA-DEV (consumed) parity with sealed R2b
    python -B -m exps.bench_g3.run cost           # latency (train/good) + per-task update cost
    python -B -m exps.bench_g3.run seal_prep
BENCH (after the task file is LOCKED):
    python -B -m exps.bench_g3.run lock
    python -B -m exps.bench_g3.run extract_test   # test tokens (images only)
    python -B -m exps.bench_g3.run score          # stage-wise energies (no label read)
    python -B -m exps.bench_g3.run metrics        # labels / masks opened once -> RESULT.json, FINAL_SEAL.json
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

from exps.calm02 import data as CD, native as N
from exps.dinomaly import encoder as E
from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1
from exps.nanm_r2 import run as R2
from exps.nanm_r2b import run as B
from exps.nar_pilot_00 import core as K

ROOT = A.ROOT
OUT = ROOT / 'results/hope_cad/bench_g3'
CACHE = Path.home() / 'hope_cad_cache/bench_g3'
FEAT, NORM, SCORE = CACHE / 'features', CACHE / 'normal', CACHE / 'scores'
TASK = ROOT / 'agents/tasks/task_bench_g3.md'
MPDD = ROOT / 'data/mpdd'
MPDD_ORDER = ('bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes')
DSETS = ('mpdd', 'visa')
D, KEY = 768, 2 * 768
SEEDS = (2711, 2712, 2713)
EXTRA_G3 = D * D * 4 + 8                  # S fp32 + n (fp64 count)            = 2,359,304
EXTRA_G1 = D * D * 4 + D * 4 + 8          # feature covariance + mean + n      = 2,362,376
C_BYTES = 30_720_000                      # CADIC 10k coreset, 10,000 x 768 fp32
ENCODER_BYTES = 144_806_400
QC = 512
BOOT_B, BOOT_SEED, BINS = 2000, 20261120, 8192
BIN_TOL = 1e-3
PARITY_TOL = 1e-6
DELTA_MIN, ALPHA = 0.005, 0.05
PUBLISHED = {'CADIC': {'p_aupr': 0.438, 'i_auroc': 0.891}, 'ReplayCAD': {'p_aupr': 0.415, 'i_auroc': 0.903},
             'UCAD': {'p_aupr': 0.300, 'i_auroc': 0.874}}
GPU_CAP_S, CPU_CAP_S, DISK_CAP = 5400, 4 * 3600, 30 * 10 ** 9
SOURCES = ('exps/bench_g3/run.py', 'exps/bench_g3/test_run.py', 'agents/tasks/task_bench_g3.md',
           'exps/nanm_r0/audit.py', 'exps/nanm_r1/run.py', 'exps/nanm_r2/run.py', 'exps/nanm_r2b/run.py',
           'exps/nar_lite_01/run.py', 'exps/nar_b14_01/run.py', 'exps/calm02/native.py', 'exps/calm02/data.py',
           'exps/cadfinal/metrics.py', 'exps/bench_aud_00/run.py', 'exps/tcs_00/run.py', 'exps/vbench_00/run.py',
           'exps/nar_pilot_00/core.py', 'exps/dinomaly/encoder.py')


# ------------------------------------------------------------------ lazy modules (heavy imports)
def T():
    from exps.tcs_00 import run as t
    return t


def V():
    from exps.vbench_00 import run as v
    return v


def B14():
    from exps.nar_b14_01 import run as b
    return b


def NL():
    from exps.nar_lite_01 import run as n
    return n


# ------------------------------------------------------------------ pure functions (unit-tested)
def a0b_sizes(n_fit):
    """Euclidean k-centre sizes buying G3's EXTRA_G3 bytes: common ratio (B.control_sizes), then the overshoot is
    trimmed one key at a time from the largest tasks (ties: earlier task), so that sum = 1 % keys + ceil(EXTRA/KEY)."""
    base = [R1.n_of(0.01, n) for n in n_fit]
    need = sum(base) + math.ceil(EXTRA_G3 / KEY)
    r, sizes = B.control_sizes(n_fit, EXTRA_G3, 0.01)
    sizes = list(sizes)
    over = sum(sizes) - need
    K.require(0 <= over <= len(sizes), f'A0b overshoot {over}')
    for i in sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))[:over]:
        sizes[i] -= 1
    K.require(sum(sizes) == need and all(s >= b for s, b in zip(sizes, base)), 'A0b sizes')
    return r, sizes


def c_sizes(n_fit, extra):
    """Largest common ratio r with sum ceil(r n_t) * KEY + extra <= C_BYTES (CADIC-bytes arms)."""
    fits = lambda r: sum(R1.n_of(r, n) for n in n_fit) * KEY + extra <= C_BYTES
    K.require(not fits(0.01), 'C budget exceeds the 1 % memory; arms would not be prefixes')
    lo, hi = 0.0, 0.01
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    sizes = [R1.n_of(lo, n) for n in n_fit]
    K.require(all(s > 0 for s in sizes) and fits(lo), 'C sizes')
    return lo, sizes


def chaudhry(mat, T_):
    """mat[(l, j)] for l >= j (0-based stages). f_j = max_{l in j..T-2} a_{l,j} - a_{T-1,j}; FM = mean_{j<T-1} f_j.
    Also the signed acquisition - final mean over all j."""
    f = [max(mat[l, j] for l in range(j, T_ - 1)) - mat[T_ - 1, j] for j in range(T_ - 1)]
    return {'fm': float(np.mean(f)), 'per_task': f,
            'acq_minus_final': float(np.mean([mat[j, j] - mat[T_ - 1, j] for j in range(T_)]))}


def holm(pvals, alpha=ALPHA):
    """Holm step-down; returns {name: (adjusted p, rejected)}."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m, out, running, stop = len(names), {}, 0., False
    for i, k in enumerate(names):
        running = max(running, min(1., (m - i) * pvals[k]))
        rej = (not stop) and pvals[k] <= alpha / (m - i)
        stop = stop or not rej
        out[k] = (running, rej)
    return out


def boot_p(d):
    """Two-sided bootstrap p-value of H0: delta = 0 from replicate deltas."""
    d = np.asarray(d)
    return float(min(1., 2 * min((d <= 0).mean(), (d >= 0).mean())))


def maps224(maps392):
    """A1 392 maps -> 224 x 224 bilinear (align_corners False, antialias False, full image)."""
    return F.interpolate(maps392[:, None].float(), size=(224, 224), mode='bilinear', align_corners=False,
                         antialias=False)[:, 0]


def hist224(maps, masks, lo, hi, bins=BINS):
    pos = np.zeros((len(maps), bins), np.float32); tot = np.zeros((len(maps), bins), np.float32)
    for i, (m, y) in enumerate(zip(maps, masks)):
        b = np.clip(((m.ravel() - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
        tot[i] = np.bincount(b, minlength=bins); pos[i] = np.bincount(b, weights=y.ravel().astype(np.float64),
                                                                       minlength=bins)
    return pos, tot


def label_of(published, lo, hi):
    return 'ABOVE_PUBLISHED' if lo > published else 'BELOW_PUBLISHED' if hi < published else 'WITHIN'


# ------------------------------------------------------------------ dataset identities (paths only)
def order(ds):
    return MPDD_ORDER if ds == 'mpdd' else tuple(CD.VISA_ORDER)


def data_root(ds):
    return MPDD if ds == 'mpdd' else ROOT / 'data/visa'


_VTEST = {}


def train_rows(ds, c):
    return CD._mvtec(MPDD, c)[0] if ds == 'mpdd' else V().train_rows(c)


def test_rows(ds, c):
    """Identities (path, label, mask path). Labels/mask paths are consumed only by the metrics phase."""
    if ds == 'mpdd':
        return CD._mvtec(MPDD, c)[1]
    if not _VTEST:
        _VTEST.update(V().test_rows())
    return _VTEST[c]


def fit_index(ds, rows):
    return K.normal_partition(rows)['FIT'] if ds == 'mpdd' else T().roles_for(rows)['FIT']


def roles(ds, rows):
    return K.normal_partition(rows) if ds == 'mpdd' else T().roles_for(rows)


def n_fit_list(ds):
    return [784 * len(fit_index(ds, train_rows(ds, c))) for c in order(ds)]


def sizes_for(ds):
    n_fit = n_fit_list(ds)
    out = {'n_fit': n_fit, 'g3': [R1.n_of(0.01, n) for n in n_fit]}
    out['a0b_ratio'], out['a0b'] = a0b_sizes(n_fit)
    if ds == 'visa':
        out['g3c_ratio'], out['g3c'] = c_sizes(n_fit, EXTRA_G3)
        out['a0c30_ratio'], out['a0c30'] = c_sizes(n_fit, 0)
        out['r2b_a0c'] = B.control_sizes(n_fit, R2.ACC_BYTES, 0.01)[1]      # parity only (R2b A0c definition)
    return out


def byte_table(sz, ds):
    k = lambda key: KEY * sum(sz[key])
    t = {'G3': k('g3') + EXTRA_G3, 'A0b': k('a0b'), 'G1b': k('g3') + EXTRA_G1, 'E-full': KEY * sum(sz['n_fit']),
         'keys_1pct': k('g3')}
    if ds == 'visa':
        t.update({'G3@C': k('g3c') + EXTRA_G3, 'A0@C': k('a0c30')})
    t['A0b_minus_G3'] = t['A0b'] - t['G3']
    return t


# ------------------------------------------------------------------ token caches
def _feat_meta(p):
    return A.rj(p.with_suffix('.json'))


def load_train(ds, c):
    if ds == 'visa':
        return V().load_train(A.rj(V().OUT / 'LOCK.json'), c)
    p = FEAT / f'mpdd_{c}_train.pt'
    K.require(A.sha(p) == _feat_meta(p)['sha256'], f'MPDD train cache changed: {c}')
    data = torch.load(p, weights_only=True, mmap=True)
    K.require([r['relative_path'] for r in data['rows']] == [r['relative_path'] for r in train_rows(ds, c)], 'rows')
    return data


@torch.no_grad()
def fit_tokens(ds, c):
    """Unit-norm fp16 FIT tokens on CUDA, identical to the R2b VisA path."""
    data = load_train(ds, c)
    idx = torch.as_tensor(fit_index(ds, data['rows']))
    fit = torch.cat([A.unit(data['features'][idx[i:i + 64]].reshape(-1, D).cuda()).half()
                     for i in range(0, len(idx), 64)])
    del data
    return fit


@torch.no_grad()
def _extract(enc, root, rows, target):
    b14 = B14()
    feats = torch.empty(len(rows), 784, D, dtype=torch.float32)
    t0 = time.monotonic()
    for i in range(0, len(rows), 8):
        x = torch.stack([E.preprocess(root / r['relative_path']) for r in rows[i:i + 8]]).cuda()
        feats[i:i + len(x)] = b14.tokens(enc, x).cpu()
    K.require(bool(torch.isfinite(feats).all()), 'feature finiteness')
    target.parent.mkdir(parents=True, exist_ok=True)
    keep = [{'relative_path': r['relative_path']} for r in rows]               # identities only, no label
    torch.save({'rows': keep, 'features': feats}, target)
    A.wj(target.with_suffix('.json'), {'sha256': A.sha(target), 'n': len(rows), 'seconds': time.monotonic() - t0,
                                       'encoder_sha256': b14.ENC_SHA, 'encoder_dtype': 'float32 (E.load)',
                                       'source_images': {r['relative_path']: A.sha(root / r['relative_path'])
                                                         for r in rows}})
    return time.monotonic() - t0


def encoder():
    from exps.nar_pilot_00 import full as R
    R.configure()
    b14 = B14()
    K.require(A.sha(b14.ENC_PATH) == b14.ENC_SHA, 'DINOv2 checkpoint hash')
    return E.load(b14.ENC_PATH, 'base', 'cuda')


# ------------------------------------------------------------------ PREP: custody
def custody():
    K.require(not (OUT / 'MPDD_MANIFEST.json').exists(), 'custody exists')
    files = sorted(p for p in MPDD.rglob('*') if p.is_file())
    man = {p.relative_to(MPDD).as_posix(): A.sha(p) for p in files}
    cats = sorted({k.split('/')[0] for k in man})
    K.require(tuple(cats) == tuple(sorted(MPDD_ORDER)), f'MPDD categories {cats}')
    counts, dup = {}, {}
    for c in MPDD_ORDER:
        tr, te = CD._mvtec(MPDD, c)
        K.require(len(tr) >= 10, f'too few train/good: {c}')
        K.require(all(r['mask_path'] is None or r['mask_path'] in man for r in te), f'missing mask: {c}')
        defects = sorted({r['relative_path'].split('/')[2] for r in te})
        counts[c] = {'train_good': len(tr), 'test': len(te), 'test_good': sum(r['label'] == 0 for r in te),
                     'test_anomalous': sum(r['label'] for r in te), 'defect_types': defects,
                     'n_fit_images': len(K.normal_partition(tr)['FIT'])}
        trh = {man[r['relative_path']] for r in tr}
        dup[c] = sorted(r['relative_path'] for r in te if man[r['relative_path']] in trh)
    layout_ok = all(len(k.split('/')) == 4 and k.split('/')[1] in ('train', 'test', 'ground_truth') for k in man)
    A.wj(OUT / 'MPDD_MANIFEST.json', {
        'source': 'github.com/stepanje/MPDD -> official VUT SharePoint share (anonymous), downloaded 2026-11-19',
        'license': 'CC BY-NC-SA 4.0 (LICENSE in github.com/stepanje/MPDD); research, non-commercial use',
        'citation': 'Jezek et al., ICUMT 2021, doi 10.1109/ICUMT54235.2021.9631567',
        'n_files': len(man), 'total_bytes': sum(p.stat().st_size for p in files), 'layout_mvtec_like': layout_ok,
        'counts': counts, 'n_images': sum(v['train_good'] + v['test'] for v in counts.values()),
        'train_test_exact_duplicates': dup, 'files': man,
        'note': 'test/ and ground_truth/ files hashed as bytes only; no image or mask decoded'})
    K.require(layout_ok, 'MPDD layout')
    A.log('CUSTODY', len(man), json.dumps({c: (v['train_good'], v['test']) for c, v in counts.items()}),
          'dups', sum(len(v) for v in dup.values()))


def extract_train():
    man = A.rj(OUT / 'MPDD_MANIFEST.json')
    enc = encoder()
    rec = {}
    for c in MPDD_ORDER:
        p = FEAT / f'mpdd_{c}_train.pt'
        if p.exists():
            continue
        rows = train_rows('mpdd', c)
        K.require(all(man['files'][r['relative_path']] == A.sha(MPDD / r['relative_path']) for r in rows), 'custody')
        rec[c] = _extract(enc, MPDD, rows, p)
        A.log('EXTRACT train', c, len(rows), round(rec[c], 1))
    A.wj(OUT / 'logs/extract_train.json', rec)


# ------------------------------------------------------------------ PREP: normal phases
@torch.no_grad()
def normal(only=None):
    """One dataset per call (only='visa' / 'mpdd'); NORMAL.json is assembled once both exist."""
    K.require(not (OUT / 'NORMAL.json').exists(), 'no recompute')
    NORM.mkdir(parents=True, exist_ok=True)
    for ds in ([only] if only else DSETS):
        if not (OUT / f'normal_{ds}.json').exists():
            _normal_ds(ds)
    if all((OUT / f'normal_{ds}.json').exists() for ds in DSETS):
        A.wj(OUT / 'NORMAL.json', {ds: A.rj(OUT / f'normal_{ds}.json') for ds in DSETS})


@torch.no_grad()
def _normal_ds(ds):
    rec, t_all = {}, time.monotonic()
    if True:
        sz = sizes_for(ds)
        rec[ds] = {'sizes': sz, 'bytes': byte_table(sz, ds), 'seeds': {}}
        K.require(0 < rec[ds]['bytes']['A0b_minus_G3'] <= KEY, 'byte parity A0b vs G3')
        for seed in SEEDS:
            files, stats = {}, []
            for t, c in enumerate(order(ds)):
                t0 = time.monotonic()
                fit = fit_tokens(ds, c)
                n = len(fit)
                K.require(n == sz['n_fit'][t], f'FIT size {ds} {c}')
                need = [sz['g3'][t], sz['a0b'][t]] + ([sz['g3c'][t], sz['a0c30'][t]] if ds == 'visa' else [])
                if ds == 'visa' and seed == 2711:
                    need.append(sz['r2b_a0c'][t])
                kc = A.greedy_coreset(fit, max(need), seed=seed)
                f32 = fit.float()
                S1, n1 = B.residual_stats(f32, kc[:sz['g3'][t]].cuda())
                st = {'kc': kc, 'keys': fit[kc.cuda()].cpu(), 'S1': S1, 'n1': n1, 'n': n}
                if ds == 'visa':
                    st['SC'], st['nC'] = B.residual_stats(f32, kc[:sz['g3c'][t]].cuda())
                Sx = torch.zeros(D, D, dtype=torch.float64, device='cuda'); s1 = torch.zeros(D, dtype=torch.float64,
                                                                                              device='cuda')
                for i in range(0, n, 8192):
                    x = f32[i:i + 8192]
                    Sx += (x.T @ x).double(); s1 += x.sum(0).double()
                st['Sx'], st['s1'] = Sx.cpu(), s1.cpu()
                p = NORM / f'{ds}_s{seed}_{c}.pt'
                torch.save(st, p)
                files[p.name] = A.sha(p)
                stats.append(st)
                A.log('NORMAL', ds, seed, c, n, len(kc), n1, round(time.monotonic() - t0, 1)); A.caps()
                del fit, f32, Sx; torch.cuda.empty_cache()
            stage, conds = {}, {}
            Tn = len(order(ds))
            for s in range(Tn):
                seen = stats[:s + 1]
                W3, c3 = B.whiten(sum(x['S1'] for x in seen), sum(x['n1'] for x in seen))
                nn_ = sum(x['n'] for x in seen); mu = sum(x['s1'] for x in seen) / nn_
                W1, c1 = R2.inv_sqrt_psd(R2.shrink(sum(x['Sx'] for x in seen) / nn_ - torch.outer(mu, mu)))
                stage[s] = {'G3': W3.float(), 'G1b': W1.float()}
                conds[s] = {'G3': c3, 'G1b': c1}
                if ds == 'visa':
                    WC, cc = B.whiten(sum(x['SC'] for x in seen), sum(x['nC'] for x in seen))
                    stage[s]['G3@C'] = WC.float(); conds[s]['G3@C'] = cc
            p = NORM / f'{ds}_s{seed}_stage.pt'
            torch.save(stage, p)
            files[p.name] = A.sha(p)
            rec[ds]['seeds'][seed] = {'files': files, 'cond': conds}
            del stats
    rec[ds]['seconds'] = time.monotonic() - t_all
    A.wj(OUT / f'normal_{ds}.json', rec[ds])


def nrec():
    return {ds: A.rj(OUT / f'normal_{ds}.json') for ds in DSETS if (OUT / f'normal_{ds}.json').exists()}


def load_norm(ds, seed, c, rec):
    p = NORM / f'{ds}_s{seed}_{c}.pt'
    K.require(A.sha(p) == rec[ds]['seeds'][str(seed)]['files'][p.name], f'normal artifact changed {p.name}')
    return torch.load(p, weights_only=True)


def load_stage(ds, seed, rec):
    p = NORM / f'{ds}_s{seed}_stage.pt'
    K.require(A.sha(p) == rec[ds]['seeds'][str(seed)]['files'][p.name], f'stage artifact changed {p.name}')
    return torch.load(p, weights_only=True)


# ------------------------------------------------------------------ scoring kernels (shared by parity and score)
@torch.no_grad()
def euclid(q16, keys, out=None):
    """0.5 min ||q - m||^2, fp32, chunked; q16 on CPU (fp16 unit tokens), keys fp32 on CUDA."""
    out = torch.empty(len(q16)) if out is None else out
    for i in range(0, len(q16), QC):
        out[i:i + QC] = (0.5 * R2.sqdist(q16[i:i + QC].cuda().float(), keys).min(1).values).cpu()
    return out


@torch.no_grad()
def whitened(q16, keys, W):
    mw = keys @ W
    out = torch.empty(len(q16))
    for i in range(0, len(q16), QC):
        out[i:i + QC] = (0.5 * R2.sqdist(q16[i:i + QC].cuda().float() @ W, mw).min(1).values).cpu()
    return out


# ------------------------------------------------------------------ PREP: VisA-DEV parity (consumed DEV)
@torch.no_grad()
def parity():
    K.require(not (OUT / 'PARITY.json').exists(), 'no rerun')
    rec = nrec()
    sz = rec['visa']['sizes']
    vis = A.rj(B.OUT / 'VISA_COMPLETE.json')
    K.require(vis['control_sizes'] == sz['r2b_a0c'], 'R2b A0c sizes')
    r1lk = A.rj(R1.OUT / 'LOCK.json')
    W = load_stage('visa', 2711, rec)
    ords = order('visa')
    store = [load_norm('visa', 2711, c, rec) for c in ords]
    out, par_g3, par_a0c, par_chunk = {}, 0., 0., 0.
    grids = {}
    for i, c in enumerate(ords):
        p = R1.DEVR / f'devr_{c}.pt'
        K.require(A.sha(p) == r1lk['visa_devr'][c], 'devr hash')
        d = torch.load(p, weights_only=False)
        q = A.unit(d['features'].float().reshape(-1, D)).half().cuda().float(); n = len(d['rows'])
        a0c = torch.empty(12, len(q))
        for t in range(12):                                                    # R2b code path, unchunked
            a0c[t] = (0.5 * R2.sqdist(q, store[t]['keys'][:sz['r2b_a0c'][t]].cuda().float()).min(1).values).cpu()
        Wg = W[11]['G3'].cuda()
        mw = torch.cat([x['keys'][:sz['g3'][j]] for j, x in enumerate(store)]).cuda().float() @ Wg
        g3 = (0.5 * R2.sqdist(q @ Wg, mw).min(1).values).cpu().reshape(n, 28, 28)
        ref = torch.load(B.OUT / f'visa_{c}.pt', weights_only=False)
        par_g3 = max(par_g3, float((g3 - ref['g3']['final']).abs().max()))
        par_a0c = max(par_a0c, float((a0c.reshape(12, n, 28, 28) - ref['per_task']['A0c']).abs().max()))
        q16 = A.unit(d['features'].float().reshape(-1, D)).half()               # benchmark kernel, chunked
        g3k = whitened(q16, torch.cat([x['keys'][:sz['g3'][j]] for j, x in enumerate(store)]).cuda().float(), Wg)
        par_chunk = max(par_chunk, float((g3k.reshape(n, 28, 28) - g3).abs().max()))
        grids[c] = {'G3': g3, 'A0c': a0c.reshape(12, n, 28, 28).amin(0), 'rows': d['rows']}
        del d, q
    macro = {}
    for arm in ('G3', 'A0c'):
        aps = []
        for c in ords:
            rows = grids[c]['rows']
            hws = [N.native_hw(V().VISA / r['relative_path']) for r in rows]
            masks = [N.native_mask(V().VISA / r['mask_path'] if r['mask_path'] else None, hw)
                     for r, hw in zip(rows, hws)]
            aps.append(A.exact_ap(R2.maps_of(grids[c][arm]), hws, masks))
        macro[arm] = float(np.mean(aps))
    r2b = A.rj(B.OUT / 'RESULT.json')['macro']
    ref = {'G3': r2b['visa|G3|final']['pixel_ap'], 'A0c': r2b['visa|A0c|final']['pixel_ap']}
    res = {'macro_dev_pixel_ap': macro, 'r2b_reference': ref,
           'abs_diff': {k: abs(macro[k] - ref[k]) for k in macro},
           'grid_max_abs_diff_vs_r2b': {'G3_final': par_g3, 'A0c_per_task': par_a0c},
           'benchmark_kernel_vs_r2b_path_max_abs': par_chunk,
           'pass': all(abs(macro[k] - ref[k]) <= PARITY_TOL for k in macro)}
    A.wj(OUT / 'PARITY.json', res)
    A.log('PARITY', json.dumps(res))
    K.require(res['pass'], 'VisA-DEV parity failed')


# ------------------------------------------------------------------ PREP: cost (train/good only)
def _q(ts):
    ts = np.asarray(ts) * 1e3
    return {'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, .95)), 'mean_ms': float(ts.mean())}


@torch.no_grad()
def cost():
    K.require(not (OUT / 'COST.json').exists(), 'no rerun')
    rec = A.rj(OUT / 'NORMAL.json')
    nl = NL()
    out = {'device': torch.cuda.get_device_name(0)}
    for ds in DSETS:
        sz = rec[ds]['sizes']; ords = order(ds)
        W = load_stage(ds, 2711, rec)
        store = [load_norm(ds, 2711, c, rec) for c in ords]
        upd, acc, n_acc = {}, torch.zeros(D, D, dtype=torch.float64, device='cuda'), 0
        for t, c in enumerate(ords):                                   # per-task update cost from cached FIT tokens
            fit = fit_tokens(ds, c).float()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            kc = A.greedy_coreset(fit, sz['g3'][t], seed=2711)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            S, cnt = B.residual_stats(fit, kc.cuda())
            acc += S.cuda(); n_acc += cnt
            torch.cuda.synchronize(); t2 = time.perf_counter()
            Wt, _ = B.whiten(acc.cpu(), n_acc)
            torch.cuda.synchronize(); t3 = time.perf_counter()
            mw = torch.cat([x['keys'][:sz['g3'][j]] for j, x in enumerate(store[:t + 1])]).cuda().float() @ \
                Wt.float().cuda()
            torch.cuda.synchronize(); t4 = time.perf_counter()
            K.require(torch.equal(kc, store[t]['kc'][:sz['g3'][t]]), 'k-centre prefix reproducibility')
            upd[c] = {'kcentre_s': t1 - t0, 'residual_stats_s': t2 - t1, 'geometry_s': t3 - t2,
                      'union_rebuild_s': t4 - t3, 'total_s': t4 - t0, 'union_keys': len(mw)}
            A.log('UPD', ds, c, {k: round(v, 3) for k, v in upd[c].items()})
            del fit, mw; torch.cuda.empty_cache()
        enc = nl.lite_encoder(torch.bfloat16)
        c0 = ords[0]
        rows = train_rows(ds, c0)
        idx = roles(ds, rows)['EVAL'][:50]
        K.require(len(idx) == 50, 'latency images')
        paths = [data_root(ds) / rows[i]['relative_path'] for i in idx]
        inf = {}
        for arm in ('A0b', 'G3'):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            key = 'a0b' if arm == 'A0b' else 'g3'
            memc = torch.cat([x['keys'][:sz[key][j]] for j, x in enumerate(store)]).cuda().float()
            Wg = W[len(ords) - 1]['G3'].cuda()
            if arm == 'G3':
                memc = memc @ Wg
            tt, te, tr = [], [], []
            for k in range(53):
                pth = paths[max(0, k - 3)]
                t0 = time.perf_counter(); x = E.preprocess(pth)[None]; t1 = time.perf_counter()
                torch.cuda.synchronize(); t2 = time.perf_counter()
                z = A.unit(nl.tokens(enc, x.cuda().to(torch.bfloat16))[0].float())
                torch.cuda.synchronize(); t3 = time.perf_counter()
                if arm == 'G3':
                    z = z @ Wg
                e = 0.5 * R2.sqdist(z, memc).min(1).values.reshape(1, 28, 28)
                sm = R2.maps_of(e.cpu())
                sm.flatten(1).topk(max(1, int(nl.TOP * 392 * 392)), dim=1).values.mean(1)
                torch.cuda.synchronize(); t4 = time.perf_counter()
                if k >= 3:
                    tt.append(t4 - t2 + t1 - t0); te.append(t3 - t2); tr.append(t4 - t3)
            inf[arm] = {'end_to_end': _q(tt), 'encoder': _q(te), 'retrieval_and_readout': _q(tr),
                        'memory_keys': len(memc), 'peak_cuda_bytes_above_encoder':
                            torch.cuda.max_memory_allocated() - base}
            A.log('INF', ds, arm, round(inf[arm]['end_to_end']['p50_ms'], 1))
            del memc
        out[ds] = {'update_per_task': upd, 'update_total': {k: sum(v[k] for v in upd.values())
                                                            for k in ('kcentre_s', 'residual_stats_s',
                                                                      'geometry_s', 'union_rebuild_s', 'total_s')},
                   'inference_batch1': inf, 'latency_images': f'{c0} train/good EVAL role, first 50 (3 warm-up)',
                   'whitened_cache_bytes_fp32': 4 * D * sum(sz['g3']),
                   'whitened_cache_bytes_fp16': KEY * sum(sz['g3'])}
        del enc, store; torch.cuda.empty_cache()
    out['encoder_bytes'] = ENCODER_BYTES
    out['note'] = 'lite bf16 encoder for latency (R2b cost method); benchmark tokens use the fp32 encoder'
    A.wj(OUT / 'COST.json', out)


def seal_prep():
    K.require(not (OUT / 'PREP_SEAL.json').exists(), 'sealed')
    files = {n: A.sha(OUT / n) for n in ('MPDD_MANIFEST.json', 'NORMAL.json', 'PARITY.json', 'COST.json')}
    feats = {p.name: _feat_meta(p)['sha256'] for p in sorted(FEAT.glob('mpdd_*_train.pt'))}
    for n, h in feats.items():
        K.require(A.sha(FEAT / n) == h, 'feature cache changed ' + n)
    K.require(A.rj(OUT / 'PARITY.json')['pass'], 'parity')
    A.wj(OUT / 'PREP_SEAL.json', {'files': files, 'mpdd_train_features': feats,
                                 'task_sha256_at_prep_seal': A.sha(TASK)})
    A.log('PREP_SEAL', A.sha(OUT / 'PREP_SEAL.json'))


# ------------------------------------------------------------------ lock
def test_manifest(ds):
    return {c: [r['relative_path'] for r in test_rows(ds, c)] for c in order(ds)}


def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    K.require('Status: LOCKED' in TASK.read_text(), 'task file not LOCKED')
    K.require(not any(FEAT.glob('*_test.pt')), 'test tokens exist before lock')
    ps = A.rj(OUT / 'PREP_SEAL.json')
    for n, h in ps['files'].items():
        K.require(A.sha(OUT / n) == h, 'prep artifact changed ' + n)
    tm = {ds: test_manifest(ds) for ds in DSETS}
    K.require(sum(len(v) for v in tm['visa'].values()) == 2162, 'VisA test count')
    A.wj(OUT / 'LOCK.json', {
        'sources': {s: A.sha(ROOT / s) for s in SOURCES}, 'prep_seal_sha256': A.sha(OUT / 'PREP_SEAL.json'),
        'test_manifest_sha256': {ds: K_text(tm[ds]) for ds in DSETS},
        'test_counts': {ds: {c: len(v) for c, v in tm[ds].items()} for ds in DSETS},
        'seeds': SEEDS, 'boot': [BOOT_B, BOOT_SEED, BINS], 'bin_tol': BIN_TOL, 'delta_min': DELTA_MIN,
        'alpha': ALPHA, 'published': PUBLISHED, 'caps': {'gpu_s': GPU_CAP_S, 'cpu_s': CPU_CAP_S, 'disk': DISK_CAP},
        'torch': torch.__version__})
    A.log('LOCK', A.sha(OUT / 'LOCK.json'))


def K_text(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def verify_lock():
    lk = A.rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(ROOT / p) == h, 'locked source changed: ' + p)
    ps = A.rj(OUT / 'PREP_SEAL.json')
    K.require(A.sha(OUT / 'PREP_SEAL.json') == lk['prep_seal_sha256'], 'prep seal changed')
    for n, h in ps['files'].items():
        K.require(A.sha(OUT / n) == h, 'prep artifact changed ' + n)
    return lk


# ------------------------------------------------------------------ extract test tokens (images only)
def extract_test():
    lk = verify_lock()
    K.require(not (OUT / 'EXTRACT_COMPLETE.json').exists(), 'one shot')
    enc = encoder()
    rec, t0 = {}, time.monotonic()
    for ds in DSETS:
        tm = test_manifest(ds)
        K.require(K_text(tm) == lk['test_manifest_sha256'][ds], 'test manifest changed')
        for c in order(ds):
            p = FEAT / f'{ds}_{c}_test.pt'
            if not p.exists():
                s = _extract(enc, data_root(ds), [{'relative_path': r} for r in tm[c]], p)
                A.log('EXTRACT test', ds, c, len(tm[c]), round(s, 1))
            rec[p.name] = _feat_meta(p)['sha256']
    A.wj(OUT / 'EXTRACT_COMPLETE.json', {'files': rec, 'seconds': time.monotonic() - t0})


def load_test(ds, c, ex):
    p = FEAT / f'{ds}_{c}_test.pt'
    K.require(A.sha(p) == ex['files'][p.name], 'test cache changed ' + p.name)
    d = torch.load(p, weights_only=True)
    return A.unit(d['features'].reshape(-1, D)).half(), [r['relative_path'] for r in d['rows']]


# ------------------------------------------------------------------ score (no label read)
def arms_for(ds):
    whit = ['G3', 'G1b'] + (['G3@C'] if ds == 'visa' else [])
    euc = ['A0b'] + (['A0@C'] if ds == 'visa' else [])
    return whit, euc


@torch.no_grad()
def score():
    verify_lock()
    K.require(not (OUT / 'SCORES_COMPLETE.json').exists(), 'one shot')
    ex = A.rj(OUT / 'EXTRACT_COMPLETE.json')
    rec = A.rj(OUT / 'NORMAL.json')
    SCORE.mkdir(parents=True, exist_ok=True)
    files, t_all = {}, time.monotonic()
    for ds in DSETS:
        ords, sz, Tn = order(ds), rec[ds]['sizes'], len(order(ds))
        whit, euc = arms_for(ds)
        Q = {c: load_test(ds, c, ex) for c in ords}
        out = {c: {} for c in ords}
        bank_key = {'G3': 'g3', 'G1b': 'g3', 'G3@C': 'g3c', 'A0b': 'a0b', 'A0@C': 'a0c30'}
        # E-full (seed independent): per-task energies against all FIT tokens of the task
        per = {c: torch.empty(Tn, len(Q[c][0])) for c in ords}
        for t, m in enumerate(ords):
            keys = fit_tokens(ds, m).float()
            for c in ords:
                euclid(Q[c][0], keys, per[c][t])
            A.log('SCORE E-full', ds, m, len(keys), round(time.monotonic() - t_all, 1)); A.caps()
            del keys; torch.cuda.empty_cache()
        for j, c in enumerate(ords):
            out[c]['E-full'] = torch.stack([per[c][:l + 1].amin(0) for l in range(j, Tn)])
        for seed in SEEDS:
            store = [load_norm(ds, seed, m, rec) for m in ords]
            W = load_stage(ds, seed, rec)
            for a in euc:
                for t, m in enumerate(ords):
                    keys = store[t]['keys'][:sz[bank_key[a]][t]].cuda().float()
                    for c in ords:
                        euclid(Q[c][0], keys, per[c][t])
                for j, c in enumerate(ords):
                    out[c][f'{a}|s{seed}'] = torch.stack([per[c][:l + 1].amin(0) for l in range(j, Tn)])
            for a in whit:
                stg = {c: [] for c in ords}
                for l in range(Tn):
                    keys = torch.cat([x['keys'][:sz[bank_key[a]][t]] for t, x in enumerate(store[:l + 1])]).cuda()
                    Wg = W[l][a].cuda()
                    for c in ords[:l + 1]:
                        stg[c].append(whitened(Q[c][0], keys.float(), Wg))
                for c in ords:
                    out[c][f'{a}|s{seed}'] = torch.stack(stg[c])
            A.log('SCORE', ds, seed, round(time.monotonic() - t_all, 1)); A.caps()
            del store
        for j, c in enumerate(ords):
            n = len(Q[c][1])
            g = {k: v.reshape(len(v), n, 28, 28) for k, v in out[c].items()}
            for k, v in g.items():
                K.require(len(v) == Tn - j and bool(torch.isfinite(v).all()), f'grid {ds} {c} {k}')
            p = SCORE / f'{ds}_{c}.pt'
            torch.save({'rows': Q[c][1], 'first_stage': j, 'grids': g}, p)
            files[p.name] = A.sha(p)
        del Q, out, per
    A.wj(OUT / 'SCORES_COMPLETE.json', {'files': files, 'seconds': time.monotonic() - t_all})


# ------------------------------------------------------------------ metrics (labels / masks opened here only)
def full_stage(key):
    return key == 'E-full' or key.endswith('|s2711')


def metrics():
    from sklearn.metrics import roc_auc_score
    from exps.bench_aud_00 import run as AUD
    from exps.cadfinal import metrics as MX
    lk = verify_lock()
    K.require(not (OUT / 'RESULT.json').exists(), 'one shot')
    sc = A.rj(OUT / 'SCORES_COMPLETE.json')
    torch.set_num_threads(8)
    t_all = time.monotonic()
    cells, hist, ihist, labels_all, dev_all = {}, {}, {}, {}, {}
    boot_arms = {'G3|s2711', 'A0b|s2711', 'G1b|s2711', 'G3@C|s2711', 'A0@C|s2711', 'E-full'}
    for ds in DSETS:
        ords, Tn = order(ds), len(order(ds))
        for j, c in enumerate(ords):
            p = SCORE / f'{ds}_{c}.pt'
            K.require(A.sha(p) == sc['files'][p.name], 'score file changed ' + p.name)
            d = torch.load(p, weights_only=True)
            ids = {r['relative_path']: r for r in test_rows(ds, c)}
            rows = [ids[r] for r in d['rows']]
            root = data_root(ds)
            hws = [N.native_hw(root / r['relative_path']) for r in rows]
            masks = [N.native_mask(root / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
            m224 = [AUD.mask224(m) for m in masks]
            y = np.array([r['label'] for r in rows])
            labels_all[ds, c] = y
            dev_all[ds, c] = np.array([bool(r.get('dev', False)) for r in rows])
            for key, g in d['grids'].items():
                stages = range(j, Tn) if full_stage(key) else [Tn - 1]
                hi = float(g.max()) + 1e-12
                for l in stages:
                    maps = R2.maps_of(g[l - j])
                    img = A.image_scores(maps)
                    s224 = maps224(maps).numpy()
                    cell = {'i_auroc': float(roc_auc_score(y, img)),
                            'i_auroc_mapmax': float(roc_auc_score(y, maps.flatten(1).amax(1).numpy())),
                            'ap224': float(MX.exact_ap(list(s224), m224)['ap'])}
                    if full_stage(key):
                        h = A.image_hists(maps, hws, masks, 0., hi)
                        cell['ap_native_binned'] = A.binned_ap(h[0].sum(0), h[1].sum(0))
                        if l == Tn - 1 and key in boot_arms:
                            hist[ds, key, c] = h
                            ihist[ds, key, c] = (hist224(s224, m224, 0., hi), img)
                    if l == Tn - 1:
                        cell['ap_native'] = float(A.exact_ap(maps, hws, masks))
                    cells[ds, key, l, c] = cell
                A.log('METRIC', ds, c, key, round(cells[ds, key, Tn - 1, c]['ap_native'], 4),
                      round(cells[ds, key, Tn - 1, c]['i_auroc'], 4), round(time.monotonic() - t_all))
            del d, masks, m224; A.caps()
    A.wj(OUT / 'CELLS.json', {f'{ds}|{k}|{l}|{c}': v for (ds, k, l, c), v in cells.items()})
    res = summarize(cells, hist, ihist, labels_all, dev_all)
    res['seconds_metrics'] = time.monotonic() - t_all
    res['lock_sha256'] = A.sha(OUT / 'LOCK.json')
    A.wj(OUT / 'RESULT.json', res)
    A.wj(OUT / 'FINAL_SEAL.json', {'lock_sha256': A.sha(OUT / 'LOCK.json'),
                                   'files': {p.name: A.sha(p) for p in OUT.iterdir() if p.is_file()}})
    A.log('DONE', json.dumps(res['decisions']))


def summarize(cells, hist, ihist, labels_all, dev_all):
    from sklearn.metrics import roc_auc_score
    res = {'macro': {}, 'decisions': {}, 'fm': {}, 'bootstrap': {}, 'binned_validation': {}}
    keys_of = lambda ds: sorted({k for (d, k, l, c) in cells if d == ds})
    for ds in DSETS:
        ords, Tn = order(ds), len(order(ds))
        mac = lambda k, met, l=Tn - 1: float(np.mean([cells[ds, k, l, c][met] for c in ords]))
        for k in keys_of(ds):
            res['macro'][f'{ds}|{k}'] = {m: mac(k, m) for m in ('ap_native', 'ap224', 'i_auroc', 'i_auroc_mapmax')}
            if full_stage(k):
                res['macro'][f'{ds}|{k}']['ap_native_binned'] = mac(k, 'ap_native_binned')
                fm = {}
                for met in ('ap_native_binned', 'ap224', 'i_auroc'):
                    mat = {(l, j): cells[ds, k, l, c][met] for j, c in enumerate(ords) for l in range(j, Tn)}
                    fm[met] = chaudhry(mat, Tn)
                    fm[met]['acquisition_macro'] = float(np.mean([mat[j, j] for j in range(Tn)]))
                res['fm'][f'{ds}|{k}'] = fm
                dv = abs(res['macro'][f'{ds}|{k}']['ap_native_binned'] - res['macro'][f'{ds}|{k}']['ap_native'])
                res['binned_validation'][f'{ds}|{k}'] = dv
        K.require(max(v for kk, v in res['binned_validation'].items() if kk.startswith(ds)) <= BIN_TOL,
                  f'binned AP validation failed ({ds})')
        # paired category-stratified image bootstrap (seed 2711 arms), one resample shared by all arms and metrics
        barms = sorted({k for (d, k, c) in hist if d == ds})
        rng = np.random.default_rng(BOOT_SEED)
        reps = {(a, m): np.empty(BOOT_B) for a in barms for m in ('ap', 'ap224', 'auroc')}
        for r in range(BOOT_B):
            per = {key: [] for key in reps}
            for c in ords:
                y = labels_all[ds, c]; n = len(y)
                w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
                for a in barms:
                    h = hist[ds, a, c]; (h2, img) = ihist[ds, a, c]
                    per[a, 'ap'].append(A.binned_ap(w @ h[0], w @ h[1]))
                    per[a, 'ap224'].append(A.binned_ap(w @ h2[0], w @ h2[1]))
                    per[a, 'auroc'].append(float(roc_auc_score(y, img, sample_weight=w))
                                           if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0 else np.nan)
            for key in reps:
                reps[key][r] = np.nanmean(per[key])
        q = lambda v: {'lo': float(np.quantile(v, .025)), 'hi': float(np.quantile(v, .975))}
        res['bootstrap'][ds] = {f'{a}|{m}': q(v) for (a, m), v in reps.items()}
        # matched contrasts, Holm over two per dataset
        con = {'G3-A0b': ('G3|s2711', 'A0b|s2711'), 'G3-G1b': ('G3|s2711', 'G1b|s2711')}
        dl = {n: reps[a, 'ap'] - reps[b, 'ap'] for n, (a, b) in con.items()}
        hp = holm({n: boot_p(v) for n, v in dl.items()})
        dec = {}
        for n, (a, b) in con.items():
            point = {s: res['macro'][f'{ds}|{a[:-6]}|s{s}']['ap_native'] - res['macro'][f'{ds}|{b[:-6]}|s{s}']['ap_native']
                     for s in SEEDS}
            rank = sorted(dl, key=lambda k: boot_p(dl[k])).index(n)
            a_k = ALPHA / (len(dl) - rank)
            lo_adj = float(np.quantile(dl[n], a_k / 2))
            confirmed = point[2711] >= DELTA_MIN and hp[n][1] and lo_adj > 0 and all(v > 0 for v in point.values())
            label = 'CONFIRMED' if confirmed else 'REFUTED' if point[2711] <= 0 else 'INCONCLUSIVE'
            dec[n] = {'delta_seed2711': point[2711], 'delta_by_seed': point, 'ci95': q(dl[n]),
                      'p_boot': boot_p(dl[n]), 'p_holm': hp[n][0], 'holm_rejected': hp[n][1],
                      'holm_adjusted_lower': lo_adj, 'label': ('EXPOSED_' if ds == 'visa' else '') + label}
        res['decisions'][ds] = dec
        if ds == 'visa':
            pub = {}
            for arm in ('G3|s2711', 'G3@C|s2711'):
                for m, (rk, pk) in {'p_aupr_224': ('ap224', 'p_aupr'), 'i_auroc': ('auroc', 'i_auroc'),
                                    'p_aupr_native': ('ap', 'p_aupr')}.items():
                    b = res['bootstrap'][ds][f'{arm}|{rk}']
                    for meth, vals in PUBLISHED.items():
                        pub[f'{arm}|{m}|{meth}'] = {'g3_ci': b, 'published': vals[pk],
                                                    'label': label_of(vals[pk], b['lo'], b['hi'])}
            res['published_comparison'] = pub
            res['published_qualifier'] = ('historically exposed cohort; unmatched backbone, resolution and '
                                          'preprocessing; published value not reproduced')
            fin = {}
            for a in barms:                                   # VisA-FINAL only (DEV images excluded), descriptive
                aps, aus = [], []
                for c in ords:
                    keep = ~dev_all[ds, c]
                    h = hist[ds, a, c]
                    aps.append(A.binned_ap(h[0][keep].sum(0), h[1][keep].sum(0)))
                    aus.append(float(roc_auc_score(labels_all[ds, c][keep], ihist[ds, a, c][1][keep])))
                fin[a] = {'ap_native_binned': float(np.mean(aps)), 'i_auroc': float(np.mean(aus)),
                          'n_images': int(sum((~dev_all[ds, c]).sum() for c in ords))}
            res['visa_final_only'] = fin
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    phases = ('custody', 'extract_train', 'normal', 'parity', 'cost', 'seal_prep', 'lock', 'extract_test', 'score',
              'metrics')
    p.add_argument('phase', choices=phases)
    p.add_argument('--only', choices=DSETS)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    globals()[a.phase](a.only) if a.phase == 'normal' else globals()[a.phase]()
