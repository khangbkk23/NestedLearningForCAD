"""NANM-R0 §6: same-feature complementarity audit (F decoder vs non-parametric normal patch memory) on MVTec.

No training. Pre-registration: agents/tasks/task_nanm_r0_feasibility.md.

    python -B -m exps.nanm_r0.audit lock
    python -B -m exps.nanm_r0.audit normal     # normal-only: FIT memory, coresets, EVAL statistics inputs
    python -B -m exps.nanm_r0.audit test       # kNN distances on MVTec test tokens (needs LOCK)
    python -B -m exps.nanm_r0.audit metrics    # one-shot metrics + decision
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exps.calm02 import native as N
from exps.nar_lite_01 import run as NL
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.detector import prefix_grids

ROOT = R.ROOT
OUT = ROOT / 'results/hope_cad/nanm_r0'
FEAT = Path.home() / 'hope_cad_cache/nar_b14_01/features'
CACHE = Path.home() / 'hope_cad_cache/nanm_r0'
LITE = ROOT / 'results/hope_cad/nar_lite_01'
ORDER = tuple(R.ORDER)
SEED = 2711
CORESET_RATIO, PROJ_DIM = 0.01, 128
BOOT_B, BOOT_SEED, BINS = 1000, 20261119, 8192
SCREEN, NULL_BAND = 0.015, 0.005
SOURCES = ('exps/nanm_r0/audit.py', 'exps/nanm_r0/test_audit.py', 'agents/tasks/task_nanm_r0_feasibility.md',
           'exps/nar_lite_01/run.py', 'exps/calm02/native.py', 'exps/nar_pilot_00/core.py',
           'exps/nar_pilot_00/detector.py')
CUDA_CAP, RSS_CAP = 5 * 2**30, 9_000_000_000


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def wj(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n')


def rj(p):
    return json.loads(Path(p).read_text())


def caps():
    K.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < RSS_CAP, 'RSS cap')
    if torch.cuda.is_initialized():
        K.require(torch.cuda.max_memory_allocated() <= CUDA_CAP, 'CUDA cap')


def load_feat(c, split, lk=None):
    p = FEAT / f'{c}_{split}.pt'
    want = rj(p.with_suffix('.json'))['sha256']
    if lk is not None:
        K.require(lk['inputs'][f'{c}_{split}'] == want, 'cache meta changed')
    K.require(sha(p) == want, f'feature cache hash {p.name}')
    return torch.load(p, weights_only=True)


# ------------------------------------------------------------------ pure functions (unit-tested)
def unit(x):
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def min_cos_dist(q, mem, qc=2048):
    """q [Nq,D], mem [Nm,D], both unit-norm (any float dtype, same device). Returns (1 - max cos) [Nq] fp32."""
    out = torch.empty(len(q), dtype=torch.float32, device=q.device)
    for i in range(0, len(q), qc):
        out[i:i + qc] = (1 - (q[i:i + qc] @ mem.T).amax(1).float()).clamp_min(0)
    return out


@torch.no_grad()
def greedy_coreset(x, n, seed=SEED, dim=PROJ_DIM):
    """PatchCore-style greedy k-centre on a seeded Gaussian projection. x [N,D] on device. Returns indices [n]."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x.shape[1], dim, generator=g) / math.sqrt(dim)).to(x.device, torch.float32)
    z = x.float() @ P
    idx = torch.empty(n, dtype=torch.long)
    cur = int(torch.randint(len(z), (1,), generator=g))
    mind = torch.full((len(z),), float('inf'), device=x.device)
    for i in range(n):
        idx[i] = cur
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        cur = int(mind.argmax())
    return idx


def smooth(grid):
    """A1 readout of NAR-LITE-01 (identical function)."""
    return NL.smooth392(grid)


def binned_ap(pos, tot):
    """AP from per-bin positive / total counts (bins ascending in score); each bin = one tie group."""
    pos, tot = pos[::-1], tot[::-1]
    P = pos.sum()
    if P <= 0:
        return float('nan')
    tp, n = np.cumsum(pos), np.cumsum(tot)
    keep = tot > 0
    return float(np.sum(pos[keep] / P * (tp[keep] / n[keep])))


def image_hists(maps, hws, masks, lo, hi, bins=BINS):
    """Per-image native-resolution (pos, tot) histograms of 392 maps over a fixed [lo, hi] range."""
    pos = np.zeros((len(maps), bins), np.float32); tot = np.zeros((len(maps), bins), np.float32)
    for i, (m, hw, y) in enumerate(zip(maps, hws, masks)):
        v = NL.map_native(m, hw).ravel()
        b = np.clip(((v - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
        tot[i] = np.bincount(b, minlength=bins); pos[i] = np.bincount(b, weights=y.ravel().astype(np.float64),
                                                                       minlength=bins)
    return pos, tot


def exact_ap(maps, hws, masks):
    original = N.native_map
    N.native_map = NL.map_native
    try:
        return N.native_ap(list(maps), hws, masks)['ap']
    finally:
        N.native_map = original


def image_scores(maps):
    k = max(1, int(NL.TOP * maps.shape[-1] * maps.shape[-2]))
    return maps.flatten(1).topk(k, dim=1).values.mean(1).numpy()


def decide(delta, lo, per_cat_delta):
    guard = (min(per_cat_delta) >= -0.05) and (sum(d < -0.01 for d in per_cat_delta) <= 3)
    if delta >= SCREEN and lo > 0 and guard:
        return 'H1_SUPPORTED', guard
    if delta < NULL_BAND:
        return 'H1_NOT_SUPPORTED', guard
    return 'H1_INCONCLUSIVE', guard


# ------------------------------------------------------------------ lock
def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    K.require(not any(OUT.glob('knn_*')), 'test outputs exist before lock')
    lite = rj(LITE / 'SCORES_COMPLETE.json')
    inputs = {f'{c}_{s}': rj(FEAT / f'{c}_{s}.json')['sha256'] for c in ORDER for s in ('train', 'test')}
    wj(OUT / 'LOCK.json', {
        'sources': {s: sha(ROOT / s) for s in SOURCES}, 'inputs': inputs,
        'lite_scores': lite['files'], 'lite_cells_smooth_sha256': sha(LITE / 'cells_smooth.json'),
        'lite_stage15_sha256': sha(NL.REF / 'stage15.pt'), 'base_sha256': sha(NL.W.HIST / 'BASE.pt'),
        'coreset_ratio': CORESET_RATIO, 'proj_dim': PROJ_DIM, 'seed': SEED, 'boot': [BOOT_B, BOOT_SEED, BINS],
        'decision': {'screen': SCREEN, 'null_band': NULL_BAND, 'guard': 'min cat delta >= -0.05, <=3 cats < -0.01'},
        'torch': torch.__version__})
    log('LOCK', sha(OUT / 'LOCK.json'))


def verify_lock():
    lk = rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(sha(ROOT / p) == h, 'locked source changed: ' + p)
    for name, h in lk['lite_scores'].items():
        K.require(sha(LITE / name) == h, 'sealed lite score changed: ' + name)
    return lk


# ------------------------------------------------------------------ normal-only phase
@torch.no_grad()
def normal():
    lk = verify_lock()
    CACHE.mkdir(parents=True, exist_ok=True)
    NL.verify_reference()
    model, loras, archive, _ = NL.lite_detector(torch.bfloat16)
    rec = {}
    for c in ORDER:
        t0 = time.monotonic()
        d = load_feat(c, 'train', lk)
        rows = d['rows']
        K.require([r['relative_path'] for r in rows] == [r['relative_path'] for r in R.rows_for(c, 'train')],
                  'train rows')
        roles = K.normal_partition(rows)
        feats = d['features']
        fit = unit(feats[roles['FIT']].reshape(-1, 768).cuda()).half()
        n_cs = math.ceil(CORESET_RATIO * len(fit))
        cs_idx = greedy_coreset(fit, n_cs)
        ev = feats[roles['EVAL']]
        f_eval = prefix_grids(NL.expert_grids(model, loras, archive, ev.to(torch.bfloat16), torch.bfloat16))
        torch.save({'fit': fit.cpu(), 'coreset': fit[cs_idx.cuda()].cpu(), 'coreset_idx': cs_idx,
                    'fit_images': [rows[i]['relative_path'] for i in roles['FIT']]}, CACHE / f'norm_{c}.pt')
        torch.save({'eval_unit': unit(ev.reshape(-1, 768)).half(), 'f_eval_prefix': f_eval,
                    'eval_images': [rows[i]['relative_path'] for i in roles['EVAL']]}, CACHE / f'eval_{c}.pt')
        rec[c] = {'n_fit_images': len(roles['FIT']), 'n_eval_images': len(roles['EVAL']), 'n_fit_patches': len(fit),
                  'n_coreset': n_cs, 'seconds': time.monotonic() - t0, 'sha256': sha(CACHE / f'norm_{c}.pt'),
                  'eval_sha256': sha(CACHE / f'eval_{c}.pt')}
        log('NORMAL', c, rec[c])
        del d, feats, fit; torch.cuda.empty_cache(); caps()
    wj(OUT / 'NORMAL.json', {'per_task': rec, 'lock_sha256': sha(OUT / 'LOCK.json')})


def load_norm(c, normal_rec, kind='norm'):
    p = CACHE / f'{kind}_{c}.pt'
    K.require(sha(p) == normal_rec['per_task'][c]['sha256' if kind == 'norm' else 'eval_sha256'], 'cache changed')
    return torch.load(p, weights_only=True)


# ------------------------------------------------------------------ kNN on test + EVAL queries
@torch.no_grad()
def test():
    lk = verify_lock()
    nr = rj(OUT / 'NORMAL.json')
    K.require(not (OUT / 'KNN_COMPLETE.json').exists(), 'no rescoring')
    queries = {}
    for c in ORDER:                                    # test queries, row order = sealed F scores
        d = load_feat(c, 'test', lk)
        lite_rows = torch.load(LITE / f'scores_{c}.pt', weights_only=True)['rows']
        K.require([r['relative_path'] for r in d['rows']] == [r['relative_path'] for r in lite_rows], 'test rows')
        queries['test', c] = unit(d['features'].reshape(-1, 768)).half()
        del d; caps()
    for c in ORDER:
        queries['eval', c] = load_norm(c, nr, 'eval')['eval_unit']
    full = {k: torch.empty(len(ORDER), len(q)) for k, q in queries.items()}
    cs = {k: torch.empty(len(ORDER), len(q)) for k, q in queries.items()}
    t0 = time.monotonic()
    for t, m in enumerate(ORDER):
        nm = load_norm(m, nr)
        mem = nm['fit'].cuda(); memc = nm['coreset'].cuda(); del nm
        for k, q in queries.items():
            qg = q.cuda()
            full[k][t] = min_cos_dist(qg, mem).cpu()
            cs[k][t] = min_cos_dist(qg, memc).cpu()
            del qg
        log('KNN memory task', t + 1, m, round(time.monotonic() - t0, 1)); caps()
        del mem, memc; torch.cuda.empty_cache()
    files = {}
    for (split, c), q in queries.items():
        n = len(q) // 784
        path = OUT / f'knn_{split}_{c}.pt'
        torch.save({'full': full[split, c].reshape(15, n, 28, 28), 'coreset': cs[split, c].reshape(15, n, 28, 28)},
                   path)
        files[path.name] = sha(path)
    K.require(all(bool(torch.isfinite(v).all()) for v in list(full.values()) + list(cs.values())), 'nonfinite kNN')
    wj(OUT / 'KNN_COMPLETE.json', {'files': files, 'seconds': time.monotonic() - t0,
                                   'peak_cuda': torch.cuda.max_memory_allocated()})


@torch.no_grad()
def latency():
    """kNN-only cost per image on EVAL normal tokens (no test data); encoder shared with F, excluded."""
    nr = rj(OUT / 'NORMAL.json')
    q = load_norm('bottle', nr, 'eval')['eval_unit'][:784 * 20].reshape(-1, 784, 768)

    def timed(mg):
        ts = []
        for i in range(len(q) + 3):
            qq = q[max(0, i - 3)].cuda()
            torch.cuda.synchronize(); t = time.perf_counter()
            min_cos_dist(qq, mg)
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        return float(np.mean(ts[3:]))
    full_ms, cs, n_full = 0., [], 0
    for c in ORDER:                      # full memory does not fit the 6 GB GPU at once: sum of resident chunks
        nm = load_norm(c, nr); mg = nm['fit'].cuda(); n_full += len(mg)
        full_ms += 1e3 * timed(mg); cs.append(nm['coreset']); del mg, nm; torch.cuda.empty_cache()
    csm = torch.cat(cs).cuda()
    out = {'full_patches': n_full, 'coreset_patches': len(csm), 'full_bytes_fp16': n_full * 768 * 2,
           'coreset_bytes_fp16': len(csm) * 768 * 2, 'full_ms_per_image_sum_of_chunks': full_ms,
           'coreset_ms_per_image': 1e3 * timed(csm),
           'note': 'kNN search only (encoder shared with F); EVAL normal tokens; fp16 matmul; batch 1 image'}
    return out


# ------------------------------------------------------------------ metrics
def stage_stats(feval, kn, s):
    """(mu, sigma) of smoothed EVAL maps of tasks <= s for arms F, K, Kcs at stage s (normal only)."""
    acc = {a: [0., 0., 0] for a in ('F', 'K', 'Kcs')}
    for c in ORDER[:s + 1]:
        g = {'F': feval[c][s], 'K': kn['eval', c]['full'][s],
             'Kcs': kn['eval', c]['coreset'][s]}
        for a, grid in g.items():
            m = smooth(grid).double()
            acc[a][0] += float(m.sum()); acc[a][1] += float((m ** 2).sum()); acc[a][2] += m.numel()
    out = {}
    for a, (s1, s2, n) in acc.items():
        mu = s1 / n
        out[a] = (mu, math.sqrt(max(s2 / n - mu * mu, 1e-30)))
    return out


def patch_labels(masks):
    out = []
    for y in masks:
        t = torch.as_tensor(y, dtype=torch.float32)[None, None]
        t = F.interpolate(t, size=(392, 392), mode='area')
        out.append(F.avg_pool2d(t, 14)[0, 0])
    return torch.stack(out)                       # [N,28,28] anomalous area fraction


def metrics():
    lk = verify_lock()
    K.require(not (OUT / 'RESULT.json').exists(), 'one-shot: RESULT exists')
    nr = rj(OUT / 'NORMAL.json'); kc = rj(OUT / 'KNN_COMPLETE.json')
    for name, h in kc['files'].items():
        K.require(sha(OUT / name) == h, 'knn file changed')
    kn = {(sp, c): torch.load(OUT / f'knn_{sp}_{c}.pt', weights_only=True) for sp in ('test', 'eval') for c in ORDER}
    sealed = rj(LITE / 'cells_smooth.json')
    feval = {c: load_norm(c, nr, 'eval')['f_eval_prefix'] for c in ORDER}
    stats = {s: stage_stats(feval, kn, s) for s in range(15)}
    log('STATS done')
    arms = ('A0', 'A1', 'A1cs', 'A2', 'A2cs', 'A2min', 'A2max')
    cells, hists, extra, parity = {}, {}, {}, {}
    for i, c in enumerate(ORDER):
        rows = torch.load(LITE / f'scores_{c}.pt', weights_only=True)
        fgrid = rows['prefix_grids']; rows = rows['rows']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        for phase, s in (('acquisition', i), ('final', 14)):
            st = stats[s]
            m0 = smooth(fgrid[s]); m1 = smooth(kn['test', c]['full'][s]); m1c = smooth(kn['test', c]['coreset'][s])
            z0 = (m0 - st['F'][0]) / st['F'][1]; z1 = (m1 - st['K'][0]) / st['K'][1]
            z1c = (m1c - st['Kcs'][0]) / st['Kcs'][1]
            maps = {'A0': m0, 'A1': m1, 'A1cs': m1c, 'A2': z0 + z1, 'A2cs': z0 + z1c,
                    'A2min': torch.minimum(z0, z1), 'A2max': torch.maximum(z0, z1)}
            for a, m in maps.items():
                m = m - min(0., float(m.min()))          # rank-invariant shift for the non-negative AP accumulator
                from sklearn.metrics import roc_auc_score
                ap = exact_ap(m, hws, masks)
                cells[a, phase, c] = {'pixel_ap': ap, 'image_auroc': float(roc_auc_score(labels, image_scores(m)))}
                if phase == 'final' and a in ('A0', 'A1', 'A1cs', 'A2', 'A2cs'):
                    hists[a, c] = image_hists(m, hws, masks, float(m.min()), float(m.max()) + 1e-12)
                    cells[a, phase, c]['binned_ap'] = binned_ap(hists[a, c][0].sum(0), hists[a, c][1].sum(0))
            ref = sealed[f'{s + 1:02}_{c}']['pixel_ap']
            parity[phase, c] = cells['A0', phase, c]['pixel_ap'] - ref
            K.require(abs(parity[phase, c]) <= 1e-6, f'A0 parity {c} {phase} {parity[phase, c]}')
            log('CELL', c, phase, {a: round(cells[a, phase, c]['pixel_ap'], 4) for a in arms})
        extra[c] = diagnostics(c, fgrid[14], kn, feval, stats[14], masks, labels)
        del masks; caps()
    result = summarise(cells, hists, extra, parity, arms)
    result['latency'] = latency()
    result['normal'] = nr['per_task']
    result['stats_final'] = stats[14]
    wj(OUT / 'RESULT.json', result)
    wj(OUT / 'FINAL_SEAL.json', {'lock_sha256': sha(OUT / 'LOCK.json'),
                                 'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}})
    log('DECISION', json.dumps(result['decision']))


def diagnostics(c, fgrid, kn, feval, st, masks, labels):
    """Final-stage patch-level complementarity, retrieval identity and defect-size strata (raw 28x28 grids)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score
    k = kn['test', c]['full']
    kgrid = k[14]
    nn_task = k.argmin(0)                                   # task whose memory holds the nearest neighbour
    frac = patch_labels(masks)
    lab = torch.as_tensor(labels, dtype=torch.bool)
    norm_p = (~lab)[:, None, None].expand_as(frac)
    anom_p = frac >= .5
    # q99 thresholds from raw EVAL grids at the final stage (all tasks)
    fe = torch.cat([feval[t][14].flatten() for t in ORDER])
    ke = torch.cat([kn['eval', t]['full'][14].flatten() for t in ORDER])
    tf, tk = float(torch.quantile(fe[::7], .99)), float(torch.quantile(ke[::7], .99))
    df, dk = fgrid > tf, kgrid > tk
    ci = ORDER.index(c)
    out = {'q99_F': tf, 'q99_K': tk,
           'anom_patches': int(anom_p.sum()),
           'detect_F_only': int((df & ~dk & anom_p).sum()), 'detect_K_only': int((dk & ~df & anom_p).sum()),
           'detect_both': int((df & dk & anom_p).sum()), 'detect_neither': int((~df & ~dk & anom_p).sum()),
           'fp_rate_F_normal_images': float(df[norm_p].float().mean()),
           'fp_rate_K_normal_images': float(dk[norm_p].float().mean()),
           'nn_same_task_normal_images': float((nn_task[norm_p] == ci).float().mean()),
           'nn_same_task_anom_patches': float((nn_task[anom_p] == ci).float().mean()) if anom_p.any() else None}
    if anom_p.sum() > 10:
        out['spearman_anom'] = float(spearmanr(fgrid[anom_p].numpy(), kgrid[anom_p].numpy())[0])
    sub = norm_p[:, :, :]
    idx = torch.nonzero(sub.flatten())[:, 0][::13]
    out['spearman_normal'] = float(spearmanr(fgrid.flatten()[idx].numpy(), kgrid.flatten()[idx].numpy())[0])
    # defect-size strata: within-image AP on the 392 smoothed map (localisation inside the image, calibration-free)
    m0, m1 = smooth(fgrid), smooth(kgrid)
    z = (m0 - st['F'][0]) / st['F'][1] + (m1 - st['K'][0]) / st['K'][1]
    strata = {'small(<1%)': [], 'medium(1-5%)': [], 'large(>=5%)': []}
    for j in np.flatnonzero(labels):
        y = masks[j]; a = float(y.mean())
        y392 = F.interpolate(torch.as_tensor(y, dtype=torch.float32)[None, None], size=(392, 392),
                             mode='area')[0, 0].numpy().ravel() >= .5
        if y392.sum() == 0:
            continue
        key = 'small(<1%)' if a < .01 else 'medium(1-5%)' if a < .05 else 'large(>=5%)'
        strata[key].append([average_precision_score(y392, mm[j].numpy().ravel()) for mm in (m0, m1, z)])
    out['size_strata'] = {k: {'n': len(v), 'A0': float(np.mean([x[0] for x in v])) if v else None,
                              'A1': float(np.mean([x[1] for x in v])) if v else None,
                              'A2': float(np.mean([x[2] for x in v])) if v else None} for k, v in strata.items()}
    return out


def boot_delta(hists, a, b, rng_seed=BOOT_SEED, B=BOOT_B):
    rng = np.random.default_rng(rng_seed)
    reps = np.empty(B)
    for r in range(B):
        da = []
        for c in ORDER:
            n = len(hists[a, c][0]); w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
            pa = binned_ap(w @ hists[a, c][0], w @ hists[a, c][1])
            pb = binned_ap(w @ hists[b, c][0], w @ hists[b, c][1])
            da.append(pa - pb)
        reps[r] = np.mean(da)
    return {'lo': float(np.quantile(reps, .025)), 'hi': float(np.quantile(reps, .975)), 'B': B, 'unit': 'image'}


def cat_boot(x, y, seed=20261117, B=2000):
    rng = np.random.default_rng(seed); d = np.asarray(x) - np.asarray(y)
    sm = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return {'lo': float(np.quantile(sm, .025)), 'hi': float(np.quantile(sm, .975)), 'unit': 'category'}


def summarise(cells, hists, extra, parity, arms):
    macro = {a: {ph: {m: float(np.mean([cells[a, ph, c][m] for c in ORDER])) for m in ('pixel_ap', 'image_auroc')}
                 for ph in ('acquisition', 'final')} for a in arms}
    per = {a: {c: cells[a, 'final', c]['pixel_ap'] for c in ORDER} for a in arms}
    binned_err = max(abs(cells[a, 'final', c]['binned_ap'] - cells[a, 'final', c]['pixel_ap'])
                     for a in ('A0', 'A1', 'A1cs', 'A2', 'A2cs') for c in ORDER)

    def comp(a, b):
        d = macro[a]['final']['pixel_ap'] - macro[b]['final']['pixel_ap']
        pc = [per[a][c] - per[b][c] for c in ORDER]
        return {'delta': d, 'image_boot': boot_delta(hists, a, b), 'cat_boot': cat_boot(list(per[a].values()),
                list(per[b].values())), 'per_category': dict(zip(ORDER, pc)), 'n_cat_pos': int(sum(x > 0 for x in pc))}

    best = 'A0' if macro['A0']['final']['pixel_ap'] >= macro['A1']['final']['pixel_ap'] else 'A1'
    bestc = 'A0' if macro['A0']['final']['pixel_ap'] >= macro['A1cs']['final']['pixel_ap'] else 'A1cs'
    primary = comp('A2', best)
    verdict, guard = decide(primary['delta'], primary['image_boot']['lo'], list(primary['per_category'].values()))
    k_vs_f = comp('A1', 'A0')
    flag = ('KNN_DOMINATES_F' if k_vs_f['delta'] >= SCREEN and k_vs_f['image_boot']['lo'] > 0 else
            'F_DOMINATES_KNN' if -k_vs_f['delta'] >= SCREEN and k_vs_f['image_boot']['hi'] < 0 else 'NEITHER_DOMINATES')
    sec = comp('A2cs', bestc)
    sec_verdict, sec_guard = decide(sec['delta'], sec['image_boot']['lo'], list(sec['per_category'].values()))
    oracle = float(np.mean([max(per['A0'][c], per['A1'][c]) for c in ORDER]))
    return {'macro': macro, 'per_category_final_ap': per,
            'per_category_cells': {f'{a}|{ph}|{c}': v for (a, ph, c), v in cells.items()},
            'best_standalone': best, 'primary_A2_vs_best': primary,
            'decision': {'H1': verdict, 'guard_ok': guard, 'flag': flag,
                         'secondary_coreset_A2cs_vs_' + bestc: sec_verdict, 'secondary_guard_ok': sec_guard},
            'A1_vs_A0': k_vs_f, 'A1cs_vs_A0': comp('A1cs', 'A0'), 'A2cs_vs_' + bestc: sec,
            'A2_vs_A0': comp('A2', 'A0') if best != 'A0' else primary,
            'A3_oracle_final_ap': oracle, 'A3_minus_best': oracle - macro[best]['final']['pixel_ap'],
            'binned_vs_exact_max_abs': binned_err, 'a0_parity_max_abs': max(abs(v) for v in parity.values()),
            'diagnostics': extra,
            'exposure': 'MVTec official test, historically exposed; one F seed (2711), one order; exploratory'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'normal', 'test', 'metrics'))
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {'lock': lock, 'normal': normal, 'test': test, 'metrics': metrics}[a.phase]()
