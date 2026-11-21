"""NANM-R1: bytes-quality frontier of task-free normal memory on the same B/14 tokens (no training).

Pre-registration: agents/tasks/task_nanm_r1_memory_frontier.md (owner authorised 2026-11-19).

    python -B -m exps.nanm_r1.run lock
    python -B -m exps.nanm_r1.run coresets   # MVTec, normal-only
    python -B -m exps.nanm_r1.run score      # MVTec kNN distances (per task; prefix applied at use)
    python -B -m exps.nanm_r1.run visa       # VisA normal-only memory + DEV/EVAL distances + F EVAL grids
    python -B -m exps.nanm_r1.run metrics    # one-shot decision
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
from exps.nanm_r0.audit_r1 import prefix
from exps.nar_lite_01 import run as L
from exps.nar_pilot_00 import core as K, full as R
from exps.nar_pilot_00.detector import prefix_grids

ROOT = A.ROOT
OUT = ROOT / 'results/hope_cad/nanm_r1'
CACHE = Path.home() / 'hope_cad_cache/nanm_r1'
R0 = A.OUT
DEVR = Path.home() / 'hope_cad_cache/cmsb'
RATIOS = (0.0025, 0.005, 0.01, 0.02, 0.04)
FUSE = (0.005, 0.01, 0.02)
VISA_EVAL_N = 40
SEED = 2711
SOURCES = ('exps/nanm_r1/run.py', 'exps/nanm_r1/test_run.py', 'agents/tasks/task_nanm_r1_memory_frontier.md',
           'exps/nanm_r0/audit.py', 'exps/nanm_r0/audit_r1.py')


def tag(r):
    return f'{100 * r:g}'


def n_of(r, n):
    return math.ceil(r * n)


def VB():
    from exps.vbench_00 import run as vb
    return vb


def T():
    from exps.tcs_00 import run as t
    return t


def verify_lock():
    lk = A.rj(OUT / 'LOCK.json')
    for p, h in lk['sources'].items():
        K.require(A.sha(ROOT / p) == h, 'locked source changed: ' + p)
    return lk


# ------------------------------------------------------------------ lock
def lock():
    K.require(not (OUT / 'LOCK.json').exists(), 'LOCK exists')
    vb = VB()
    r0 = A.rj(R0 / 'RESULT_r1.json')
    K.require(r0['decision']['H1'] == 'H1_NOT_SUPPORTED', 'R0 r1 decision of record')
    vlock = A.rj(vb.OUT / 'LOCK.json')
    A.wj(OUT / 'LOCK.json', {
        'sources': {s: A.sha(ROOT / s) for s in SOURCES},
        'r0': {n: A.sha(R0 / n) for n in ('LOCK.json', 'LOCK_r1.json', 'NORMAL.json', 'KNN_COMPLETE.json',
                                          'RESULT_r1.json')},
        'mvtec_test_inputs': {c: A.rj(A.FEAT / f'{c}_test.json')['sha256'] for c in A.ORDER},
        'visa_train_inputs': {c: vlock['inputs'][c]['sha256'] for c in vb.ORDER},
        'visa_devr': {c: A.sha(DEVR / f'devr_{c}.pt') for c in vb.ORDER},
        'visa_scores': A.rj(vb.OUT / 'SCORES_COMPLETE.json')['files'],
        'visa_final_sha256': A.sha(vb.OUT / 'final.pt'),
        'ratios': RATIOS, 'fuse': FUSE, 'rand': 0.01, 'visa_eval_n': VISA_EVAL_N, 'seed': SEED,
        'boot': [A.BOOT_B, A.BOOT_SEED, A.BINS],
        'decision': {'BYTES_SUFFICE': 'G(2%) <= 0.005',
                     'CONSOLIDATION_HEADROOM': 'G(2%) >= 0.010 and image-boot LB > 0 and G(1%) >= 0.015',
                     'else': 'INCONCLUSIVE'},
        'torch': torch.__version__})
    A.log('LOCK', A.sha(OUT / 'LOCK.json'))


# ------------------------------------------------------------------ MVTec normal-only coresets
@torch.no_grad()
def coresets():
    verify_lock()
    K.require(not (OUT / 'CORESETS.json').exists(), 'no recompute')
    nr = A.rj(R0 / 'NORMAL.json')
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {}
    for c in A.ORDER:
        nm = A.load_norm(c, nr)
        fit = nm['fit'].cuda(); n = len(fit)
        t0 = time.monotonic()
        idx = A.greedy_coreset(fit, n_of(RATIOS[-1], n))
        K.require(torch.equal(idx[:n_of(0.01, n)], nm['coreset_idx']), f'1% prefix parity with R0 {c}')
        rand = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))[:n_of(0.01, n)]
        p = CACHE / f'mv_{c}.pt'
        torch.save({'kc_idx': idx, 'rand_idx': rand}, p)
        rec[c] = {'n_fit_patches': n, 'sizes': {tag(r): n_of(r, n) for r in RATIOS}, 'sha256': A.sha(p),
                  'seconds': time.monotonic() - t0}
        A.log('CORESET', c, rec[c]['sizes'], round(rec[c]['seconds'], 1))
        del fit, nm; torch.cuda.empty_cache()
    A.wj(OUT / 'CORESETS.json', rec)


def variants(fit, cs):
    out = {'full': fit}
    n = len(fit)
    for r in RATIOS:
        out['kc' + tag(r)] = fit[cs['kc_idx'][:n_of(r, n)].to(fit.device)]
    out['rand1'] = fit[cs['rand_idx'].to(fit.device)]
    return out


# ------------------------------------------------------------------ MVTec distances
@torch.no_grad()
def score():
    lk = verify_lock()
    K.require(not (OUT / 'SCORE_COMPLETE.json').exists(), 'no rescoring')
    nr = A.rj(R0 / 'NORMAL.json'); cr = A.rj(OUT / 'CORESETS.json')
    queries = {}
    for c in A.ORDER:
        p = A.FEAT / f'{c}_test.pt'
        K.require(A.sha(p) == lk['mvtec_test_inputs'][c], 'test cache hash')
        d = torch.load(p, weights_only=True)
        lite_rows = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)['rows']
        K.require([r['relative_path'] for r in d['rows']] == [r['relative_path'] for r in lite_rows], 'rows')
        queries['test', c] = A.unit(d['features'].reshape(-1, 768)).half()
        del d; A.caps()
    for c in A.ORDER:
        queries['eval', c] = A.load_norm(c, nr, 'eval')['eval_unit']
    names = ['full'] + ['kc' + tag(r) for r in RATIOS] + ['rand1']
    dist = {k: {v: torch.empty(15, len(q)) for v in names} for k, q in queries.items()}
    t0 = time.monotonic()
    for t, m in enumerate(A.ORDER):
        nm = A.load_norm(m, nr)
        cp = CACHE / f'mv_{m}.pt'
        K.require(A.sha(cp) == cr[m]['sha256'], 'coreset cache')
        mems = {v: x.cuda() for v, x in variants(nm['fit'], torch.load(cp, weights_only=True)).items()}
        del nm
        for k, q in queries.items():
            qg = q.cuda()
            for v, mem in mems.items():
                dist[k][v][t] = A.min_cos_dist(qg, mem).cpu()
            del qg
        A.log('KNN task', t + 1, m, round(time.monotonic() - t0, 1)); A.caps()
        del mems; torch.cuda.empty_cache()
    files, par = {}, 0.
    for (sp, c), dv in dist.items():
        n = len(queries[sp, c]) // 784
        out = {v: x.reshape(15, n, 28, 28) for v, x in dv.items()}
        r0 = torch.load(R0 / f'knn_{sp}_{c}.pt', weights_only=True)
        par = max(par, float((out['full'] - r0['full']).abs().max()), float((out['kc1'] - r0['coreset']).abs().max()))
        p = OUT / f'mvknn_{sp}_{c}.pt'
        torch.save(out, p); files[p.name] = A.sha(p)
    K.require(par <= 1e-6, f'R0 distance parity {par}')
    K.require(all(bool(torch.isfinite(x).all()) for dv in dist.values() for x in dv.values()), 'nonfinite')
    A.wj(OUT / 'SCORE_COMPLETE.json', {'files': files, 'r0_distance_parity_max_abs': par,
                                       'seconds': time.monotonic() - t0})


# ------------------------------------------------------------------ VisA (normal-only memory + DEV/EVAL distances)
@torch.no_grad()
def visa():
    lk = verify_lock()
    K.require(not (OUT / 'VISA_COMPLETE.json').exists(), 'no rescoring')
    vb, tt = VB(), T()
    vlock = A.rj(vb.OUT / 'LOCK.json')
    K.require(A.sha(vb.OUT / 'final.pt') == lk['visa_final_sha256'], 'VBENCH checkpoint')
    model, loras, archive, _ = vb.detector(torch.bfloat16)
    dev = {}
    for c in vb.ORDER:
        p = DEVR / f'devr_{c}.pt'
        K.require(A.sha(p) == lk['visa_devr'][c], 'devr hash')
        d = torch.load(p, weights_only=False)
        dev[c] = {'q': A.unit(d['features'].float().reshape(-1, 768)).half(), 'rows': d['rows']}
    evq, fev, rec = {}, {}, {}
    names = ('full', 'kc1')
    t0 = time.monotonic()
    for m in vb.ORDER:                          # pass 1: EVAL normals (first VISA_EVAL_N in role order) + F grids
        data = vb.load_train(vlock, m)
        ev = data['features'][tt.roles_for(data['rows'])['EVAL'][:VISA_EVAL_N]].clone()
        fev[m] = prefix_grids(L.expert_grids(model, loras, archive, ev.to(torch.bfloat16), torch.bfloat16))
        evq[m] = A.unit(ev.reshape(-1, 768)).half()
        A.log('VISA eval', m, len(ev)); del data, ev
    del model, loras; torch.cuda.empty_cache()
    dd = {c: {v: torch.empty(12, len(dev[c]['q'])) for v in names} for c in vb.ORDER}
    de = {c: {v: torch.empty(12, len(evq[c])) for v in names} for c in vb.ORDER}
    for t, m in enumerate(vb.ORDER):            # pass 2: memory of task t vs all DEV and EVAL queries
        data = vb.load_train(vlock, m)
        roles = tt.roles_for(data['rows'])
        fit_idx = torch.as_tensor(roles['FIT'])
        fit = torch.cat([A.unit(data['features'][fit_idx[i:i + 64]].reshape(-1, 768).cuda()).half()
                         for i in range(0, len(fit_idx), 64)])
        cs = A.greedy_coreset(fit, n_of(0.01, len(fit)))
        mems = {'full': fit, 'kc1': fit[cs.cuda()]}
        for c in vb.ORDER:
            qg = dev[c]['q'].cuda()
            for v, mem in mems.items():
                dd[c][v][t] = A.min_cos_dist(qg, mem, qc=1024).cpu()
            qg = evq[c].cuda()
            for v, mem in mems.items():
                de[c][v][t] = A.min_cos_dist(qg, mem, qc=1024).cpu()
        rec[m] = {'n_fit_patches': len(fit), 'n_coreset': len(cs), 'n_eval_images': len(evq[m]) // 784}
        A.log('VISA task', t + 1, m, rec[m], round(time.monotonic() - t0, 1)); A.caps()
        del data, fit, mems; torch.cuda.empty_cache()
    files = {}
    for c in vb.ORDER:
        n = len(dev[c]['rows'])
        p = OUT / f'visa_{c}.pt'
        torch.save({'dev': {v: x.reshape(12, n, 28, 28) for v, x in dd[c].items()}, 'dev_rows': dev[c]['rows'],
                    'eval': {v: x.reshape(12, -1, 28, 28) for v, x in de[c].items()}, 'f_eval_prefix': fev[c]}, p)
        files[p.name] = A.sha(p)
    A.wj(OUT / 'VISA_COMPLETE.json', {'files': files, 'per_task': rec, 'seconds': time.monotonic() - t0})


# ------------------------------------------------------------------ metrics helpers
def stats(fe, ke, s, tasks):
    """(mu, sigma) of smoothed EVAL maps at stage s pooled over tasks[:s+1]; fe: {c: [S,N,28,28]}, ke: {c: {v: ...}}."""
    acc = {}
    for c in tasks[:s + 1]:
        grids = {'F': fe[c][s], **{v: g[s] for v, g in ke[c].items()}}
        for a, g in grids.items():
            m = A.smooth(g).double(); x = acc.setdefault(a, [0., 0., 0])
            x[0] += float(m.sum()); x[1] += float((m ** 2).sum()); x[2] += m.numel()
    return {a: (s1 / n, math.sqrt(max(s2 / n - (s1 / n) ** 2, 1e-30))) for a, (s1, s2, n) in acc.items()}


def score_maps(maps, hws, masks, labels, want_hist):
    from sklearn.metrics import roc_auc_score
    out, hists = {}, {}
    for a, m in maps.items():
        m = m - min(0., float(m.min()))
        out[a] = {'pixel_ap': A.exact_ap(m, hws, masks), 'image_auroc': float(roc_auc_score(labels, A.image_scores(m)))}
        if want_hist:
            hists[a] = A.image_hists(m, hws, masks, float(m.min()), float(m.max()) + 1e-12)
            out[a]['binned_ap'] = A.binned_ap(hists[a][0].sum(0), hists[a][1].sum(0))
    return out, hists


def boot_reps(hists, arms, cats, B=A.BOOT_B, seed=A.BOOT_SEED):
    """Paired image-level bootstrap: one resample per (replicate, category), shared by all arms."""
    rng = np.random.default_rng(seed)
    reps = {a: np.empty(B) for a in arms}
    for r in range(B):
        per = {a: [] for a in arms}
        for c in cats:
            n = len(hists[arms[0], c][0])
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32)
            for a in arms:
                per[a].append(A.binned_ap(w @ hists[a, c][0], w @ hists[a, c][1]))
        for a in arms:
            reps[a][r] = np.mean(per[a])
    return reps


def ci(reps, a, b):
    d = reps[a] - reps[b]
    return {'lo': float(np.quantile(d, .025)), 'hi': float(np.quantile(d, .975)), 'B': len(d), 'unit': 'image'}


def decide(g1, g2, g2_lo):
    if g2 <= 0.005:
        return 'BYTES_SUFFICE'
    if g2 >= 0.010 and g2_lo > 0 and g1 >= 0.015:
        return 'CONSOLIDATION_HEADROOM'
    return 'INCONCLUSIVE'


def fuse(f, k, st, key):
    return (A.smooth(f) - st['F'][0]) / st['F'][1] + (A.smooth(k) - st[key][0]) / st[key][1]


# ------------------------------------------------------------------ metrics
def metrics():
    lk = verify_lock()
    K.require(not (OUT / 'RESULT.json').exists(), 'one-shot')
    sc, vc = A.rj(OUT / 'SCORE_COMPLETE.json'), A.rj(OUT / 'VISA_COMPLETE.json')
    for name, h in list(sc['files'].items()) + list(vc['files'].items()):
        K.require(A.sha(OUT / name) == h, 'artifact changed: ' + name)
    for name, h in lk['r0'].items():
        K.require(A.sha(R0 / name) == h, 'R0 changed: ' + name)
    nr = A.rj(R0 / 'NORMAL.json'); r0 = A.rj(R0 / 'RESULT_r1.json')['per_category_cells']
    result = {'mvtec': mvtec_metrics(nr, r0), 'visa_dev': visa_metrics()}
    m = result['mvtec']
    result['decision'] = {'R1': decide(m['G']['1']['delta'], m['G']['2']['delta'], m['G']['2']['image_boot']['lo']),
                          'visa_flag': result['visa_dev']['flag']}
    A.wj(OUT / 'RESULT.json', result)
    A.wj(OUT / 'FINAL_SEAL.json', {'lock_sha256': A.sha(OUT / 'LOCK.json'),
                                   'files': {p.name: A.sha(p) for p in OUT.iterdir() if p.is_file()}})
    A.log('DECISION', json.dumps(result['decision']))


def mvtec_metrics(nr, r0):
    kn = {(sp, c): prefix(torch.load(OUT / f'mvknn_{sp}_{c}.pt', weights_only=True))
          for sp in ('test', 'eval') for c in A.ORDER}
    fe = {c: A.load_norm(c, nr, 'eval')['f_eval_prefix'] for c in A.ORDER}
    fuse_keys = ['kc' + tag(r) for r in FUSE]
    ke = {c: {v: kn['eval', c][v] for v in fuse_keys} for c in A.ORDER}
    st = {s: stats(fe, ke, s, A.ORDER) for s in range(15)}
    sealed = A.rj(A.LITE / 'cells_smooth.json')
    cells, hists, parity = {}, {}, []
    karms = ['full'] + ['kc' + tag(r) for r in RATIOS] + ['rand1']
    for i, c in enumerate(A.ORDER):
        d = torch.load(A.LITE / f'scores_{c}.pt', weights_only=True)
        fg, rows = d['prefix_grids'], d['rows']
        hws = [N.native_hw(R.DATA / r['relative_path']) for r in rows]
        masks = [N.native_mask(R.DATA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        for phase, s in (('acquisition', i), ('final', 14)):
            q = kn['test', c]
            arms = ['full', 'kc1', 'kc2'] if phase == 'acquisition' else karms
            maps = {'F': A.smooth(fg[s]), **{a: A.smooth(q[a][s]) for a in arms}}
            if phase == 'final':
                maps.update({'F+' + v: fuse(fg[s], q[v][s], st[s], v) for v in fuse_keys})
            out, h = score_maps(maps, hws, masks, labels, phase == 'final')
            for a, v in out.items():
                cells[a, phase, c] = v
            for a, v in h.items():
                hists[a, c] = v
            parity.append(abs(cells['F', phase, c]['pixel_ap'] - sealed[f'{s + 1:02}_{c}']['pixel_ap']))
            A.log('MV', c, phase, {a: round(v['pixel_ap'], 4) for a, v in out.items()})
        for a, ref in (('full', 'A1'), ('kc1', 'A1cs'), ('F+kc1', 'A2cs')):
            parity.append(abs(cells[a, 'final', c]['pixel_ap'] - r0[f'{ref}|final|{c}']['pixel_ap']))
        K.require(max(parity) <= 1e-6, f'R0 parity {c} {max(parity)}')
        del masks; A.caps()
    farms = karms + ['F+' + v for v in fuse_keys]
    reps = boot_reps(hists, farms, A.ORDER)
    macro = {a: {ph: {k: float(np.mean([cells[a, ph, c][k] for c in A.ORDER])) for k in ('pixel_ap', 'image_auroc')}
                 for ph in ('acquisition', 'final') if (a, ph, A.ORDER[0]) in cells} for a in ['F'] + farms}
    ap = lambda a: macro[a]['final']['pixel_ap']
    G = {tag(r): {'delta': ap('full') - ap('kc' + tag(r)), 'image_boot': ci(reps, 'full', 'kc' + tag(r)),
                  'n_cat_pos': int(sum(cells['full', 'final', c]['pixel_ap'] > cells['kc' + tag(r), 'final', c]
                                       ['pixel_ap'] for c in A.ORDER))} for r in RATIOS}
    fusion = {v: {'delta': ap('F+' + v) - ap(v), 'image_boot': ci(reps, 'F+' + v, v)} for v in fuse_keys}
    sizes = A.rj(OUT / 'CORESETS.json')
    bytes_ = {tag(r): 2 * 768 * sum(sizes[c]['sizes'][tag(r)] for c in A.ORDER) for r in RATIOS}
    return {'macro': macro, 'per_category': {f'{a}|{ph}|{c}': v for (a, ph, c), v in cells.items()},
            'G_full_minus_coreset': G, 'G': G, 'fusion_gain_over_coreset': fusion,
            'kc1_minus_rand1': {'delta': ap('kc1') - ap('rand1'), 'image_boot': ci(reps, 'kc1', 'rand1')},
            'memory_bytes_fp16': {**bytes_, 'full': 2 * 768 * sum(sizes[c]['n_fit_patches'] for c in A.ORDER)},
            'parity_max_abs': max(parity),
            'binned_vs_exact_max_abs': max(abs(v['binned_ap'] - v['pixel_ap']) for v in cells.values()
                                           if 'binned_ap' in v)}


def visa_metrics():
    vb = VB()
    order = vb.ORDER
    data = {c: torch.load(OUT / f'visa_{c}.pt', weights_only=False) for c in order}
    kd = {c: prefix(data[c]['dev']) for c in order}
    ke = {c: {'kc1': prefix(data[c]['eval'])['kc1']} for c in order}
    fe = {c: data[c]['f_eval_prefix'] for c in order}
    s = len(order) - 1
    st = stats(fe, ke, s, order)
    cells, hists = {}, {}
    for i, c in enumerate(order):
        sc = torch.load(vb.OUT / f'scores_{c}.pt', weights_only=True)
        pos = {r['relative_path']: j for j, r in enumerate(sc['rows'])}
        rows = data[c]['dev_rows']
        K.require({r['relative_path'] for r in rows} == {r['relative_path'] for r in sc['rows'] if r['dev']},
                  f'DEV row parity {c}')
        idx = torch.as_tensor([pos[r['relative_path']] for r in rows])
        fg = sc['prefix_grids'][:, idx]
        hws = [N.native_hw(vb.VISA / r['relative_path']) for r in rows]
        masks = [N.native_mask(vb.VISA / r['mask_path'] if r['mask_path'] else None, hw) for r, hw in zip(rows, hws)]
        labels = np.array([r['label'] for r in rows])
        for phase, t in (('acquisition', i), ('final', s)):
            maps = {'F': A.smooth(fg[t]), 'full': A.smooth(kd[c]['full'][t]), 'kc1': A.smooth(kd[c]['kc1'][t])}
            if phase == 'final':
                maps['F+kc1'] = fuse(fg[t], kd[c]['kc1'][t], st, 'kc1')
            out, h = score_maps(maps, hws, masks, labels, phase == 'final')
            for a, v in out.items():
                cells[a, phase, c] = v
            for a, v in h.items():
                hists[a, c] = v
            A.log('VISA', c, phase, {a: round(v['pixel_ap'], 4) for a, v in out.items()})
    arms = ['F', 'full', 'kc1', 'F+kc1']
    reps = boot_reps(hists, arms, order)
    macro = {a: {ph: {k: float(np.mean([cells[a, ph, c][k] for c in order])) for k in ('pixel_ap', 'image_auroc')}
                 for ph in ('acquisition', 'final') if (a, ph, order[0]) in cells} for a in arms}
    ap = lambda a: macro[a]['final']['pixel_ap']
    kf = {'delta': ap('full') - ap('F'), 'image_boot': ci(reps, 'full', 'F')}
    flag = 'KNN_DOMINATES_F' if kf['delta'] >= .015 and kf['image_boot']['lo'] > 0 else \
        'F_DOMINATES_KNN' if -kf['delta'] >= .015 and kf['image_boot']['hi'] < 0 else 'NEITHER_DOMINATES'
    best = 'full' if ap('full') >= ap('F') else 'F'
    return {'macro': macro, 'per_category': {f'{a}|{ph}|{c}': v for (a, ph, c), v in cells.items()},
            'full_minus_F': kf, 'kc1_minus_F': {'delta': ap('kc1') - ap('F'), 'image_boot': ci(reps, 'kc1', 'F')},
            'Fkc1_minus_kc1': {'delta': ap('F+kc1') - ap('kc1'), 'image_boot': ci(reps, 'F+kc1', 'kc1')},
            'Fkc1_minus_best': {'best': best, 'delta': ap('F+kc1') - ap(best), 'image_boot': ci(reps, 'F+kc1', best)},
            'flag': flag, 'stats_final': st,
            'f_check_vs_cmsb_s0_0370': ap('F') - 0.370,
            'exposure': 'VisA-DEV, consumed repeatedly; diagnostic only; query tokens bf16 lite, memory fp32'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('lock', 'coresets', 'score', 'visa', 'metrics'))
    torch.backends.cuda.matmul.allow_tf32 = False
    {'lock': lock, 'coresets': coresets, 'score': score, 'visa': visa, 'metrics': metrics}[p.parse_args().phase]()
