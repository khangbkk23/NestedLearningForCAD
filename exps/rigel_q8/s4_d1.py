"""RIGEL-D1-01 DEV (PREREG_D1.md): CMS medium level on the MVTec AD 2 test_public stream (same DEV set as S3).

    python -B -m exps.rigel_q8.s4_d1 stream | metrics | gate
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_gg01 import dev as GD
from exps.rigel_q8 import d1, q8
from exps.rigel_q8.common import CACHE, OUT, ROOT, log, sha, wj
from exps.rigel_q8.s3_dev import SEED, load_fit
from rigel_cad import ops

ARMS = ['V1', 'Q8T', 'Q4T', 'D1', 'F8', 'U']
DEV = CACHE / 'dev_d1'
S3DEV = CACHE / 'dev'
PREREG = ROOT / 'exps/rigel_q8/PREREG_D1.md'
PREREG_SHA = 'f57997fc1d9dfd853c317a510bcea75609017b6262e7400a984c85649e822380'


def prereg_sha():
    s = sha(PREREG)
    if s != PREREG_SHA:
        raise SystemExit('STOP: PREREG_D1.md changed after registration')
    return s


@torch.no_grad()
def int4_error(K16, W, samples):
    q, s = d1.quantize4(K16)
    Kh = q8.dequantize(q, s)
    q2, s2 = d1.quantize4(Kh)
    rel = []
    for x in samples:
        e0 = ops.energy_whitened(x, K16.float(), W)
        e1 = ops.energy_whitened(x, Kh, W)
        rel.append(((e1 - e0).abs() / e0.clamp_min(1e-12)).numpy())
    rel = np.concatenate(rel)
    return {'idempotent': bool(torch.equal(q, q2) and torch.equal(s, s2)),
            'cos_min': float(torch.nn.functional.cosine_similarity(K16.float(), Kh, dim=1).min()),
            'median_rel_dE': float(np.median(rel)), 'p99_rel_dE': float(np.percentile(rel, 99))}


@torch.no_grad()
def stream_stage():
    sha0 = prereg_sha()
    s3 = json.loads((OUT / 'S3_DEV_VERDICT.json').read_text())
    if s3['verdict'] != 'Q8T_DEV_PASS':
        raise SystemExit('STOP: D1 runs only after Q8T_DEV_PASS')
    GD.guard()
    t0 = time.time()
    scens = GD.scenarios()
    T = len(scens)
    st = d1.Stream(ARMS, SEED)
    for l, s in enumerate(scens):
        st.task(l, load_fit(s))
        r = st.arms['D1'].log[-1]
        log('D1 TASK', l, s, {a: len(st.arms[a].keys) for a in ARMS}, 'D1 fast/med/evicted', r['n_fast'], r['n_medium'],
            r['evicted'], {a: st.arms[a].payload() for a in ARMS if st.arms[a].cap is not None})
    t_learn = time.time() - t0
    lat = {a: [] for a in ARMS}
    parity, hit = {}, {}
    for j, s in enumerate(scens):
        rows = GD.test_rows(s)
        q = GD.tokens(f'ad2/{s}_test_public', rows).reshape(-1, q8.D)
        for a in ARMS:
            gl = []
            for l in range(j, T):
                torch.cuda.synchronize()
                t1 = time.time()
                gl.append(st.energy(a, l, q))
                torch.cuda.synchronize()
                lat[a].append((time.time() - t1) / len(rows))
            grid = torch.stack(gl).reshape(T - j, len(rows), 28, 28).contiguous()
            p = DEV / a / f'{s}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rows, 'grid': grid, 'stages': list(range(j, T)), 'task': s, 'pos': j}, p)
            if a == 'V1':
                rg = torch.load(GD.LOCKED_AD2 / f'{s}__test_public.pt', weights_only=False)['grid']
                parity.setdefault(s, {})['V1_locked'] = bool(rg.dim() == 4 and torch.equal(grid, rg))
            if a == 'Q8T':
                rg = torch.load(S3DEV / 'Q8T' / f'{s}.pt', weights_only=True)['grid']
                parity.setdefault(s, {})['Q8T_s3'] = bool(torch.equal(grid, rg))
        # medium hit rate at the final stage: share of test tokens whose nearest key lies in the medium level
        K, W = st.state('D1', T - 1)
        K, W = K.cuda(), W.cuda()
        mw, nf = K @ W, st.n_fast[T - 1]
        idx = torch.cat([ops.sqdist(q[i:i + 512].cuda().float() @ W, mw).argmin(1) for i in range(0, len(q), 512)])
        hit[s] = float((idx >= nf).float().mean())
        del q
        log('D1 GRIDS', s, parity[s], 'medium hit %.3f' % hit[s])
    rt = {}
    for a in ('Q4T', 'D1'):
        m = st.arms[a]
        p = DEV / 'states' / f'{a}.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        sd = m.state_dict()
        sd['n'] = st.n
        torch.save(sd, p)
        dd = torch.load(p, weights_only=True)
        if a == 'Q4T':
            K = q8.dequantize(d1.unpack4(dd['q4']), dd['scale'])
        else:
            K = torch.cat([q8.dequantize(dd['fq'], dd['fs']), q8.dequantize(d1.unpack4(dd['mq4']), dd['ms'])])
        W = st._W(q8.tri_unpack(dd['S_tri']), dd['n'])[0]
        rt[a] = {'keys_equal': bool(torch.equal(K, m.keys.cpu())), 'W_equal': bool(torch.equal(W, st.W_tri[a][-1])),
                 'file_bytes': p.stat().st_size}
    # int4 / int8 idempotence on the final D1 levels, and the int4 energy error (label-free, diagnostic)
    m = st.arms['D1']
    q2, s2 = d1.quantize4(q8.dequantize(m.mq, m.ms))
    q3, s3_ = q8.quantize(q8.dequantize(m.fq, m.fs))
    idem = bool(torch.equal(q2, m.mq) and torch.equal(s2, m.ms) and torch.equal(q3, m.fq) and torch.equal(s3_, m.fs))
    v1s = torch.load(GD.CACHE / 'dev' / 'states' / 'V1.pt', weights_only=True)
    Wv = ops.whitener(v1s['S'], v1s['n'])[0].float().cuda()
    from rigel_cad import data as DATA
    smp = []
    for l, s in enumerate(scens):
        x = GD.tokens(f'ad2/{s}_train', DATA.ad2_files(GD.AD2_ROOT, s, 'train')).reshape(-1, q8.D)
        g = torch.Generator().manual_seed(20261126 + l)
        smp.append(x[torch.randperm(len(x), generator=g)[:4096]].cuda())
        del x
    i4 = int4_error(v1s['keys'].cuda(), Wv, smp)
    rec = {'scenarios': scens, 'seed': SEED, 'arms': ARMS,
           'select_log': {a: st.arms[a].log for a in ARMS}, 'select_s': {a: st.arms[a].select_s for a in ARMS},
           'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
           'cond': st.cond, 'parity': parity, 'roundtrip': rt, 'idempotent': idem, 'medium_hit_rate': hit,
           'int4_error_on_v1_keys': i4,
           'payload_max': {a: st.arms[a].max_payload for a in ARMS if st.arms[a].cap is not None},
           'learn_s': t_learn, 'seconds': time.time() - t0, 'prereg_sha256': sha0}
    assert prereg_sha() == sha0
    wj(OUT / 'D1_STREAM.json', rec)
    log('D1 STREAM DONE', '%.0fs' % rec['seconds'], rt, 'idem', idem, 'int4', i4)


def _metric_task(args):
    s, j = args
    from rigel_cad import metrics as M
    rows = GD.test_rows(s)
    mm = GD.mask_map(s)

    class Cat:
        pass
    cat = Cat()
    cat.rows = rows
    cat.hws = [M.native_hw(GD.AD2_ROOT / r) for r in rows]
    cat.masks = [M.native_mask(mm[Path(r).stem] if Path(r).parts[2] == 'bad' else None, hw)
                 for r, hw in zip(rows, cat.hws)]
    cat.y = np.array([int(m.any()) for m in cat.masks])
    bw = M.boot_weights(len(rows), j)
    res, boots = {}, {}
    t0 = time.time()
    for a in ARMS:
        d = torch.load(DEV / a / f'{s}.pt', weights_only=True)
        res[a], boots[a] = M.evaluate_category(cat, d['grid'], d['stages'], bw)
    return s, res, boots, time.time() - t0


def metrics_stage():
    sha0 = prereg_sha()
    rec = json.loads((OUT / 'D1_STREAM.json').read_text())
    scens = rec['scenarios']
    res, boots, secs = {}, {}, {}
    with mp.get_context('spawn').Pool(2) as pool:
        for s, r, b, dt in pool.imap_unordered(_metric_task, [(s, j) for j, s in enumerate(scens)]):
            res[s], boots[s], secs[s] = r, b, dt
            log('D1 METRIC', s, {a: round(r[a][len(scens) - 1]['ap'], 4) for a in ARMS}, '%.0fs' % dt)
    bp = DEV / 'BOOT_d1.npz'
    np.savez_compressed(bp, **{f'{s}||{a}||{m}': v[m] for s, ba in boots.items() for a, v in ba.items() for m in v})
    wj(OUT / 'D1_METRICS.json', {'cells': res, 'seconds': secs, 'boot_file': str(bp), 'boot_sha256': sha(bp),
                                 'prereg_sha256': sha0})
    log('D1 METRICS DONE')


def gate_stage():
    from rigel_cad import metrics as M
    sha0 = prereg_sha()
    st = json.loads((OUT / 'D1_STREAM.json').read_text())
    me = json.loads((OUT / 'D1_METRICS.json').read_text())
    order = st['scenarios']
    T = len(order)
    bf = np.load(me['boot_file'])
    S = {a: M.stream_summary(order, {t: {int(l): v for l, v in me['cells'][t][a].items()} for t in order}) for a in ARMS}
    boot = {a: np.mean([bf[f'{s}||{a}||ap'] for s in order], axis=0) for a in ARMS}
    bau = {a: np.nanmean([bf[f'{s}||{a}||auroc'] for s in order], axis=0) for a in ARMS}

    def paired(x, y):
        d, da = boot[x] - boot[y], bau[x] - bau[y]
        return {'delta': S[x]['pixel_ap'] - S[y]['pixel_ap'], 'ci95': [float(v) for v in np.nanpercentile(d, [2.5, 97.5])],
                'delta_auroc': S[x]['image_auroc'] - S[y]['image_auroc'],
                'ci95_auroc': [float(v) for v in np.nanpercentile(da, [2.5, 97.5])]}
    cmp_ = {k: paired(*k.split('-')) for k in ('D1-Q8T', 'D1-Q4T', 'D1-F8', 'D1-V1', 'Q4T-Q8T', 'Q8T-V1', 'U-D1')}
    per = {s: me['cells'][s]['D1'][str(T - 1)]['ap'] - me['cells'][s]['Q8T'][str(T - 1)]['ap'] for s in order}
    d = cmp_['D1-Q8T']
    parity_ok = all(v['V1_locked'] and v['Q8T_s3'] for v in st['parity'].values())
    rt_ok = all(v['keys_equal'] and v['W_equal'] for v in st['roundtrip'].values())
    g = {
        'G1_delta_ge_0.005': d['delta'] >= 0.005,
        'G2_ci_lower_gt_0': d['ci95'][0] > 0,
        'G3_no_scenario_loss_gt_0.03': min(per.values()) >= -0.03,
        'G4_forgetting_not_worse_gt_0.005': S['D1']['forgetting_ap'] - S['Q8T']['forgetting_ap'] <= 0.005,
        'G5_payload_le_8MiB': all(v <= q8.BUDGET for v in st['payload_max'].values()),
        'G6_correctness': bool(parity_ok and rt_ok and st['idempotent'] and all(c > 0 for c in st['cond'])),
        'G7_hierarchy_D1_gt_Q4T': cmp_['D1-Q4T']['delta'] > 0,
        'G8_nl_D1_gt_F8': cmp_['D1-F8']['delta'] > 0,
    }
    others = all(v for k, v in g.items() if not k.startswith(('G1', 'G2')))
    verdict = ('D1_REJECTED_DEV' if (not others or not g['G1_delta_ge_0.005'])
               else 'D1_INCONCLUSIVE_DEV' if not g['G2_ci_lower_gt_0'] else 'D1_DEV_PASS')
    out = {'verdict': verdict, 'gates': g, 'comparisons': cmp_, 'per_scenario_delta_ap_vs_Q8T': per,
           'summaries': {a: {k: v for k, v in S[a].items() if k != 'per_task_final'} for a in ARMS},
           'per_task_final': {a: S[a]['per_task_final'] for a in ARMS}, 'medium_hit_rate': st['medium_hit_rate'],
           'int4_error_on_v1_keys': st['int4_error_on_v1_keys'], 'latency_ms_per_img_p50': st['latency_ms_per_img_p50'],
           'prereg_sha256': sha0, 'time': time.strftime('%FT%T')}
    wj(OUT / 'D1_DEV_VERDICT.json', out)
    log('D1 VERDICT', verdict, g, {k: round(v['delta'], 4) for k, v in cmp_.items()})


if __name__ == '__main__':
    {'stream': stream_stage, 'metrics': metrics_stage, 'gate': gate_stage}[sys.argv[1]]()
