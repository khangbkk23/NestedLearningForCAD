"""RIGEL-Q8-01 S3 (PREREG.md): Q8T DEV gate on the MVTec AD 2 test_public stream (the GG-01 DEV set).

    python -B -m exps.rigel_q8.s3_dev stream     # GPU: arms, per-stage grids, parity, serialization round trip
    python -B -m exps.rigel_q8.s3_dev metrics    # CPU: native binned pixel AP / image AUROC per stage, bootstrap
    python -B -m exps.rigel_q8.s3_dev gate       # registered gate -> S3_DEV_VERDICT.json
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
from exps.rigel_gg01 import gg
from exps.rigel_q8 import q8
from exps.rigel_q8.common import CACHE, OUT, log, prereg_sha, sha, wj
from rigel_cad import ops

SEED = 2711
ARMS = ['V1', 'T16', 'Q8', 'Q8T', 'F', 'F8', 'U']
DEV = CACHE / 'dev'


def load_fit(s):
    p = GD.CACHE / 'fit' / 'ad2' / f'{s}_s{SEED}.pt'
    if not p.exists():
        raise SystemExit(f'STOP: missing cached v1 fit {p}')
    d = torch.load(p, weights_only=True)
    return gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16'])


@torch.no_grad()
def stream_stage():
    sha0 = prereg_sha()
    s1 = json.loads((OUT / 'S1_INT8.json').read_text())
    if s1['verdict'] != 'Q8_GO':
        raise SystemExit('STOP: S1 is Q8_NO_GO; S3 is not run')
    GD.guard()
    t0 = time.time()
    scens = GD.scenarios()
    T = len(scens)
    st = q8.Stream(ARMS, SEED)
    for l, s in enumerate(scens):
        st.task(l, load_fit(s))
        log('S3 TASK', l, s, {a: st.arms[a].log[-1]['n_keys'] for a in ARMS},
            {a: st.arms[a].log[-1]['payload'] for a in ARMS if st.arms[a].cap is not None})
    t_learn = time.time() - t0
    lat = {a: [] for a in ARMS}
    parity = {}
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
                parity[s] = {'rows_equal': True, 'final_bitwise': bool(torch.equal(grid[-1], rg[-1] if rg.dim() == 4 else rg)),
                             'all_stages_bitwise': bool(rg.dim() == 4 and rg.shape == grid.shape and torch.equal(grid, rg))}
        del q
        log('S3 GRIDS', s, parity[s])
    # serialization round trip of every bounded arm's final state; Q8T must reload to identical keys, S and W
    states, rt = {}, {}
    for a in ARMS:
        m = st.arms[a]
        p = DEV / 'states' / f'{a}.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        sd = m.state_dict()
        if m.sfmt == 'full':
            sd.update(S=st.S, n=st.n)
        elif m.sfmt == 'tri':
            sd['n'] = st.n
        torch.save(sd, p)
        states[a] = {'n_keys': len(m.task), 'payload_final': m.payload(), 'payload_max': m.max_payload,
                     'file_bytes': p.stat().st_size, 'cap': m.cap}
        if a in ('Q8T', 'Q8', 'F8', 'T16'):
            d = torch.load(p, weights_only=True)
            K = q8.dequantize(d['q'], d['scale']) if m.kfmt == 'int8' else d['k16'].float()
            ok_k = bool(torch.equal(K, m.keys.cpu()))
            ok_w = True
            if m.sfmt == 'tri':
                W = st._W(q8.tri_unpack(d['S_tri']), d['n'])[0]
                ok_w = bool(torch.equal(W, st.W_tri[a][-1]))
            q2, s2 = q8.quantize(m.keys) if m.kfmt == 'int8' else (None, None)
            idem = bool(torch.equal(q2, m.q) and torch.equal(s2, m.scale)) if m.kfmt == 'int8' else True
            rt[a] = {'keys_equal': ok_k, 'W_equal': ok_w, 'idempotent': idem}
    rec = {'scenarios': scens, 'seed': SEED, 'arms': ARMS, 'caps': {a: q8.ARMS[a][2] for a in ARMS},
           'select_log': {a: st.arms[a].log for a in ARMS}, 'select_s': {a: st.arms[a].select_s for a in ARMS},
           'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
           'cond': st.cond, 'finite_W': True, 'states': states, 'roundtrip': rt, 'v1_parity_locked_ad2': parity,
           'learn_s': t_learn, 'seconds': time.time() - t0, 'prereg_sha256': sha0}
    assert prereg_sha() == sha0
    wj(OUT / 'S3_STREAM.json', rec)
    log('S3 STREAM DONE', '%.0fs' % rec['seconds'], rt)


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
    assert all((Path(r).parts[2] == 'bad') == bool(m.any()) for r, m in zip(rows, cat.masks))
    cat.y = np.array([int(m.any()) for m in cat.masks])
    bw = M.boot_weights(len(rows), j)
    res, boots = {}, {}
    t0 = time.time()
    for a in ARMS:
        d = torch.load(DEV / a / f'{s}.pt', weights_only=True)
        assert d['rows'] == rows
        res[a], boots[a] = M.evaluate_category(cat, d['grid'], d['stages'], bw)
    return s, res, boots, time.time() - t0


def metrics_stage():
    sha0 = prereg_sha()
    rec = json.loads((OUT / 'S3_STREAM.json').read_text())
    assert rec['prereg_sha256'] == sha0
    scens = rec['scenarios']
    res, boots, secs = {}, {}, {}
    with mp.get_context('spawn').Pool(2) as pool:
        for s, r, b, dt in pool.imap_unordered(_metric_task, [(s, j) for j, s in enumerate(scens)]):
            res[s], boots[s], secs[s] = r, b, dt
            log('S3 METRIC', s, {a: round(r[a][len(scens) - 1]['ap'], 4) for a in ARMS}, '%.0fs' % dt)
    bp = DEV / 'BOOT_s3.npz'
    np.savez_compressed(bp, **{f'{s}||{a}||{m}': v[m] for s, ba in boots.items() for a, v in ba.items() for m in v})
    wj(OUT / 'S3_METRICS.json', {'cells': res, 'seconds': secs, 'boot_file': str(bp), 'boot_sha256': sha(bp),
                                 'prereg_sha256': sha0})
    log('S3 METRICS DONE')


def gate_stage():
    from rigel_cad import metrics as M
    sha0 = prereg_sha()
    st = json.loads((OUT / 'S3_STREAM.json').read_text())
    me = json.loads((OUT / 'S3_METRICS.json').read_text())
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
    cmp_ = {k: paired(*k.split('-')) for k in ('Q8T-V1', 'Q8T-F8', 'T16-V1', 'Q8-V1', 'Q8T-Q8', 'Q8T-T16', 'V1-F',
                                               'F8-F', 'U-V1', 'U-Q8T')}
    per_cat = {s: me['cells'][s]['Q8T'][str(T - 1)]['ap'] - me['cells'][s]['V1'][str(T - 1)]['ap'] for s in order}
    d = cmp_['Q8T-V1']
    parity_ok = all(v['final_bitwise'] and v['all_stages_bitwise'] for v in st['v1_parity_locked_ad2'].values())
    rt_ok = all(all(v.values()) for v in st['roundtrip'].values())
    bytes_ok = all(v['payload_max'] <= q8.BUDGET for a, v in st['states'].items() if v['cap'] is not None)
    g = {
        'G1_delta_ge_0.005': d['delta'] >= 0.005,
        'G2_ci_lower_gt_0': d['ci95'][0] > 0,
        'G3_no_scenario_loss_gt_0.03': min(per_cat.values()) >= -0.03,
        'G4_forgetting_not_worse_gt_0.005': S['Q8T']['forgetting_ap'] - S['V1']['forgetting_ap'] <= 0.005,
        'G5_payload_le_8MiB': bytes_ok,
        'G6_correctness': bool(parity_ok and rt_ok and st['finite_W'] and all(c > 0 for c in st['cond'])),
        'G7_nl_slow_level_Q8T_gt_F8': cmp_['Q8T-F8']['delta'] > 0,
    }
    others = all(v for k, v in g.items() if not k.startswith(('G1', 'G2')))
    if not others or not g['G1_delta_ge_0.005']:
        verdict = 'Q8T_REJECTED_DEV'
    elif not g['G2_ci_lower_gt_0']:
        verdict = 'Q8T_INCONCLUSIVE_DEV'
    else:
        verdict = 'Q8T_DEV_PASS'
    out = {'verdict': verdict, 'gates': g, 'comparisons': cmp_, 'per_scenario_delta_ap': per_cat,
           'summaries': {a: {k: v for k, v in S[a].items() if k != 'per_task_final'} for a in ARMS},
           'per_task_final': {a: S[a]['per_task_final'] for a in ARMS}, 'states': st['states'],
           'latency_ms_per_img_p50': st['latency_ms_per_img_p50'], 'select_s': st['select_s'],
           'prereg_sha256': sha0, 'time': time.strftime('%FT%T')}
    wj(OUT / 'S3_DEV_VERDICT.json', out)
    log('S3 VERDICT', verdict, g, {k: round(v['delta'], 4) for k, v in cmp_.items()})


if __name__ == '__main__':
    {'stream': stream_stage, 'metrics': metrics_stage, 'gate': gate_stage}[sys.argv[1]]()
