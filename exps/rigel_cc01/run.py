"""RIGEL-CC-01 driver (SPEC.md). Stages:

    python -B -m exps.rigel_cc01.run audit              # correctness audit (CPU + MPDD restoration on GPU)
    python -B -m exps.rigel_cc01.run stream <ds> [o s]  # GPU: 7 arms, per-stage grids, accounting, distortion
    python -B -m exps.rigel_cc01.run metrics <ds> [o s] # CPU: rigel_cad metrics, bootstrap (order a only)
    python -B -m exps.rigel_cc01.run gate               # provisional selection gate (primary cell a|2711)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_cc01 import cc
from exps.rigel_gg01 import gg
from exps.rigel_q8 import d1, q8
from rigel_cad import ops

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/rigel_cc01'
CACHE = Path('/mnt/k/NLCAD/cache/rigel_cc01')
GG_FIT = Path('/mnt/k/NLCAD/cache/rigel_gg01/fit')
LOCK_GRID = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench')
LOCK_MET = ROOT / 'results/hope_cad/nl03_multirate_01/bench'
SPEC = ROOT / 'exps/rigel_cc01/SPEC.md'
SPEC_SHA = 'f1a26f661e4020e8acf76abe9010442f24b8f15925247734624eabd93245698c'
DATA = {'mvtec': 'data/mvtec', 'visa': 'data/visa', 'mpdd': 'data/mpdd'}
DSETS = ['mvtec', 'visa', 'mpdd']


def sha(p):
    from exps.rigel_gg01.dev import sha as _s
    return _s(p)


def spec_ok():
    s = sha(SPEC)
    if s != SPEC_SHA:
        raise SystemExit('STOP: SPEC.md changed after registration')
    return s


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'cc.log', 'a') as f:
        f.write(s + '\n')


def wj(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1, default=float))


def guard():
    for m in ('/mnt/d', '/mnt/e', '/mnt/k'):
        s = os.statvfs(m)
        if s.f_bavail * s.f_frsize / 2 ** 30 < 30:
            raise SystemExit(f'STOP: {m} below 30 GiB')


# ---------------------------------------------------------------------------------------------- canonical inputs
_G = None


def G_():
    global _G
    if _G is None:
        from exps.nl03_pressure_00 import core as PC
        _G = PC.bind()
    return _G


def order_of(ds, ok):
    from exps.nl03_multirate_01 import final as FN
    return FN.bench_order(ds, ok)


def train_x16(t):
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    G = G_()
    return RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))[0]


def fit_of(ds, c, seed, x16=None):
    """v1 per-task fit (gg.fit_task); GG-01 cache for mvtec/visa seed 2711, else computed and cached here."""
    for p in (GG_FIT / ds / f'{c}_s{seed}.pt', CACHE / 'fit' / ds / f'{c}_s{seed}.pt'):
        if p.exists():
            d = torch.load(p, weights_only=True)
            return gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16']), d['fit_s'], str(p)
    t0 = time.time()
    f = gg.fit_task(x16, seed)
    torch.cuda.synchronize()
    dt = time.time() - t0
    p = CACHE / 'fit' / ds / f'{c}_s{seed}.pt'
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kc': f.kc, 'S': f.S, 'n': f.n, 'n_tokens': f.n_tokens, 'keys16': f.keys16, 'fit_s': dt}, p)
    return f, dt, str(p)


def test_tokens(ds, c):
    from exps.nl03_pressure_00 import core as PC
    G = G_()
    PC.use(G, ds)
    return G.load_test(ds, c, PC.extract_rec(G, ds))


# ---------------------------------------------------------------------------------------------- correctness audit
@torch.no_grad()
def audit():
    sha0 = spec_ok()
    from rigel_cad import data as DATA_
    res = {}
    # 1. AD 2 masks
    root = Path('/mnt/k/mvtec_ad_2')
    bad = good = wrong = 0
    for s in DATA_.ad2_scenarios(root):
        for r in DATA_.ad2_files(root, s, 'test_public'):
            m = DATA_.ad2_mask_of(root, r)
            isgood = Path(r).parts[2] == 'good'
            good += isgood
            bad += not isgood
            wrong += (m is not None) if isgood else (m is None or m.name != Path(r).stem + '_mask.png')
    res['ad2_mask'] = {'good': good, 'bad': bad, 'wrong': wrong, 'ok': wrong == 0 and good == 379 and bad == 705}
    log('AUDIT ad2 mask', res['ad2_mask'])
    # 2. representations
    g = torch.Generator().manual_seed(0)
    x = ops.unit(torch.randn(2000, 768, generator=g)).half()
    q, s = q8.quantize(x)
    q4, s4 = d1.quantize4(x)
    rep = {'fp16_lossless': bool(torch.equal(x.float().half(), x)),
           'int8_range': int(q.abs().max()) <= 127, 'int8_idem': bool(torch.equal(q8.quantize(q8.dequantize(q, s))[0], q)),
           'int4_range': int(q4.abs().max()) <= 7, 'int4_idem': bool(torch.equal(d1.quantize4(q8.dequantize(q4, s4))[0], q4)),
           'int4_pack_roundtrip': bool(torch.equal(d1.unpack4(d1.pack4(q4)), q4)),
           'int4_packed_bytes_per_key': d1.pack4(q4).shape[1] + 2, 'int8_bytes_per_key': q.shape[1] + 2}
    res['representations'] = {**rep, 'ok': all(v for k, v in rep.items() if not k.endswith('per_key'))}
    log('AUDIT representations', res['representations'])
    # 3. triangular S: W from the fp32 triangle == W from the full fp32 matrix (bitwise)
    a = torch.randn(768, 4000, generator=g, dtype=torch.float64)
    S = a @ a.T
    t = q8.tri_pack(S)
    W_tri = ops.whitener(q8.tri_unpack(t), 4000)[0]
    W_f32 = ops.whitener(S.float().double(), 4000)[0]
    W_64 = ops.whitener(S, 4000)[0]
    res['triangle'] = {'roundtrip_equals_fp32_full': bool(torch.equal(q8.tri_unpack(t), S.float().double())),
                       'W_tri_equals_W_fp32_full': bool(torch.equal(W_tri, W_f32)),
                       'W_fp32_vs_fp64_rel': float((W_f32 - W_64).norm() / W_64.norm()),
                       'packed_numel': t.numel()}
    res['triangle']['ok'] = res['triangle']['roundtrip_equals_fp32_full'] and res['triangle']['W_tri_equals_W_fp32_full']
    log('AUDIT triangle', res['triangle'])
    # 4. save / load / continue (MPDD order a, seed 2711): tasks 0-2, persist, restore, continue 3-5
    ds, seed = 'mpdd', 2711
    order = order_of(ds, 'a')
    fits = []
    for t in order:
        c = t.split('/')[1]
        x16 = None if (GG_FIT / ds / f'{c}_s{seed}.pt').exists() or (CACHE / 'fit' / ds / f'{c}_s{seed}.pt').exists() else train_x16(t)
        fits.append(fit_of(ds, c, seed, x16)[0])
        del x16
    full = cc.Stream(cc.ARMS, seed)
    for l, f in enumerate(fits):
        full.task(l, f)
    half = cc.Stream(cc.ARMS, seed)
    for l, f in enumerate(fits[:3]):
        half.task(l, f)
    qtest, _ = test_tokens(ds, order[-1].split('/')[1])
    qtest = qtest.reshape(-1, q8.D)
    rt = {}
    for a in cc.ARMS:
        sd = cc.persist(half, a)
        p = CACHE / 'audit' / f'{a}_after3.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, p)
        sd2 = torch.load(p, weights_only=True)
        st = cc.restore(a, sd2, seed)
        for l, f in enumerate(fits[3:], start=3):
            st.task(l, f)
        K0, W0 = full.state(a, len(order) - 1)
        K1, W1 = st.state(a, 2)                                  # restored stream has 3 snapshots (tasks 3-5)
        e0, e1 = full.energy(a, len(order) - 1, qtest), st.energy(a, 2, qtest)
        rel = float(((e1 - e0).abs() / e0.clamp_min(1e-12)).max())
        r = {'keys_bitwise': bool(torch.equal(K0, K1)),
             'W_bitwise': (W0 is None and W1 is None) or bool(torch.equal(W0, W1)),
             'W_rel_diff': 0.0 if W0 is None else float((W0 - W1).norm() / W0.norm()),
             'energy_max_rel_diff': rel, 'payload_bytes_after3': cc.payload_bytes(sd), 'file_bytes_after3': p.stat().st_size}
        r['class'] = 'numerical_approximation' if a == 'Q8' else 'mathematical_equivalence'
        r['ok'] = (r['keys_bitwise'] and rel <= 1e-4) if a == 'Q8' else (r['keys_bitwise'] and r['W_bitwise'] and rel == 0.0)
        rt[a] = r
        log('AUDIT restore', a, r)
    res['restore_continue_mpdd'] = rt
    res['ok'] = res['ad2_mask']['ok'] and res['representations']['ok'] and res['triangle']['ok'] and all(v['ok'] for v in rt.values())
    res['spec_sha256'] = sha0
    wj(OUT / 'AUDIT.json', res)
    log('AUDIT', 'PASS' if res['ok'] else 'FAIL')


# ---------------------------------------------------------------------------------------------- stream (GPU)
def tag_of(ds, ok, seed):
    return f'{ds}_{ok}{seed}'


@torch.no_grad()
def stream(ds, ok='a', seed=2711, arms=None):
    sha0 = spec_ok()
    arms = arms or cc.ARMS
    guard()
    tag = tag_of(ds, ok, seed)
    order = order_of(ds, ok)
    T = len(order)
    st = cc.Stream(arms, seed)
    t0 = time.time()
    fit_s, fit_src, samples, logical = 0.0, {}, [], {a: [] for a in arms}
    primary = (ok, seed) == ('a', 2711)
    for l, t in enumerate(order):
        c = t.split('/')[1]
        cached = any(p.exists() for p in (GG_FIT / ds / f'{c}_s{seed}.pt', CACHE / 'fit' / ds / f'{c}_s{seed}.pt'))
        x16 = train_x16(t) if (not cached or primary) else None
        f, dt, src = fit_of(ds, c, seed, x16)
        fit_s += dt
        fit_src[c] = src
        if primary:
            g = torch.Generator().manual_seed(20261126 + l)
            samples.append(x16[torch.randperm(len(x16), generator=g)[:4096].to(x16.device)].cpu())
        del x16
        torch.cuda.empty_cache()
        st.task(l, f)
        for a in arms:
            m = st.arms[a]
            L = cc.logical_bytes(a, len(m.keys))
            assert L <= q8.BUDGET, (a, L)
            logical[a].append(L)
        log('STREAM', tag, l, c, {a: len(st.arms[a].keys) for a in arms})
    t_learn = time.time() - t0
    lat = {a: [] for a in arms}
    parity, runtime = {}, {}
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = test_tokens(ds, c)
        q = q.reshape(-1, q8.D)
        for a in arms:
            gl = []
            if j == T - 1:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            for l in range(j, T):
                torch.cuda.synchronize()
                t1 = time.time()
                gl.append(st.energy(a, l, q))
                torch.cuda.synchronize()
                lat[a].append((time.time() - t1) / len(rows))
            if j == T - 1:
                runtime[a] = int(torch.cuda.max_memory_allocated() - base)
            grid = torch.stack(gl).reshape(T - j, len(rows), 28, 28).contiguous()
            p = CACHE / 'grids' / tag / a / f'{c}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': list(rows), 'grid': grid, 'stages': list(range(j, T)), 'task': t, 'pos': j}, p)
            if a == 'V1' and ok in ('a', 'b', 'c') and (LOCK_GRID / f'{tag}' / 'E' / f'{c}.pt').exists():
                rg = torch.load(LOCK_GRID / tag / 'E' / f'{c}.pt', weights_only=False)
                parity[c] = bool(list(rg['rows']) == list(rows) and torch.equal(rg['grid'], grid))
        del q
        torch.cuda.empty_cache()
        log('GRIDS', tag, c, 'V1 parity', parity.get(c))
    acct = {}
    for a in arms:
        sd = cc.persist(st, a)
        p = CACHE / 'states' / tag / f'{a}.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, p)
        n = len(st.arms[a].keys)
        acct[a] = {'n_keys': n, 'L_logical_final': logical[a][-1], 'L_logical_max': max(logical[a]),
                   'P_persisted': cc.payload_bytes(sd), 'F_file': p.stat().st_size,
                   'R_runtime_scoring_peak': runtime.get(a),
                   'R_resident_fp32_keys_W_mW': n * 768 * 4 * (2 if st.arms[a].sfmt else 1) + (768 * 768 * 4 if st.arms[a].sfmt else 0)}
    dist = None
    if primary:
        K16 = st.state('V1', T - 1)[0].cuda()
        W = st.state('V1', T - 1)[1].cuda()
        dist = {}
        for name, Q in (('int8', q8.quantize), ('int4', d1.quantize4)):
            qq, ss = Q(K16)
            Kh = q8.dequantize(qq, ss)
            mw0, mw1 = K16.float() @ W, Kh @ W
            rel, chg = [], []
            for smp in samples:
                for i in range(0, len(smp), 1024):
                    z = smp[i:i + 1024].cuda().float() @ W
                    d0, d1_ = ops.sqdist(z, mw0), ops.sqdist(z, mw1)
                    v0, i0 = d0.min(1)
                    v1, i1 = d1_.min(1)
                    rel.append(((v1 - v0).abs() / v0.clamp_min(1e-12)).cpu().numpy())
                    chg.append((i0 != i1).cpu().numpy())
            rel, chg = np.concatenate(rel), np.concatenate(chg)
            cos = torch.nn.functional.cosine_similarity(K16.float(), Kh, dim=1)
            dist[name] = {'median_rel_dE': float(np.median(rel)), 'p99_rel_dE': float(np.percentile(rel, 99)),
                          'nearest_changed': float(chg.mean()), 'cos_min': float(cos.min()), 'cos_median': float(cos.median()),
                          'rel_l2_median': float(((K16.float() - Kh).norm(dim=1) / K16.float().norm(dim=1)).median())}
    rec = {'tag': tag, 'dataset': ds, 'order': order, 'seed': seed, 'arms': arms, 'fit_s_shared': fit_s, 'fit_src': fit_src,
           'select_s': {a: st.arms[a].select_s for a in arms},
           'update_s': {a: fit_s + st.arms[a].select_s for a in arms},
           'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
           'logical_bytes_per_stage': logical, 'accounting': acct, 'v1_grid_parity_locked': parity,
           'distortion_on_v1_keys': dist, 'learn_s': t_learn, 'seconds': time.time() - t0, 'spec_sha256': sha0}
    wj(OUT / f'STREAM_{tag}.json', rec)
    log('STREAM DONE', tag, '%.0fs' % rec['seconds'], 'parity all', all(parity.values()) if parity else None)


# ---------------------------------------------------------------------------------------------- metrics (CPU)
def _metric_task(args):
    ds, tag, t, j, arms, boot = args
    from rigel_cad import data as DATA_
    from rigel_cad import metrics as M
    c = t.split('/')[1]
    root = ROOT / DATA[ds]
    cat = M.Category(root, DATA_.category_split(ds, root, c)[1])
    bw = M.boot_weights(len(cat.rows), j) if boot else None
    res, boots = {}, {}
    t0 = time.time()
    for a in arms:
        d = torch.load(CACHE / 'grids' / tag / a / f'{c}.pt', weights_only=True)
        assert d['rows'] == cat.rows, (tag, a, c)
        res[a], b = M.evaluate_category(cat, d['grid'], d['stages'], bw)
        if b is not None:
            boots[a] = b
    return t, res, boots, time.time() - t0


def metrics(ds, ok='a', seed=2711):
    sha0 = spec_ok()
    tag = tag_of(ds, ok, seed)
    rec = json.loads((OUT / f'STREAM_{tag}.json').read_text())
    boot = ok == 'a'
    args = [(ds, tag, t, j, rec['arms'], boot) for j, t in enumerate(rec['order'])]
    res, boots, secs = {}, {}, {}
    with mp.get_context('spawn').Pool(2) as pool:
        for t, r, b, dt in pool.imap_unordered(_metric_task, args):
            res[t], boots[t], secs[t] = r, b, dt
            log('METRIC', tag, t, {a: round(r[a][len(rec['order']) - 1]['ap'], 4) for a in rec['arms']}, '%.0fs' % dt)
    out = {'tag': tag, 'cells': res, 'seconds': secs, 'spec_sha256': sha0}
    if boot:
        bp = CACHE / 'boot' / f'boot_{tag}.npz'
        bp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bp, **{f'{t}||{a}||{m}': v[m] for t, ba in boots.items() for a, v in ba.items() for m in v})
        out.update(boot_file=str(bp), boot_sha256=sha(bp))
    # V1 metric parity with the locked METRICS (arm E), every stage
    lp = LOCK_MET / f'METRICS_{tag}.json'
    if 'V1' in rec['arms'] and lp.exists():
        lm = json.loads(lp.read_text())['cells']
        mism = [(t, l) for t in rec['order'] for l, v in res[t]['V1'].items()
                if v['ap'] != lm[t]['E'][str(l)]['ap'] or v['i_auroc'] != lm[t]['E'][str(l)]['i_auroc']]
        out['v1_metric_parity_locked'] = {'file': str(lp), 'sha256': sha(lp), 'mismatches': mism, 'ok': not mism}
    wj(OUT / f'METRICS_{tag}.json', out)
    log('METRICS DONE', tag, out.get('v1_metric_parity_locked', {}).get('ok'))


# ---------------------------------------------------------------------------------------------- provisional gate
def summary(order, cells, a):
    from rigel_cad import metrics as M
    return M.stream_summary(order, {t: {int(l): v for l, v in cells[t][a].items()} for t in order})


def gate():
    sha0 = spec_ok()
    au = json.loads((OUT / 'AUDIT.json').read_text())
    tab, deltas, boots_ci, percat = {}, {}, {}, {}
    acct, lat = {}, {}
    for ds in DSETS:
        tag = tag_of(ds, 'a', 2711)
        st = json.loads((OUT / f'STREAM_{tag}.json').read_text())
        me = json.loads((OUT / f'METRICS_{tag}.json').read_text())
        order, T = st['order'], len(st['order'])
        S = {a: summary(order, me['cells'], a) for a in st['arms']}
        tab[ds] = {a: {k: v for k, v in S[a].items() if k != 'per_task_final'} for a in st['arms']}
        bf = np.load(me['boot_file'])
        bap = {a: np.mean([bf[f'{t}||{a}||ap'] for t in order], axis=0) for a in st['arms']}
        bau = {a: np.nanmean([bf[f'{t}||{a}||auroc'] for t in order], axis=0) for a in st['arms']}
        ci = lambda x: [float(v) for v in np.nanpercentile(x, [2.5, 97.5])]
        deltas[ds], boots_ci[ds] = {}, {}
        pairs = [(c, 'V1') for c in cc.ELIGIBLE] + [('Q8T', 'F8'), ('Q8', 'F8'), ('Q4T', 'F4'), ('F8', 'V1'), ('F4', 'V1')]
        for x, y in pairs:
            deltas[ds][f'{x}-{y}'] = {'ap': S[x]['pixel_ap'] - S[y]['pixel_ap'], 'auroc': S[x]['image_auroc'] - S[y]['image_auroc'],
                                      'fm': S[x]['forgetting_ap'] - S[y]['forgetting_ap'],
                                      'acq': S[x]['acquisition_ap'] - S[y]['acquisition_ap'],
                                      'preq': S[x]['prequential_ap'] - S[y]['prequential_ap'],
                                      'ap_ci95': ci(bap[x] - bap[y]), 'auroc_ci95': ci(bau[x] - bau[y])}
        percat[ds] = {a: {t: {'ap': me['cells'][t][a][str(T - 1)]['ap'], 'auroc': me['cells'][t][a][str(T - 1)]['i_auroc']}
                          for t in order} for a in st['arms']}
        acct[ds], lat[ds] = st['accounting'], st['latency_ms_per_img_p50']
        tab[ds]['_meta'] = {'v1_grid_parity': all(st['v1_grid_parity_locked'].values()) and len(st['v1_grid_parity_locked']) == T,
                            'v1_metric_parity': me.get('v1_metric_parity_locked', {}).get('ok'),
                            'update_s': st['update_s'], 'distortion': st['distortion_on_v1_keys']}
    correct = au['ok'] and all(tab[ds]['_meta']['v1_grid_parity'] and tab[ds]['_meta']['v1_metric_parity'] for ds in DSETS)
    gates, eligible = {}, []
    for c in cc.ELIGIBLE:
        d = {ds: deltas[ds][f'{c}-V1'] for ds in DSETS}
        worst_cat = min(percat[ds][c][t]['ap'] - percat[ds]['V1'][t]['ap'] for ds in DSETS for t in percat[ds]['V1'])
        acc_ok = all(acct[ds][c]['L_logical_max'] <= q8.BUDGET and acct[ds][c]['P_persisted'] <= q8.BUDGET for ds in DSETS)
        g = {'G1_mean_dAP_ge_0.005': float(np.mean([d[ds]['ap'] for ds in DSETS])) >= 0.005,
             'G2_ge2_datasets_improve': sum(d[ds]['ap'] > 0 for ds in DSETS) >= 2,
             'G3_no_dataset_loss_gt_0.01': min(d[ds]['ap'] for ds in DSETS) >= -0.01,
             'G4_auroc_drop_le_0.005': min(d[ds]['auroc'] for ds in DSETS) >= -0.005,
             'G5_fm_worsen_le_0.005': max(d[ds]['fm'] for ds in DSETS) <= 0.005,
             'G6_no_category_loss_gt_0.03': worst_cat >= -0.03,
             'G7_accounting_and_correctness': bool(acc_ok and correct)}
        gates[c] = {'gates': g, 'pass': all(g.values()), 'mean_dAP': float(np.mean([d[ds]['ap'] for ds in DSETS])),
                    'worst_dataset_dAP': min(d[ds]['ap'] for ds in DSETS), 'worst_category_dAP': worst_cat,
                    'mean_latency_ms': float(np.mean([lat[ds][c] for ds in DSETS]))}
        if gates[c]['pass']:
            eligible.append(c)
    ranked = sorted(eligible, key=lambda c: (-gates[c]['mean_dAP'], -gates[c]['worst_dataset_dAP'], gates[c]['mean_latency_ms']))
    sel = ranked[0] if ranked else None
    out = {'provisional': 'SELECTED' if sel else 'COMPRESSION_REJECTED', 'selected': sel, 'ranking': ranked,
           'gates': gates, 'correctness_ok': correct, 'table': tab, 'deltas': deltas, 'per_category': percat,
           'accounting': acct, 'latency_ms_per_img_p50': lat, 'audit': au, 'spec_sha256': sha0, 'time': time.strftime('%FT%T')}
    wj(OUT / 'GATE.json', out)
    log('GATE', out['provisional'], sel, {c: (gates[c]['pass'], round(gates[c]['mean_dAP'], 4)) for c in cc.ELIGIBLE})


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'audit':
        audit()
    elif cmd == 'gate':
        gate()
    else:
        ds = sys.argv[2]
        ok = sys.argv[3] if len(sys.argv) > 3 else 'a'
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2711
        if cmd == 'stream':
            stream(ds, ok, seed, sys.argv[5].split(',') if len(sys.argv) > 5 else None)
        else:
            metrics(ds, ok, seed)
