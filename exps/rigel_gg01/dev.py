"""GG-01 DEV stage (registered in exps/rigel_gg01/PREREG.md): MVTec AD 2 continual stream, test_public as the
development set. Implementation (gg.py) is separate from this evaluation driver.

    python -B -m exps.rigel_gg01.dev stream     # GPU: learn the 8 scenarios, per-stage test_public grids, diagnostics
    python -B -m exps.rigel_gg01.dev metrics    # CPU: native binned pixel AP / image AUROC per stage, paired bootstrap
    python -B -m exps.rigel_gg01.dev gate       # registered DEV gate -> DEV_VERDICT.json
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_gg01 import gg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/hope_cad/rigel_gg01/dev'
CACHE = Path('/mnt/k/NLCAD/cache/rigel_gg01')
AD2_ROOT = Path('/mnt/k/mvtec_ad_2')
TOK = Path('/mnt/k/NLCAD/cache/rigel_cad')
LOCKED_AD2 = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/ad2/grids/E')
PREREG = ROOT / 'exps/rigel_gg01/PREREG.md'
SEED = 2711
ARMS = ['V1', 'GG01', 'F', 'U', 'CCW', 'CCE']
COVER_N = 4096
MIN_FREE_GIB = 30


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'dev.log', 'a') as f:
        f.write(s + '\n')


def wj(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1, default=float))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def guard():
    for m in ('/mnt/d', '/mnt/e', '/mnt/k'):
        s = os.statvfs(m)
        if s.f_bavail * s.f_frsize / 2 ** 30 < MIN_FREE_GIB:
            raise SystemExit(f'STOP: {m} below {MIN_FREE_GIB} GiB')


def scenarios():
    from rigel_cad import data as DATA
    return list(DATA.ad2_scenarios(AD2_ROOT))


def test_rows(s):
    from rigel_cad import data as DATA
    return DATA.ad2_files(AD2_ROOT, s, 'test_public')


def mask_map(s):
    """{image stem: mask path} for one scenario's test_public ground truth (one directory walk)."""
    gt = AD2_ROOT / s / 'test_public' / 'ground_truth'
    out = {}
    for q in gt.rglob('*'):
        if q.is_file():
            assert q.parent.name == 'bad', q
            st = q.stem[:-5] if q.stem.endswith('_mask') else q.stem
            out[st] = q
    return out


def tokens(key, rows):
    from rigel_cad.pipeline import TokenStore
    return TokenStore('dinov2_vitb14_reg4', 'cuda', TOK).get(key, AD2_ROOT, rows)


def fit_cached(s, x16):
    """Canonical per-task fit (C_t, S_t, n_t) cached per (scenario, seed); order-independent."""
    p = CACHE / 'fit' / 'ad2' / f'{s}_s{SEED}.pt'
    if p.exists():
        d = torch.load(p, weights_only=True)
        return gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16']), 0.0
    t0 = time.time()
    f = gg.fit_task(x16, SEED)
    torch.cuda.synchronize()
    dt = time.time() - t0
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'kc': f.kc, 'S': f.S, 'n': f.n, 'n_tokens': f.n_tokens, 'keys16': f.keys16, 'fit_s': dt}, p)
    return f, dt


@torch.no_grad()
def coverage(stream, arm, l, samples, keys=None):
    """Mean normal-token energy (0.5 min_m d_W_l^2, or Euclidean for fast-only arms) of each seen task's held-aside
    sample of training tokens under the arm's state after task l (keys override: pre-consolidation candidates)."""
    K, W = stream.state(arm, l)
    if keys is not None:
        K = keys.cpu()
    kk = K.cuda().float()
    out = []
    for t in range(l + 1):
        q = samples[t]
        e = gg.ops.energy_euclidean(q, kk) if W is None else gg.ops.energy_whitened(q, kk, W.cuda())
        out.append(float(e.mean()))
    return out


@torch.no_grad()
def stream_stage():
    from rigel_cad import data as DATA
    assert PREREG.exists(), 'STOP: pre-registration missing'
    guard()
    t_all = time.time()
    scens = scenarios()
    T = len(scens)
    st = gg.Stream(ARMS, SEED)
    samples, diag = [], {'coverage_after': {a: [] for a in ARMS}, 'coverage_before': {a: [] for a in ARMS},
                         'peak_vram_write': {a: [] for a in ARMS}}
    for l, s in enumerate(scens):
        tr = DATA.ad2_files(AD2_ROOT, s, 'train')
        x16 = tokens(f'ad2/{s}_train', tr).reshape(-1, gg.D).cuda()
        g = torch.Generator().manual_seed(20261126 + l)
        samples.append(x16[torch.randperm(len(x16), generator=g)[:COVER_N].to(x16.device)].cpu())
        f, fs = fit_cached(s, x16)
        cands = {}
        orig_write = gg.ArmMemory.write

        def write(self, l_, C, x, W_prev, _o=orig_write):
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            new = C
            if self.rule == 'cc' and len(self.keys) > 0:
                new = None                                   # recorded after the call (expansion inside write)
            pre = torch.cat([self.keys, C]) if new is not None else None
            r = _o(self, l_, C, x, W_prev)
            diag['peak_vram_write'][self.name].append(int(torch.cuda.max_memory_allocated() - base))
            cands[self.name] = pre
            return r
        gg.ArmMemory.write = write
        try:
            st.task(l, x16, f, fs)
        finally:
            gg.ArmMemory.write = orig_write
        for a in ARMS:
            diag['coverage_after'][a].append(coverage(st, a, l, samples))
            diag['coverage_before'][a].append(coverage(st, a, l, samples, cands[a]) if cands[a] is not None else None)
        log('TASK', l, s, 'k', len(f.kc), {a: st.arms[a].log[-1]['n_keys'] for a in ARMS},
            {a: st.arms[a].log[-1]['metric'] for a in ARMS}, 'cond %.1f' % st.cond[-1])
        del x16
        torch.cuda.empty_cache()
        guard()
    t_learn = time.time() - t_all
    lat = {a: [] for a in ARMS}
    parity = {}
    for j, s in enumerate(scens):
        rows = test_rows(s)
        q = tokens(f'ad2/{s}_test_public', rows).reshape(-1, gg.D)
        ref = torch.load(LOCKED_AD2 / f'{s}__test_public.pt', weights_only=False)
        for a in ARMS:
            gl = []
            for l in range(j, T):
                torch.cuda.synchronize()
                t1 = time.time()
                gl.append(st.energy(a, l, q))
                torch.cuda.synchronize()
                lat[a].append((time.time() - t1) / len(rows))
            grid = torch.stack(gl).reshape(T - j, len(rows), 28, 28).contiguous()
            p = CACHE / 'dev' / a / f'{s}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rows, 'grid': grid, 'stages': list(range(j, T)), 'task': s, 'pos': j}, p)
            if a == 'V1':
                rg = ref['grid']
                parity[s] = {'rows_equal': list(ref['rows']) == rows,
                             'final_bitwise': bool(torch.equal(grid[-1], rg[-1] if rg.dim() == 4 else rg)),
                             'all_stages_bitwise': bool(rg.dim() == 4 and rg.shape == grid.shape and torch.equal(grid, rg))}
        del q
        log('GRIDS', s, parity[s])
    m = {a: st.arms[a] for a in ARMS}
    states = {}
    for a in ARMS:
        K, W = st.state(a, T - 1)
        p = CACHE / 'dev' / 'states' / f'{a}.pt'
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'keys': K, 'S': st.S if m[a].slow else None, 'n': st.n if m[a].slow else None}, p)
        states[a] = {'n_keys': len(K), 'nominal_bytes': m[a].bytes(), 'physical': gg.physical_bytes(len(K), m[a].slow),
                     'serialized_bytes': p.stat().st_size, 'inference_W_fp32': (gg.D * gg.D * 4) if m[a].slow else 0,
                     'inference_whitened_keys_fp32': len(K) * gg.D * 4 if m[a].slow else 0}
    rec = {'scenarios': scens, 'seed': SEED, 'arms': ARMS, 'k_per_task': st.k_per_task, 'cond': st.cond,
           'drift': st.drift(), 'select_log': {a: m[a].log for a in ARMS},
           'update_s': {a: st.fit_s + m[a].select_s for a in ARMS}, 'fit_s': st.fit_s,
           'select_s': {a: m[a].select_s for a in ARMS},
           'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
           'states': states, 'diag': diag, 'v1_parity_locked_ad2': parity,
           'learn_s': t_learn, 'seconds': time.time() - t_all, 'peak_vram': torch.cuda.max_memory_allocated(),
           'prereg_sha256': sha(PREREG)}
    wj(OUT / 'STREAM.json', rec)
    log('STREAM DONE', '%.0fs' % rec['seconds'])


# ---------------------------------------------------------------------------------------------- CPU metrics
def _metric_task(args):
    s, j, arms = args
    from rigel_cad import metrics as M
    rows = test_rows(s)
    mm = mask_map(s)

    class Cat:
        pass
    cat = Cat()
    cat.rows = rows
    cat.hws = [M.native_hw(AD2_ROOT / r) for r in rows]
    # only images under test_public/bad/ have a mask: good and bad images share file names (000_regular.png ...),
    # so a stem-only lookup (the pre-existing ad2_mask_of) would give good images a defect mask
    cat.masks = [M.native_mask(mm[Path(r).stem] if Path(r).parts[2] == 'bad' else None, hw)
                 for r, hw in zip(rows, cat.hws)]
    assert all((Path(r).parts[2] == 'bad') == bool(m.any()) for r, m in zip(rows, cat.masks)), 'mask/label mismatch'
    cat.y = np.array([int(m.any()) for m in cat.masks])
    bw = M.boot_weights(len(rows), j)
    res, boots = {}, {}
    t0 = time.time()
    for a in arms:
        d = torch.load(CACHE / 'dev' / a / f'{s}.pt', weights_only=True)
        assert d['rows'] == rows
        r, b = M.evaluate_category(cat, d['grid'], d['stages'], bw)
        res[a], boots[a] = r, b
    return s, res, boots, {'n': len(rows), 'n_bad': int(cat.y.sum()), 'seconds': time.time() - t0}


def metrics_stage():
    rec = json.loads((OUT / 'STREAM.json').read_text())
    assert rec['prereg_sha256'] == sha(PREREG), 'STOP: pre-registration changed after the stream stage'
    scens = rec['scenarios']
    res, boots, info = {}, {}, {}
    with mp.get_context('spawn').Pool(2) as pool:
        for s, r, b, inf in pool.imap_unordered(_metric_task, [(s, j, ARMS) for j, s in enumerate(scens)]):
            res[s], boots[s], info[s] = r, b, inf
            log('METRIC', s, {a: round(r[a][len(scens) - 1]['ap'], 4) for a in ARMS}, '%.0fs' % inf['seconds'])
    bp = CACHE / 'dev' / 'BOOT_dev.npz'
    np.savez_compressed(bp, **{f'{s}||{a}||{m}': v[m] for s, ba in boots.items() for a, v in ba.items() for m in v})
    wj(OUT / 'METRICS.json', {'cells': res, 'info': info, 'boot_file': str(bp), 'boot_sha256': sha(bp)})
    log('METRICS DONE')


# ---------------------------------------------------------------------------------------------- registered gate
def summary(order, cells, arm):
    from rigel_cad import metrics as M
    per = {t: {int(l): v for l, v in cells[t][arm].items()} for t in order}
    return M.stream_summary(order, per)


def gate_stage():
    st = json.loads((OUT / 'STREAM.json').read_text())
    me = json.loads((OUT / 'METRICS.json').read_text())
    assert st['prereg_sha256'] == sha(PREREG)
    order = st['scenarios']
    T = len(order)
    bf = np.load(me['boot_file'])
    S = {a: summary(order, me['cells'], a) for a in ARMS}
    boot = {a: np.mean([bf[f'{s}||{a}||ap'] for s in order], axis=0) for a in ARMS}

    def paired(x, y):
        d = boot[x] - boot[y]
        return {'delta': S[x]['pixel_ap'] - S[y]['pixel_ap'], 'ci95': [float(v) for v in np.nanpercentile(d, [2.5, 97.5])]}
    cmp_ = {k: paired(*k.split('-')) for k in ('GG01-V1', 'CCW-V1', 'CCE-F', 'GG01-CCW', 'V1-F', 'U-V1')}
    per_cat = {s: me['cells'][s]['GG01'][str(T - 1)]['ap'] - me['cells'][s]['V1'][str(T - 1)]['ap'] for s in order}
    d = cmp_['GG01-V1']
    numerics_ok = all(np.isfinite(st['cond'])) and all(c > 0 for c in st['cond'])
    parity_ok = all(v['rows_equal'] and v['final_bitwise'] for v in st['v1_parity_locked_ad2'].values())
    g = {
        'G1_delta_ge_0.005': d['delta'] >= 0.005,
        'G2_ci_lower_gt_0': d['ci95'][0] > 0,
        'G3_no_category_loss_gt_0.03': min(per_cat.values()) >= -0.03,
        'G4_forgetting_not_worse_gt_0.005': S['GG01']['forgetting_ap'] - S['V1']['forgetting_ap'] <= 0.005,
        'G5_physical_bytes_le_v1': st['states']['GG01']['physical']['total'] <= st['states']['V1']['physical']['total']
                                   and st['states']['GG01']['nominal_bytes'] <= st['states']['V1']['nominal_bytes'],
        'G6_correctness': bool(numerics_ok and parity_ok),
    }
    others = all(g[k] for k in g if not k.startswith(('G1', 'G2')))
    if not others or not g['G1_delta_ge_0.005']:
        verdict = 'GG01_REJECTED_DEV'
    elif not g['G2_ci_lower_gt_0']:
        verdict = 'GG01_INCONCLUSIVE_DEV'
    else:
        verdict = 'GG01_DEV_PASS'
    out = {'verdict': verdict, 'gates': g, 'comparisons': cmp_, 'per_category_delta_ap': per_cat,
           'summaries': {a: {k: v for k, v in S[a].items() if k != 'per_task_final'} for a in ARMS},
           'per_task_final': {a: S[a]['per_task_final'] for a in ARMS},
           'update_s': st['update_s'], 'latency_ms_per_img_p50': st['latency_ms_per_img_p50'],
           'prereg_sha256': st['prereg_sha256'], 'time': time.strftime('%FT%T')}
    wj(OUT / 'DEV_VERDICT.json', out)
    log('VERDICT', verdict, g)


if __name__ == '__main__':
    {'stream': stream_stage, 'metrics': metrics_stage, 'gate': gate_stage}[sys.argv[1]]()
