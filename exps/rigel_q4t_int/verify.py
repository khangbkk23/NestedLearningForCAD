"""RIGEL-Q4T-INT: verification of the official rigel_cad 'int4_triangular' storage backend against the research Q4T
implementation (exps/rigel_q8/d1.py ArmQ4, run through exps/rigel_cc01/cc.Stream) and of the default fp16 backend
against the locked RIGEL-CAD v1 benchmark. Integration check only: no new method, hyperparameter or result search.

    python -B -m exps.rigel_q4t_int.verify stream <ds>      # GPU: mvtec | visa | mpdd, order a, seed 2711
    python -B -m exps.rigel_q4t_int.verify summary

Per dataset (canonical K: token caches, read-only; nothing is written to existing caches or result files):
  * the package fits each task itself (RigelCAD.fit_task) for both backends; its per-task coreset keys, S_t and n_t are
    compared with the cached v1 fits that the research Q4T run used;
  * the research ArmQ4 is fed the cached fits; after every task the package int4 codes, fp16 scales, task order of the
    survivors (implicitly, via bitwise codes), fp32 triangle, count and whitener are compared bitwise;
  * every stage's grid is compared bitwise with the cached research Q4T grids (RIGEL-CC-01) and, for the fp16 backend,
    with the locked v1 grids (arm E of NL03-MULTIRATE-01);
  * metrics per stage (rigel_cad.metrics) are compared with the locked RIGEL-CC-01 METRICS (Q4T) and the locked v1
    METRICS (E); stream summaries (final AP / AUROC / forgetting / acquisition / prequential) are reported;
  * the final package checkpoint is compared tensor by tensor with the research Q4T persisted state;
  * save -> load -> continue (MVTec: after task 7 of 15, so consolidation binds after restoration; MPDD: after task 3);
  * accounting (logical, tensor payload, file size, resident and peak GPU memory) and update / scoring time.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.rigel_cc01 import cc
from exps.rigel_gg01 import gg
from exps.rigel_q8 import q8
from rigel_cad import RigelCAD, RigelConfig
from rigel_cad import data as DATA
from rigel_cad import metrics as M
from rigel_cad import ops
from rigel_cad import storage as STO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/rigel_cad/q4t_integration'
CC_CACHE = Path('/mnt/k/NLCAD/cache/rigel_cc01')
GG_FIT = Path('/mnt/k/NLCAD/cache/rigel_gg01/fit')
LOCK_GRID = Path('/mnt/k/NLCAD/cache/nl03_multirate_01/bench')
LOCK_MET = ROOT / 'results/hope_cad/nl03_multirate_01/bench'
CC_RES = ROOT / 'results/hope_cad/rigel_cc01'
Q4 = 'int4_triangular'
REF = {'mvtec': (0.6304, 0.9926), 'visa': (0.4749, 0.9707), 'mpdd': (0.5170, 0.9598)}   # owner-locked, 4 d.p.
RESUME_AT = {'mvtec': 7, 'mpdd': 3}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'verify.log', 'a') as f:
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


_G = None


def G_():
    global _G
    if _G is None:
        from exps.nl03_pressure_00 import core as PC
        _G = PC.bind()
    return _G


def train_x16(t):
    from exps.nl03_multirate_01 import final as FN
    from exps.nl03_multirate_01 import run as RN
    G = G_()
    return RN.fit_tokens(G, t, order=FN.train_rows_sorted(G, t))[0]


def test_tokens(ds, c):
    from exps.nl03_pressure_00 import core as PC
    G = G_()
    PC.use(G, ds)
    return G.load_test(ds, c, PC.extract_rec(G, ds))


def cached_fit(ds, c, seed=2711):
    for p in (GG_FIT / ds / f'{c}_s{seed}.pt', CC_CACHE / 'fit' / ds / f'{c}_s{seed}.pt'):
        if p.exists():
            d = torch.load(p, weights_only=True)
            return gg.Fit(kc=d['kc'], S=d['S'], n=d['n'], n_tokens=d['n_tokens'], keys16=d['keys16'])
    raise SystemExit(f'BLOCKED_SOURCE: no cached v1 fit for {ds}/{c}')


class Recorder:
    """Wraps a model's fast / slow write to capture the per-task fit the package computed (diagnostic only)."""

    def __init__(self, model):
        self.keys, self.S, self.n = None, None, None
        fw, sw = model.fast.write, model.slow.write

        def fast_write(k):
            self.keys = k.detach().cpu()
            fw(k)

        def slow_write(S_t, n_t):
            self.S, self.n = S_t.detach().cpu(), int(n_t)
            sw(S_t, n_t)
        model.fast.write, model.slow.write = fast_write, slow_write


def sync():
    torch.cuda.synchronize()


def eq(a, b) -> bool:
    return bool(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu()))


def model_gpu_bytes(m) -> int:
    ts = [m.fast.keys] + ([m.fast.q, m.fast.scale] if m.config.storage == Q4 else [])
    return int(sum(t.numel() * t.element_size() for t in ts if t.is_cuda))


@torch.no_grad()
def stream(ds):
    from exps.nl03_multirate_01 import final as FN
    guard()
    torch.manual_seed(0)
    order = FN.bench_order(ds, 'a')
    T = len(order)
    tag = f'{ds}_a2711'
    mq = RigelCAD(RigelConfig(budget_mib=8, storage=Q4), keep_history=True)
    mv = RigelCAD(RigelConfig(budget_mib=8), keep_history=True)
    rq, rv = Recorder(mq), Recorder(mv)
    ref = cc.Stream(['Q4T'], 2711)
    arm = ref.arms['Q4T']
    per_task, upd = [], {'Q4T': [], 'V1': []}
    resume_at = RESUME_AT.get(ds)
    resumed = None
    x_keep = []
    t0 = time.time()
    for l, t in enumerate(order):
        c = t.split('/')[1]
        x16 = train_x16(t)
        rec = {'task': t}
        for name, m, r in (('Q4T', mq, rq), ('V1', mv, rv)):
            sync()
            t1 = time.time()
            m.fit_task(x16, t)
            sync()
            upd[name].append(time.time() - t1)
            f = cached_fit(ds, c)
            rec[f'{name}_fit_equal_cached'] = bool(eq(r.keys, f.keys16) and eq(r.S, f.S) and r.n == f.n)
        f = cached_fit(ds, c)
        ref.task(l, f)
        rec['state_equal_research'] = {
            'codes': eq(mq.fast.q, arm.q), 'scale': eq(mq.fast.scale, arm.scale),
            'keys_fp32': eq(mq.fast.keys, arm.keys), 'S_tri': eq(mq.slow.S_tri, arm.S_tri), 'n': mq.slow.n == ref.n,
            'W': eq(mq.history[l].W, ref.W_tri['Q4T'][l])}
        rec['n_keys'] = len(mq.fast)
        rec['bytes_logical'] = mq.history[l].bytes
        rec['v1_n_keys'] = len(mv.fast)
        if resumed is not None:
            resumed.fit_task(x16, t)
            rec['resumed_equal'] = bool(eq(resumed.fast.q, mq.fast.q) and eq(resumed.fast.scale, mq.fast.scale)
                                        and eq(resumed.slow.S_tri, mq.slow.S_tri) and resumed.slow.n == mq.slow.n)
        if resume_at is not None and l == resume_at - 1:
            p = mq.save(OUT / 'states' / f'{tag}_after{resume_at}_int4tri.rigel')
            resumed = RigelCAD.load(p)
            rec['saved_for_resume'] = str(p.relative_to(ROOT))
        per_task.append(rec)
        log('TASK', tag, l, c, 'keys', len(mq.fast), {k: v for k, v in rec.items() if k not in ('task',)})
        del x16
        torch.cuda.empty_cache()
    t_learn = time.time() - t0

    # ------------------------------------------------------------------ scoring, grids, metrics
    data_root = G_().data_root(ds)
    lat = {'Q4T': [], 'V1': []}
    grid_eq = {'Q4T': {}, 'V1': {}}
    cells = {'Q4T': {}, 'V1': {}}
    peak = {}
    for j, t in enumerate(order):
        c = t.split('/')[1]
        q, rows = test_tokens(ds, c)
        q = q.reshape(-1, 768)
        _, test = DATA.category_split(ds, data_root, c)
        cat = M.Category(data_root, test)
        assert cat.rows == list(rows), (t, 'row order')
        for name, m, refp in (('Q4T', mq, CC_CACHE / 'grids' / tag / 'Q4T' / f'{c}.pt'),
                              ('V1', mv, LOCK_GRID / tag / 'E' / f'{c}.pt')):
            gl = []
            if j == T - 1:
                sync()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            for l in range(j, T):
                st = m.state_at(l)
                sync()
                t1 = time.time()
                gl.append(m.patch_energy_grid(q, st))
                sync()
                lat[name].append((time.time() - t1) / len(rows))
            if j == T - 1:
                peak[name] = {'peak_scoring_alloc_above_base': int(torch.cuda.max_memory_allocated() - base),
                              'base_allocated': int(base)}
            grid = torch.stack(gl)
            rg = torch.load(refp, weights_only=False)
            grid_eq[name][c] = bool(list(rg['rows']) == list(rows) and rg['stages'] == list(range(j, T))
                                    and torch.equal(rg['grid'], grid))
            cells[name][t], _ = M.evaluate_category(cat, grid, list(range(j, T)), None)
        del q
        torch.cuda.empty_cache()
        log('GRIDS', tag, c, {n: grid_eq[n][c] for n in grid_eq})

    lq = json.loads((CC_RES / f'METRICS_{tag}.json').read_text())['cells']
    lv = json.loads((LOCK_MET / f'METRICS_{tag}.json').read_text())['cells']
    met_eq = {'Q4T': all(cells['Q4T'][t][l] == lq[t]['Q4T'][str(l)] for t in order for l in cells['Q4T'][t]),
              'V1': all(cells['V1'][t][l] == lv[t]['E'][str(l)] for t in order for l in cells['V1'][t])}
    summ = {n: M.stream_summary(order, cells[n]) for n in cells}

    # ------------------------------------------------------------------ checkpoint vs research persisted state
    pq = mq.save(OUT / 'states' / f'{tag}_int4tri.rigel')
    d = torch.load(pq, weights_only=True)
    rs = torch.load(CC_CACHE / 'states' / tag / 'Q4T.pt', weights_only=True)
    ckpt_eq = {k: eq(d[k], rs[k]) for k in ('q4', 'scale', 'S_tri')}
    ckpt_eq['n'] = int(d['n']) == int(rs['n'])
    payload = sum(d[k].numel() * d[k].element_size() for k in ('q4', 'scale', 'S_tri')) + 8
    pv = OUT / 'states' / f'{tag}_fp16_scratch.rigel'
    mv.save(pv)
    dv = torch.load(pv, weights_only=True)
    payload_v = dv['keys'].numel() * 2 + dv['S'].numel() * 8 + 8          # v1 persists S in fp64 (limitation L1)
    loaded = RigelCAD.load(pq)
    qq, _ = test_tokens(ds, order[-1].split('/')[1])
    qq = qq.reshape(-1, 768)[:784 * 8]
    load_eq = bool(torch.equal(loaded.patch_energy(qq), mq.patch_energy(qq)))

    acct = {
        'Q4T': {'n_keys': len(mq.fast), 'L_logical': mq.persistent_bytes(), 'L_logical_max': max(r['bytes_logical'] for r in per_task),
                'P_tensor_payload': payload, 'F_file': pq.stat().st_size,
                'R_model_resident_gpu': model_gpu_bytes(mq),
                'R_scoring_resident_keys_W_mW': len(mq.fast) * 768 * 4 * 2 + 768 * 768 * 4, **peak['Q4T']},
        'V1': {'n_keys': len(mv.fast), 'L_logical': mv.persistent_bytes(), 'P_tensor_payload': payload_v,
               'F_file': pv.stat().st_size, 'R_model_resident_gpu': model_gpu_bytes(mv),
               'R_scoring_resident_keys_W_mW': len(mv.fast) * 768 * 4 * 2 + 768 * 768 * 4, **peak['V1']}}
    pv.unlink()                                                              # scratch copy only (v1 is not the subject)

    out = {
        'tag': tag, 'dataset': ds, 'order': order,
        'package_sha256': {f: sha(ROOT / 'rigel_cad' / f) for f in ('config.py', 'model.py', 'storage.py', 'ops.py')},
        'research_sha256': {'exps/rigel_q8/d1.py': sha(ROOT / 'exps/rigel_q8/d1.py'),
                            'exps/rigel_q8/q8.py': sha(ROOT / 'exps/rigel_q8/q8.py'),
                            'exps/rigel_cc01/cc.py': sha(ROOT / 'exps/rigel_cc01/cc.py')},
        'per_task': per_task,
        'fit_equal_cached_all': all(r['Q4T_fit_equal_cached'] and r['V1_fit_equal_cached'] for r in per_task),
        'state_equal_research_all': all(all(r['state_equal_research'].values()) for r in per_task),
        'resume': ({'after_task': resume_at, 'equal_all': all(r.get('resumed_equal', True) for r in per_task),
                    'n_compared': sum('resumed_equal' in r for r in per_task)} if resume_at else None),
        'grid_equal': {n: {'all': all(v.values()), 'n': len(v)} for n, v in grid_eq.items()},
        'metric_cells_equal_locked': met_eq,
        'summary': summ,
        'reference_owner_4dp': {'pixel_ap': REF[ds][0], 'image_auroc': REF[ds][1],
                                'match': (round(summ['Q4T']['pixel_ap'], 4), round(summ['Q4T']['image_auroc'], 4)) == REF[ds]},
        'checkpoint_equal_research': ckpt_eq, 'load_energy_equal': load_eq,
        'accounting': acct,
        'update_seconds_total': {n: float(np.sum(v)) for n, v in upd.items()},
        'update_seconds_per_task_median': {n: float(np.median(v)) for n, v in upd.items()},
        'latency_ms_per_img_p50': {n: float(np.median(v) * 1e3) for n, v in lat.items()},
        'learn_s': t_learn, 'seconds': time.time() - t0, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0)}
    wj(OUT / f'VERIFY_{tag}.json', out)
    log('DONE', tag, 'fit', out['fit_equal_cached_all'], 'state', out['state_equal_research_all'], 'grids',
        out['grid_equal'], 'metrics', met_eq, 'ckpt', ckpt_eq, 'resume', out['resume'],
        'Q4T AP %.4f AUROC %.4f' % (summ['Q4T']['pixel_ap'], summ['Q4T']['image_auroc']))


def summary():
    rows = {}
    for ds in ('mvtec', 'visa', 'mpdd'):
        p = OUT / f'VERIFY_{ds}_a2711.json'
        if p.exists():
            v = json.loads(p.read_text())
            rows[ds] = {'pass': bool(v['fit_equal_cached_all'] and v['state_equal_research_all']
                                     and all(g['all'] for g in v['grid_equal'].values())
                                     and all(v['metric_cells_equal_locked'].values())
                                     and all(v['checkpoint_equal_research'].values()) and v['load_energy_equal']
                                     and (v['resume'] is None or v['resume']['equal_all'])
                                     and v['reference_owner_4dp']['match']),
                        'Q4T': {k: v['summary']['Q4T'][k] for k in ('pixel_ap', 'image_auroc', 'forgetting_ap', 'acquisition_ap', 'prequential_ap')
                                if k in v['summary']['Q4T']},
                        'V1': {k: v['summary']['V1'][k] for k in ('pixel_ap', 'image_auroc', 'forgetting_ap', 'acquisition_ap', 'prequential_ap')
                               if k in v['summary']['V1']}}
    wj(OUT / 'VERIFY_SUMMARY.json', rows)
    print(json.dumps(rows, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'stream':
        stream(sys.argv[2])
    else:
        summary()
