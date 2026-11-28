"""RIGEL-ENG-01: bitwise check of the rigel_cad 1.1.1 engineering changes on the real streams (no new method).

    python -B -m exps.rigel_eng_01.check <ds>        # GPU: mvtec | visa | mpdd (order a, seed 2711)
    python -B -m exps.rigel_eng_01.check summary

Per dataset (canonical K: token caches, read-only; nothing existing is overwritten):
  * every task's package fit (RigelCAD.compute_fit, CUDA-graph k-centre) == the cached v1 fit (coreset keys, S_t, n_t)
    that produced the locked results; fit time new vs the plain-loop k-centre (greedy_kcenter_reference) on the same
    tokens;
  * the fits written into fp16 @ 8 MiB and int4_triangular @ 8 MiB models (write_task): number of keys per stage ==
    RIGEL-Q4T-INT, final state == the saved 1.1 checkpoints (bench E state, Q4T integration state), tensor by tensor;
  * scoring: last category, every stage, grid with the state == grid with RigelCAD.prepare(state) == locked grid
    (v1: NL03-MULTIRATE-01 arm E; Q4T: RIGEL-CC-01);
  * batch-1 head latency of the final state: RigelState path (keys moved and multiplied per call) vs Prepared.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from exps.rigel_q4t_int import verify as V
from rigel_cad import RigelCAD, RigelConfig
from rigel_cad import ops
from rigel_cad.model import TaskFit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/rigel_cad/eng_01'
Q4 = 'int4_triangular'
V1_STATE = str(ROOT / 'results/rigel_cad/bench/states/{ds}_E.rigel')
Q4_STATE = str(ROOT / 'results/rigel_cad/q4t_integration/states/{ds}_a2711_int4tri.rigel')
Q4_VERIFY = str(ROOT / 'results/rigel_cad/q4t_integration/VERIFY_{ds}_a2711.json')


def run(ds):
    from exps.nl03_multirate_01 import final as FN
    V.guard()
    order = FN.bench_order(ds, 'a')
    T = len(order)
    tag = f'{ds}_a2711'
    mv = RigelCAD(RigelConfig(budget_mib=8), keep_history=True)
    mq = RigelCAD(RigelConfig(budget_mib=8, storage=Q4), keep_history=True)
    tasks = []
    for l, t in enumerate(order):
        c = t.split('/')[1]
        x16 = V.train_x16(t).cuda()
        V.sync()
        t0 = time.time()
        fit = mv.compute_fit(x16)
        V.sync()
        t_new = time.time() - t0
        k = ops.coreset_size(len(x16))
        t0 = time.time()
        kc_ref = ops.greedy_kcenter_reference(x16, k, seed=2711)
        V.sync()
        t_ref_kc = time.time() - t0
        ref = V.cached_fit(ds, c)
        rec = {'task': t, 'n_tokens': len(x16), 'k': k,
               'fit_equal_cached': bool(V.eq(fit.keys16, ref.keys16) and V.eq(fit.S, ref.S) and fit.n == ref.n),
               'kcenter_equal_reference': bool(V.eq(fit.keys16, x16[kc_ref.to(x16.device)])),
               'fit_s_new': t_new, 'kcenter_s_reference': t_ref_kc}
        for name, m in (('v1', mv), ('q4t', mq)):
            V.sync()
            t0 = time.time()
            m.write_task(TaskFit(fit.keys16, fit.S, fit.n, fit.n_tokens), t)
            V.sync()
            rec[f'write_s_{name}'] = time.time() - t0
            rec[f'n_keys_{name}'] = len(m.fast)
        tasks.append(rec)
        V.log('TASK', tag, l, c, {k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in rec.items() if k_ != 'task'})
        del x16, fit
        torch.cuda.empty_cache()

    vp = Path(Q4_VERIFY.format(ds=ds))
    vq = json.loads(vp.read_text()) if vp.exists() else None
    keys_per_stage_equal = None
    if vq is not None:
        keys_per_stage_equal = [r['n_keys'] for r in vq['per_task']] == [r['n_keys_q4t'] for r in tasks]

    # final states vs the saved 1.1 checkpoints
    sv = torch.load(V1_STATE.format(ds=ds), weights_only=True)
    sq = torch.load(Q4_STATE.format(ds=ds), weights_only=True)
    q4, scale = mq.fast.packed()
    final = {'v1_keys': V.eq(mv.fast.keys, sv['keys']), 'v1_S': V.eq(mv.slow.S, sv['S']), 'v1_n': mv.slow.n == sv['n'],
             'q4t_codes': V.eq(q4, sq['q4']), 'q4t_scale': V.eq(scale, sq['scale']),
             'q4t_S_tri': V.eq(mq.slow.S_tri, sq['S_tri']), 'q4t_n': mq.slow.n == sq['n']}

    # scoring: last category, every stage
    c = order[-1].split('/')[1]
    q, rows = V.test_tokens(ds, c)
    q = q.reshape(-1, 768)
    score = {}
    for name, m, refp in (('v1', mv, V.LOCK_GRID / tag / 'E' / f'{c}.pt'),
                          ('q4t', mq, V.CC_CACHE / 'grids' / tag / 'Q4T' / f'{c}.pt')):
        rg = torch.load(refp, weights_only=False)
        st = m.state_at(T - 1)
        g_state = m.patch_energy_grid(q, st)
        g_prep = m.patch_energy_grid(q, m.prepare(st))
        g_cur = m.patch_energy_grid(q)
        score[name] = {'state_eq_prepared': V.eq(g_state, g_prep), 'state_eq_current_cached': V.eq(g_state, g_cur),
                       'state_eq_locked': bool(list(rg['rows']) == list(rows) and V.eq(g_state, rg['grid'][-1]))}
        # batch-1 head latency (median over 100 images), state path vs prepared
        p = m.prepare(st)
        lat = {'state': [], 'prepared': []}
        for i in range(100):
            qi = q[(i % len(rows)) * 784:(i % len(rows) + 1) * 784].cuda()
            for mode, s_ in (('state', st), ('prepared', p)):
                V.sync()
                t0 = time.time()
                m.patch_energy_grid(qi, s_)
                V.sync()
                lat[mode].append(time.time() - t0)
        score[name]['head_ms_p50'] = {k_: float(torch.tensor(v[10:]).median() * 1e3) for k_, v in lat.items()}
    out = {'dataset': ds, 'tag': tag, 'per_task': tasks,
           'fit_equal_cached_all': all(r['fit_equal_cached'] for r in tasks),
           'kcenter_equal_reference_all': all(r['kcenter_equal_reference'] for r in tasks),
           'fit_s_new_total': sum(r['fit_s_new'] for r in tasks),
           'kcenter_s_reference_total': sum(r['kcenter_s_reference'] for r in tasks),
           'q4t_keys_per_stage_equal_integration': keys_per_stage_equal, 'final_state_equal_saved': final,
           'final_state_equal_saved_all': all(final.values()), 'scoring_last_category': {'category': c, **score},
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    V.wj(OUT / f'CHECK_{tag}.json', out)
    V.log('DONE', tag, 'fit', out['fit_equal_cached_all'], 'kc', out['kcenter_equal_reference_all'], 'final',
          out['final_state_equal_saved_all'], 'scoring', score, 'fit new %.0fs, kcenter ref %.0fs' % (
              out['fit_s_new_total'], out['kcenter_s_reference_total']))


def summary():
    rows = {}
    for ds in ('mvtec', 'visa', 'mpdd'):
        p = OUT / f'CHECK_{ds}_a2711.json'
        if p.exists():
            d = json.loads(p.read_text())
            sc = d['scoring_last_category']
            rows[ds] = {'fit_equal_cached_all': d['fit_equal_cached_all'],
                        'kcenter_equal_reference_all': d['kcenter_equal_reference_all'],
                        'q4t_keys_per_stage_equal_integration': d['q4t_keys_per_stage_equal_integration'],
                        'final_state_equal_saved_all': d['final_state_equal_saved_all'],
                        'scoring_all_equal': all(v for n in ('v1', 'q4t') for k, v in sc[n].items() if k != 'head_ms_p50'),
                        'fit_s_new_total': d['fit_s_new_total'], 'kcenter_s_reference_total': d['kcenter_s_reference_total'],
                        'head_ms_p50': {n: sc[n]['head_ms_p50'] for n in ('v1', 'q4t')}}
    ok = bool(rows) and all(r['fit_equal_cached_all'] and r['kcenter_equal_reference_all'] and r['final_state_equal_saved_all']
                            and r['scoring_all_equal'] and r['q4t_keys_per_stage_equal_integration'] is not False
                            for r in rows.values())
    V.wj(OUT / 'CHECK_SUMMARY.json', {'datasets': rows, 'ALL_BITWISE': ok})
    print(json.dumps({'datasets': rows, 'ALL_BITWISE': ok}, indent=1))


if __name__ == '__main__':
    summary() if sys.argv[1] == 'summary' else run(sys.argv[1])
