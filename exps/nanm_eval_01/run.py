"""NANM-EVAL-01 driver (registration: agents/tasks/task_nanm_eval_01_registration.md; hash-locked by `lock`).

    source docs/recovery/recovery_env.sh; export CUDA_VISIBLE_DEVICES=0
    python -B -m exps.nanm_eval_01.audit                 # SOURCE_AUDIT.json (done before the lock)
    python -B -m exps.nanm_eval_01.run lock              # PREREG_LOCK.json (create-exclusive)
    python -B -m exps.nanm_eval_01.run states            # extended per-task states (GPU) -> K:
    python -B -m exps.nanm_eval_01.run score  <tier>     # grids (GPU) -> K:; Stage-A grid parity in order-a cells
    python -B -m exps.nanm_eval_01.run metrics <tier>    # oracle metrics (CPU pool) -> E: METRICS_<ds>_<tier>.json
    python -B -m exps.nanm_eval_01.run latency           # retrieval latency (GPU, normal FIT queries only)
    python -B -m exps.nanm_eval_01.run summary           # all registered JSON outputs + VERDICT.json

Only MPDD and VisA (verified G3 caches) are executable; MVTec AD / Real-IAD stop at the owner GO gate.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nanm_eval_01 import arms as AR
from exps.nanm_eval_01 import common as U
from exps.nanm_eval_01 import metrics as M

CODE = ['exps/nanm_eval_01/common.py', 'exps/nanm_eval_01/audit.py', 'exps/nanm_eval_01/arms.py',
        'exps/nanm_eval_01/metrics.py', 'exps/nanm_eval_01/run.py', 'exps/nanm_eval_01/summary.py',
        'exps/nanm_eval_01/test_eval.py']
BUDGET = {'gpu_hours': 4.0, 'cpu_wall_hours': 14.0, 'k_new_bytes': 40 * U.GiB, 'rss_bytes': int(9.5 * U.GiB),
          'd_free_min': 30 * U.GiB}
PRIMARY = ('N0', 'A1', 'A2', 'B1', 'N1', 'N2', 'N3', 'N4')
POOL = 3


# ------------------------------------------------------------------ registered plan
def sizes_info(G, ds):
    rec = G.nrec()[ds]
    order = list(G.order(ds))
    n_fit = rec['sizes']['n_fit']
    return order, {'n_fit': dict(zip(order, n_fit)), 'k1': dict(zip(order, rec['sizes']['g3'])),
                   'a0b': dict(zip(order, rec['sizes']['a0b'])), 'n_fit_list': n_fit}


def plan(G, ds):
    """Registered cells and arms. Each arm: name -> (kind, sizes{task: k}, exact_final, boot). Pure from metadata."""
    base, si = sizes_info(G, ds)
    od = U.orders(base)
    k1, a0b = si['k1'], si['a0b']
    nf = si['n_fit_list']
    as_map = lambda lst: dict(zip(base, lst))
    cells = {}
    for s in U.SEEDS:
        for o in ('a', 'b', 'c'):
            arms = {a: ('A1' if a == 'A1' else a, a0b if a == 'A1' else k1, o == 'a', o == 'a' and s == 2711) for a in PRIMARY}
            if o == 'a':
                arms['B5'] = ('B5', k1, False, s == 2711)
                arms['B6'] = ('B6', k1, False, s == 2711)
            cells[f'{o}|{s}'] = {'order': od[o], 'seed': s, 'tier': 1, 'arms': arms}
    t2 = {}
    for a in ('B3', 'B4a', 'B4b', 'N5P2', 'N5P4'):
        t2[a] = (a, k1, False, True)
    infeasible = {}
    for r in U.RATIOS:
        if r == 0.01:
            continue
        tag = f'@r{r * 100:g}'
        t2['G3r' + tag] = ('G3r', as_map(U.ratio_sizes(nf, r)), False, True)
        t2['A0r' + tag] = ('A0r', as_map(U.a0b_like_sizes(nf, r)[1]), False, True)
        t2['G1r' + tag] = ('G1r', as_map(U.ratio_sizes(nf, r)), False, True)
    for cap in U.CAPS_MIB:
        tag = f'@cap{cap}'
        for name, kind, extra in (('G3r', 'G3r', U.S_B), ('N1r', 'N1r', U.S_B), ('N4r', 'N4r', U.S_B),
                                  ('N5P2r', 'N5P2r', U.S_B), ('A0cap', 'A0cap', 0), ('G1r', 'G1r', U.G1_B)):
            r, sz = U.cap_ratio(nf, cap * U.MiB, extra)
            if r is None:
                infeasible[name + tag] = 'INFEASIBLE: cap exceeds all FIT tokens or below one key per task'
                continue
            t2[name + tag] = (kind, as_map(sz), False, True)
    eqb = k1[od['a'][0]] * U.KEY_B + U.S_B                          # N2/N3 final persistent bytes (order a)
    r, sz = U.cap_ratio(nf, eqb, 0)
    if r is not None:
        t2['EQB'] = ('EQB', as_map(sz), False, True)
    cells['a|2711']['arms_t2'] = t2
    return {'cells': cells, 'infeasible': infeasible, 'sizes': si, 'orders': {k: list(v) for k, v in od.items()},
            'eqb_bytes': eqb}


def arm_bytes(kind, sizes, n_tasks):
    nk = sum(sizes.values())
    if kind in ('A1', 'A0r', 'A0cap', 'EQB', 'B1'):
        return U.bytes_of('euclid', nk)
    if kind in ('A2', 'G1r'):
        return U.bytes_of('g1', nk)
    if kind == 'B5':
        return U.bytes_of('pertask', nk, n_tasks)                   # one acquisition-time W (or S prefix) per task
    return U.bytes_of('g3', nk)


def final_bytes(kind, sizes, order):
    if kind in ('N2', 'N3'):
        return U.bytes_of('g3', sizes[order[0]])
    return arm_bytes(kind, sizes, len(order))


# ------------------------------------------------------------------ lock
def lock():
    p = U.OUT / 'PREREG_LOCK.json'
    assert not p.exists(), 'lock exists'
    au = U.rj(U.OUT / 'SOURCE_AUDIT.json')
    assert au['pass'], 'SOURCE_AUDIT failed'
    G = U.bind()
    plans = {ds: plan(G, ds) for ds in U.EXEC_DATASETS}
    rec = {'registration': {str(U.REGFILE.relative_to(U.ROOT)): U.sha(U.REGFILE)},
           'task': {str(U.TASKFILE.relative_to(U.ROOT)): U.sha(U.TASKFILE)},
           'code': {f: U.sha(U.ROOT / f) for f in CODE},
           'canonical_sources': au['canonical_sources'], 'source_audit_sha256': U.sha(U.OUT / 'SOURCE_AUDIT.json'),
           'repro_artifacts': {n: U.sha(U.REPRO / n) for n in ('bench_g3/NORMAL.json', 'bench_g3/SCORES_COMPLETE.json',
                                                               'bench_g3/EXTRACT_COMPLETE.json', 'bench_g3/LOCK.json',
                                                               'vbench_00/LOCK.json', 'bench_g3/CELLS_mpdd.json',
                                                               'bench_g3/CELLS_visa.json', 'bench_g3/RESULT_mpdd.json',
                                                               'bench_g3/RESULT_visa.json')},
           'env': au['env'], 'plan': {ds: {k: v for k, v in pl.items()} for ds, pl in plans.items()},
           'budget': BUDGET, 'storage': U.storage(), 'time': time.strftime('%FT%T'),
           'git_head': au['git']['head'],
           'exposure': 'all MPDD/VisA metrics are EXPOSED-DEV; stored G3/A0b/G1b grids are pre-existing exposed artifacts'}
    U.wj(p, rec, exclusive=True)
    U.log('LOCK', U.sha(p)[:16])


def verify_lock():
    lk = U.rj(U.OUT / 'PREREG_LOCK.json')
    bad = [f for f, h in {**lk['registration'], **lk['task'], **lk['code'], **lk['canonical_sources']}.items()
           if h is not None and U.sha(U.ROOT / f) != h]
    bad += [n for n, h in lk['repro_artifacts'].items() if U.sha(U.REPRO / n) != h]
    assert not bad, f'LOCK_FAIL: {bad}'
    return lk


def guard_storage(stage):
    s = U.storage()
    assert s['D_free'] >= BUDGET['d_free_min'], f'D: free below 30 GiB at {stage}'
    with open(U.OUT / 'logs/storage.jsonl', 'a') as f:
        f.write(json.dumps({'stage': stage, **s}) + '\n')
    return s


# ------------------------------------------------------------------ states (GPU)
def states():
    lk = verify_lock()
    G = U.bind()
    torch.backends.cuda.matmul.allow_tf32 = False
    guard_storage('states_start')
    rec = {}
    for ds in U.EXEC_DATASETS:
        pl = plan(G, ds)
        base_order, si = sizes_info(G, ds)
        nrec = G.nrec()
        for s in U.SEEDS:
            arms_all = {**pl['cells'][f'a|{s}']['arms'], **pl['cells'][f'a|{s}'].get('arms_t2', {})}
            for c in base_order:
                p = U.CACHE / 'states' / f'{ds}_s{s}_{c}.pt'
                if p.exists():
                    continue
                base = G.load_norm(ds, s, c, nrec)
                need_k = max(sz[c] for (_, sz, _, _) in arms_all.values())
                s_ks = [sz[c] for (kind, sz, _, _) in arms_all.values()
                        if kind in ('G3r', 'N1r', 'N4r', 'N5P2r') and sz[c] != si['k1'][c]]
                ext = AR.build_ext(G, ds, s, c, base, need_k, s_ks, rand_k=si['k1'][c])
                # determinism: recompute the canonical 1 % S and compare with the stored S1 (seed 2711, first task)
                if s == 2711 and c == base_order[0]:
                    from exps.nanm_r2b import run as B
                    fit = G.fit_tokens(ds, c)
                    S1, n1 = B.residual_stats(fit.float(), base['kc'][:si['k1'][c]].cuda())
                    ext['check_S1_bitwise'] = bool(torch.equal(S1.cpu(), base['S1'])) and int(n1) == base['n1']
                    del fit
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(ext, p)
                rec[p.name] = {'sha256': U.sha(p), 'bytes': p.stat().st_size, 'seconds': ext['seconds'],
                               'kc_len': len(ext['kc']), 's_ks': sorted(ext['S']), 'time_kcentre_s': ext['time_kcentre_s'],
                               'time_S_s': ext['time_S_s'], 'check_S1_bitwise': ext.get('check_S1_bitwise')}
                U.log('STATE', ds, s, c, need_k, len(s_ks), '%.1fs' % ext['seconds'])
    prev = U.rj(U.OUT / 'STATES.json') if (U.OUT / 'STATES.json').exists() else {}
    U.wj(U.OUT / 'STATES.json', {**prev, **rec})
    guard_storage('states_end')


def load_banks(G, ds, s):
    base_order, _ = sizes_info(G, ds)
    nrec = G.nrec()
    banks = {}
    for c in base_order:
        base = G.load_norm(ds, s, c, nrec)
        p = U.CACHE / 'states' / f'{ds}_s{s}_{c}.pt'
        ext = torch.load(p, weights_only=True) if p.exists() else None
        banks[c] = AR.TaskBank(base, ext)
    return banks


# ------------------------------------------------------------------ score (GPU)
def grid_path(ds, cell, arm, c):
    return U.CACHE / 'grids' / ds / cell.replace('|', '_') / arm / f'{c}.pt'


def score(tier):
    verify_lock()
    tier = int(tier)
    G = U.bind()
    torch.backends.cuda.matmul.allow_tf32 = False
    guard_storage(f'score{tier}_start')
    ex = U.rj(G.OUT / 'EXTRACT_COMPLETE.json')
    meta_all = U.rj(U.OUT / f'SCORE_META_t{tier}.json') if (U.OUT / f'SCORE_META_t{tier}.json').exists() else {}
    t_all = time.time()
    for ds in U.EXEC_DATASETS:
        pl = plan(G, ds)
        base_order, si = sizes_info(G, ds)
        Q = {}
        for c in base_order:
            q, rows = G.load_test(ds, c, ex)
            Q[c] = (q, rows)
        stored = {}
        for cell, cd in pl['cells'].items():
            key = f'{ds}|{cell}'
            if key in meta_all:
                continue
            arms_def = cd['arms'] if tier == 1 else cd.get('arms_t2')
            if not arms_def:
                continue
            banks = load_banks(G, ds, cd['seed'])
            cfg = {'k1': si['k1']}
            arms = {a: (kind, sz, cfg) for a, (kind, sz, _, _) in arms_def.items()}
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            grids, meta = AR.score_cell(G, cd['order'], banks, Q, arms)
            parity = {}
            if cell.startswith('a|') and tier == 1:                # Stage-A: bitwise vs stored REPRO grids
                for c in base_order:
                    if c not in stored:
                        stored[c] = torch.load(G.SCORE / f'{ds}_{c}.pt', weights_only=True)['grids']
                    for a, hk in (('N0', 'G3'), ('A1', 'A0b'), ('A2', 'G1b')):
                        ref = stored[c][f'{hk}|s{cd["seed"]}']
                        parity[f'{a}|{c}'] = {'equal': bool(torch.equal(grids[a][c], ref)),
                                              'max_abs': float((grids[a][c] - ref).abs().max())}
            for a, gd in grids.items():
                for j, c in enumerate(cd['order']):
                    p = grid_path(ds, cell, a, c)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({'rows': Q[c][1], 'grid': gd[c].contiguous(), 'order': list(cd['order']), 'pos': j}, p)
            meta_all[key] = {'seconds': time.time() - t0, 'peak_vram': torch.cuda.max_memory_allocated(),
                             'arms': {a: {'seconds': m['seconds'], 'cond': m['cond']} for a, m in meta.items()},
                             'parity': parity, 'parity_all_equal': all(v['equal'] for v in parity.values()) if parity else None}
            U.wj(U.OUT / f'SCORE_META_t{tier}.json', meta_all)
            U.log('SCORE', ds, cell, 't%d' % tier, '%.0fs' % (time.time() - t0), 'parity', meta_all[key]['parity_all_equal'])
            if parity and not meta_all[key]['parity_all_equal']:
                bad = {k: v for k, v in parity.items() if not v['equal']}
                U.wj(U.OUT / 'STAGEA_PARITY_FAIL.json', bad)
                raise SystemExit(f'EVAL01_PROTOCOL_BLOCKED: Stage-A grid parity failed ({len(bad)} cells)')
            del grids
        del Q, stored
        guard_storage(f'score{tier}_{ds}')
    U.log('SCORE DONE', tier, '%.0fs' % (time.time() - t_all))


# ------------------------------------------------------------------ metrics (CPU pool)
def metrics(tier):
    verify_lock()
    tier = int(tier)
    G = U.bind()
    guard_storage(f'metrics{tier}_start')
    for ds in U.EXEC_DATASETS:
        outp = U.OUT / f'METRICS_{ds}_t{tier}.json'
        if outp.exists():
            continue
        pl = plan(G, ds)
        base_order, _ = sizes_info(G, ds)
        T = len(base_order)
        tasks = []
        for ci, c in enumerate(base_order):
            jobs = []
            for cell, cd in pl['cells'].items():
                arms_def = cd['arms'] if tier == 1 else cd.get('arms_t2') or {}
                for a, (kind, sz, exact, boot) in arms_def.items():
                    jobs.append((f'{cell}|{a}', str(grid_path(ds, cell, a, c)), list(cd['order']).index(c), exact, boot))
            if jobs:
                tasks.append((ds, c, ci, T, jobs, 4))
        t0 = time.time()
        res, boots, info = {}, {}, {}
        ctx = mp.get_context('spawn')
        with ctx.Pool(POOL) as pool:
            for ds_, c, r, b, inf in pool.imap_unordered(M.run_category, sorted(tasks, key=lambda x: -len(x[4]))):
                res[c], boots[c], info[c] = r, b, inf
                U.log('METRIC', ds, c, len(r), '%.0fs' % inf['seconds'])
        bp = U.CACHE / 'boot' / f'{ds}_t{tier}.npz'
        bp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bp, **{f'{c}||{j}||{m}': v[m] for c, bj in boots.items() for j, v in bj.items() for m in v})
        U.wj(outp, {'dataset': ds, 'tier': tier, 'cells': res, 'categories': info, 'boot_file': str(bp),
                    'boot_sha256': U.sha(bp), 'seconds': time.time() - t0,
                    'peak_rss_parent': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    'peak_rss_children': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024})
        guard_storage(f'metrics{tier}_{ds}')


# ------------------------------------------------------------------ latency (GPU; normal FIT queries only)
@torch.no_grad()
def latency():
    verify_lock()
    G = U.bind()
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {}
    for ds in U.EXEC_DATASETS:
        pl = plan(G, ds)
        base_order, si = sizes_info(G, ds)
        banks = load_banks(G, ds, 2711)
        q = G.fit_tokens(ds, base_order[0])[:784 * 50].cpu()            # 50 FIT images of the first task (normal)
        cfg = {'k1': si['k1']}
        arms = {**{a: v for a, v in pl['cells']['a|2711']['arms'].items() if a in ('N0', 'A1', 'A2')},
                **{a: v for a, v in pl['cells']['a|2711']['arms_t2'].items() if a.startswith(('G3r', 'A0cap', 'A0r'))}}
        for a, (kind, sz, _, _) in arms.items():
            spec = AR.stage_specs(kind, pl['cells']['a|2711']['order'], len(base_order) - 1, banks, sz, cfg)
            ts = []
            if spec[0] == 'union_w':
                keys = torch.cat(spec[1]).cuda().float(); W = spec[2].cuda()
                f = lambda x: G.whitened(x, keys, W)
            else:
                kb = [banks[t].keys(spec[2][t]).cuda().float() for t in spec[1]]
                f = lambda x: torch.stack([G.euclid(x, k) for k in kb]).amin(0)
            for i in range(60):
                x = q[(i % 50) * 784:(i % 50 + 1) * 784]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                f(x); torch.cuda.synchronize()
                if i >= 10:
                    ts.append(time.perf_counter() - t0)
            ts = np.array(ts) * 1000
            out[f'{ds}|{a}'] = {'p50_ms': float(np.median(ts)), 'p95_ms': float(np.quantile(ts, .95)), 'mean_ms': float(ts.mean()),
                                'n_keys': int(sum(sz.values())), 'bytes': arm_bytes(kind, sz, len(base_order))}
            U.log('LAT', ds, a, '%.2f ms' % out[f'{ds}|{a}']['p50_ms'])
    out['protocol'] = ('retrieval only (energy of 784 query tokens vs the final-stage memory; excludes encoder and A1 '
                       'readout); batch 1 image; 10 warm-up + 50 timed; RTX 4050 Laptop; TF32 off; queries = FIT tokens '
                       'of the first task (normal training data, no test access)')
    U.wj(U.OUT / 'LATENCY.json', out)


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'summary':
        from exps.nanm_eval_01 import summary as S
        S.main()
    else:
        {'lock': lock, 'states': states, 'score': score, 'metrics': metrics, 'latency': latency}[ph](*sys.argv[2:])
