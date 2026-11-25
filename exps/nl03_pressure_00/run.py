"""NL03-MEMORY-PRESSURE-00 driver (registration: agents/tasks/task_nl03_pressure_00_registration.md).

    export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
    python -B -m exps.nl03_pressure_00.run tests          # pytest T1-T6 -> TESTS.json (GPU: T1 parity)
    python -B -m exps.nl03_pressure_00.run lock           # PREREG_LOCK.json (create-exclusive)
    python -B -m exps.nl03_pressure_00.run cover_prep     # CREDIT-role coverage queries -> K:
    python -B -m exps.nl03_pressure_00.run smoke          # MPDD a|2711, all arms @ 4 MiB: score + metrics -> SMOKE.json
    python -B -m exps.nl03_pressure_00.run project        # PROJECTION.json; exit 4 if projected GPU wall > ceiling
    python -B -m exps.nl03_pressure_00.run score <tier>   # grids + eviction logs + coverage (GPU); budget stop
    python -B -m exps.nl03_pressure_00.run metrics <tier> # CPU pool
    python -B -m exps.nl03_pressure_00.run summary        # RESULTS.json + VERDICT.json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from exps.nanm_eval_01 import common as U
from exps.nl03_pressure_00 import core as C
from exps.nl03_pressure_00 import evaluate as EV

OUT, CACHE = C.OUT, C.CACHE
REG = U.ROOT / 'agents/tasks/task_nl03_pressure_00_registration.md'
TASK = U.ROOT / 'agents/tasks/task_nl03_pressure_00.md'
CODE = ['exps/nl03_pressure_00/__init__.py', 'exps/nl03_pressure_00/core.py', 'exps/nl03_pressure_00/evaluate.py',
        'exps/nl03_pressure_00/run.py', 'exps/nl03_pressure_00/summary.py', 'exps/nl03_pressure_00/test_core.py']
BUDGET = {'gpu_wall_hours': 6.0, 'k_new_bytes': 30 * 10 ** 9, 'rss_bytes': int(9.5 * U.GiB), 'd_free_min': 30 * U.GiB}
SAFETY = 1.5
POOL = 3
EVAL01_GRIDS = U.CACHE / 'grids'                       # /mnt/k/NLCAD/cache/nanm_eval_01/grids (read-only, M2)


# ------------------------------------------------------------------ plan (pure, metadata only)
def arms_t1():
    a = {'U': ('U', None)}
    for b in C.BUDGETS_MIB:
        for arm in C.ARMS[1:]:
            a[f'{arm}@{b}'] = (arm, b * C.MiB)
    return a


def plan():
    cells = []
    for st in C.STREAMS:
        cells.append({'stream': st, 'tier': 1, 'cell': 'a|2711', 'arms': arms_t1()})
    for st in C.STREAMS:                                 # VisA first, then S33 (registered priority)
        for s in C.SEEDS:
            for o in ('a', 'b', 'c'):
                if (o, s) == ('a', 2711):
                    continue
                cells.append({'stream': st, 'tier': 2, 'cell': f'{o}|{s}',
                              'arms': {f'{a}@{C.PRIMARY_MIB}': (a, C.PRIMARY_MIB * C.MiB) for a in C.T2_ARMS}})
    return cells


def smoke_cell():
    return {'stream': 'mpdd', 'tier': 0, 'cell': 'a|2711',
            'arms': {'U': ('U', None), **{f'{a}@4': (a, 4 * C.MiB) for a in C.ARMS[1:]}}}


def grid_path(stream, cell, name, task):
    return CACHE / 'grids' / stream / cell.replace('|', '_') / name / f"{task.replace('/', '__')}.pt"


def ck(cd):
    return f"{cd['stream']}|{cd['cell']}"


# ------------------------------------------------------------------ guards, ledger, lock
def guard(stage):
    s = U.storage()
    assert s['D_free'] >= BUDGET['d_free_min'], f'D: free below 30 GiB at {stage}'
    s['K_new_bytes'] = sum(f.stat().st_size for f in CACHE.rglob('*') if f.is_file()) if CACHE.exists() else 0
    assert s['K_new_bytes'] <= BUDGET['k_new_bytes'], f'new K: bytes above ceiling at {stage}'
    OUT.joinpath('logs').mkdir(parents=True, exist_ok=True)
    with open(OUT / 'logs/storage.jsonl', 'a') as f:
        f.write(json.dumps({'stage': stage, **s}) + '\n')
    return s


def ledger_add(phase, seconds, gpu, **kw):
    with open(OUT / 'BUDGET_LEDGER.jsonl', 'a') as f:
        f.write(json.dumps({'phase': phase, 'seconds': seconds, 'gpu': gpu, 'time': time.strftime('%FT%T'), **kw}) + '\n')


def gpu_used():
    p = OUT / 'BUDGET_LEDGER.jsonl'
    if not p.exists():
        return 0.0
    return sum(json.loads(l)['seconds'] for l in p.read_text().splitlines() if l.strip() and json.loads(l)['gpu'])


def lock():
    p = OUT / 'PREREG_LOCK.json'
    assert not p.exists(), 'lock exists'
    tests = U.rj(OUT / 'TESTS.json')
    assert tests['pass'], 'tests failed'
    au = U.rj(U.OUT / 'SOURCE_AUDIT.json')
    canon = au['canonical_sources']
    bad = [f for f, h in canon.items() if h is not None and U.sha(U.ROOT / f) != h]
    assert not bad, f'canonical sources changed: {bad}'
    from exps.nanm_eval_01 import mvtec as MV
    eval01 = ['exps/nanm_eval_01/common.py', 'exps/nanm_eval_01/arms.py', 'exps/nanm_eval_01/metrics.py',
              'exps/nanm_eval_01/mvtec.py']
    rec = {'registration': {str(REG.relative_to(U.ROOT)): U.sha(REG)}, 'task': {str(TASK.relative_to(U.ROOT)): U.sha(TASK)},
           'code': {f: U.sha(U.ROOT / f) for f in CODE}, 'eval01_code': {f: U.sha(U.ROOT / f) for f in eval01},
           'canonical_sources': canon,
           'eval01_lock_sha256': U.sha(U.OUT / 'PREREG_LOCK.json'), 'a1_lock_sha256': U.sha(MV.MV_OUT / 'PREREG_LOCK.json'),
           'tests_sha256': U.sha(OUT / 'TESTS.json'), 'plan': [{k: v for k, v in cd.items()} for cd in plan()],
           'smoke': smoke_cell(), 'orders': {s: C.orders(s) for s in ('visa', 's33', 'mpdd')},
           'caps': {f'{a}@{b}': C.cap_keys(b * C.MiB, C.ARM_SLOW[a]) for a in C.ARMS[1:] for b in C.BUDGETS_MIB},
           'budget': BUDGET, 'storage': U.storage(), 'time': time.strftime('%FT%T'),
           'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'exposure': 'all MPDD / VisA / MVTec AD metrics are EXPOSED-DEV; diagnostic only'}
    U.wj(p, rec, exclusive=True)
    U.log('LOCK', U.sha(p)[:16])


def verify_lock():
    lk = U.rj(OUT / 'PREREG_LOCK.json')
    bad = [f for f, h in {**lk['registration'], **lk['task'], **lk['code'], **lk['eval01_code'],
                          **lk['canonical_sources']}.items() if h is not None and U.sha(U.ROOT / f) != h]
    assert not bad, f'LOCK_FAIL: {bad}'
    return lk


def tests():
    t0 = time.time()
    r = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-s', '-p', 'no:cacheprovider',
                        'exps/nl03_pressure_00/test_core.py'], capture_output=True, text=True)
    OUT.joinpath('logs').mkdir(parents=True, exist_ok=True)
    (OUT / 'logs/tests_stdout.log').write_text(r.stdout + '\n' + r.stderr)
    last = [l for l in r.stdout.splitlines() if l.strip()][-1] if r.stdout.strip() else ''
    U.wj(OUT / 'TESTS.json', {'command': 'pytest -q -s -p no:cacheprovider exps/nl03_pressure_00/test_core.py',
                              'returncode': r.returncode, 'summary': last, 'pass': r.returncode == 0,
                              'seconds': time.time() - t0})
    ledger_add('tests', time.time() - t0, True)
    U.log('TESTS', last)
    raise SystemExit(r.returncode)


# ------------------------------------------------------------------ coverage queries (CREDIT role)
def cover_prep():
    verify_lock()
    t0 = time.time()
    G = C.bind()
    man = {}
    for t in C.base_order('s33'):
        p = CACHE / 'cover' / f"{t.replace('/', '__')}.pt"
        if not p.exists():
            x, paths = C.credit_tokens(G, t)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'x': x, 'paths': paths}, p)
        d = torch.load(p, weights_only=True)
        man[t] = {'n_images': len(d['paths']), 'n_tokens': len(d['x']), 'sha256': U.sha(p),
                  'excluded_nl02': len(C.nl02_excluded(G, t))}
        U.log('COVER', t, man[t]['n_images'])
    U.wj(OUT / 'COVER_MANIFEST.json', man)
    ledger_add('cover_prep', time.time() - t0, True)


# ------------------------------------------------------------------ score one cell (GPU)
@torch.no_grad()
def score_cell(G, cd, check_eval01=False):
    stream, cell = cd['stream'], cd['cell']
    o, s = cell.split('|')
    seed = int(s)
    order = C.orders(stream)[o]
    T = len(order)
    stages = C.logged_stages(stream, T)
    t0 = time.time()
    mat = C.load_material(G, order, seed)
    pool = C.Pool(order, mat)
    Ws = C.whiteners(order, mat)
    trajs, ev_meta, t_arm = {}, {}, {}
    for name, (arm, budget) in cd['arms'].items():
        t1 = time.time()
        traj, logs = C.run_arm(arm, pool, seed, budget)
        t_arm[name] = time.time() - t1
        if not all(x['invariant_ok'] for x in logs):
            raise SystemExit(f'NL03_PROTOCOL_BLOCKED: eviction invariant failed {stream} {cell} {name}')
        trajs[name] = traj
        ep = OUT / 'evict' / stream / cell.replace('|', '_') / f'{name}.npz'
        ep.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ep, **{f'{x["stage"]}|{k}': np.asarray(x[k]) for x in logs
                                   for k in ('evicted_ids', 'evicted_task', 'evicted_age', 'survivors_per_task')},
                            **{f'{l}|K': trajs[name][l].numpy() for l in range(T)})
        ev_meta[name] = [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in x.items()
                          if k not in ('evicted_ids', 'evicted_task', 'evicted_age')} | {
            'n_evicted': int(len(x['evicted_ids'])),
            'survivors_per_task': x['survivors_per_task'].tolist(),
            'evicted_age_mean': float(x['evicted_age'].mean()) if len(x['evicted_age']) else None,
            'evicted_by_task': np.bincount(x['evicted_task'], minlength=T)[:x['stage'] + 1].tolist()}
            for x in logs]
    if 'E@8' in trajs and 'E4@8' in trajs:
        assert all(torch.equal(a, b) for a, b in zip(trajs['E@8'], trajs['E4@8'])), 'E/E4 trajectories differ'
    t_state = time.time() - t0
    exs = {}
    cover, parity, work = {}, {}, {}
    for j, t in enumerate(order):
        ds, c = t.split('/')
        if ds not in exs:
            exs[ds] = C.extract_rec(G, ds)
        C.use(G, ds)
        q, rows = G.load_test(ds, c, exs[ds])
        q = q.cuda()
        cq = torch.load(CACHE / 'cover' / f"{t.replace('/', '__')}.pt", weights_only=True)['x'].cuda()
        st = [l for l in stages if l >= j]
        for name, (arm, budget) in cd['arms'].items():
            p = grid_path(stream, cell, name, t)
            geom = C.ARM_GEOM[arm]
            gl = []
            for l in st:
                keys16 = pool.keys[trajs[name][l].cuda()]
                gl.append(C.energy(G, q, keys16, None if geom is None else Ws[geom][l]))
                cover[f'{name}|{t}|{l}'] = C.coverage(G, cq, keys16)
                work.setdefault(name, [0, 0])
                work[name][0] += len(q) * len(keys16); work[name][1] += len(q)
            grid = torch.stack(gl).reshape(len(st), len(rows), 28, 28).contiguous()
            if check_eval01 and name == 'U':
                ref = torch.load(EVAL01_GRIDS / 'visa' / cell.replace('|', '_') / 'N0' / f'{c}.pt', weights_only=True)
                assert ref['rows'] == rows
                parity[t] = bool(torch.equal(ref['grid'], grid))
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'rows': rows, 'grid': grid, 'stages': st, 'task': t, 'pos': j, 'order': order}, p)
        del q, cq
        torch.cuda.empty_cache()
    meta = {'seconds': time.time() - t0, 'state_seconds': t_state, 'arm_state_seconds': t_arm, 'work': work,
            'peak_vram': torch.cuda.max_memory_allocated(), 'stages': stages, 'order': order,
            'cond': Ws['cond'], 'evict': ev_meta, 'parity_eval01_U': parity or None,
            'n_keys_final': {n: len(tr[-1]) for n, tr in trajs.items()},
            'bytes_final': {n: C.bytes_of(len(tr[-1]), C.ARM_SLOW[cd['arms'][n][0]]) for n, tr in trajs.items()}}
    U.wj(OUT / 'cells' / f"{stream}_{cell.replace('|', '_')}.json", meta)
    U.wj(OUT / 'cover' / f"{stream}_{cell.replace('|', '_')}.json", cover)
    if parity and not all(parity.values()):
        raise SystemExit(f'NL03_PROTOCOL_BLOCKED: M2 U vs EVAL-01 N0 grid parity failed {parity}')
    return meta


def score(tier):
    verify_lock()
    tier = int(tier)
    torch.backends.cuda.matmul.allow_tf32 = False
    G = C.bind()
    guard(f'score{tier}_start')
    proj = U.rj(OUT / 'PROJECTION.json')
    done = U.rj(OUT / f'SCORE_t{tier}.json') if (OUT / f'SCORE_t{tier}.json').exists() else {}
    for cd in [x for x in plan() if x['tier'] == tier]:
        key = ck(cd)
        if key in done:
            continue
        if tier == 2 and cd['stream'] == 's33' and not proj['fits']:      # registered priority: S33 tier 2 NOT_RUN
            done[key] = {'status': 'NOT_RUN (budget projection)'}
            U.wj(OUT / f'SCORE_t{tier}.json', done)
            continue
        need = proj['cells'][key]['gpu_s_projected']
        used = gpu_used()
        if used + need > BUDGET['gpu_wall_hours'] * 3600:
            U.wj(OUT / 'BUDGET_STOP.json', {'cell': key, 'used_s': used, 'projected_s': need,
                                            'ceiling_s': BUDGET['gpu_wall_hours'] * 3600, 'time': time.strftime('%FT%T')})
            U.log('BUDGET_STOP', key, used, need)
            raise SystemExit(4)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            meta = score_cell(G, cd, check_eval01=(cd['stream'] == 'visa' and cd['cell'] == 'a|2711'))
        finally:
            ledger_add(f'score{tier}', time.time() - t0, True, cell=key)
        done[key] = {'seconds': meta['seconds'], 'peak_vram': meta['peak_vram'],
                     'parity_eval01_U_all': (all(meta['parity_eval01_U'].values()) if meta['parity_eval01_U'] else None)}
        U.wj(OUT / f'SCORE_t{tier}.json', done)
        U.log('SCORE', key, '%.0fs' % meta['seconds'], 'parity', done[key]['parity_eval01_U_all'])
        guard(f'score{tier}_{key}')


# ------------------------------------------------------------------ metrics (CPU pool)
def metric_jobs(cd):
    stream, cell = cd['stream'], cd['cell']
    order = C.orders(stream)[cell.split('|')[0]]
    jobs = {t: [] for t in order}
    for name, (arm, budget) in cd['arms'].items():
        boot = cd['tier'] in (0, 1)
        exact = (stream == 'visa' and cd['tier'] == 1 and name in ('U', 'Q@8', 'E@8', 'E4@8'))
        for t in order:
            jobs[t].append((f'{cell}|{name}', str(grid_path(stream, cell, name, t)), boot, exact))
    return jobs


def run_metrics(cds, tag):
    t0 = time.time()
    by_stream = {}
    for cd in cds:
        for t, js in metric_jobs(cd).items():
            by_stream.setdefault(cd['stream'], {}).setdefault(t, []).extend(js)
    for stream, tasks in by_stream.items():
        outp = OUT / f'METRICS_{stream}_{tag}.json'
        if outp.exists():
            continue
        args = [(t, C.cat_index(stream, t), js, 4) for t, js in tasks.items()]
        res, boots, info = {}, {}, {}
        ctx = mp.get_context('spawn')
        with ctx.Pool(POOL) as pool:
            for t, r, b, inf in pool.imap_unordered(EV.run_task, sorted(args, key=lambda a: -len(a[2]))):
                res[t], boots[t], info[t] = r, b, inf
                U.log('METRIC', stream, t, len(r), '%.0fs' % inf['seconds'])
        bp = CACHE / 'boot' / f'{stream}_{tag}.npz'
        bp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bp, **{f'{t}||{j}||{m}': v[m] for t, bj in boots.items() for j, v in bj.items() for m in v})
        U.wj(outp, {'stream': stream, 'tag': tag, 'cells': res, 'tasks': info, 'boot_file': str(bp), 'boot_sha256': U.sha(bp),
                    'seconds': time.time() - t0,
                    'peak_rss_children': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024})
    return time.time() - t0


def metrics(tier):
    verify_lock()
    tier = int(tier)
    guard(f'metrics{tier}_start')
    done = U.rj(OUT / f'SCORE_t{tier}.json')
    cds = [cd for cd in plan() if cd['tier'] == tier and ck(cd) in done and 'seconds' in done[ck(cd)]]
    sec = run_metrics(cds, f't{tier}')
    ledger_add(f'metrics{tier}', sec, False)
    guard(f'metrics{tier}_end')


# ------------------------------------------------------------------ smoke + projection
def smoke():
    verify_lock()
    torch.backends.cuda.matmul.allow_tf32 = False
    G = C.bind()
    guard('smoke_start')
    cd = smoke_cell()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        meta = score_cell(G, cd)
    finally:
        ledger_add('smoke_score', time.time() - t0, True)
    t_score = time.time() - t0
    t_met = run_metrics([cd], 'smoke')
    ledger_add('smoke_metrics', t_met, False)
    U.wj(OUT / 'SMOKE.json', {'score_seconds': t_score, 'metrics_seconds': t_met, 'meta': {k: meta[k] for k in (
        'seconds', 'state_seconds', 'arm_state_seconds', 'work', 'peak_vram', 'n_keys_final', 'bytes_final')}})
    guard('smoke_end')


def n_test(G, t):
    ds, c = t.split('/')
    C.use(G, ds)
    return len(G.test_rows(ds, c))


def planned_work(G, cd, ntest):
    """GPU work units of one cell (registered caps, pure metadata): (sum nq*nk, sum nq, k-centre iterations*cand)."""
    stream, cell = cd['stream'], cd['cell']
    order = C.orders(stream)[cell.split('|')[0]]
    T = len(order)
    stages = C.logged_stages(stream, T)
    recs = {}
    k1 = []
    for t in order:
        ds, c = t.split('/')
        if ds not in recs:
            recs[ds] = C.nrec(G, ds)[ds]
        C.use(G, ds)
        k1.append(recs[ds]['sizes']['g3'][list(G.order(ds)).index(c)])
    cum = np.cumsum(k1)
    w1 = w0 = wk = 0
    for name, (arm, budget) in cd['arms'].items():
        cap = None if arm == 'U' else C.cap_keys(budget, C.ARM_SLOW[arm])
        nk = [int(cum[l]) if cap is None else int(min(cum[l], cap)) for l in range(T)]
        for j, t in enumerate(order):
            nq = (ntest[t] + C.COVER_IMGS) * 784
            for l in stages:
                if l >= j:
                    w1 += nq * nk[l]; w0 += nq
        if cap is not None:
            for l in range(T):
                cand = int(cum[l]) if arm == 'Q' else (nk[l - 1] if l else 0) + k1[l]
                if cand > cap and arm != 'FIFO':
                    wk += cap * cand
    return w1, w0, wk


def project():
    verify_lock()
    G = C.bind()
    sm = U.rj(OUT / 'SMOKE.json')
    order_all = C.base_order('s33') + [t for t in C.base_order('mpdd') if t not in C.base_order('s33')]
    ntest = {t: n_test(G, t) for t in set(order_all)}
    sw1, sw0, swk = planned_work(G, smoke_cell(), ntest)
    # conservative single-rate model: smoke GPU seconds split by measured state (k-centre) vs scoring time
    s_state = sm['meta']['state_seconds']; s_score = sm['score_seconds'] - s_state
    r_k = s_state / max(swk, 1)
    r_1 = s_score / max(sw1, 1)                         # all scoring time attributed to nq*nk work (conservative)
    r_0 = s_score / max(sw0, 1)
    cells, tot, tot_t = {}, 0.0, {1: 0.0, 2: 0.0}
    for cd in plan():
        w1, w0, wk = planned_work(G, cd, ntest)
        g = SAFETY * (max(r_1 * w1, r_0 * w0) + r_k * wk + 60)
        cells[ck(cd)] = {'tier': cd['tier'], 'w1': w1, 'w0': w0, 'wk': wk, 'gpu_s_projected': g}
        tot += g; tot_t[cd['tier']] += g
    used = gpu_used()
    ceiling = BUDGET['gpu_wall_hours'] * 3600
    met_rate = sm['metrics_seconds'] / max(sw0, 1)
    cpu = {k: met_rate * sum(v['w0'] for v in cells.values() if v['tier'] == k) for k in (1, 2)}
    rec = {'rates': {'r_k': r_k, 'r_1': r_1, 'r_0': r_0, 'metrics_per_query_token_stage': met_rate},
           'safety': SAFETY, 'cells': cells, 'gpu_projected_s': {'tier1': tot_t[1], 'tier2': tot_t[2], 'total': tot},
           'gpu_used_s_before': used, 'ceiling_s': ceiling, 'fits': used + tot <= ceiling,
           'fits_tier1_plus_visa_t2': used + tot_t[1] + sum(v['gpu_s_projected'] for k, v in cells.items()
                                                            if v['tier'] == 2 and k.startswith('visa')) <= ceiling,
           'cpu_metrics_projected_s': cpu, 'time': time.strftime('%FT%T')}
    U.wj(OUT / 'PROJECTION.json', rec)
    U.log('PROJECT gpu total %.2f h (+used %.2f h) ceiling %.1f h fits=%s; cpu metrics t1 %.1f h t2 %.1f h' % (
        tot / 3600, used / 3600, ceiling / 3600, rec['fits'], cpu[1] / 3600, cpu[2] / 3600))
    if not rec['fits_tier1_plus_visa_t2']:
        raise SystemExit(4)


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'summary':
        from exps.nl03_pressure_00 import summary as S
        verify_lock()
        S.main()
    else:
        {'tests': tests, 'lock': lock, 'cover_prep': cover_prep, 'smoke': smoke, 'project': project, 'score': score,
         'metrics': metrics}[ph](*sys.argv[2:])
