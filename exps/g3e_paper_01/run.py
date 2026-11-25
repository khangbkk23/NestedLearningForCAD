"""G3E-PAPER-01 driver (owner task 2026-11-25, agents/tasks/task_ad2_dthr_g3e_paper_01.md, PART 2).

Protocol = NL03-MULTIRATE-01 §17 exactly: full train/good, full official test sets, orders a/b/c (final.bench_order),
final model; native pixel AP (binned, BENCH-G3 primitives) with paired image bootstrap (B = 2000, stratified by
category, index = position in the order), image AUROC (A1), FM, acquisition, prequential, bytes, update time, latency.
Scoring and metrics reuse final.CO_energy and the NL03-PRESSURE-00 evaluator unchanged.

    python -B -m exps.g3e_paper_01.run reuse | tests | plan | cell <name> | metrics <tag> | tables
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.g3e_paper_01 import learner as LR
from exps.nl03_multirate_01 import final as FN
from exps.nl03_multirate_01 import run as RN
from exps.nl03_multirate_01 import spec as SP

ROOT = SP.ROOT
OUT = ROOT / 'results/hope_cad/g3e_paper_01'
CACHE = Path('/mnt/k/NLCAD/cache/g3e_paper_01')
LEDGER = OUT / 'BUDGET_LEDGER.jsonl'
CEILING_S = 10 * 3600
MR_BENCH = FN.BENCH_OUT                          # results/hope_cad/nl03_multirate_01/bench
MR_CACHE = FN.CACHE / 'bench'
DSETS = ('mvtec', 'visa', 'mpdd')
MIN_FREE_GIB = 30


def log(*a):
    s = time.strftime('%H:%M:%S') + ' ' + ' '.join(str(x) for x in a)
    print(s, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'logs' / 'run.log', 'a') as f:
        f.write(s + '\n')


def wj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=1, default=float))


def rj(p):
    return json.loads(Path(p).read_text())


def ledger_add(rec):
    rec['time'] = time.strftime('%FT%T')
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def gpu_used():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)['seconds'] for l in LEDGER.read_text().splitlines() if json.loads(l).get('gpu'))


class BudgetStop(RuntimeError):
    pass


def guard(extra=0.0):
    if gpu_used() + extra > CEILING_S:
        raise BudgetStop(f'G3E-PAPER-01 GPU ceiling ({(gpu_used() + extra) / 3600:.2f} h)')
    import os
    for m in ('/mnt/d', '/mnt/e', '/mnt/k'):
        s = os.statvfs(m)
        if s.f_bavail * s.f_frsize / 2**30 < MIN_FREE_GIB:
            raise SystemExit(f'STOP: {m} below {MIN_FREE_GIB} GiB')


# ------------------------------------------------------------------ registered plan (priority order A1 > A4 > A3 > B > A2)
SEEDS = (2711, 2712, 2713)
A4_ARMS = ['Q', 'E4', 'F', 'G', 'FIFO', 'B1']


def plan():
    P = [('T1', 'dinov2', 'mpdd', 'a', 2711, ['U', 'E@8', 'E@inf'])]
    P += [('A1', 'dinov2', 'mpdd', 'c', 2711, ['U', 'E@8'])]
    P += [('A4', 'dinov2', ds, 'a', 2711, A4_ARMS) for ds in DSETS]
    P += [('A3', 'dinov2', ds, 'a', 2711, ['E@4', 'E@16', 'E@32']) for ds in DSETS]
    P += [('A3', 'dinov2', ds, 'a', s, ['U', 'E@4', 'E@8', 'E@16', 'E@32']) for s in (2712, 2713) for ds in DSETS]
    P += [('B', 'vitb8', ds, 'a', 2711, ['U', 'E@8', 'F']) for ds in ('mvtec', 'visa')]
    P += [('A2', 'dinov2', ds, ok, s, ['U', 'E@8']) for s in (2712, 2713) for ok in 'bc' for ds in DSETS]
    return P


def tag_of(c):
    name, bb, ds, ok, seed, arms = c
    return f'{name}_{bb}_{ds}_{ok}{seed}'


# reference GPU seconds of a MULTIRATE-01 U+E+P benchmark cell (orders b/c), used only for the budget projection
REF_S = {'mvtec': 520.0, 'visa': 1590.0, 'mpdd': 85.0}


def est(c):
    name, bb, ds, ok, seed, arms = c
    f = 1.0 if 'U' in arms else 0.55 + 0.08 * len(arms)
    if any(a in ('E@32', 'E@inf') for a in arms):
        f += 0.25
    extra = 900.0 if bb == 'vitb8' and not (CACHE / 'vitb8' / f'{ds}_DONE').exists() else 0.0
    return 1.25 * f * REF_S[ds] + extra


# ------------------------------------------------------------------ one cell (GPU)
def _train_rows(G, t):
    return FN.train_rows_sorted(G, t)


@torch.no_grad()
def cell(c):
    from exps.nl03_pressure_00 import core as PC
    from exps.g3e_paper_01 import vitb8 as V8
    name, bb, ds, ok, seed, arms = c
    tag = tag_of(c)
    done = OUT / 'cells' / f'SCORE_{tag}.json'
    if done.exists():
        log('SKIP', tag); return
    FN.verify_final()
    G = PC.bind()
    order = FN.bench_order(ds, ok)
    counts = [len(_train_rows(G, t)) for t in order]
    t0 = time.time()
    try:
        guard()
        if bb == 'vitb8':
            V8.extract(G, ds, guard=lambda s: guard(time.time() - t0))
            (CACHE / 'vitb8' / f'{ds}_DONE').write_text(time.strftime('%FT%T'))
        L = LR.Learner(arms, seed)
        for l, t in enumerate(order):
            if bb == 'dinov2':
                x16, n = RN.fit_tokens(G, t, order=_train_rows(G, t))
            else:
                x16, n = V8.fit_tokens(ds, t.split('/')[1], _train_rows(G, t))
            L.task(x16, l)
            del x16; torch.cuda.empty_cache()
            log('CELL', tag, l, t, {a: L.bytes[a][-1] for a in arms})
            guard(time.time() - t0)
        t_state = time.time() - t0
        ex = PC.extract_rec(G, ds) if bb == 'dinov2' else None
        lat = {a: [] for a in arms}
        for j, t in enumerate(order):
            cname = t.split('/')[1]
            PC.use(G, ds)
            if bb == 'dinov2':
                q, rows = G.load_test(ds, cname, ex)
            else:
                q, rows = V8.load_test(ds, cname)
                ref = torch.load(MR_CACHE / f'{ds}_a2711' / 'U' / f'{cname}.pt', weights_only=True)['rows']
                assert rows == ref, f'vitb8 test rows differ from the DINOv2 benchmark rows {cname}'
            for a in arms:
                gl = []
                for l in range(j, len(order)):
                    K, W = L.keys_at(a, l)
                    t1 = time.time()
                    gl.append(FN.CO_energy(q, K, W))
                    lat[a].append((time.time() - t1) / len(rows))
                grid = torch.stack(gl).reshape(len(gl), len(rows), 28, 28).contiguous()
                p = CACHE / 'cells' / tag / a / f'{cname}.pt'
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'rows': rows, 'grid': grid, 'stages': list(range(j, len(order))), 'task': t, 'pos': j,
                            'order': order}, p)
            del q; torch.cuda.empty_cache()
            guard(time.time() - t0)
    finally:
        ledger_add({'phase': 'g3e_cell', 'cell': tag, 'seconds': time.time() - t0, 'gpu': True})
    wj(done, {'tag': tag, 'name': name, 'backbone': bb, 'dataset': ds, 'order_key': ok, 'seed': seed, 'order': order,
              'arms': arms, 'counts': counts, 'bytes': L.bytes, 'update_s': L.timing, 'state_s': t_state,
              'select_log': {a: v for a, v in L.select_log.items()}, 'seconds': time.time() - t0,
              'latency_ms_per_img_p50': {a: float(np.median(v) * 1e3) for a, v in lat.items()},
              'peak_vram': torch.cuda.max_memory_allocated(),
              'label': 'MATCHED_BACKBONE_ABLATION' if bb == 'vitb8' else 'G3E_PAPER'})
    log('CELL DONE', tag, '%.0fs' % (time.time() - t0))


# ------------------------------------------------------------------ metrics (CPU, pool 2)
def metrics(tag):
    from exps.nl03_pressure_00 import evaluate as PEV
    sc = rj(OUT / 'cells' / f'SCORE_{tag}.json')
    outp = OUT / 'cells' / f'METRICS_{tag}.json'
    if outp.exists():
        return
    boot = sc['order_key'] == 'a'
    args = []
    for j, t in enumerate(sc['order']):
        cname = t.split('/')[1]
        jobs = [(a, str(CACHE / 'cells' / tag / a / f'{cname}.pt'), boot, False) for a in sc['arms']]
        args.append((t, j, jobs, 4))
    res, boots, info = {}, {}, {}
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for t, r, b, inf in pool.imap_unordered(PEV.run_task, args):
            res[t], boots[t], info[t] = r, b, inf
            log('METRIC', tag, t, '%.0fs' % inf['seconds'])
    rec = {'tag': tag, 'cells': res, 'tasks': info, 'seconds': time.time() - t0}
    if boot:
        bp = CACHE / 'cells' / f'boot_{tag}.npz'
        np.savez_compressed(bp, **{f'{t}||{a}||{m}': v[m] for t, bj in boots.items() for a, v in bj.items() for m in v})
        rec['boot_file'] = str(bp); rec['boot_sha256'] = FN.sha(bp)
    wj(outp, rec)
    ledger_add({'phase': 'cpu_metrics', 'cell': tag, 'seconds': time.time() - t0, 'gpu': False})


# ------------------------------------------------------------------ reuse of NL03-MULTIRATE-01 benchmark cells (seed 2711)
def reuse():
    FN.verify_final()
    rec = {}
    for p in sorted(MR_BENCH.glob('*.json')):
        d = rj(p)
        rec[p.name] = {'sha256': FN.sha(p)}
        if 'boot_file' in d:
            assert FN.sha(d['boot_file']) == d['boot_sha256'], f'boot changed {p.name}'
            rec[p.name]['boot_sha256_verified'] = d['boot_sha256']
    wj(OUT / 'REUSE.json', {'source': str(MR_BENCH.relative_to(ROOT)), 'files': rec,
                            'final_lock': FN.sha(FN.FINAL), 'note': 'U/E (8 MiB, seed 2711) cells of MULTIRATE-01 §17; not recomputed',
                            'time': time.strftime('%FT%T')})
    log('REUSE', len(rec))


# ------------------------------------------------------------------ tests (before running the plan)
def tests():
    import subprocess
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'exps/g3e_paper_01/test_g3e.py'],
                       capture_output=True, text=True, cwd=ROOT)
    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
    log('PYTEST rc', r.returncode, tail[-1] if tail else '')
    assert r.returncode == 0, 'STOP: unit tests failed'
    c = plan()[0]
    cell(c)
    tag = tag_of(c)
    out = {'pytest': {'rc': r.returncode, 'tail': tail}, 'cells': {}}
    for cname in [t.split('/')[1] for t in FN.bench_order('mpdd', 'a')]:
        u = torch.load(CACHE / 'cells' / tag / 'U' / f'{cname}.pt', weights_only=True)
        ei = torch.load(CACHE / 'cells' / tag / 'E@inf' / f'{cname}.pt', weights_only=True)
        e8 = torch.load(CACHE / 'cells' / tag / 'E@8' / f'{cname}.pt', weights_only=True)
        mu = torch.load(MR_CACHE / 'mpdd_a2711' / 'U' / f'{cname}.pt', weights_only=True)
        me = torch.load(MR_CACHE / 'mpdd_a2711' / 'E' / f'{cname}.pt', weights_only=True)
        out['cells'][cname] = {
            'Einf_eq_U_bitwise': bool(torch.equal(u['grid'], ei['grid'])),
            'U_eq_MR_U_bitwise': bool(torch.equal(u['grid'], mu['grid'])) and u['rows'] == mu['rows'],
            'E8_eq_MR_E_bitwise': bool(torch.equal(e8['grid'], me['grid'])) and e8['rows'] == me['rows'],
            'max_abs_U': float((u['grid'] - mu['grid']).abs().max()), 'max_abs_E': float((e8['grid'] - me['grid']).abs().max())}
    sc = rj(OUT / 'cells' / f'SCORE_{tag}.json')
    out['bytes_Einf_eq_U'] = sc['bytes']['E@inf'] == sc['bytes']['U']
    ok_inf = all(v['Einf_eq_U_bitwise'] for v in out['cells'].values()) and out['bytes_Einf_eq_U']
    ok_rep = all(v['U_eq_MR_U_bitwise'] and v['E8_eq_MR_E_bitwise'] for v in out['cells'].values())
    if not ok_rep:                                     # fallback: AP within 1e-6 against the stored MULTIRATE metrics
        metrics(tag)
        me_new = rj(OUT / 'cells' / f'METRICS_{tag}.json')['cells']
        me_old = rj(MR_BENCH / 'METRICS_mpdd_a2711.json')['cells']
        diffs = [abs(me_new[t][a][l]['ap'] - me_old[t][b][l]['ap']) for t in me_old for a, b in (('U', 'U'), ('E@8', 'E'))
                 for l in me_old[t][b]]
        out['ap_max_abs_diff'] = float(max(diffs))
        ok_rep = out['ap_max_abs_diff'] <= 1e-6
    out['E_inf_equals_U'] = ok_inf
    out['reproduces_multirate'] = ok_rep
    out['pass'] = bool(ok_inf and ok_rep)
    wj(OUT / 'TESTS.json', out)
    log('TESTS', out['pass'], {k: out[k] for k in ('E_inf_equals_U', 'reproduces_multirate')})
    assert out['pass'], 'STOP: driver tests failed'


def run_plan():
    """Priority order; a cell whose projection exceeds the remaining ceiling is NOT_RUN (budget); later smaller cells
    may still run. Metrics (CPU) follow each cell."""
    assert rj(OUT / 'TESTS.json')['pass']
    status = rj(OUT / 'PLAN_STATUS.json')['status'] if (OUT / 'PLAN_STATUS.json').exists() else {}
    P = plan()
    proj = {tag_of(c): est(c) for c in P[1:]}
    wj(OUT / 'PROJECTION.json', {'est_s': proj, 'total_h': sum(proj.values()) / 3600, 'used_h': gpu_used() / 3600,
                                 'ceiling_h': CEILING_S / 3600, 'time': time.strftime('%FT%T'),
                                 'rule': 'priority A1 > A4 > A3 > B > A2; A2 order a is inside the A3 seed-2712/2713 cells'})
    for c in P[1:]:
        tag = tag_of(c)
        if status.get(tag) == 'DONE':
            continue
        if not (OUT / 'cells' / f'SCORE_{tag}.json').exists() and gpu_used() + est(c) > CEILING_S:
            status[tag] = f'NOT_RUN (budget: projected {est(c) / 3600:.2f} h, used {gpu_used() / 3600:.2f} h)'
            log('NOT_RUN', tag); wj(OUT / 'PLAN_STATUS.json', {'status': status}); continue
        try:
            cell(c)
            metrics(tag)
            status[tag] = 'DONE'
        except BudgetStop as e:
            status[tag] = f'NOT_RUN (budget: stopped in cell) {e}'
            log('BUDGET_STOP', tag)
        wj(OUT / 'PLAN_STATUS.json', {'status': status, 'gpu_used_h': gpu_used() / 3600, 'time': time.strftime('%FT%T')})
    log('PLAN DONE', status)


if __name__ == '__main__':
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'cell':
        cell(next(c for c in plan() if tag_of(c) == sys.argv[2]))
    elif cmd == 'metrics':
        metrics(sys.argv[2])
    else:
        {'reuse': reuse, 'tests': tests, 'plan': run_plan}[cmd]()
