"""NL03-MULTIRATE-01 development driver (Phases G-K). One GPU job at a time; resumable at task boundaries.

    python -B -m exps.nl03_multirate_01.run synth <stream>        # build SYNTHETIC_DEV query caches (K:)
    python -B -m exps.nl03_multirate_01.run smoke                  # VisA a|2711, first 2 tasks, all arms, 8 MiB
    python -B -m exps.nl03_multirate_01.run project
    python -B -m exps.nl03_multirate_01.run dev                    # registered priority plan with GPU auto-stop
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import torch

from exps.nl03_multirate_01 import core as CO
from exps.nl03_multirate_01 import evaluate as EV
from exps.nl03_multirate_01 import spec as SP
from exps.nl03_multirate_01 import synth as SY

OUT, CACHE = SP.OUT, SP.CACHE
LEDGER = OUT / 'BUDGET_LEDGER.jsonl'
ALL_ARMS = ('C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'P', 'P4', 'P16')


def log(*a):
    from exps.nanm_eval_01 import common as U
    U.log(*a)


def wj(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def rj(p):
    return json.loads(p.read_text())


def ledger_add(rec):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, 'a') as f:
        f.write(json.dumps({**rec, 'time': time.strftime('%FT%T')}) + '\n')


def gpu_used(phase_prefix=('dev',)):
    if not LEDGER.exists():
        return 0.0
    return sum(r['seconds'] for r in map(json.loads, LEDGER.read_text().splitlines())
               if r.get('gpu') and r['phase'].startswith(phase_prefix))


def bind():
    from exps.nl03_pressure_00 import core as PC
    return PC.bind()


def orders(stream):
    from exps.nl03_pressure_00 import core as PC
    return PC.orders(stream)


def fit_dev_counts(stream):
    a = rj(OUT / 'AUDIT.json')['tasks']
    return [a[t]['FIT_dev'] for t in orders(stream)['a']]


def tau(stream):
    return SP.tau_of(fit_dev_counts(stream))


@torch.no_grad()
def fit_tokens(G, task, order=None):
    """Unit fp16 FIT_dev tokens on CUDA in registered chronological order (or a given path order)."""
    from exps.nanm_r0 import audit as A
    from exps.nl03_pressure_00 import core as PC
    ds, c = task.split('/')
    PC.use(G, ds)
    paths = order if order is not None else SP.splits(G, task)['FIT']
    data = G.load_train(ds, c)
    idx = {r['relative_path']: i for i, r in enumerate(data['rows'])}
    sel = torch.as_tensor([idx[p] for p in paths])
    x = torch.cat([A.unit(data['features'][sel[i:i + 64]].reshape(-1, SP.D).cuda()).half()
                   for i in range(0, len(sel), 64)])
    del data
    return x, len(paths)


# ------------------------------------------------------------------ one cell
class Cell:
    def __init__(self, stream, ok, seed, mib, arms, tag, ntasks=None):
        self.stream, self.ok, self.seed, self.mib, self.arms, self.tag = stream, ok, seed, mib, tuple(arms), tag
        self.order = orders(stream)[ok][:ntasks] if ntasks else orders(stream)[ok]
        self.budget = SP.BUDGETS[mib]
        self.tau = tau(stream)
        self.fasts, self.slows, self.bindk = CO.plan_arms([a for a in arms if a != 'C0'], self.budget, self.tau, seed)
        self.c0 = CO.TaskLevel(self.budget, seed) if 'C0' in arms else None
        self.step, self.l = 0, 0
        self.theta = {a: {} for a in arms}
        self.res = {a: {} for a in arms}            # arm -> stage -> task -> metrics (scalars + arrays)
        self.fast_log = {k: [] for k in self.fasts}
        self.evict_log = {k: [] for k in self.fasts}
        self.timing = {'fast_s': {str(k): 0.0 for k in self.fasts}, 'cand_s': 0.0, 'eval_s': 0.0, 'c0_s': 0.0,
                       'slow_s': 0.0, 'infer_img_s': []}
        self.causal = []

    def ckpt_path(self):
        return CACHE / 'ckpt' / f'{self.tag}.pt'

    def save(self):
        p = self.ckpt_path(); p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp'); torch.save(self, tmp); tmp.replace(p)

    @torch.no_grad()
    def run_task(self, G):
        task = self.order[self.l]
        x16, nimg = fit_tokens(G, task)
        need_cnt = any(f.counts for f in self.fasts.values())
        for b in range(0, nimg, SP.MB_IMAGES):
            xb = x16[b * 784:min(nimg, b + SP.MB_IMAGES) * 784]
            t0 = time.time()
            keys, S_b, n_b, cnt = CO.write_candidates(xb, self.seed, need_cnt)
            self.timing['cand_s'] += time.time() - t0
            self.causal.append((self.step, self.l, b))
            for k, f in self.fasts.items():
                t0 = time.time()
                ev = f.step(keys, cnt, self.l, self.step)
                torch.cuda.synchronize(); self.timing['fast_s'][str(k)] += time.time() - t0
                self.fast_log[k].append((self.step, len(f.K), ev['hash'], ev['n_admitted'], len(ev['ids'])))
                if len(ev['ids']):
                    self.evict_log[k].append({'step': self.step, 'ids': ev['ids'].numpy(), 'task': ev['task'].numpy(),
                                              'age': ev['age'].numpy(),
                                              'cnt': ev['cnt'].numpy() if 'cnt' in ev else None})
                t0 = time.time()
                for arm, kk in self.bindk.items():
                    if kk == k:
                        self.slows[arm].after_fast(f, ev, S_b, n_b, self.step)
                self.timing['slow_s'] += time.time() - t0
            self.step += 1
        if self.c0 is not None:
            t0 = time.time(); self.c0.boundary(x16, self.l); self.timing['c0_s'] += time.time() - t0
        del x16; torch.cuda.empty_cache()
        t0 = time.time(); self.evaluate(); self.timing['eval_s'] += time.time() - t0
        self.l += 1

    def state(self, arm):
        if arm == 'C0':
            return self.c0.K.float(), self.c0.W(), len(self.c0.K), 0
        f = self.fasts[self.bindk[arm]]
        s = self.slows[arm]
        return f.K.float(), s.W(), len(f.K), s.n_pending()

    @torch.no_grad()
    def evaluate(self):
        l, T = self.l, len(self.order)
        Qs = {j: SY.load(self.order[j]) for j in range(l + 1)}
        for arm in self.arms:
            keys, W, nk, npend = self.state(arm)
            st = {}
            for j in range(l + 1):
                keep = (j == l) or (l == T - 1)
                t0 = time.time()
                r = EV.eval_task(Qs[j], keys, W, self.theta[arm].get(j), keep=keep, cover=(l == T - 1))
                self.timing['infer_img_s'].append((time.time() - t0) / (3 * len(Qs[j]['tokens'])))
                if j == l:
                    self.theta[arm][j] = r['theta_calib']
                    r['fpr'] = float((r['s_cln'] > r['theta_calib']).mean())
                st[j] = r
            self.res[arm][l] = {'tasks': st, 'n_keys': nk, 'n_pending': npend,
                                'bytes': (nk * SP.KEY_B + SP.S_B) if arm == 'C0' else SP.bytes_of(arm, nk, npend)}
        log('CELL', self.tag, 'stage', l, self.order[l], 'step', self.step,
            {a: round(float(np.mean([v['auroc'] for v in self.res[a][l]['tasks'].values()])), 4) for a in self.arms})


class BudgetStop(RuntimeError):
    pass


def run_cell(stream, ok, seed, mib, arms, tag, ntasks=None, phase='dev', ceiling=None):
    G = bind()
    used0 = gpu_used(('dev',))
    p = CACHE / 'ckpt' / f'{tag}.pt'
    done = OUT / 'cells' / f'{tag}.json'
    if done.exists():
        log('SKIP done', tag); return rj(done)
    cell = torch.load(p, weights_only=False) if p.exists() else Cell(stream, ok, seed, mib, arms, tag, ntasks)
    t0 = time.time()
    try:
        while cell.l < len(cell.order):
            if ceiling is not None and used0 + (time.time() - t0) > ceiling:
                raise BudgetStop(f'{tag}: GPU ceiling reached at stage {cell.l}')
            cell.run_task(G)
            cell.save()
    finally:
        ledger_add({'phase': phase, 'cell': tag, 'seconds': time.time() - t0, 'gpu': True, 'stage': cell.l})
    return finish(cell)


def finish(cell):
    """Write the cell summary JSON (scalars) and the array record (K:)."""
    arr = {'res': cell.res, 'theta': cell.theta, 'fast_log': {str(k): v for k, v in cell.fast_log.items()},
           'evict_log': {str(k): v for k, v in cell.evict_log.items()},
           'slow_log': {a: s.log for a, s in cell.slows.items()}, 'causal': cell.causal,
           'c0_log': cell.c0.log if cell.c0 else None,
           'fast_final': {str(k): {'ids': f.ids.numpy(), 'task': f.task.numpy(), 'born': f.born.numpy(),
                                   'cnt': f.cnt.numpy() if f.counts else None} for k, f in cell.fasts.items()},
           'c0_final_task': cell.c0.task.numpy() if cell.c0 else None, 'bind': {a: str(k) for a, k in cell.bindk.items()}}
    ap = CACHE / 'cells' / f'{cell.tag}.pt'
    ap.parent.mkdir(parents=True, exist_ok=True)
    torch.save(arr, ap)
    T = len(cell.order)
    summ = {'tag': cell.tag, 'stream': cell.stream, 'order': cell.ok, 'seed': cell.seed, 'mib': cell.mib,
            'tasks': cell.order, 'tau': cell.tau, 'arms': cell.arms, 'fast_steps': cell.step,
            'caps': CO.caps_table(cell.budget), 'timing': {k: v for k, v in cell.timing.items() if k != 'infer_img_s'},
            'infer_img_ms_p50': float(np.median(cell.timing['infer_img_s']) * 1e3),
            'max_bytes': {a: (cell.c0.max_bytes if a == 'C0' else cell.slows[a].max_bytes) for a in cell.arms},
            'slow_writes': {a: s.writes for a, s in cell.slows.items()},
            'fast_writes': {str(k): f.writes for k, f in cell.fasts.items()},
            'metrics': {a: {str(l): {'n_keys': v['n_keys'], 'n_pending': v['n_pending'], 'bytes': v['bytes'],
                                     'tasks': {str(j): {m: r[m] for m in ('auroc', 'ap', 'fpr', 'theta_calib',
                                                                          'ap_hb', 'cover_median', 'cover_p95')
                                                        if m in r} for j, r in v['tasks'].items()}}
                            for l, v in cell.res[a].items()} for a in cell.arms},
            'array_record': str(ap), 'array_sha256': SY._fsha(ap)}
    wj(OUT / 'cells' / f'{cell.tag}.json', summ)
    log('CELL DONE', cell.tag, 'steps', cell.step)
    return summ


# ------------------------------------------------------------------ phases
def synth(stream):
    from exps.bench_g3 import run as G0
    G = bind(); enc = G0.encoder()
    t0 = time.time()
    for t in orders(stream)['a']:
        m = SY.build_task(G, enc, t)
        log('SYNTH', t, m['n'], round(m['mask_frac_mean'], 4))
    ledger_add({'phase': 'dev_synth', 'stream': stream, 'seconds': time.time() - t0, 'gpu': True})


def smoke():
    rec = run_cell('visa', 'a', 2711, 8, ALL_ARMS, 'smoke_visa_a2711_8_2tasks', ntasks=2, phase='dev_smoke')
    wj(OUT / 'SMOKE.json', {'tag': rec['tag'], 'timing': rec['timing'], 'fast_steps': rec['fast_steps'],
                            'max_bytes': rec['max_bytes'], 'slow_writes': rec['slow_writes'], 'caps': rec['caps']})


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'synth':
        synth(sys.argv[2])
    elif ph == 'smoke':
        smoke()
    else:
        from exps.nl03_multirate_01 import plan as PL
        getattr(PL, ph)(*sys.argv[2:])
