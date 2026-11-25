"""NANM-DYN-01: Gate A (stronger conventional controls) and conditional Gate B (delayed prototype consolidation).

Registration: agents/tasks/task_nanm_dyn_01.md. Normal-only; DYN-00 unit caches reused read-only; nanm/ read-only.
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_dyn_01.policies selftest                 # CPU: operator / allocation checks
    python -B -m exps.nanm_dyn_01.policies prep_seed SEED           # candidates for seeds 2712/2713 (+ timing audit)
    python -B -m exps.nanm_dyn_01.policies run GATE STREAM MIB SEED # GATE in {A, B}
    python -B -m exps.nanm_dyn_01.policies analyze_a | analyze_b
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nanm_dyn_00 import stress as D
from exps.nanm_sys import final as F
from exps.nanm_sys import regen as RG

ROOT = Path(__file__).resolve().parents[2]
CACHE = RG.CACHE.parent / 'nanm_dyn_01'
KEY_B, ID_B, CNT_B = 1536, 2, 4
S_BYTES = D.S_BYTES
K_P, K_B = 4096, 2048


def out():
    o = ROOT / 'results/hope_cad/nanm_dyn_01'
    o.mkdir(parents=True, exist_ok=True)
    return o


# ------------------------------------------------------------------ operators
def wquotas(groups, ub, w, cap):
    """Weighted water-filling: q_g = min(ub_g, floor(lam*w_g)), largest lam that fits; leftovers +1 in arrival order."""
    if sum(ub[g] for g in groups) <= cap:
        return {g: ub[g] for g in groups}
    lo, hi = 0.0, max(ub[g] / w[g] for g in groups) + 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(min(ub[g], math.floor(mid * w[g])) for g in groups) <= cap else (lo, mid)
    q = {g: min(ub[g], math.floor(lo * w[g])) for g in groups}
    left = cap - sum(q.values())
    for g in groups:
        if left <= 0:
            break
        if q[g] < ub[g]:
            q[g] += 1; left -= 1
    return q


@torch.no_grad()
def _dist(a, b):
    a, b = a.float(), b.float()
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1)[None] - 2 * a @ b.T).clamp_min(0).sqrt()


@torch.no_grad()
def medoid(X16, cnt, q, dev='cuda', col=1024):
    """Weighted greedy-BUILD medoids (registered operator): returns (selected row indices, merged counts).
    Greedily reduces sum_i c_i * min_m d(i, m); no claim about quantiles. Ties -> lowest index."""
    n = len(X16)
    if q >= n:
        return torch.arange(n), cnt.clone()
    X = X16.to(dev).float(); c = cnt.to(dev).double()
    Dm = torch.empty(n, n, device=dev)
    for i in range(0, n, col):
        Dm[i:i + col] = _dist(X[i:i + col], X)
    Dm = Dm.double()
    j = int(torch.argmin((c[:, None] * Dm).sum(0)))
    cur = Dm[:, j].clone(); sel = [j]; used = torch.zeros(n, dtype=torch.bool, device=dev); used[j] = True
    for _ in range(q - 1):
        gain = torch.empty(n, dtype=torch.float64, device=dev)
        for i in range(0, n, col):
            gain[i:i + col] = (c[:, None] * (cur[:, None] - Dm[:, i:i + col]).clamp_min(0)).sum(0)
        gain[used] = -1.0
        j = int(torch.argmax(gain)); sel.append(j); used[j] = True
        cur = torch.minimum(cur, Dm[:, j])
    s = torch.tensor(sel, device=dev)
    assign = Dm[:, s].argmin(1)
    newc = torch.zeros(q, dtype=torch.float64, device=dev).index_add_(0, assign, c)
    return s.cpu(), newc.round().long().cpu()


# ------------------------------------------------------------------ units
def load_unit(u, seed):
    if seed == 2711:
        d = D.load_unit(u)
        return {'keys': d['keys'], 'S': d['S'], 'n': d['n']}
    dd, c = u.split('/')
    return torch.load(CACHE / f'units_s{seed}' / f'{dd}_{c}.pt', weights_only=True)


@torch.no_grad()
def prep_seed(seed):
    from nanm import G3Config
    from nanm.geometry import residual_moment
    from nanm.memory import BoundaryExemplarWriter, greedy_kcenter, quota
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    wr = BoundaryExemplarWriter(G3Config(), torch.device('cuda'))
    d = CACHE / f'units_s{seed}'; d.mkdir(parents=True, exist_ok=True)
    o, man = D.orders(G), {}
    for ds in ('mvtec', 'visa', 'mpdd', 'dtd'):
        for c in o[ds]:
            _, feats, roles = D.train_tokens(G, ds, c)
            tok = feats[torch.as_tensor(list(roles['FIT']))]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fit16 = wr.prepare(tok); k = quota(0.01, len(fit16))
            torch.cuda.synchronize(); t1 = time.perf_counter()
            kc = greedy_kcenter(fit16, k, seed, 128); torch.cuda.synchronize(); t2 = time.perf_counter()
            S, n = residual_moment(fit16.float(), kc.cuda(), 2048); torch.cuda.synchronize(); t3 = time.perf_counter()
            p = d / f'{ds}_{c}.pt'
            torch.save({'keys': fit16[kc.cuda()].cpu(), 'S': S, 'n': n}, p)
            man[f'{ds}/{c}'] = {'sha256': G.A.sha(p), 'k': k, 'n_tokens': len(fit16), 'prepare_s': t1 - t0,
                                'kcentre_s': t2 - t1, 'residual_s': t3 - t2}
            del fit16
        G.A.log('PREP', seed, ds)
    G.A.wj(out() / f'UNITS_s{seed}.json', {'units': man, 'device': dev})


# ------------------------------------------------------------------ arms
class QuotaArm:
    """Bounded exemplar memory with weighted allocation; prefix trimming or weighted-medoid compression."""

    def __init__(self, name, cap, weight, compress=False, merge=False, seed=2711, has_S=True):
        self.name, self.cap, self.weight, self.compress, self.merge, self.seed = name, cap, weight, compress, merge, seed
        self.has_S = has_S                                       # C1 (Euclidean) holds no residual statistic
        self.groups, self.keep, self.cnt, self.w = [], {}, {}, {}
        self.removed = 0; self.policy_s = 0.0; self.medoid_calls = 0

    def boundary(self, unit, d):
        from nanm.memory import greedy_kcenter
        t0 = time.perf_counter()
        g = D.task_of(unit) if self.merge else unit
        new_keys = d['keys']
        if g in self.keep:                                       # recurrence-aware merge (C3m only)
            pool = torch.cat([self.keep[g], new_keys])
            order = greedy_kcenter(pool.cuda(), len(pool), self.seed, 128)
            self.keep[g] = pool[order]; self.cnt[g] = torch.ones(len(pool), dtype=torch.long)
            ub_override = len(pool)
        else:
            self.groups.append(g); self.keep[g] = new_keys; self.cnt[g] = torch.ones(len(new_keys), dtype=torch.long)
            ub_override = None
        N_u = d['n'] + len(new_keys)
        self.w[g] = self.weight(N_u, d)
        ub = {h: len(self.keep[h]) for h in self.groups}
        if ub_override is not None:
            ub[g] = ub_override
        q = wquotas(self.groups, ub, self.w, self.cap)
        trims = []
        for h in self.groups:
            if q[h] < len(self.keep[h]):
                self.removed += len(self.keep[h]) - q[h]
                if self.compress:
                    idx, c = medoid(self.keep[h], self.cnt[h], q[h]); self.medoid_calls += 1
                    self.keep[h], self.cnt[h] = self.keep[h][idx], c
                else:
                    trims.append(self.keep[h][q[h]:])
                    self.keep[h], self.cnt[h] = self.keep[h][:q[h]], self.cnt[h][:q[h]]
        torch.cuda.synchronize(); self.policy_s += time.perf_counter() - t0
        return trims

    def bank(self):
        return torch.cat([self.keep[h] for h in self.groups if len(self.keep[h])])

    def nkeys(self):
        return sum(len(v) for v in self.keep.values())

    def bytes(self):
        return self.nkeys() * (KEY_B + ID_B + (CNT_B if self.compress else 0)) + (S_BYTES if self.has_S else 0)


class DynArm:
    """Fast bounded exemplars + slow retrievable prototypes consolidated from a counted buffer of trimmed keys."""

    def __init__(self, name, capF, weight, m, mode, n_units, seed=2711):
        self.name, self.m, self.mode, self.T = name, m, mode, n_units
        self.fast = QuotaArm(name + '/F', capF, weight, seed=seed)
        self.P = torch.empty(0, 768, dtype=torch.float16); self.Pc = torch.empty(0, dtype=torch.long)
        self.B = torch.empty(0, 768, dtype=torch.float16); self.Bc = torch.empty(0, dtype=torch.long)
        self.t = -1; self.events = []; self.overflow_dropped = 0; self.cons_s = 0.0

    @property
    def policy_s(self):
        return self.fast.policy_s + self.cons_s

    def boundary(self, unit, d):
        self.t += 1
        trims = self.fast.boundary(unit, d)
        if self.mode == 'disc':
            return
        if trims:
            tk = torch.cat(trims)
            self.B = torch.cat([self.B, tk]); self.Bc = torch.cat([self.Bc, torch.ones(len(tk), dtype=torch.long)])
        if len(self.B) > K_B:                                    # never an early consolidation: drop the oldest
            self.overflow_dropped += len(self.B) - K_B
            self.B, self.Bc = self.B[-K_B:], self.Bc[-K_B:]
        if (self.t + 1) % self.m == 0 or self.t == self.T - 1:
            t0 = time.perf_counter()
            U = torch.cat([self.P, self.B]); Uc = torch.cat([self.Pc, self.Bc])
            if len(U) <= K_P:
                self.P, self.Pc = U, Uc
            else:
                idx, c = medoid(U, Uc, K_P)
                self.P, self.Pc = U[idx], c
            torch.cuda.synchronize(); dt = time.perf_counter() - t0; self.cons_s += dt
            self.events.append({'stage': self.t, 'union': len(U), 'P': len(self.P), 'seconds': dt})
            self.B = torch.empty(0, 768, dtype=torch.float16); self.Bc = torch.empty(0, dtype=torch.long)

    def bank(self):
        return torch.cat([self.fast.bank(), self.P]) if len(self.P) else self.fast.bank()

    def nkeys(self):
        return self.fast.nkeys() + len(self.P)

    def bytes(self):
        return self.fast.nkeys() * (KEY_B + ID_B) + (len(self.P) + len(self.B)) * (KEY_B + CNT_B) + S_BYTES

    @staticmethod
    def reserved(budget):
        return budget


W_EQ = lambda N, d: 1.0
W_SQRT = lambda N, d: math.sqrt(N)
W_CPLX = lambda N, d: math.sqrt(N * float(torch.trace(d['S'])) / d['n'])
WEIGHTS = {'C3': W_EQ, 'C4': W_SQRT, 'C5': W_CPLX}


def caps(budget):
    return {'C1': budget // (KEY_B + ID_B), 'Cq': (budget - S_BYTES) // (KEY_B + ID_B),
            'C6': (budget - S_BYTES) // (KEY_B + ID_B + CNT_B),
            'F': (budget - S_BYTES - (K_P + K_B) * (KEY_B + CNT_B)) // (KEY_B + ID_B)}


def build_arms(gate, stream, budget, seed, n_units):
    c = caps(budget)
    if gate == 'A':
        arms = {'C1': QuotaArm('C1', c['C1'], W_EQ, seed=seed, has_S=False), 'C3': QuotaArm('C3', c['Cq'], W_EQ, seed=seed),
                'C4': QuotaArm('C4', c['Cq'], W_SQRT, seed=seed), 'C5': QuotaArm('C5', c['Cq'], W_CPLX, seed=seed),
                'C6': QuotaArm('C6', c['C6'], W_EQ, compress=True, seed=seed)}
        if stream == 'REC':
            arms['C3m'] = QuotaArm('C3m', c['Cq'], W_EQ, merge=True, seed=seed)
        return arms
    ga = json.loads((out() / 'GATE_A.json').read_text())
    best, fast_rule = ga['best_conventional'], ga['best_quota_rule']
    arms = {'BEST': (QuotaArm('BEST', c['C6'], W_EQ, compress=True, seed=seed) if best == 'C6'
                     else QuotaArm('BEST', c['Cq'], WEIGHTS[best], seed=seed)),
            'D4': DynArm('D4', c['F'], WEIGHTS[fast_rule], 4, 'dyn', n_units, seed),
            'D1': DynArm('D1', c['F'], WEIGHTS[fast_rule], 1, 'dyn', n_units, seed),
            'DISC': DynArm('DISC', c['F'], WEIGHTS[fast_rule], 4, 'disc', n_units, seed)}
    if best != 'C6':
        arms['C6'] = QuotaArm('C6', c['C6'], W_EQ, compress=True, seed=seed)
    return arms


# ------------------------------------------------------------------ run
@torch.no_grad()
def score(queries, bank16, W, chunk=4096):
    m = bank16.cuda().float()
    if W is not None:
        m = m @ W
    m2 = m.pow(2).sum(1)
    res = {}
    for tk, q16 in queries.items():
        e = torch.empty(len(q16))
        for i in range(0, len(q16), chunk):
            q = q16[i:i + chunk].cuda().float()
            if W is not None:
                q = q @ W
            e[i:i + chunk] = (0.5 * (q.pow(2).sum(1, keepdim=True) + m2[None] - 2 * q @ m.T).clamp_min(0)
                              .min(1).values).cpu()
        res[tk] = [float(torch.quantile(e, .95)), float(torch.quantile(e, .99))]
    return res


def run(gate, stream, mib, seed):
    from nanm.geometry import whiten
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    units = D.streams(G)[stream]
    budget = mib * 2**20
    arms = build_arms(gate, stream, budget, seed, len(units))
    queries, keys0, S, n, log, t_all = {}, [], 0, 0, [], time.monotonic()
    for t, u in enumerate(units):
        d = load_unit(u, seed)
        tk = D.task_of(u)
        if tk not in queries:
            dsn, c = tk.split('/')
            queries[tk] = torch.load(D.CACHE / 'units' / f'{dsn}_{c}_eval.pt', weights_only=True)
        S = S + d['S']; n = n + d['n']
        W = whiten(S, n, 0.5, 1e-12)[0].float().cuda()
        keys0.append(d['keys'])
        rec = {'stage': t, 'unit': u, 'stats': {}, 'bytes': {}, 'keys': {}, 'removed_cum': {}}
        banks = {'C0': torch.cat(keys0)}
        rec['bytes']['C0'] = len(banks['C0']) * KEY_B + S_BYTES
        for name, a in arms.items():
            a.boundary(u, d)
            banks[name] = a.bank(); rec['bytes'][name] = a.bytes(); rec['keys'][name] = a.nkeys()
            rec['removed_cum'][name] = (a.fast.removed if isinstance(a, DynArm) else a.removed)
            G.K.require(a.bytes() <= budget, f'{name} exceeds budget')
        for name, bk in banks.items():
            for g, Wg in (('E', None), ('W', W)):
                rec['stats'][f'{name}|{g}'] = score(queries, bk, Wg)
        log.append(rec)
        G.A.log('STAGE', gate, stream, mib, seed, t, u, round(time.monotonic() - t_all))
    meta = {}
    for name, a in arms.items():
        m = {'policy_s': a.policy_s, 'final_keys': a.nkeys(), 'final_bytes': a.bytes()}
        if isinstance(a, DynArm):
            m.update({'events': a.events, 'overflow_dropped': a.overflow_dropped, 'fast_cap': a.fast.cap,
                      'consolidation_s': a.cons_s, 'P_final': len(a.P), 'clock_m': a.m, 'mode': a.mode})
        else:
            m.update({'cap': a.cap, 'medoid_calls': a.medoid_calls})
        meta[name] = m
    p = out() / 'runs' / f'{gate}_{stream}_{mib}_s{seed}.json'
    G.A.wj(p, {'gate': gate, 'stream': stream, 'mib': mib, 'seed': seed, 'units': units, 'budget': budget,
               'caps': caps(budget), 'arms': meta, 'stages': log, 'seconds': time.monotonic() - t_all,
               'device': dev, 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


# ------------------------------------------------------------------ analysis
def summarise(path):
    r = json.loads(Path(path).read_text())
    st, units = r['stages'], r['units']
    cur = D.task_of(units[-1])
    arms = [a for a in st[-1]['bytes'] if a != 'C0']
    res = {}
    for a in arms:
        for g in ('W', 'E'):
            fin = st[-1]['stats']
            L = {tk: math.log(fin[f'{a}|{g}'][tk][0]) - math.log(fin[f'C0|{g}'][tk][0]) for tk in fin[f'C0|{g}']}
            hist = {k: v for k, v in L.items() if k != cur}
            acq = [math.log(s['stats'][f'{a}|{g}'][D.task_of(s['unit'])][0]) -
                   math.log(s['stats'][f'C0|{g}'][D.task_of(s['unit'])][0]) for s in st]
            res[f'{a}|{g}'] = {'HL': float(np.mean(list(hist.values()))), 'AG': float(np.mean(acq)),
                               'WT': max(hist.items(), key=lambda kv: kv[1]), 'per_task': hist,
                               'n_ge_0.05': sum(v >= 0.05 for v in hist.values())}
    return r, res


def analyze_a():
    G, _ = RG.setup()
    out_ = {}
    for p in sorted((out() / 'runs').glob('A_*.json')):
        r, s = summarise(p)
        out_[p.stem] = {'summary': {k: {x: v[x] for x in ('HL', 'AG', 'WT', 'n_ge_0.05')} for k, v in s.items()},
                        'arms': r['arms'], 'seconds': r['seconds'], 'per_task_W': {k: v['per_task'] for k, v in s.items()
                                                                                  if k.endswith('|W')}}
    m, mr = out_['A_MIX_32_s2711']['summary'], out_['A_MIX_REV_32_s2711']['summary']
    cand = ['C3', 'C4', 'C5', 'C6']
    best = min(cand, key=lambda a: m[f'{a}|W']['HL'])
    best_quota = min(['C3', 'C4', 'C5'], key=lambda a: m[f'{a}|W']['HL'])
    hb, hbr = m[f'{best}|W']['HL'], mr[f'{best}|W']['HL']
    if hb < 0.02 and hbr < 0.02:
        dec = 'DYN_CONVENTIONAL_SUFFICIENT'
    elif hb < 0.03 or hbr < 0.03:
        dec = 'DYN_CONVENTIONAL_SUFFICIENT (headroom < 0.03: registered primary contrast not attainable)'
    else:
        dec = 'RUN_GATE_B'
    res = {'decision': dec, 'best_conventional': best, 'best_quota_rule': best_quota,
           'HL_best_MIX': hb, 'HL_best_MIX_REV': hbr, 'runs': out_}
    G.A.wj(out() / 'GATE_A.json', res)
    print(dec, 'best', best, 'quota', best_quota, round(hb, 4), round(hbr, 4))
    for k, v in out_.items():
        print(k, {a: (round(s['HL'], 4), round(s['AG'], 4), round(s['WT'][1], 3)) for a, s in v['summary'].items()
                  if a.endswith('|W')})


def analyze_b():
    G, _ = RG.setup()
    rng = np.random.default_rng(20261120)
    runs = {}
    for p in sorted((out() / 'runs').glob('B_*.json')):
        r, s = summarise(p)
        con = {}
        for x, y in (('BEST', 'D4'), ('D1', 'D4'), ('DISC', 'D4'), ('C6', 'D4')):
            if f'{x}|W' not in s:
                continue
            tasks = sorted(s['D4|W']['per_task'])
            diff = np.array([s[f'{x}|W']['per_task'][k] - s['D4|W']['per_task'][k] for k in tasks])
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
            con[f'{x}-D4'] = {'delta_HL': float(diff.mean()), 'ci95_task_bootstrap': [float(np.quantile(bs, .025)),
                                                                                      float(np.quantile(bs, .975))]}
        runs[p.stem] = {'summary': {k: {x: v[x] for x in ('HL', 'AG', 'WT', 'n_ge_0.05')} for k, v in s.items()},
                        'contrasts': con, 'arms': r['arms'], 'seconds': r['seconds'],
                        'per_task_W': {k: v['per_task'] for k, v in s.items() if k.endswith('|W')}}
    g = {}
    for st in ('MIX', 'MIX_REV'):
        k = f'B_{st}_32_s2711'
        if k not in runs:
            continue
        sm, cn = runs[k]['summary'], runs[k]['contrasts']
        g[st] = {'primary': cn['BEST-D4']['delta_HL'] >= 0.03, 'nl_specific': cn['D1-D4']['delta_HL'] >= 0.02,
                 'worst_task': sm['D4|W']['WT'][1] <= sm['BEST|W']['WT'][1] + 0.05}
    seeds = {}
    for sd in (2712, 2713):
        for st in ('MIX', 'MIX_REV'):
            k = f'B_{st}_32_s{sd}'
            if k in runs:
                cn = runs[k]['contrasts']
                seeds[k] = {'BEST-D4': cn['BEST-D4']['delta_HL'], 'D1-D4': cn['D1-D4']['delta_HL']}
    prim = all(v['primary'] for v in g.values()) and len(g) == 2
    nl = all(v['nl_specific'] for v in g.values()) and len(g) == 2
    wt = all(v['worst_task'] for v in g.values())
    sd_prim = all(v['BEST-D4'] > 0 for v in seeds.values()) and len(seeds) == 4
    sd_nl = all(v['D1-D4'] > 0 for v in seeds.values()) and len(seeds) == 4
    if prim and nl and wt and sd_prim and sd_nl:
        verdict = 'DYN_CLOCK_BENEFIT_SUPPORTED_NORMAL_ONLY'
    elif prim and wt and sd_prim:
        verdict = 'DYN_COMPRESSION_ONLY'
    else:
        verdict = 'DYN_NOT_SUPPORTED'
    G.A.wj(out() / 'GATE_B.json', {'verdict': verdict, 'gates': g, 'seed_signs': seeds, 'runs': runs})
    print(verdict, g, seeds)
    for k, v in runs.items():
        print(k, {a: (round(s['HL'], 4), round(s['WT'][1], 3)) for a, s in v['summary'].items() if a.endswith('|W')},
              {c: round(x['delta_HL'], 4) for c, x in v['contrasts'].items()})


def selftest():
    q = wquotas(['a', 'b', 'c'], {'a': 100, 'b': 10, 'c': 100}, {'a': 1, 'b': 1, 'c': 1}, 120)
    assert q == {'a': 55, 'b': 10, 'c': 55}, q
    q = wquotas(['a', 'b'], {'a': 1000, 'b': 1000}, {'a': 1.0, 'b': 3.0}, 400)
    assert q == {'a': 100, 'b': 300}, q
    g = torch.Generator().manual_seed(0)
    X = torch.nn.functional.normalize(torch.randn(300, 768, generator=g), dim=1).half()
    c = torch.ones(300, dtype=torch.long)
    i1, c1 = medoid(X, c, 40, dev='cpu'); i2, c2 = medoid(X, c, 40, dev='cpu')
    assert torch.equal(i1, i2) and int(c1.sum()) == 300 and len(set(i1.tolist())) == 40
    # brute-force check of the first greedy step
    Dm = _dist(X, X).double()
    assert int(i1[0]) == int(torch.argmin(Dm.sum(0)))
    print('selftest ok')


if __name__ == '__main__':
    ph, *a = sys.argv[1:]
    if ph == 'run':
        run(a[0], a[1], int(a[2]), int(a[3]))
    elif ph == 'prep_seed':
        prep_seed(int(a[0]))
    else:
        globals()[ph]()
