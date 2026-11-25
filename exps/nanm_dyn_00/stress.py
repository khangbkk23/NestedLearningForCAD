"""NANM-DYN-00: normal-only bounded-memory stress test (registration agents/tasks/task_nanm_dyn_00.md).

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
    python -B -m exps.nanm_dyn_00.stress validate_c2          # CPU: incremental CADIC rule == local reference
    python -B -m exps.nanm_dyn_00.stress extract_mvtec        # nanm encoder, MVTec train/good tokens
    python -B -m exps.nanm_dyn_00.stress prep                 # per-unit candidates, S_u, EVAL queries, FIT fp16
    python -B -m exps.nanm_dyn_00.stress run STREAM MIB       # arms C0/C1/C2/C3, all stages, E and W geometry
    python -B -m exps.nanm_dyn_00.stress analyze              # registered gates and verdict
No anomaly label, mask or test image is read. nanm/ is used read-only.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from exps.nanm_sys import final as F
from exps.nanm_sys import regen as RG

ROOT = Path(__file__).resolve().parents[2]
CACHE = RG.CACHE.parent / 'nanm_dyn_00'
RD_CACHE = RG.CACHE.parent / 'nanm_rd_01'
ORDERS = None
N_EVAL, SEED, KEY_B, TID_B = 8, 2711, 1536, 2
S_BYTES = 768 * 768 * 8 + 8


def out():
    o = ROOT / 'results/hope_cad/nanm_dyn_00'
    o.mkdir(parents=True, exist_ok=True)
    return o


def orders(G):
    from exps.calm02 import data as CD
    return {'mvtec': tuple(CD.MVTEC_ORDER), 'visa': G.order('visa'), 'mpdd': G.MPDD_ORDER,
            'dtd': tuple(json.loads((ROOT / 'results/hope_cad/nanm_rd_01/rd01b/DATA_AUDIT.json').read_text())['order'])}


def streams(G):
    o = orders(G)
    visa = [f'visa/{c}' for c in o['visa']]
    lists = [[f'{d}/{c}' for c in o[d]] for d in ('mvtec', 'visa', 'mpdd', 'dtd')]
    mix = [x for i in range(max(map(len, lists))) for l in lists if i < len(l) for x in [l[i]]]
    rec = [f'visaA/{c}' for c in o['visa']] + [f'visaB/{c}' for c in o['visa']]
    return {'VISA': visa, 'VISA_REV': visa[::-1], 'MIX': mix, 'MIX_REV': mix[::-1], 'REC': rec}


def task_of(unit):
    d, c = unit.split('/')
    return f'{d.rstrip("AB") if d.startswith("visa") else d}/{c}'


# ------------------------------------------------------------------ data
def extract_mvtec():
    from exps.calm02 import data as CD
    from exps.nar_pilot_00 import full as R
    from nanm import FrozenPatchEncoder
    F.guard(); G, _ = RG.setup(); R.configure()
    B14 = G.B14()
    enc = FrozenPatchEncoder.load(B14.ENC_PATH, B14.ENC_SHA, 'cuda')
    (CACHE / 'features').mkdir(parents=True, exist_ok=True)
    man = {}
    for c in CD.MVTEC_ORDER:
        tr, _ = CD._mvtec(ROOT / 'data/mvtec', c)                       # train/good identities only are used
        tok = enc.tokens([ROOT / 'data/mvtec' / r['relative_path'] for r in tr], batch=8)
        p = CACHE / 'features' / f'mvtec_{c}_train.pt'
        torch.save({'rows': [r['relative_path'] for r in tr], 'features': tok}, p)
        man[p.name] = G.A.sha(p)
        G.A.log('MVTEC', c, len(tr))
    G.A.wj(out() / 'MVTEC_FEATURES.json', man)


def train_tokens(G, ds, c):
    """(row paths, fp32 tokens [n,784,768], roles) from the sha-verified caches."""
    if ds in ('visa', 'mpdd'):
        d = G.load_train(ds, c)
        paths = [r['relative_path'] for r in d['rows']]
        return paths, d['features'], G.roles(ds, d['rows'])
    p = (CACHE / 'features' / f'mvtec_{c}_train.pt') if ds == 'mvtec' else (RD_CACHE / 'features' / f'{c}_train.pt')
    want = (G.A.rj(out() / 'MVTEC_FEATURES.json') if ds == 'mvtec' else
            G.A.rj(ROOT / 'results/hope_cad/nanm_rd_01/rd01b/FEATURES.json')['files'])[p.name]
    G.K.require(G.A.sha(p) == want, f'cache {p.name}')
    d = torch.load(p, weights_only=True, mmap=True)
    return d['rows'], d['features'], G.K.normal_partition([{'relative_path': r} for r in d['rows']])


@torch.no_grad()
def prep():
    from nanm import G3Config
    from nanm.geometry import residual_moment
    from nanm.memory import BoundaryExemplarWriter, greedy_kcenter, quota, unit
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    writer = BoundaryExemplarWriter(G3Config(), torch.device('cuda'))
    (CACHE / 'units').mkdir(parents=True, exist_ok=True)
    man, o = {}, orders(G)
    for ds in ('mvtec', 'visa', 'mpdd', 'dtd'):
        for c in o[ds]:
            paths, feats, roles = train_tokens(G, ds, c)
            fit = list(roles['FIT'])
            q = unit(feats[torch.as_tensor(roles['EVAL'][:N_EVAL])].reshape(-1, 768)).half()
            torch.save(q, CACHE / 'units' / f'{ds}_{c}_eval.pt')
            parts = {ds: fit} if ds != 'visa' else {'visa': fit, 'visaA': fit[:len(fit) // 2],
                                                    'visaB': fit[len(fit) // 2:]}
            for name, idx in parts.items():
                fit16 = writer.prepare(feats[torch.as_tensor(idx)])
                k = quota(0.01, len(fit16))
                kc = greedy_kcenter(fit16, k, SEED, 128)
                S, n = residual_moment(fit16.float(), kc.cuda(), 2048)
                p = CACHE / 'units' / f'{name}_{c}.pt'
                torch.save({'keys': fit16[kc.cuda()].cpu(), 'S': S, 'n': n, 'fit16': fit16.cpu(),
                            'fit_paths': [paths[i] for i in idx]}, p)
                man[f'{name}/{c}'] = {'sha256': G.A.sha(p), 'k': k, 'n_fit_images': len(idx), 'n_res': n}
                del fit16
            G.A.log('PREP', ds, c)
    G.A.wj(out() / 'UNITS.json', {'units': man, 'device': dev})


def load_unit(unit):
    d, c = unit.split('/')
    return torch.load(CACHE / 'units' / f'{d}_{c}.pt', weights_only=True)


# ------------------------------------------------------------------ C2: CADIC rule with incremental bookkeeping
class CadicIncremental:
    """CADIC eq. (1)-(7) (local reference: models/cadic_paperfaithful.CADICPatchCoresetV1.update): fill first-come;
    then repeatedly take the batch row farthest from C and, while d_max > |C|_min, replace the i-member of the
    closest pair (i < j). Bookkeeping (per-element nearest-other distance) replaces O(N^2) rescans."""

    def __init__(self, cap, device, dist=None):
        from models.cadic_paperfaithful.cadic_patch_coreset import euclidean_distance_mm
        self.cap, self.dev, self.dist = cap, device, dist or euclidean_distance_mm
        self.C = torch.empty(0, 768, device=device)
        self.nn_d = self.nn_i = None
        self.replaced = 0

    def _rows(self, X, chunk=4096):
        d = torch.empty(len(X), len(self.C), device=self.dev)
        for i in range(0, len(X), chunk):
            d[i:i + chunk] = self.dist(X[i:i + chunk], self.C)
        return d

    def _init_nn(self):
        n = len(self.C)
        self.nn_d = torch.empty(n, device=self.dev); self.nn_i = torch.empty(n, dtype=torch.long, device=self.dev)
        for i in range(0, n, 2048):
            d = self.dist(self.C[i:i + 2048], self.C)
            d[torch.arange(len(d)), torch.arange(i, i + len(d))] = float('inf')
            self.nn_d[i:i + 2048], self.nn_i[i:i + 2048] = d.min(1)

    def _closest(self):
        v = self.nn_d.min()
        rows = torch.nonzero(self.nn_d == v).flatten()                       # lexicographic first pair (i<j)
        pairs = torch.stack([torch.minimum(rows, self.nn_i[rows]), torch.maximum(rows, self.nn_i[rows])], 1)
        key = pairs[:, 0] * (len(self.C) + 1) + pairs[:, 1]
        return v, int(pairs[int(key.argmin()), 0])

    @torch.no_grad()
    def update(self, x):
        x = x.to(self.dev, torch.float32)
        if len(self.C) < self.cap:
            take = min(self.cap - len(self.C), len(x))
            self.C = torch.cat([self.C, x[:take]]); x = x[take:]
            if len(self.C) == self.cap:
                self._init_nn()
        if not len(x) or len(self.C) < self.cap:
            return 0
        xd = self._rows(x); xn_d, xn_i = xd.min(1)
        alive = torch.ones(len(x), dtype=torch.bool, device=self.dev)
        rep = 0
        while bool(alive.any()):
            vals = torch.where(alive, xn_d, torch.full_like(xn_d, -float('inf')))
            fv, fr = vals.max(0)
            cv, c1 = self._closest()
            if not bool(fv > cv):
                break
            fr = int(fr)
            self.C[c1] = x[fr]; alive[fr] = False; rep += 1
            dn = self.dist(self.C[c1:c1 + 1], self.C)[0]; dn[c1] = float('inf')
            stale = torch.nonzero(self.nn_i == c1).flatten()
            better = dn < self.nn_d
            self.nn_d = torch.where(better, dn, self.nn_d); self.nn_i = torch.where(better, torch.full_like(self.nn_i, c1),
                                                                                    self.nn_i)
            self.nn_d[c1], self.nn_i[c1] = dn.min(0)
            if len(stale):
                d = self.dist(self.C[stale], self.C); d[torch.arange(len(stale)), stale] = float('inf')
                self.nn_d[stale], self.nn_i[stale] = d.min(1)
            dx = self.dist(x, self.C[c1:c1 + 1])[:, 0]
            st = torch.nonzero((xn_i == c1) & alive).flatten()
            upd = dx < xn_d
            xn_d = torch.where(upd, dx, xn_d); xn_i = torch.where(upd, torch.full_like(xn_i, c1), xn_i)
            if len(st):
                d = self.dist(x[st], self.C); xn_d[st], xn_i[st] = d.min(1)
        self.replaced += rep
        return rep


def validate_c2():
    from models.cadic_paperfaithful.cadic_patch_coreset import CADICPatchCoresetConfig, CADICPatchCoresetV1
    torch.manual_seed(0)
    res = []
    for trial in range(3):
        cap = 64 + 32 * trial
        ref = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=cap), device='cpu')
        inc = CadicIncremental(cap, torch.device('cpu'))
        ok = True
        for b in range(12):
            x = torch.nn.functional.normalize(torch.randn(40, 768) + 0.5 * (b % 3), dim=1).half().float()
            r1 = ref.update(x)['replaced']; r2 = inc.update(x)
            ok &= (r1 == r2) and torch.equal(ref.features, inc.C)
        res.append({'cap': cap, 'identical_state_and_counts': bool(ok), 'replacements': inc.replaced})
    G, _ = RG.setup()
    G.A.wj(out() / 'C2_VALIDATION.json', res)
    print(res)
    assert all(r['identical_state_and_counts'] for r in res)


# ------------------------------------------------------------------ bounded quota policy (C1 / C3)
def quotas(avail, prev, cap):
    """Water-filling equal share, non-increasing per unit (trimmed keys are gone), leftover to earlier units."""
    ub = [min(a, prev.get(u, a)) for u, a in avail]
    lo, hi = 0, max(ub)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if sum(min(b, mid) for b in ub) <= cap else (lo, mid - 1)
    q = [min(b, lo) for b in ub]
    left = cap - sum(q)
    for i, b in enumerate(ub):
        if left <= 0:
            break
        if q[i] < b:
            q[i] += 1; left -= 1
    return {u: qi for (u, _), qi in zip(avail, q)}


# ------------------------------------------------------------------ scoring
@torch.no_grad()
def energies(q16, bank16, W=None, chunk=4096):
    m = bank16.cuda().float()
    if W is not None:
        m = m @ W
    m2 = m.pow(2).sum(1)
    out = torch.empty(len(q16))
    for i in range(0, len(q16), chunk):
        q = q16[i:i + chunk].cuda().float()
        if W is not None:
            q = q @ W
        d = (q.pow(2).sum(1, keepdim=True) + m2[None] - 2 * q @ m.T).clamp_min(0)
        out[i:i + chunk] = (0.5 * d.min(1).values).cpu()
    return out


def run(stream, mib):
    from nanm.geometry import whiten
    dev = F.guard(); RG.numerics(); G, _ = RG.setup()
    units = streams(G)[stream]
    budget = mib * 2**20
    cap = {'C1': budget // (KEY_B + TID_B), 'C2': budget // KEY_B, 'C3': (budget - S_BYTES) // (KEY_B + TID_B)}
    do_c2 = stream in ('VISA', 'MIX') or '--c2' in sys.argv
    queries = {}
    data = {}
    c2 = CadicIncremental(cap['C2'], torch.device('cuda')) if do_c2 else None
    q_prev = {'C1': {}, 'C3': {}}
    banks_prev = {}
    S, n = 0, 0
    log, t_all = [], time.monotonic()
    for t, u in enumerate(units):
        rec = {'stage': t, 'unit': u}
        d = load_unit(u); data[u] = {'keys': d['keys']}
        tk = task_of(u)
        if tk not in queries:
            dsn, c = tk.split('/')
            queries[tk] = torch.load(CACHE / 'units' / f'{dsn}_{c}_eval.pt', weights_only=True)
        S = S + d['S']; n = n + d['n']
        W64, cond = whiten(S, n, 0.5, 1e-12); W = W64.float().cuda()
        banks = {'C0': torch.cat([data[v]['keys'] for v in units[:t + 1]])}
        for arm in ('C1', 'C3'):
            t0 = time.perf_counter()
            q = quotas([(v, len(data[v]['keys'])) for v in units[:t + 1]], q_prev[arm], cap[arm])
            banks[arm] = torch.cat([data[v]['keys'][:q[v]] for v in units[:t + 1] if q[v] > 0])
            removed = sum(q_prev[arm].get(v, 0) - q[v] for v in q_prev[arm])
            rec[f'{arm}_update_s'] = time.perf_counter() - t0
            rec[f'{arm}_turnover'] = removed / cap[arm]
            rec[f'{arm}_utilisation'] = len(banks[arm]) / cap[arm]
            q_prev[arm] = q
        if c2 is not None:
            t0 = time.perf_counter(); r0 = c2.replaced
            f16 = d['fit16']
            for i in range(0, len(f16), 784):
                c2.update(f16[i:i + 784])
            torch.cuda.synchronize()
            rec['C2_update_s'] = time.perf_counter() - t0
            rec['C2_turnover'] = (c2.replaced - r0) / cap['C2']
            rec['C2_utilisation'] = len(c2.C) / cap['C2']
            banks['C2'] = c2.C.half().cpu()
        rec['bank_keys'] = {a: len(b) for a, b in banks.items()}
        rec['bytes'] = {'C0': len(banks['C0']) * KEY_B + S_BYTES, 'C1': len(banks['C1']) * (KEY_B + TID_B),
                        'C3': len(banks['C3']) * (KEY_B + TID_B) + S_BYTES}
        if 'C2' in banks:
            rec['bytes']['C2'] = len(banks['C2']) * KEY_B
        stats = {}
        for arm, bank in banks.items():
            for g, Wg in (('E', None), ('W', W)):
                for tk2, qq in queries.items():
                    e = energies(qq, bank, Wg)
                    stats[f'{arm}|{g}|{tk2}'] = [float(torch.quantile(e, .95)), float(torch.quantile(e, .99))]
        rec['p95_p99'] = stats
        rec['cond'] = cond
        log.append(rec)
        G.A.log('STAGE', stream, mib, t, u, {a: len(b) for a, b in banks.items()}, round(time.monotonic() - t_all))
        del d
    final = {}
    from safetensors.torch import save as st_save
    for arm in ('C0', 'C1', 'C2', 'C3'):
        if arm not in banks:
            continue
        tens = {'keys': banks[arm].contiguous()}
        if arm in ('C1', 'C3'):
            tens['task_id'] = torch.cat([torch.full((q_prev[arm][v],), i, dtype=torch.int16)
                                         for i, v in enumerate(units) if q_prev[arm][v] > 0])
        if arm in ('C0', 'C3'):
            tens['S'] = S.contiguous(); tens['n'] = torch.tensor([n], dtype=torch.int64)
        final[arm] = {'tensor_bytes': sum(v.numel() * v.element_size() for v in tens.values()),
                      'serialized_bytes': len(st_save(tens))}
    G.A.wj(out() / 'runs' / f'{stream}_{mib}.json', {'stream': stream, 'mib': mib, 'units': units, 'cap': cap,
                                                     'c2_run': c2 is not None, 'stages': log, 'final_state': final,
                                                     'seconds': time.monotonic() - t_all, 'device': dev,
                                                     'peak_cuda_bytes': torch.cuda.max_memory_allocated()})


# ------------------------------------------------------------------ analysis (registered gates)
def _loss(stages, arm, g, at=-1):
    st = stages[at]
    tasks = list(dict.fromkeys(task_of(u) for u in [s['unit'] for s in stages[:len(stages) if at == -1 else at + 1]]))
    p = st['p95_p99']
    return {tk: math.log(p[f'{arm}|{g}|{tk}'][0]) - math.log(p[f'C0|{g}|{tk}'][0]) for tk in tasks
            if f'{arm}|{g}|{tk}' in p}


def analyze():
    G, _ = RG.setup()
    res = {}
    for f in sorted((out() / 'runs').glob('*.json')):
        r = json.loads(f.read_text())
        st, units = r['stages'], r['units']
        cur = task_of(units[-1])
        arms = [a for a in ('C1', 'C2', 'C3') if f'{a}|W|{cur}' in st[-1]['p95_p99']]
        x = {}
        for a in arms:
            for g in ('W', 'E'):
                L = _loss(st, a, g)
                hist = {k: v for k, v in L.items() if k != cur}
                acq = [math.log(s['p95_p99'][f'{a}|{g}|{task_of(s["unit"])}'][0]) -
                       math.log(s['p95_p99'][f'C0|{g}|{task_of(s["unit"])}'][0]) for s in st]
                x[f'{a}|{g}'] = {'HL': float(np.mean(list(hist.values()))) if hist else 0.0,
                                 'n_ge_0.05': sum(v >= 0.05 for v in hist.values()),
                                 'worst': max(hist.items(), key=lambda kv: kv[1]) if hist else None,
                                 'AG': float(np.mean(acq)), 'per_task_final': L}
        res[f.stem] = {'arms': x, 'final_state': r['final_state'], 'cap': r['cap'], 'seconds': r['seconds'],
                       'update_s': {a: float(np.sum([s.get(f'{a}_update_s', 0) for s in st])) for a in arms},
                       'turnover_mean': {a: float(np.mean([s.get(f'{a}_turnover', 0) for s in st])) for a in arms}}
    m, mr = res.get('MIX_32', {}).get('arms', {}), res.get('MIX_REV_32', {}).get('arms', {})
    g1 = lambda a: bool(a and a['C3|W']['HL'] >= 0.02 and a['C3|W']['n_ge_0.05'] >= 5)
    gates = {'G1': g1(m), 'G2': bool(m and m['C3|W']['HL'] >= 2 * max(m['C3|W']['AG'], 0)),
             'G3': bool(m and all(k in m for k in ('C1|W', 'C2|W', 'C3|W')) and
                        min(m[k]['HL'] for k in ('C1|W', 'C2|W', 'C3|W')) >= 0.02),
             'G4': g1(mr)}
    verdict = 'DYN_HEADROOM_CONFIRMED' if all(gates.values()) else 'DYN_HEADROOM_NOT_ESTABLISHED'
    G.A.wj(out() / 'DYN00_RESULT.json', {'verdict': verdict, 'gates': gates, 'runs': res})
    print(verdict, gates)
    for k, v in res.items():
        print(k, {a: (round(s['HL'], 4), s['n_ge_0.05'], round(s['AG'], 4)) for a, s in v['arms'].items()})


if __name__ == '__main__':
    ph = sys.argv[1]
    if ph == 'run':
        run(sys.argv[2], int(sys.argv[3]))
    else:
        globals()[ph]()
