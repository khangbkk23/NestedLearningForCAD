"""NANM-EVAL-01 evaluation-only arms (registration: agents/tasks/task_nanm_eval_01_registration.md §R4-R6).

Every numerical primitive is the canonical oracle's own function object:
  k-centre A.greedy_coreset | residual S B.residual_stats | W B.whiten / R2.shrink / R2.inv_sqrt_psd
  whitened energy G.whitened | Euclidean energy G.euclid | test tokens G.load_test | FIT tokens G.fit_tokens
Canonical-equivalent arms (N0 / A1 / A2 in order a) must reproduce the stored REPRO-01 grids bit-for-bit.
"""
from __future__ import annotations

import math
import time

import torch

from exps.nanm_eval_01 import common as U

D = U.D


def _mods():
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    from exps.nanm_r2b import run as B
    return A, R2, B


# ------------------------------------------------------------------ whiteners (all fp64 -> fp32 as canonical stage W)
def w_g3(states, key='S1', nkey='n1', alpha=None):
    """Canonical: B.whiten(sum S, sum n).float(); alpha overrides the shrinkage (B4); alpha == 0 -> no shrinkage (B3)."""
    A, R2, B = _mods()
    S = sum(x[key] for x in states); n = sum(x[nkey] for x in states)
    if alpha is None:
        W, cond = B.whiten(S, n)
    elif alpha == 0:
        W, cond = R2.inv_sqrt_psd(S / n)
    else:
        W, cond = R2.inv_sqrt_psd(R2.shrink(S / n, alpha))
    return W.float(), cond


def w_g1(states):
    """Canonical G1b (bench_g3._normal_ds): centred feature covariance of FIT_{<=t}, shrink 0.5."""
    A, R2, B = _mods()
    nn_ = sum(x['n'] for x in states); mu = sum(x['s1'] for x in states) / nn_
    W, cond = R2.inv_sqrt_psd(R2.shrink(sum(x['Sx'] for x in states) / nn_ - torch.outer(mu, mu)))
    return W.float(), cond


def cond_of(S, n):
    w = torch.linalg.eigvalsh((S / n).double())
    return float(w.max() / w.min()) if float(w.min()) > 0 else math.inf


# ------------------------------------------------------------------ extended per-task states (K: cache)
@torch.no_grad()
def build_ext(G, ds, seed, c, base, need_k, s_ks, rand_k=None):
    """Extended state of task c: k-centre order to need_k (prefix-checked against the stored canonical kc), residual S
    at each k in s_ks (B.residual_stats on FIT, members excluded), optional random-key bank of size rand_k with its own
    residual S (B6). base: stored canonical state (REPRO normal artifact). Returns dict + timings."""
    A, R2, B = _mods()
    t0 = time.time()
    fit = G.fit_tokens(ds, c)
    n = len(fit)
    assert n == base['n'], (ds, c, n, base['n'])
    kc0 = base['kc']
    if need_k > len(kc0):
        kc = A.greedy_coreset(fit, need_k, seed=seed)
        assert torch.equal(kc[:len(kc0)], kc0), f'k-centre prefix mismatch {ds} {c} {seed}'
    else:
        kc = kc0
    t_kc = time.time() - t0
    f32 = fit.float()
    out = {'kc': kc, 'keys': fit[kc.cuda()].cpu(), 'S': {}, 'n': n, 'time_kcentre_s': t_kc, 'time_S_s': {}}
    for k in sorted(set(s_ks)):
        t1 = time.time()
        S, nr = B.residual_stats(f32, kc[:k].cuda())
        out['S'][k] = (S.cpu(), int(nr)); out['time_S_s'][k] = time.time() - t1
    if rand_k:
        g = torch.Generator().manual_seed(int(U.ssha(f'NANM-EVAL-01/random/v1|{ds}|{seed}|{c}'), 16) % 2 ** 62)
        idx = torch.randperm(n, generator=g)[:rand_k]
        S, nr = B.residual_stats(f32, idx.cuda())
        out.update(rand_idx=idx, rand_keys=fit[idx.cuda()].cpu(), rand_S=S.cpu(), rand_n=int(nr))
    del fit, f32; torch.cuda.empty_cache()
    out['seconds'] = time.time() - t0
    return out


# ------------------------------------------------------------------ arm specifications
class TaskBank:
    """Per-task material for one (dataset, seed): canonical stored states + optional extension."""

    def __init__(self, base, ext=None):
        self.base, self.ext = base, ext

    def keys(self, k):
        src = self.base['keys'] if k <= len(self.base['keys']) else self.ext['keys']
        assert k <= len(src)
        return src[:k]

    def S(self, k, k1):
        """(S, n) for the residual statistic against the first k centres (k1 = canonical 1 % size -> stored S1)."""
        if k == k1:
            return self.base['S1'], self.base['n1']
        return self.ext['S'][k]


def st(S, n):
    return {'S1': S, 'n1': n}


def stage_specs(arm, order, l, banks, sizes, cfg):
    """Return a scoring spec for `arm` at stage l of `order`:
    ('union_w', keys_list, W)       canonical concatenated-key whitened energy (G.whitened)
    ('groups_e', [task...], ksz)    per-group Euclidean energies, min over the listed groups (canonical A0b style)
    ('groups_w', [(task, W)], ksz)  per-group whitened energies with a group-specific W, min over groups
    sizes: task -> key count for this arm. cfg: dict with k1 (canonical 1 % sizes) and options."""
    seen = list(order[:l + 1]); first = order[0]; k1 = cfg['k1']
    if arm in ('N0', 'A2', 'B1', 'B3', 'B4a', 'B4b', 'N1', 'N4', 'N5P2', 'N5P4', 'B6', 'G3r', 'G1r', 'N1r', 'N4r', 'N5P2r'):
        if arm == 'B6':
            keys = [banks[t].ext['rand_keys'] for t in seen]
            W, c = w_g3([st(banks[t].ext['rand_S'], banks[t].ext['rand_n']) for t in seen])
            return ('union_w', keys, W, c)
        keys = [banks[t].keys(sizes[t]) for t in seen]
        sts = lambda ts: [st(*banks[t].S(sizes[t], k1[t])) for t in ts]
        if arm in ('N0', 'G3r'):
            W, c = w_g3(sts(seen))
        elif arm == 'A2' or arm == 'G1r':
            W, c = w_g1([banks[t].base for t in seen])
        elif arm == 'B1':
            return ('groups_e', seen, sizes)
        elif arm == 'B3':
            W, c = w_g3(sts(seen), alpha=0)
        elif arm in ('B4a', 'B4b'):
            W, c = w_g3(sts(seen), alpha={'B4a': 0.25, 'B4b': 0.75}[arm])
        elif arm in ('N1', 'N1r'):
            W, c = w_g3(sts([first]))
        elif arm in ('N4', 'N4r'):
            W, c = w_g3(sts([order[l]]))
        else:                                                   # N5: stale W refreshed every P boundaries (+ final)
            P = 4 if arm == 'N5P4' else 2
            r = refresh_stage(l, P, len(order))
            W, c = w_g3(sts(list(order[:r + 1])))
        return ('union_w', keys, W, c)
    if arm in ('A1', 'A0r', 'A0cap', 'EQB'):
        return ('groups_e', seen, sizes)
    if arm in ('N2', 'N3'):
        keys = [banks[first].keys(sizes[first])]
        W, c = w_g3([st(*banks[t].S(sizes[t], k1[t])) for t in (seen if arm == 'N2' else [first])])
        return ('union_w', keys, W, c)
    if arm == 'B5':                                             # each key group under its acquisition-time W
        grp = []
        for t in seen:
            p = order.index(t)
            W, _ = w_g3([st(*banks[u].S(sizes[u], k1[u])) for u in order[:p + 1]])
            grp.append((t, W))
        return ('groups_w', grp, sizes)
    raise KeyError(arm)


def refresh_stage(l, P, T):
    """Last boundary m <= l at which a stale-W arm materialises W: m == 0, (m + 1) % P == 0, or m == T - 1."""
    return max(m for m in range(l + 1) if m == 0 or (m + 1) % P == 0 or m == T - 1)


# ------------------------------------------------------------------ scoring one cell
@torch.no_grad()
def score_cell(G, order, banks, Q, arms, log=None):
    """arms: {name: (arm_kind, sizes, cfg)}. Q: task -> (q16 CPU fp16 [N*784, D], rows). Returns
    grids[name][task] = tensor [T - pos, n_img, 28, 28] fp32 and meta (W condition numbers, seconds)."""
    T = len(order)
    grids = {a: {c: [] for c in order} for a in arms}
    meta = {a: {'cond': [], 'seconds': 0.0} for a in arms}
    cache_e = {}                                                 # (arm, group task, query task) -> energies
    for l in range(T):
        for a, (kind, sizes, cfg) in arms.items():
            t0 = time.time()
            spec = stage_specs(kind, order, l, cfg['banks'] if 'banks' in cfg else banks, sizes, cfg)
            if spec[0] == 'union_w':
                keys = torch.cat(spec[1]).cuda().float(); W = spec[2].cuda()
                meta[a]['cond'].append(spec[3])
                for c in order[:l + 1]:
                    grids[a][c].append(G.whitened(Q[c][0], keys, W))
                del keys, W
            else:
                for c in order[:l + 1]:
                    es = []
                    for item in spec[1]:
                        t, W = (item, None) if spec[0] == 'groups_e' else item
                        ck = (a, t, c)
                        if ck not in cache_e:
                            kb = (cfg.get('banks', banks))[t].keys(spec[2][t]).cuda().float()
                            cache_e[ck] = G.euclid(Q[c][0], kb) if W is None else G.whitened(Q[c][0], kb, W.cuda())
                            del kb
                        es.append(cache_e[ck])
                    grids[a][c].append(torch.stack(es).amin(0))
            meta[a]['seconds'] += time.time() - t0
        if log:
            log('stage', l, 'done')
    out = {}
    for a in arms:
        out[a] = {}
        for c in order:
            n = len(Q[c][1])
            out[a][c] = torch.stack(grids[a][c]).reshape(len(grids[a][c]), n, 28, 28)
    return out, meta
