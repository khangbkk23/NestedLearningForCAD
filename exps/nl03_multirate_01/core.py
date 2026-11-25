"""NL03-MULTIRATE-01 core: bounded multi-rate residual memory (BMRM) and matched controls.

Canonical primitives imported unchanged:
  per-write coreset + residual statistic : nanm.g3.fit_task (A.greedy_coreset + B.residual_stats)
  eviction k-centre                      : exps.nl03_pressure_00.core.kc_select (A.greedy_coreset, insertion order)
  whitening                              : exps.nanm_r2b.run.whiten (shrink 0.5, inv-sqrt, fp64 -> fp32)
  energy                                 : nanm.g3.score (whitened) / exps.bench_g3.run.euclid (W = I)

Fast level (frequency 1 = once per minibatch of MB_IMAGES FIT images):
  C_b = canonical 1 % coreset of the minibatch tokens (fit_task), counts n_c = tokens assigned to each centre;
  K'  = kc_select(K u C_b, N_fast); new candidates not admitted -> their mass is reassigned to the nearest key of K';
  old keys not kept are EVICTED: freed (no-pending arms) or moved to the pending buffer P (charged until the slow write).
Slow level: pooled (S += S_b every step), EMA (every C fast steps, S <- beta S + (1-beta) H_hat), or none.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from exps.nl03_multirate_01 import spec as SP

D = SP.D


def _mods():
    from exps.nanm_r0 import audit as A
    from exps.nanm_r2 import run as R2
    from exps.nanm_r2b import run as B
    from exps.nl03_pressure_00 import core as PC
    import nanm.g3 as g3
    return A, R2, B, PC, g3


def sha_t(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


class _Ids:
    """Minimal pool adaptor so that PC.kc_select runs on a local key matrix (pool.keys[cand])."""

    def __init__(self, keys):
        self.keys = keys


@torch.no_grad()
def write_candidates(x16, seed, counts):
    """Canonical per-write coreset of one minibatch. Returns centre keys fp16, S_b (fp64 cpu), n_b, counts or None."""
    A, R2, B, PC, g3 = _mods()
    k = g3.task_size(len(x16))
    st = g3.fit_task(x16, k, seed)
    keys = x16[st.kc[:k].to(x16.device)]
    cnt = None
    if counts:
        kf = keys.float()
        j = torch.cat([R2.sqdist(x16[i:i + 4096].float(), kf).argmin(1) for i in range(0, len(x16), 4096)])
        cnt = torch.bincount(j, minlength=k).long()
    return keys, st.S, int(st.n), cnt


class Fast:
    """Fast level shared by every arm with the same (N_fast, counts); evictions are returned to the caller."""

    def __init__(self, n_fast, counts, seed):
        self.n_fast, self.counts, self.seed = n_fast, counts, seed
        self.K = torch.empty(0, D, dtype=torch.float16, device='cuda')
        self.ids = torch.empty(0, dtype=torch.long)
        self.task = torch.empty(0, dtype=torch.long)
        self.born = torch.empty(0, dtype=torch.long)
        self.cnt = torch.empty(0, dtype=torch.long) if counts else None
        self.next_id = 0
        self.writes = 0

    @torch.no_grad()
    def step(self, keys, cnt, task, step):
        A, R2, B, PC, g3 = _mods()
        k = len(keys)
        nid = torch.arange(self.next_id, self.next_id + k); self.next_id += k
        candK = torch.cat([self.K, keys])
        cand = torch.arange(len(candK))
        keep = PC.kc_select(cand, self.n_fast, _Ids(candK), self.seed)          # survivors keep insertion order
        n_old = len(self.K)
        kmask = torch.zeros(len(candK), dtype=torch.bool); kmask[keep] = True
        ev = torch.nonzero(~kmask[:n_old]).flatten()                             # evicted old keys
        drop_new = torch.nonzero(~kmask[n_old:]).flatten()                       # candidates never admitted
        all_ids = torch.cat([self.ids, nid]); all_task = torch.cat([self.task, torch.full((k,), task)])
        all_born = torch.cat([self.born, torch.full((k,), step)])
        out = {'keys': self.K[ev.to(self.K.device)], 'ids': self.ids[ev], 'task': self.task[ev],
               'age': step - self.born[ev], 'n_admitted': k - len(drop_new), 'n_dropped_new': len(drop_new)}
        if self.counts:
            all_cnt = torch.cat([self.cnt, cnt.cpu()])
            out['cnt'] = self.cnt[ev]
            if len(drop_new):                                                    # reassign dropped-candidate mass
                kk = candK[keep.to(candK.device)].float()
                j = R2.sqdist(keys[drop_new.to(keys.device)].float(), kk).argmin(1).cpu()
                add = torch.bincount(j, weights=cnt.cpu()[drop_new].double(), minlength=len(keep)).long()
            self.cnt = all_cnt[keep] + (add if len(drop_new) else 0)
        self.K = candK[keep.to(candK.device)]
        self.ids, self.task, self.born = all_ids[keep], all_task[keep], all_born[keep]
        self.writes += 1
        out['hash'] = hashlib.sha256(self.ids.numpy().tobytes() + (self.cnt.numpy().tobytes() if self.counts else b'')
                                     ).hexdigest()
        return out


class Slow:
    """Per-arm slow level + pending buffer + window accumulator, with continuous byte accounting."""

    def __init__(self, arm, budget, tau):
        self.arm, self.a, self.budget = arm, SP.ARMS[arm], budget
        self.C = self.a['C']
        self.beta = SP.beta(self.C, tau) if self.a['slow'] == 'ema' else None
        self.S = None                       # stored fp32 (persistent)
        self.n = 0.0                        # pooled mass (fp64)
        self.acc, self.acc_n, self.acc_steps = None, 0.0, []
        self.P = {'keys': [], 'ids': [], 'cnt': [], 'task': [], 'age': []}
        self.writes, self.log, self.fast_steps = 0, [], 0
        self.max_bytes = 0

    def n_pending(self):
        return int(sum(len(x) for x in self.P['ids']))

    def check(self, n_fast, where):
        b = SP.assert_budget(self.arm, where, n_fast, self.n_pending(), self.budget)
        self.max_bytes = max(self.max_bytes, b)
        return b

    @torch.no_grad()
    def after_fast(self, fast, ev, S_b, n_b, step):
        """Called after every fast write of this arm's fast level (and before any slow write of the same step)."""
        self.fast_steps += 1
        kind = self.a['slow']
        if self.a['pending'] and len(ev['ids']):
            for f in ('keys', 'ids', 'cnt', 'task', 'age'):
                self.P[f].append(ev[f])
        b_fast = self.check(len(fast.K), f'fast@{step}')
        if kind == 'pooled':
            S64 = (self.S.double().cpu() if self.S is not None else 0) + S_b
            self.S = S64.float().cuda(); self.n += n_b
            self.writes += 1
            self.log.append({'step': step, 'kind': 'pooled', 'n': self.n, 'hash': sha_t(self.S), 'bytes': b_fast,
                             'trace': float(torch.trace(self.S.double()) / self.n)})
            return
        if kind != 'ema':
            return
        if self.C > 1:                                                           # window accumulator (stored fp32)
            a64 = (self.acc.double().cpu() if self.acc is not None else 0) + S_b
            self.acc = a64.float().cuda(); self.acc_n += n_b; self.acc_steps.append(step)
            S_in, n_in = self.acc.double(), self.acc_n
        else:
            S_in, n_in = S_b.cuda(), float(n_b); self.acc_steps = [step]
        if self.fast_steps % self.C:
            return
        self._slow_write(fast, S_in, n_in, step)

    @torch.no_grad()
    def _slow_write(self, fast, S_in, n_in, step):
        A, R2, B, PC, g3 = _mods()
        H_in = (S_in / n_in).double()
        w_e, n_ev, ev_ids = 0.0, 0, np.zeros(0, np.int64)
        if self.a['transfer'] and self.n_pending():
            Pk = torch.cat(self.P['keys']).float(); Pc = torch.cat(self.P['cnt']).double().cuda()
            ev_ids = torch.cat(self.P['ids']).numpy()
            Kf = fast.K.float()
            j = torch.cat([R2.sqdist(Pk[i:i + 4096], Kf).argmin(1) for i in range(0, len(Pk), 4096)])
            r = (Pk - Kf[j]).double()
            n_ev = float(Pc.sum())
            H_ev = (r * Pc[:, None]).T @ r / n_ev
            w_e = n_ev / (n_ev + n_in)
            H_hat = w_e * H_ev + (1 - w_e) * H_in
        else:
            H_hat = H_in
        first = self.S is None
        S_new = H_hat if first else self.beta * self.S.double() + (1 - self.beta) * H_hat
        delta = float('nan') if first else float((S_new - self.S.double()).norm())
        self.S = S_new.float()
        released = self.n_pending()
        self.P = {'keys': [], 'ids': [], 'cnt': [], 'task': [], 'age': []}
        steps = self.acc_steps
        self.acc, self.acc_n, self.acc_steps = None, 0.0, []
        self.writes += 1
        b = self.check(len(fast.K), f'slow@{step}')
        self.log.append({'step': step, 'kind': 'ema', 'write': self.writes, 'first': first, 'w_e': w_e,
                         'n_evict_mass': n_ev, 'n_in': n_in, 'released_keys': released, 'beta': self.beta,
                         'in_steps': [int(steps[0]), int(steps[-1])], 'n_in_steps': len(steps),
                         'evict_ids_sha': hashlib.sha256(ev_ids.tobytes()).hexdigest(), 'n_evict_ids': len(ev_ids),
                         'evict_ids': ev_ids, 'hash': sha_t(self.S), 'bytes': b, 'trace': float(torch.trace(S_new)),
                         'delta_fro': delta, 'trace_H_in': float(torch.trace(H_in)),
                         'trace_H_hat': float(torch.trace(H_hat))})

    def W(self):
        A, R2, B, PC, g3 = _mods()
        kind = self.a['slow']
        if kind == 'none':
            return None
        assert self.S is not None, f'{self.arm}: no slow state yet'
        n = self.n if kind == 'pooled' else 1.0
        W, cond = B.whiten(self.S.double().cpu(), n)
        return W.float()


class TaskLevel:
    """C0: historical G3-E at task boundaries (pressure-00 rule E on the development FIT set)."""

    def __init__(self, budget, seed):
        self.cap = (budget - SP.S_B) // SP.KEY_B
        self.seed = seed
        self.K = torch.empty(0, D, dtype=torch.float16, device='cuda')
        self.task = torch.empty(0, dtype=torch.long)
        self.S, self.n, self.budget, self.max_bytes, self.log = None, 0.0, budget, 0, []

    @torch.no_grad()
    def boundary(self, x16, task):
        A, R2, B, PC, g3 = _mods()
        k = g3.task_size(len(x16))
        st = g3.fit_task(x16, k, self.seed)
        keys = x16[st.kc[:k].to(x16.device)]
        candK = torch.cat([self.K, keys])
        keep = PC.kc_select(torch.arange(len(candK)), self.cap, _Ids(candK), self.seed)
        ev = len(candK) - len(keep)
        self.K = candK[keep.to(candK.device)]
        self.task = torch.cat([self.task, torch.full((k,), task)])[keep]
        S64 = (self.S.double().cpu() if self.S is not None else 0) + st.S
        self.S = S64.float().cuda(); self.n += st.n
        b = len(self.K) * SP.KEY_B + SP.S_B
        if b > self.budget:
            raise SP.BudgetViolation(f'C0 {task} {b}')
        self.max_bytes = max(self.max_bytes, b)
        self.log.append({'task': task, 'n_keys': len(self.K), 'evicted': ev, 'bytes': b, 'hash': sha_t(self.S)})

    def W(self):
        A, R2, B, PC, g3 = _mods()
        W, cond = B.whiten(self.S.double().cpu(), self.n)
        return W.float()


def plan_arms(arms, budget, tau, seed):
    """Build shared fast levels: arms with identical (N_fast, counts) share one trajectory."""
    fasts, slows, bind = {}, {}, {}
    for arm in arms:
        if arm == 'C0':
            continue
        a = SP.ARMS[arm]
        ntot = SP.total_cap(arm, budget)
        R = (a['C'] * kmax()) if a['pending'] else 0
        n_fast = ntot - R
        key = (n_fast, a['counts'])
        if key not in fasts:
            fasts[key] = Fast(n_fast, a['counts'], seed)
        slows[arm] = Slow(arm, budget, tau)
        bind[arm] = key
    return fasts, slows, bind


def kmax():
    """Largest per-write candidate count = canonical 1 % of a full minibatch (MB_IMAGES * 784 tokens)."""
    A, R2, B, PC, g3 = _mods()
    return g3.task_size(SP.MB_IMAGES * 784)


def caps_table(budget):
    rows = {}
    for arm, a in SP.ARMS.items():
        if arm == 'C0':
            rows[arm] = {'n_tot': (budget - SP.S_B) // SP.KEY_B, 'n_fast': (budget - SP.S_B) // SP.KEY_B,
                         'reserve': 0, 'per_key_B': SP.KEY_B, 'slow_B': SP.S_B}
            continue
        ntot = SP.total_cap(arm, budget)
        R = (a['C'] * kmax()) if a['pending'] else 0
        rows[arm] = {'n_tot': ntot, 'n_fast': ntot - R, 'reserve': R, 'per_key_B': SP.per_key_bytes(arm),
                     'slow_B': SP.slow_bytes(arm)}
    return rows
