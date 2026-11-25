"""G3E-PAPER-01 stream learner: the proposed method G3-E and its references/ablations over full-train task tokens.

U and E reproduce exps.nl03_multirate_01.final.Learner exactly (same g3.fit_task call, same PC.kc_select call through
CO._Ids, same B.whiten, same byte accounting); the other arms reuse the NL03-PRESSURE-00 rules on the same per-task
canonical material:

    U        canonical G3, unbounded union of per-task 1 % coresets, pooled S            bytes n*1536 + S_B
    E@<MiB>  G3-E: task-boundary greedy k-centre (kc_select) of K u C_t to cap (B - S_B)/1536, pooled S
    E@inf    E with no budget (must equal U bitwise)
    Q        hindsight k-centre over the union of all seen coresets to E's cap, pooled S
    E4       E's keys, W from the current task's S_t only                               bytes as E
    F        Euclidean (W = I), E rule at cap B/1536 (all bytes as keys)
    G        generic covariance W (canonical G1b: centred FIT feature covariance, shrink 0.5), E rule at (B - G1_B)/1536
    FIFO     oldest keys evicted first (PC.fifo_select) at E's cap, pooled S
    B1       E's keys, no whitening (W = I)                                             bytes n*1536 (E's keys only)
"""
from __future__ import annotations

import time

import torch

from exps.nl03_multirate_01 import core as CO
from exps.nl03_multirate_01 import spec as SP

MiB = SP.MiB
KEY_B = SP.KEY_B
S_B = SP.S_B
G1_B = SP.D * SP.D * 4 + SP.D * 4 + 8                 # 2,362,376 (BENCH-G3 G1b)


class BudgetViolation(AssertionError):
    pass


def budget_of(arm):
    if '@' in arm:
        v = arm.split('@')[1]
        return None if v == 'inf' else int(v) * MiB
    return None if arm == 'U' else 8 * MiB


def slow_bytes(arm):
    base = arm.split('@')[0]
    return {'U': S_B, 'E': S_B, 'Q': S_B, 'E4': S_B, 'FIFO': S_B, 'F': 0, 'G': G1_B, 'B1': 0}[base]


def cap_of(arm):
    b = budget_of(arm)
    base = arm.split('@')[0]
    if b is None:
        return None
    if base in ('E4', 'B1'):                          # share E@8's trajectory
        return (b - S_B) // KEY_B
    return (b - slow_bytes(arm)) // KEY_B


def bytes_of(arm, n_keys):
    return int(n_keys) * KEY_B + slow_bytes(arm)


def assert_budget(arm, stage, n_keys):
    b = bytes_of(arm, n_keys)
    lim = budget_of(arm)
    if lim is not None and b > lim:
        raise BudgetViolation(f'BUDGET_VIOLATION arm={arm} stage={stage} keys={n_keys} bytes={b} > {lim}')
    return b


class Learner:
    def __init__(self, arms, seed):
        self.names = list(arms)
        self.seed = seed
        need_e8 = any(a.split('@')[0] in ('E4', 'B1') for a in arms)
        self.traj = sorted({a for a in arms if a.split('@')[0] in ('E', 'F', 'G', 'FIFO', 'Q')} | ({'E@8'} if need_e8 else set()))
        self.K = {a: torch.empty(0, SP.D, dtype=torch.float16, device='cuda') for a in self.traj}
        self.blocks = []                                # per-task coreset keys (CPU fp16), U and Q
        self.S, self.n = 0, 0
        self.cur = None                                 # (S_t, n_t) for E4
        self.g1 = {'Sx': 0, 's1': 0, 'n': 0}
        self.snap = {a: [] for a in arms}
        self.bytes = {a: [] for a in arms}
        self.timing = {a: 0.0 for a in arms}
        self.select_log = {a: [] for a in self.traj}

    @torch.no_grad()
    def task(self, x16, l):
        import nanm.g3 as g3
        from exps.nl03_pressure_00 import core as PC
        t0 = time.time()
        k = g3.task_size(len(x16))
        st = g3.fit_task(x16, k, self.seed)
        dt = time.time() - t0
        keys = x16[st.kc[:k].to(x16.device)]
        self.blocks.append(keys.cpu())
        self.S = self.S + st.S; self.n += st.n
        self.cur = (st.S, st.n)
        tg = 0.0
        if 'G' in self.names:
            t1 = time.time()
            Sx = torch.zeros(SP.D, SP.D, dtype=torch.float64, device='cuda')
            s1 = torch.zeros(SP.D, dtype=torch.float64, device='cuda')
            for i in range(0, len(x16), 8192):
                x = x16[i:i + 8192].float()
                Sx += (x.T @ x).double(); s1 += x.sum(0).double()
            self.g1['Sx'] = self.g1['Sx'] + Sx.cpu(); self.g1['s1'] = self.g1['s1'] + s1.cpu(); self.g1['n'] += len(x16)
            tg = time.time() - t1
        tsel = {}
        for a in self.traj:
            t1 = time.time()
            base = a.split('@')[0]
            cap = (budget_of(a) - slow_bytes(a)) // KEY_B if budget_of(a) is not None else None
            if base == 'Q':
                cand = torch.cat([b.cuda() for b in self.blocks])
                keep = PC.kc_select(torch.arange(len(cand)), cap, CO._Ids(cand), self.seed)
                self.K[a] = cand[keep.to(cand.device)]
            elif base == 'FIFO':
                self.K[a] = PC.fifo_select(self.K[a], keys, cap)
            else:                                       # E rule (E@B, F, G)
                cand = torch.cat([self.K[a], keys])
                keep = PC.kc_select(torch.arange(len(cand)), cap, CO._Ids(cand), self.seed)
                self.K[a] = cand[keep.to(cand.device)]
            self.select_log[a].append({'stage': l, 'n_keys': len(self.K[a]), 'cap': cap})
            tsel[a] = time.time() - t1
        for a in self.names:
            base = a.split('@')[0]
            src = 'E@8' if base in ('E4', 'B1') else a
            self.timing[a] += dt + tsel.get(src, 0.0) + (tg if base == 'G' else 0.0)
        self.snapshot(l)

    def state(self, a):
        from exps.nanm_eval_01 import arms as AR
        from exps.nanm_r2b import run as B
        base = a.split('@')[0]
        if base == 'U':
            W, _ = B.whiten(self.S, self.n)
            return ('U_prefix', len(self.blocks)), W.float(), sum(len(b) for b in self.blocks)
        if base in ('E', 'Q', 'FIFO'):
            W, _ = B.whiten(self.S, self.n)
            return self.K[a].cpu(), W.float(), len(self.K[a])
        if base == 'E4':
            W, _ = B.whiten(*self.cur)
            return self.K['E@8'].cpu(), W.float(), len(self.K['E@8'])
        if base == 'B1':
            return self.K['E@8'].cpu(), None, len(self.K['E@8'])
        if base == 'F':
            return self.K[a].cpu(), None, len(self.K[a])
        if base == 'G':
            W, _ = AR.w_g1([self.g1])
            return self.K[a].cpu(), W.float(), len(self.K[a])
        raise KeyError(a)

    def snapshot(self, l):
        for a in self.names:
            K, W, nk = self.state(a)
            self.bytes[a].append(assert_budget(a, l, nk))
            self.snap[a].append((K, None if W is None else W.cpu()))

    def keys_at(self, a, l):
        K, W = self.snap[a][l]
        if isinstance(K, tuple):
            K = torch.cat(self.blocks[:K[1]])
        return K, W
