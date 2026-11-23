"""NANM-NL-01D core: exact greedy primitives, deployable allocation policies, privileged lattice oracles.

Protocol: agents/tasks/task_nanm_nl_01_d.md (frozen). All distances are fp64 squared Euclidean in the fixed z = v W
space. Policies are *causal by construction*: `step` receives only the current task's FIT vectors, its FEEDBACK
probes (feedback arms only) and metadata, and can only keep what it stores itself. No EVAL vector and no future
task is ever passed to a policy.
"""
from __future__ import annotations

import hashlib
import math
from itertools import combinations

import torch

N_FIT, N_FB, N_EV = 1024, 32, 2048           # per task: 16x64 FIT, 4x8 FEEDBACK, 8x256 EVAL
KEY_B, ID_B, PROBE_B = 1536, 8, 1540          # fp16 key, (task,int idx) id, fp16 probe + int32 id
W_B = 768 * 768 * 4                           # fixed geometry, fp32
UNITS = 32                                    # O1/O2a lattice: quotas are multiples of K / 32
ACQ_TOL, WORST_TOL, B4_GUARD = 1.02, 1.10, 1.02
FLOOR = 1e-12


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def start_index(seed: int, task: str, n: int) -> int:
    return int(sha(f'NANM-NL-01D/start|{seed}|{task}'), 16) % n


def sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2.0 * a @ b.T).clamp_min_(0.0)


class Counter:
    def __init__(self):
        self.dist = 0


def ffs(Z, n, start=None, base=None, ctr=None):
    """Exact farthest-first (greedy k-centre) order of n rows of Z. base: conditioning set (ContCore G_n(Z;B))."""
    N = len(Z); n = min(n, N)
    if base is not None and len(base):
        mind = sqdist(Z, base).min(1).values
        if ctr: ctr.dist += N * len(base)
        cur = int(mind.argmax())
    else:
        mind = torch.full((N,), math.inf, dtype=torch.float64); cur = int(start)
    taken = torch.zeros(N, dtype=torch.bool); idx = []
    for _ in range(n):
        idx.append(cur); taken[cur] = True
        mind = torch.minimum(mind, ((Z - Z[cur]) ** 2).sum(1))
        if ctr: ctr.dist += N
        m = mind.masked_fill(taken, -1.0)
        cur = int(m.argmax())
    return torch.tensor(idx, dtype=torch.long)


def prefix_min(Q, Kz, ctr=None):
    """[Nq, n]: column j = min distance of each query to the first j+1 rows of Kz."""
    if ctr: ctr.dist += len(Q) * len(Kz)
    return torch.cummin(sqdist(Q, Kz), 1).values


def round_quota(K, w, q_prev):
    """Largest-remainder target (ties to older), >=1 each; old quotas never increase; new task takes the rest."""
    w = [float(x) for x in w]; s = sum(w); raw = [K * x / s for x in w]
    tgt = [max(1, math.floor(r)) for r in raw]
    rem = sorted(range(len(w)), key=lambda i: (-(raw[i] - math.floor(raw[i])), i))
    k = 0
    while sum(tgt) < K:
        tgt[rem[k % len(w)]] += 1; k += 1
    while sum(tgt) > K:                                        # only if the >=1 floor pushed us over
        i = max(range(len(w)), key=lambda j: (tgt[j], -j)); tgt[i] -= 1
    q = [min(p, t) for p, t in zip(q_prev, tgt[:-1])]
    q.append(K - sum(q))
    return q


# ------------------------------------------------------------------ deployable policies
class QuotaPolicy:
    """Quota-prefix family. Each task is ordered by farthest-first at arrival (current task only); a task with
    quota q keeps the first q exemplars of its order. kind in B0 B1 B2 B4 B5."""

    def __init__(self, kind, K, seed, schedule=None):
        self.kind, self.K, self.seed, self.schedule = kind, K, seed, schedule
        self.order, self.keys, self.q, self.w, self.probes = [], [], [], [], []
        self.ctr = Counter(); self.guard_overrides = 0; self.trace = []

    def step(self, t, X, meta, P=None):
        K = self.K
        o = ffs(X, N_FIT, start=start_index(self.seed, meta['task'], len(X)), ctr=self.ctr)
        self.order.append(o); self.keys.append(X[o])                 # full order of the current task (transient)
        if self.kind == 'B1':
            self.w.append(math.sqrt(meta['n_train_images']))
        elif self.kind == 'B2':
            self.w.append(float(sqdist(X, X[o[:32]]).min(1).values.mean())); self.ctr.dist += len(X) * 32
        else:
            self.w.append(1.0)
        if self.kind == 'B4':
            self.probes.append(P); q = self._b4(t)
        elif self.kind == 'B5':
            q = round_quota(K, self.schedule[t], self.q)
        else:
            q = round_quota(K, self.w, self.q)
        assert all(a <= b for a, b in zip(q, self.q)) and sum(q) == K and min(q) >= 1, (self.kind, q, self.q)
        self.q = q
        self.keys = [k[:qi] for k, qi in zip(self.keys, q)]           # eviction: truncated rows are gone
        return [(i, self.order[i][:qi]) for i, qi in enumerate(q)]

    def _b4(self, t):
        K, n = self.K, t + 1
        P = torch.cat(self.probes)
        pm = [prefix_min(P, k, self.ctr) for k in self.keys]
        q = list(self.q) + [len(self.keys[-1])]

        def f(qv):
            m = torch.stack([pm[i][:, qv[i] - 1] for i in range(n)]).min(0).values
            return m.view(n, N_FB).mean(1)

        f0 = f(q).clamp_min(FLOOR)
        f_eq = float(f(list(self.q) + [K // n])[t])
        while sum(q) > K:
            best = None; best_any = None
            for i in range(n):
                if q[i] <= 1:
                    continue
                qq = list(q); qq[i] -= 1
                fv = f(qq); F = float((fv / f0).mean())
                if best_any is None or F < best_any[0]:
                    best_any = (F, i)
                if i == t and float(fv[t]) > B4_GUARD * f_eq:
                    continue
                if best is None or F < best[0]:
                    best = (F, i)
            if best is None:
                best = best_any; self.guard_overrides += 1
            q[best[1]] -= 1
        self.trace.append(list(q))
        return q

    def bytes(self, t):
        n = t + 1; b = W_B + self.K * (KEY_B + ID_B) + 4 * n
        if self.kind in ('B1', 'B2'):
            b += 8 * n
        if self.kind == 'B5':
            b += 64
        if self.kind == 'B4':
            b += N_FB * n * PROBE_B
        return b


class ContCorePolicy:
    """B3 (cap = K) and B6 (cap = K + extra(t)): M_t = G_cap(G_n(X_t; M_{t-1}) U M_{t-1}), exact greedy."""

    def __init__(self, K, seed, extra=None):
        self.K, self.seed, self.extra = K, seed, extra
        self.ids = []; self.z = None; self.ctr = Counter()

    def cap(self, t):
        return self.K + (self.extra(t) if self.extra else 0)

    def step(self, t, X, meta, P=None):
        cap = self.cap(t); n = min(self.K, len(X))
        e = ffs(X, n, start=start_index(self.seed, meta['task'], len(X)), base=self.z, ctr=self.ctr)
        ids = self.ids + [(t, int(j)) for j in e]
        Z = X[e] if self.z is None else torch.cat([self.z, X[e]])
        perm = sorted(range(len(ids)), key=lambda k: ids[k]); ids = [ids[k] for k in perm]; Z = Z[perm]
        if len(ids) > cap:
            keep = ffs(Z, cap, start=start_index(self.seed, meta['task'], len(ids)), ctr=self.ctr)
            ids = [ids[k] for k in keep.tolist()]; Z = Z[keep]
        self.ids, self.z = ids, Z
        return _group(ids)

    def bytes(self, t):
        return W_B + len(self.ids) * (KEY_B + ID_B)


def _group(ids):
    out = {}
    for i, j in ids:
        out.setdefault(i, []).append(j)
    return [(i, torch.tensor(sorted(v), dtype=torch.long)) for i, v in sorted(out.items())]


def b6_extra(K):
    """Extra exemplar slots so bytes(B6) <= bytes(B4) at boundary t (B4 = probes + quota ints beyond B3)."""
    return lambda t: (N_FB * (t + 1) * PROBE_B + 4 * (t + 1)) // (KEY_B + ID_B)


# ------------------------------------------------------------------ privileged lattice oracles
def compositions(total, parts):
    """All vectors of `parts` positive ints summing to total (lexicographic)."""
    out = []
    for cuts in combinations(range(1, total), parts - 1):
        b = (0,) + cuts + (total,)
        out.append([b[i + 1] - b[i] for i in range(parts)])
    return torch.tensor(out, dtype=torch.long)


def lattice_losses(C, states, lref, chunk=512):
    """C: list over task positions of [UNITS, Nev] prefix-min EVAL distances at quota = v*u (row v-1).
    states [S, n] unit quotas. Returns ell [S, n_tasks_total] (per-task normalized EVAL loss of each state)."""
    out = []
    T = len(lref)
    for s in range(0, len(states), chunk):
        st = states[s:s + chunk]
        m = torch.stack([C[i][st[:, i] - 1] for i in range(st.shape[1])]).min(0).values   # [S, Nev]
        out.append(m.view(len(st), T, -1).mean(2) / lref)
    return torch.cat(out)


def o1_dp(C, lref, ref_ell, constrain=True):
    """Exact best reachable quota path on the lattice (old quotas non-increasing), minimizing sum_{t>=1} hist_t.
    ref_ell[t] = per-task ell of SCB_K at boundary t (constraint reference). Returns (path, ell per t, flags)."""
    T = len(lref); stages, costs, back, ells, flags = [], [], [], [], []
    for t in range(T):
        S = compositions(UNITS, t + 1); ell = lattice_losses(C, S, lref)
        h = ell[:, :t].mean(1) if t else torch.zeros(len(S), dtype=torch.float64)
        feas = _feasible(ell, ref_ell[t], t) if constrain else torch.ones(len(S), dtype=torch.bool)
        if t == 0:
            prev = torch.zeros(len(S), dtype=torch.float64); bp = torch.zeros(len(S), dtype=torch.long)
        else:
            reach = (stages[-1][None, :, :] >= S[:, None, :t]).all(-1)                 # s'[i] >= s[i] for old i
            pc = costs[-1][None, :].expand(len(S), -1).masked_fill(~reach, math.inf)
            prev, bp = pc.min(1)
        cost = (h + prev).masked_fill(~feas, math.inf)
        flag = False
        if not torch.isfinite(cost).any():
            cost = h + prev; flag = True
        stages.append(S); costs.append(cost); back.append(bp); ells.append(ell); flags.append(flag)
    k = int(costs[-1].argmin()); path = []
    for t in range(T - 1, -1, -1):
        path.append((stages[t][k].tolist(), ells[t][k])); k = int(back[t][k])
    path.reverse()
    return [p for p, _ in path], [e for _, e in path], flags


def o2a_best(C, lref, ref_ell):
    """Per-boundary best lattice vector with regrowth allowed (archive access), same constraints."""
    T = len(lref); res = []
    for t in range(T):
        S = compositions(UNITS, t + 1); ell = lattice_losses(C, S, lref)
        h = ell[:, :t].mean(1) if t else ell[:, 0]
        feas = _feasible(ell, ref_ell[t], t); flag = not bool(feas.any())
        hh = h if flag else h.masked_fill(~feas, math.inf)
        k = int(hh.argmin()); res.append((S[k].tolist(), ell[k], flag))
    return res


def _feasible(ell, ref, t):
    ok = ell[:, t] <= ACQ_TOL * ref[t]
    for i in range(t):
        ok &= ell[:, i] <= WORST_TOL * ref[i]
    return ok
