"""NANM-NL-02 Stage 0 v2 — fast-optimizer layer and v2 byte ledger.

Registration: agents/tasks/task_nanm_nl_02_stage0_v2.md (amends v1 agents/tasks/task_nanm_nl_02_stage0.md; ONLY the fast
optimizer changes). The v1 objective, init, keys, W, roles, evaluator and gates are imported unchanged from
exps/nanm_nl_02/core.py (read-only, hash-locked by v1 PREREG_LOCK.json).

Optimizers (all with the v1 determinant backtracking: the proposed update is halved up to 10 times until det A > 0 and
log|det A| finite, otherwise the step is rejected and phi is unchanged):
  O0 'sgd'  : v1 plain SGD, delegated to core.sgd_step (bitwise regression oracle).                       state 0 B
  O1 'adam' : Adam (beta1 0.9, beta2 0.999, eps 1e-8, bias-corrected, no weight decay). Moments (m, v) for P and Q and
              the step counter persist across task boundaries and are charged.                           state 2*PHI_B + 8 B
  O2 'sign' : stateless sign descent  dphi = -lr * sign(g)  (element-wise normalised SGD).                 state 0 B
A non-finite gradient rejects the step before any moment update (counted as nonfinite).
"""
from __future__ import annotations

import math

import torch

from exps.nanm_nl_02 import core as C

F64 = torch.float64
KINDS = ('sgd', 'adam', 'sign')
OPT_ID = {'sgd': 'O0', 'adam': 'O1', 'sign': 'O2'}
B1, B2, EPS = 0.9, 0.999, 1e-8
MAX_HALVINGS = 10
OPT_STATE_B = {'sgd': 0, 'adam': 2 * C.PHI_B + 8, 'sign': 0}     # fp32 m, v for P and Q (196,608 B) + int64 step counter


class FastOpt:
    """Fast-update rule with explicit state. state_dict()/load_state() make the common-start copy explicit."""

    def __init__(self, kind, lr, d=C.D, r=C.RANK):
        assert kind in KINDS, kind
        self.kind, self.lr, self.t = kind, float(lr), 0
        if kind == 'adam':
            z = lambda: torch.zeros(d, r, dtype=F64)
            self.m = [z(), z()]; self.v = [z(), z()]
        self.stats = {'steps': 0, 'rejected': 0, 'nonfinite_grad': 0, 'halvings': 0, 'step_norm_sum': 0.0, 'grad_norm_sum': 0.0,
                      'grad_norm_max': 0.0, 'min_logdet': 0.0}

    def clone(self):
        o = FastOpt.__new__(FastOpt)
        o.kind, o.lr, o.t = self.kind, self.lr, self.t
        if self.kind == 'adam':
            o.m = [x.clone() for x in self.m]; o.v = [x.clone() for x in self.v]
        o.stats = {k: (0 if isinstance(v, int) else 0.0) for k, v in self.stats.items()}
        return o

    def persistent_bytes(self):
        return OPT_STATE_B[self.kind]

    def _direction(self, gP, gQ):
        if self.kind == 'sign':
            return -self.lr * torch.sign(gP), -self.lr * torch.sign(gQ)
        self.t += 1
        out = []
        for i, g in enumerate((gP, gQ)):
            self.m[i].mul_(B1).add_(g, alpha=1 - B1)
            self.v[i].mul_(B2).addcmul_(g, g, value=1 - B2)
            mh = self.m[i] / (1 - B1 ** self.t); vh = self.v[i] / (1 - B2 ** self.t)
            out.append(-self.lr * mh / (vh.sqrt() + EPS))
        return out

    def step(self, P, Q, gP, gQ, proj=None):
        """Returns (P, Q, h); h = number of halvings, -1 = rejected. proj: optional callable applied to the update
        (C4 under O1/O2; for O0 the gradient projection alone is used, exactly as v1)."""
        st = self.stats; st['steps'] += 1
        gn = float(torch.sqrt(gP.pow(2).sum() + gQ.pow(2).sum()))
        if not math.isfinite(gn):
            st['nonfinite_grad'] += 1; st['rejected'] += 1
            return P, Q, -1
        st['grad_norm_sum'] += gn; st['grad_norm_max'] = max(st['grad_norm_max'], gn)
        if self.kind == 'sgd':
            P2, Q2, h = C.sgd_step(P, Q, gP, gQ, self.lr, MAX_HALVINGS)
        else:
            dP, dQ = self._direction(gP, gQ)
            if proj is not None:
                dP, dQ = proj(dP), proj(dQ)
            P2, Q2, h = P, Q, -1
            for k in range(MAX_HALVINGS + 1):
                sc = 0.5 ** k
                tP, tQ = P + sc * dP, Q + sc * dQ
                s, la = C.logdet_A(tP, tQ)
                if float(s) > 0 and math.isfinite(float(la)):
                    P2, Q2, h = tP, tQ, k
                    break
        if h < 0:
            st['rejected'] += 1
            return P, Q, -1
        st['halvings'] += h
        st['step_norm_sum'] += float(torch.sqrt((P2 - P).pow(2).sum() + (Q2 - Q).pow(2).sum()))
        st['min_logdet'] = min(st['min_logdet'], float(C.logdet_A(P2, Q2)[1]))
        return P2, Q2, h


def phi_stats(P, Q):
    """||A - I||_F, sign and log|det A|, cond(A) (exact SVD of the 768x768 A)."""
    A = C.A_of(P, Q)
    sv = torch.linalg.svdvals(A)
    s, la = C.logdet_A(P, Q)
    return {'A_minus_I_F': float(torch.linalg.norm(P @ Q.T)), 'det_sign': float(s), 'logdet_A': float(la),
            'cond_A': float(sv[0] / sv[-1]), 'finite': bool(torch.isfinite(P).all() and torch.isfinite(Q).all())}


# ------------------------------------------------------------------ v2 ledger (v1 EXTRA + optimizer state, all C1-C5)
def extra_v2(arm, kind):
    return C.EXTRA[arm] + (OPT_STATE_B[kind] if arm in ('C1', 'C2', 'C3', 'C4', 'C5') else 0)


def reserve_alloc_v2(arm, n_seen, kind):
    """Spare-byte RESERVE probes. The optimizer state is identical for C1-C5, so the surplus vs the largest arm (C3)
    and hence the allocation equal v1 (C2 127, C4 95, C5 63; capped 32 per seen task)."""
    spare = extra_v2('C3', kind) - extra_v2(arm, kind)
    return min(spare // C.PROBE_B, 32 * n_seen) if arm in ('C2', 'C4', 'C5') else 0


def ledger_v2(arm, kind, n_tasks, keys_per_task, n_fb_probes, n_reserve_probes=0):
    keys = sum(keys_per_task) * (C.KEY_B + C.KEY_ID_B)
    probes = (n_fb_probes + n_reserve_probes) * C.PROBE_B
    ex = extra_v2(arm, kind); meta = C.META_B + 4 * n_tasks
    return {'keys': keys, 'g3_S': C.S_B, 'w_cache_uncharged': C.W_CACHE_B, 'phi_and_arm_state': C.EXTRA[arm],
            'optimizer_state': ex - C.EXTRA[arm], 'probes': probes, 'meta': meta,
            'total_charged': keys + C.S_B + ex + probes + meta}


def budget_v2(n, kind, k_keys):
    """Registered persistent budget at n seen tasks: keys + S + meta + largest arm state (C3 incl. optimizer) + FEEDBACK."""
    return n * k_keys * (C.KEY_B + C.KEY_ID_B) + C.S_B + C.META_B + 4 * n + extra_v2('C3', kind) + 32 * n * C.PROBE_B
