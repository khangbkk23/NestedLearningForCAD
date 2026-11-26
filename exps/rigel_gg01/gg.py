"""GG-01 method code: geometry-guided exemplar consolidation, and the controlled arms of the GG-01 study.

Only the metric of the task-boundary *global consolidation* changes between v1 and GG-01:

    v1     K_t = KCenter_P(K_{t-1} u C_t, m)        P = seeded Gaussian projection 768 -> 128 (rigel_cad.ops)
    GG-01  K_t = KCenter_{W_{t-1}}(K_{t-1} u C_t, m)   d^2(x, y) = ||(x - y) W_{t-1}||^2, full 768-d, exact
           (task 1, or whenever no slow statistic exists yet: the v1 rule)

C_t (1 % per-task coreset), S_t (residual moment against C_t), the shrinkage, the key capacity, the encoder, the
scoring and the read-out are the v1 objects, computed by the v1 primitives (rigel_cad.ops), unchanged.

Arms (one shared per-task fit: C_t, S_t, n_t):
    V1    frozen RIGEL-CAD: v1 consolidation, cap (B - S_B)/1536, whitened scoring with W_t
    GG01  GG-01 consolidation, same cap, same scoring
    F     fast-only control: v1 consolidation, cap B/1536, Euclidean scoring
    U     unbounded reference: union of all C_t, whitened scoring
    CCW   ContCore-style control: base-set-aware greedy expansion of the task tokens against K_{t-1} (projected
          metric P, n_t = |C_t|), then v1 consolidation; cap and scoring of V1 (isolates the selection rule)
    CCE   ContCore-style, Euclidean scoring, cap B/1536 (isolates the selection rule without whitening; compare F)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from rigel_cad import ops
from rigel_cad.config import KEY_BYTES, MiB, PROJ_DIM, SLOW_BYTES, TOKEN_DIM

D = TOKEN_DIM
BUDGET = 8 * MiB
CAP_SLOW = (BUDGET - SLOW_BYTES) // KEY_BYTES        # 3925
CAP_FAST = BUDGET // KEY_BYTES                       # 5461

ARMS = {  # name: (rule, cap, slow level used for scoring)
    'V1': ('kc', CAP_SLOW, True),
    'GG01': ('gg', CAP_SLOW, True),
    'F': ('kc', CAP_FAST, False),
    'U': ('union', None, True),
    'CCW': ('cc', CAP_SLOW, True),
    'CCE': ('cc', CAP_FAST, False),
}


# ---------------------------------------------------------------------------------------------- selection primitives
@torch.no_grad()
def kcenter_w(x: torch.Tensor, n: int, seed: int, W: torch.Tensor) -> torch.Tensor:
    """Greedy k-centre under d_W(x, y) = ||(x - y) W||, exact in the full 768-d whitened space.

    Seeded semantics identical to rigel_cad.ops.greedy_kcenter: the same generator is advanced through the same draws
    (the 768 x 128 projection is drawn and discarded, then the start index), so the first centre is the same index as
    in v1; only the distance used by every farthest-point step differs. Returns n indices (CPU long), selection order.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    _ = torch.randn(x.shape[1], PROJ_DIM, generator=g)       # consumed for identical seeded semantics (not used)
    z = x.float() @ W.to(x.device, torch.float32)
    idx = torch.empty(n, dtype=torch.long)
    cur = int(torch.randint(len(z), (1,), generator=g))
    mind = torch.full((len(z),), float('inf'), device=x.device)
    for i in range(n):
        idx[i] = cur
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        cur = int(mind.argmax())
    return idx


@torch.no_grad()
def cc_expand(x16: torch.Tensor, k: int, seed: int, base16: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """ContCore greedy expansion G_k(Z_t; B) (arXiv 2608.15277, Eq. 2): farthest-point sampling of the task tokens
    starting from the base set B = current keys; first point argmax_z d(z, B). Distances in the v1 projected space
    (same seeded P as rigel_cad.ops.greedy_kcenter), i.e. the selection metric of v1. Returns k indices (CPU long)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    P = (torch.randn(x16.shape[1], PROJ_DIM, generator=g) / math.sqrt(PROJ_DIM)).to(x16.device, torch.float32)
    z = x16.float() @ P
    zb = base16.to(x16.device).float() @ P
    mind = torch.empty(len(z), device=x16.device)
    for i in range(0, len(z), chunk):
        mind[i:i + chunk] = ops.sqdist(z[i:i + chunk], zb).min(1).values
    idx = torch.empty(k, dtype=torch.long)
    cur = int(mind.argmax())
    for i in range(k):
        idx[i] = cur
        mind = torch.minimum(mind, ((z - z[cur]) ** 2).sum(1))
        cur = int(mind.argmax())
    return idx


# ---------------------------------------------------------------------------------------------- per-task fit (shared)
@dataclass
class Fit:
    """Canonical per-task fit (v1): coreset indices (k-centre order), residual moment S_t (fp64), count n_t."""
    kc: torch.Tensor
    S: torch.Tensor
    n: int
    n_tokens: int
    keys16: Optional[torch.Tensor] = None    # C_t = x16[kc] (fp16, CPU)


@torch.no_grad()
def fit_task(x16: torch.Tensor, seed: int) -> Fit:
    """Exactly rigel_cad.model.RigelCAD.fit_task's computation (ops.coreset_size, ops.greedy_kcenter,
    ops.residual_moment)."""
    k = ops.coreset_size(len(x16))
    kc = ops.greedy_kcenter(x16, k, seed=seed)
    S, n = ops.residual_moment(x16.float(), kc[:k].to(x16.device))
    return Fit(kc=kc[:k].cpu(), S=S, n=int(n), n_tokens=len(x16), keys16=x16[kc[:k].to(x16.device)].cpu())


# ---------------------------------------------------------------------------------------------- arms
class ArmMemory:
    """Fast memory of one arm with provenance (task index and insertion stage of every key; diagnostics only)."""

    def __init__(self, name: str, seed: int, device: str = 'cuda'):
        self.name = name
        self.rule, self.cap, self.slow = ARMS[name]
        self.seed, self.device = seed, device
        self.keys = torch.empty(0, D, dtype=torch.float16, device=device)
        self.task = torch.empty(0, dtype=torch.long, device=device)
        self.log: List[dict] = []
        self.select_s = 0.0

    def bytes(self) -> int:
        """Registered nominal persistent bytes (v1 accounting: fp16 keys + S charged as fp32 + fp64 count)."""
        return len(self.keys) * KEY_BYTES + (SLOW_BYTES if self.slow else 0)

    @torch.no_grad()
    def write(self, l: int, C: torch.Tensor, x16: Optional[torch.Tensor], W_prev: Optional[torch.Tensor]) -> dict:
        """Task boundary l. C: canonical coreset keys [k, D] fp16 (device); x16: task tokens (CCW/CCE only);
        W_prev: fp32 whitener from tasks < l (GG01 only; None at l = 0). Returns the boundary log record."""
        t0 = time.time()
        k = len(C)
        new = C
        if self.rule == 'cc' and len(self.keys) > 0:
            e = cc_expand(x16, k, self.seed, self.keys)
            new = x16[e.to(x16.device)]
        if self.rule == 'union':
            cand, ctask = torch.cat([self.keys, new]), torch.cat([self.task, torch.full((k,), l, dtype=torch.long, device=self.device)])
            self.keys, self.task = cand, ctask
            rec = {'stage': l, 'n_cand': len(cand), 'n_keys': len(cand), 'evicted': 0, 'metric': 'none'}
        else:
            cand = torch.cat([self.keys, new])
            ctask = torch.cat([self.task, torch.full((len(new),), l, dtype=torch.long, device=self.device)])
            metric = 'none'
            if len(cand) > self.cap:
                if self.rule == 'gg' and W_prev is not None:
                    sel = kcenter_w(cand, self.cap, self.seed, W_prev)
                    metric = 'W_prev'
                else:
                    sel = ops.greedy_kcenter(cand, self.cap, seed=self.seed)
                    metric = 'P'
                keep = torch.arange(len(cand))[sel.sort().values].to(cand.device)   # survivors keep insertion order
                self.keys, self.task = cand[keep], ctask[keep]
            else:
                self.keys, self.task = cand, ctask
            rec = {'stage': l, 'n_cand': len(cand), 'n_keys': len(self.keys), 'evicted': len(cand) - len(self.keys),
                   'metric': metric}
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.select_s += time.time() - t0
        rec['retained_per_task'] = torch.bincount(self.task, minlength=l + 1).tolist()
        rec['bytes'] = self.bytes()
        self.log.append(rec)
        return rec


class Stream:
    """Runs several arms on one task stream from shared canonical per-task fits."""

    def __init__(self, arms: List[str], seed: int, device: str = 'cuda'):
        self.arms = {a: ArmMemory(a, seed, device) for a in arms}
        self.seed, self.device = seed, device
        self.S, self.n = None, 0
        self.W_hist: List[torch.Tensor] = []          # W_t (fp32, CPU) after each task
        self.cond: List[float] = []
        self.k_per_task: List[int] = []
        self.fit_s = 0.0
        self.snap: Dict[str, list] = {a: [] for a in arms}

    def W_now(self) -> torch.Tensor:
        W, cond = ops.whitener(self.S, self.n)
        if not torch.isfinite(W).all():
            raise FloatingPointError('non-finite whitener')
        return W.float(), cond

    @torch.no_grad()
    def task(self, l: int, x16: torch.Tensor, fit: Fit, fit_s: float = 0.0):
        x16 = x16.to(self.device) if x16 is not None else None
        C = (x16[fit.kc.to(self.device)] if x16 is not None else fit.keys16.to(self.device))
        W_prev = self.W_hist[-1].to(self.device) if self.W_hist else None
        self.fit_s += fit_s
        need_x = any(m.rule == 'cc' for m in self.arms.values())
        if need_x and x16 is None:
            raise ValueError('ContCore-style arms need the task tokens')
        for m in self.arms.values():
            m.write(l, C, x16, W_prev if m.rule == 'gg' else None)
        self.S = fit.S.clone() if self.S is None else self.S + fit.S
        self.n += fit.n
        W, cond = self.W_now()
        self.W_hist.append(W.cpu())
        self.cond.append(cond)
        self.k_per_task.append(len(fit.kc))
        for a, m in self.arms.items():
            self.snap[a].append(m.keys.cpu() if m.rule != 'union' else len(m.keys))

    def state(self, arm: str, l: int):
        """(keys fp16 CPU, W fp32 CPU or None) of arm after task l."""
        m = self.arms[arm]
        K = self.snap[arm][l]
        if isinstance(K, int):
            K = m.keys[:K].cpu()
        return K, (self.W_hist[l] if m.slow else None)

    @torch.no_grad()
    def energy(self, arm: str, l: int, q16: torch.Tensor) -> torch.Tensor:
        K, W = self.state(arm, l)
        keys = K.to(self.device).float()
        if W is None:
            return ops.energy_euclidean(q16, keys)
        return ops.energy_whitened(q16, keys, W.to(self.device))

    def drift(self) -> List[Optional[float]]:
        out = [None]
        for t in range(1, len(self.W_hist)):
            a, b = self.W_hist[t].double(), self.W_hist[t - 1].double()
            out.append(float(torch.linalg.norm(a - b) / torch.linalg.norm(b)))
        return out


def physical_bytes(n_keys: int, slow: bool) -> dict:
    """Actual tensor storage of the resumable persistent state (no serialisation overhead)."""
    keys = n_keys * KEY_BYTES
    s64 = D * D * 8 if slow else 0
    cnt = 8 if slow else 0
    return {'keys_fp16': keys, 'S_fp64': s64, 'count': cnt, 'total': keys + s64 + cnt,
            'nominal_registered': keys + (SLOW_BYTES if slow else 0)}
