"""RIGEL-D1-01 method code (registered in exps/rigel_q8/PREREG_D1.md): a CMS medium level that keeps evicted keys in
an int4 archive. Fast level int8 (written every task), medium level int4 (written only on eviction), slow level S
(fp32 triangle). Scoring: whitened nearest-key energy over fast u medium.

Arms: V1, Q8T, F8, U reuse exps.rigel_q8.q8.Arm unchanged (bitwise as in S3); Q4T and D1 are defined here.
"""
from __future__ import annotations

import time
from typing import Dict, List

import torch

from exps.rigel_q8 import q8
from rigel_cad import ops

D = q8.D
KEY4 = D // 2 + 2                                       # 386: packed int4 nibbles + fp16 scale
FREE = q8.BUDGET - q8.S_TRI                             # 7,207,416 key bytes next to a triangular S
CAP_Q4T = FREE // KEY4                                  # 18,672
CAP_D1_FAST = (FREE // 2) // q8.KEY8                    # 4,680
CAP_D1_MED = (FREE - CAP_D1_FAST * q8.KEY8) // KEY4     # 9,336
assert (CAP_Q4T, CAP_D1_FAST, CAP_D1_MED) == (18672, 4680, 9336)


@torch.no_grad()
def quantize4(x: torch.Tensor):
    """Symmetric per-row absmax int4: scale = fp16(max|x| / 7); q = clamp(round(x / scale), -7, 7) (held as int8)."""
    x = x.float()
    scale = (x.abs().amax(1) / 7.0).clamp_min(1e-12).half()
    q = torch.round(x / scale.float()[:, None]).clamp_(-7, 7).to(torch.int8)
    return q, scale


def pack4(q: torch.Tensor) -> torch.Tensor:
    """[N, 768] int8 in [-7, 7] -> [N, 384] uint8 (two nibbles per byte, offset 8)."""
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    return u[:, 0::2] | (u[:, 1::2] << 4)


def unpack4(p: torch.Tensor) -> torch.Tensor:
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = (p >> 4).to(torch.int16) - 8
    return torch.stack([lo, hi], dim=2).reshape(len(p), -1).to(torch.int8)


def _kc_keep(cand: torch.Tensor, cap: int, seed: int) -> torch.Tensor:
    """v1 consolidation: indices kept (insertion order) when the candidates exceed cap."""
    if len(cand) <= cap:
        return torch.arange(len(cand), device=cand.device)
    sel = ops.greedy_kcenter(cand, cap, seed=seed)
    return torch.arange(len(cand))[sel.sort().values].to(cand.device)


class _TriS:
    def update_S(self, S_t):
        prev = q8.tri_unpack(self.S_tri) if self.S_tri is not None else 0.
        self.S_tri = q8.tri_pack(prev + S_t.to(self.device))


class ArmQ4(_TriS):
    """Single-level int4 keys, triangular S (uniform-compression control)."""
    sfmt = 'tri'

    def __init__(self, name, seed, device='cuda'):
        self.name, self.seed, self.device, self.cap = name, seed, device, CAP_Q4T
        self.q = torch.empty(0, D, dtype=torch.int8, device=device)
        self.scale = torch.empty(0, dtype=torch.float16, device=device)
        self.task = torch.empty(0, dtype=torch.long, device=device)
        self.keys = torch.empty(0, D, device=device)
        self.S_tri, self.log, self.select_s, self.max_payload = None, [], 0.0, 0

    def payload(self):
        return len(self.task) * KEY4 + q8.S_TRI

    @torch.no_grad()
    def write(self, l, C16, S_t):
        t0 = time.time()
        qn, sn = quantize4(C16)
        cq, cs = torch.cat([self.q, qn]), torch.cat([self.scale, sn])
        ct = torch.cat([self.task, torch.full((len(C16),), l, dtype=torch.long, device=self.device)])
        keep = _kc_keep(q8.dequantize(cq, cs), self.cap, self.seed)
        self.q, self.scale, self.task = cq[keep], cs[keep], ct[keep]
        self.keys = q8.dequantize(self.q, self.scale)
        self.update_S(S_t)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.select_s += time.time() - t0
        p = self.payload()
        assert p <= q8.BUDGET, (self.name, p)
        self.max_payload = max(self.max_payload, p)
        rec = {'stage': l, 'n_keys': len(self.task), 'payload': p,
               'retained_per_task': torch.bincount(self.task, minlength=l + 1).tolist()}
        self.log.append(rec)
        return rec

    def state_dict(self):
        return {'q4': pack4(self.q).cpu(), 'scale': self.scale.cpu(), 'task': self.task.cpu(), 'S_tri': self.S_tri.cpu()}


class ArmD1(_TriS):
    """Fast int8 level (cap 4,680) + medium int4 archive of evicted keys (cap 9,336) + triangular S."""
    sfmt = 'tri'

    def __init__(self, name, seed, device='cuda'):
        self.name, self.seed, self.device = name, seed, device
        self.cap, self.cap_f, self.cap_m = CAP_D1_FAST + CAP_D1_MED, CAP_D1_FAST, CAP_D1_MED
        e8, e16, el = (torch.empty(0, D, dtype=torch.int8, device=device), torch.empty(0, dtype=torch.float16, device=device),
                       torch.empty(0, dtype=torch.long, device=device))
        self.fq, self.fs, self.ft = e8, e16, el
        self.mq, self.ms, self.mt = e8.clone(), e16.clone(), el.clone()
        self.keys = torch.empty(0, D, device=device)
        self.S_tri, self.log, self.select_s, self.max_payload = None, [], 0.0, 0

    @property
    def n_fast(self):
        return len(self.ft)

    def payload(self):
        return len(self.ft) * q8.KEY8 + len(self.mt) * KEY4 + q8.S_TRI

    @torch.no_grad()
    def write(self, l, C16, S_t):
        t0 = time.time()
        qn, sn = q8.quantize(C16)
        cq, cs = torch.cat([self.fq, qn]), torch.cat([self.fs, sn])
        ct = torch.cat([self.ft, torch.full((len(C16),), l, dtype=torch.long, device=self.device)])
        cand = q8.dequantize(cq, cs)
        keep = _kc_keep(cand, self.cap_f, self.seed)
        mask = torch.ones(len(cand), dtype=torch.bool, device=self.device)
        mask[keep] = False
        ev = mask.nonzero().flatten()                                  # evicted, insertion order
        self.fq, self.fs, self.ft = cq[keep], cs[keep], ct[keep]
        n_ev = len(ev)
        if n_ev:
            q4, s4 = quantize4(cand[ev])
            mq, ms, mt = torch.cat([self.mq, q4]), torch.cat([self.ms, s4]), torch.cat([self.mt, ct[ev]])
            mk = _kc_keep(q8.dequantize(mq, ms), self.cap_m, self.seed)
            self.mq, self.ms, self.mt = mq[mk], ms[mk], mt[mk]
        self.keys = torch.cat([q8.dequantize(self.fq, self.fs), q8.dequantize(self.mq, self.ms)])
        self.update_S(S_t)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.select_s += time.time() - t0
        p = self.payload()
        assert p <= q8.BUDGET, (self.name, p)
        self.max_payload = max(self.max_payload, p)
        rec = {'stage': l, 'n_keys': len(self.keys), 'n_fast': len(self.ft), 'n_medium': len(self.mt), 'evicted': n_ev,
               'payload': p, 'retained_per_task_fast': torch.bincount(self.ft, minlength=l + 1).tolist(),
               'retained_per_task_medium': torch.bincount(self.mt, minlength=l + 1).tolist()}
        self.log.append(rec)
        return rec

    def state_dict(self):
        return {'fq': self.fq.cpu(), 'fs': self.fs.cpu(), 'ft': self.ft.cpu(), 'mq4': pack4(self.mq).cpu(),
                'ms': self.ms.cpu(), 'mt': self.mt.cpu(), 'S_tri': self.S_tri.cpu()}


class Stream(q8.Stream):
    """q8.Stream with the D1-study arms (V1, Q8T, F8, U from q8.Arm; Q4T, D1 from here)."""

    def __init__(self, arms: List[str], seed: int, device: str = 'cuda'):
        mk = {'Q4T': ArmQ4, 'D1': ArmD1}
        self.arms = {a: (mk[a](a, seed, device) if a in mk else q8.Arm(a, seed, device)) for a in arms}
        self.device = device
        self.S, self.n = None, 0
        self.W_full: List[torch.Tensor] = []
        self.W_tri: Dict[str, List[torch.Tensor]] = {a: [] for a in arms if self.arms[a].sfmt == 'tri'}
        self.snap: Dict[str, list] = {a: [] for a in arms}
        self.cond: List[float] = []
        self.n_fast: List[int] = []

    @torch.no_grad()
    def task(self, l, fit):
        super().task(l, fit)
        if 'D1' in self.arms:
            self.n_fast.append(self.arms['D1'].n_fast)
