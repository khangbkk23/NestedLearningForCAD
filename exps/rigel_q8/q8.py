"""RIGEL-Q8-01 method code (registered in exps/rigel_q8/PREREG.md): int8 keys and a triangular slow statistic, so the
same 8 MiB holds more keys. Only storage changes; the write rule (1 % per-task k-centre coreset, v1 consolidation)
and the read-out (whitened nearest-key energy) are the v1 primitives of rigel_cad.ops.

Arms (shared per-task fit C_t, S_t, n_t from exps.rigel_gg01.gg.fit_task):
    V1   fp16 keys, cap 3,925, S v1 convention      T16  fp16 keys, cap 4,692, S fp32 triangle
    Q8   int8 keys, cap 7,830, S v1 convention      Q8T  int8 keys, cap 9,360, S fp32 triangle   (candidate)
    F    fp16 keys, cap 5,461, Euclidean            F8   int8 keys, cap 10,894, Euclidean
    U    fp16 keys, unbounded, S v1 convention
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from rigel_cad import ops
from rigel_cad.config import TOKEN_DIM

D = TOKEN_DIM
BUDGET = 8 * 2 ** 20                                   # 8,388,608
KEY16 = 2 * D                                          # 1,536
KEY8 = D + 2                                           # 770: int8 values + fp16 scale
S_FULL = D * D * 4 + 8                                 # 2,359,304 (v1 charge)
N_TRI = D * (D + 1) // 2                               # 295,296
S_TRI = N_TRI * 4 + 8                                  # 1,181,192

ARMS = {  # name: (key format, S format or None, cap)
    'V1': ('fp16', 'full', (BUDGET - S_FULL) // KEY16),
    'T16': ('fp16', 'tri', (BUDGET - S_TRI) // KEY16),
    'Q8': ('int8', 'full', (BUDGET - S_FULL) // KEY8),
    'Q8T': ('int8', 'tri', (BUDGET - S_TRI) // KEY8),
    'F': ('fp16', None, BUDGET // KEY16),
    'F8': ('int8', None, BUDGET // KEY8),
    'U': ('fp16', 'full', None),
}
assert [ARMS[a][2] for a in ('V1', 'T16', 'Q8', 'Q8T', 'F', 'F8')] == [3925, 4692, 7830, 9360, 5461, 10894]


# ---------------------------------------------------------------------------------------------- quantiser
@torch.no_grad()
def quantize(x: torch.Tensor):
    """Symmetric per-row absmax int8: scale = fp16(max|x| / 127); q = clamp(round(x / scale), -127, 127)."""
    x = x.float()
    scale = (x.abs().amax(1) / 127.0).clamp_min(1e-12).half()
    q = torch.round(x / scale.float()[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float()[:, None]


# ---------------------------------------------------------------------------------------------- triangular S
_IU = None


def _iu(device):
    global _IU
    if _IU is None or _IU[0].device != torch.device(device):
        _IU = torch.triu_indices(D, D, device=device)
    return _IU


def tri_pack(S: torch.Tensor) -> torch.Tensor:
    """fp32 upper triangle (row-major, including the diagonal) of a symmetric [D, D] matrix."""
    i, j = _iu(S.device)
    return S[i, j].float()


def tri_unpack(t: torch.Tensor) -> torch.Tensor:
    """fp64 symmetric matrix from an fp32 upper triangle."""
    i, j = _iu(t.device)
    S = torch.zeros(D, D, dtype=torch.float64, device=t.device)
    S[i, j] = t.double()
    S[j, i] = t.double()
    return S


# ---------------------------------------------------------------------------------------------- arm memory
class Arm:
    def __init__(self, name: str, seed: int, device: str = 'cuda'):
        self.name = name
        self.kfmt, self.sfmt, self.cap = ARMS[name]
        self.seed, self.device = seed, device
        self.keys = torch.empty(0, D, device=device)                    # stored vectors as scored (fp32 view)
        self.q = torch.empty(0, D, dtype=torch.int8, device=device)     # int8 storage (int8 arms)
        self.scale = torch.empty(0, dtype=torch.float16, device=device)
        self.k16 = torch.empty(0, D, dtype=torch.float16, device=device)  # fp16 storage (fp16 arms)
        self.task = torch.empty(0, dtype=torch.long, device=device)
        self.S_tri: Optional[torch.Tensor] = None                       # fp32 triangle (tri arms)
        self.log: List[dict] = []
        self.select_s = 0.0
        self.max_payload = 0

    def payload(self) -> int:
        n = len(self.task)
        kb = n * (KEY8 if self.kfmt == 'int8' else KEY16)
        sb = {'full': S_FULL, 'tri': S_TRI, None: 0}[self.sfmt]
        return kb + sb

    @torch.no_grad()
    def write(self, l: int, C16: torch.Tensor, S_t: torch.Tensor) -> dict:
        t0 = time.time()
        k = len(C16)
        if self.kfmt == 'int8':
            qn, sn = quantize(C16)
            cand_q, cand_s = torch.cat([self.q, qn]), torch.cat([self.scale, sn])
            cand = dequantize(cand_q, cand_s)
        else:
            cand16 = torch.cat([self.k16, C16.half()])
            cand = cand16.float()
        ctask = torch.cat([self.task, torch.full((k,), l, dtype=torch.long, device=self.device)])
        if self.cap is not None and len(cand) > self.cap:
            # v1 rule: seeded projected greedy k-centre; fp16 arms pass fp16 (bitwise v1), int8 arms the stored values
            sel = ops.greedy_kcenter(cand16 if self.kfmt == 'fp16' else cand, self.cap, seed=self.seed)
            keep = torch.arange(len(cand))[sel.sort().values].to(self.device)
        else:
            keep = torch.arange(len(cand), device=self.device)
        self.task = ctask[keep]
        if self.kfmt == 'int8':
            self.q, self.scale = cand_q[keep], cand_s[keep]
            self.keys = dequantize(self.q, self.scale)
        else:
            self.k16 = cand16[keep]
            self.keys = self.k16.float()
        if self.sfmt == 'tri':
            prev = tri_unpack(self.S_tri) if self.S_tri is not None else 0.
            self.S_tri = tri_pack(prev + S_t.to(self.device))
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.select_s += time.time() - t0
        p = self.payload()
        if self.cap is not None and p > BUDGET:
            raise AssertionError(f'{self.name}: payload {p} > {BUDGET} after task {l}')
        self.max_payload = max(self.max_payload, p)
        rec = {'stage': l, 'n_cand': len(cand), 'n_keys': len(self.task), 'payload': p,
               'retained_per_task': torch.bincount(self.task, minlength=l + 1).tolist()}
        self.log.append(rec)
        return rec

    def state_dict(self) -> dict:
        d = {'task': self.task.cpu()}
        if self.kfmt == 'int8':
            d.update(q=self.q.cpu(), scale=self.scale.cpu())
        else:
            d.update(k16=self.k16.cpu())
        if self.sfmt == 'tri':
            d['S_tri'] = self.S_tri.cpu()
        return d


class Stream:
    """All arms on one task stream; the v1-convention S (fp64 sum) and W history are shared."""

    def __init__(self, arms: List[str], seed: int, device: str = 'cuda'):
        self.arms = {a: Arm(a, seed, device) for a in arms}
        self.device = device
        self.S, self.n = None, 0
        self.W_full: List[torch.Tensor] = []
        self.W_tri: Dict[str, List[torch.Tensor]] = {a: [] for a in arms if ARMS[a][1] == 'tri'}
        self.snap: Dict[str, list] = {a: [] for a in arms}
        self.cond: List[float] = []

    @staticmethod
    def _W(S, n):
        W, cond = ops.whitener(S, n)
        if not torch.isfinite(W).all():
            raise FloatingPointError('non-finite whitener')
        return W.float().cpu(), float(cond)

    @torch.no_grad()
    def task(self, l: int, fit):
        C16 = fit.keys16.to(self.device)
        for a, m in self.arms.items():
            m.write(l, C16, fit.S)
        self.S = fit.S.clone() if self.S is None else self.S + fit.S
        self.n += fit.n
        W, cond = self._W(self.S, self.n)
        self.W_full.append(W)
        self.cond.append(cond)
        for a in self.W_tri:
            self.W_tri[a].append(self._W(tri_unpack(self.arms[a].S_tri).cpu(), self.n)[0])
        for a, m in self.arms.items():
            self.snap[a].append(m.keys.cpu() if m.cap is not None else len(m.keys))

    def state(self, arm: str, l: int):
        m = self.arms[arm]
        K = self.snap[arm][l]
        if isinstance(K, int):
            K = m.keys[:K].cpu()
        if m.sfmt is None:
            return K, None
        return K, (self.W_tri[arm][l] if m.sfmt == 'tri' else self.W_full[l])

    @torch.no_grad()
    def energy(self, arm: str, l: int, q16: torch.Tensor) -> torch.Tensor:
        K, W = self.state(arm, l)
        keys = K.to(self.device).float()
        if W is None:
            return ops.energy_euclidean(q16, keys)
        return ops.energy_whitened(q16, keys, W.to(self.device))
