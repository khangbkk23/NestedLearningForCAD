"""Whitened Exemplar Reader: task-free e(z) = 1/2 min_m ||zW - mW||^2 over the union bank.

Oracle: exps/nanm_r2/run.py::sqdist, exps/bench_g3/run.py::whitened (query rows flattened, chunks of 512).
"""
from __future__ import annotations

import torch


def sqdist(q, m):
    """fp32 squared Euclidean distances [n, k], clamped >= 0 (expanded form, as the oracle)."""
    return (q.pow(2).sum(1, keepdim=True) + m.pow(2).sum(1)[None] - 2 * q @ m.T).clamp_min(0)


@torch.no_grad()
def whitened_min_energy(q16, mw, W, qc):
    """q16: CPU fp16 unit queries; mw = keys32 @ W (cached per boundary); W fp32 on the bank device."""
    out = torch.empty(len(q16))
    for i in range(0, len(q16), qc):
        out[i:i + qc] = (0.5 * sqdist(q16[i:i + qc].to(W.device).float() @ W, mw).min(1).values).cpu()
    return out


class WhitenedExemplarReader:
    def __init__(self, cfg):
        self.cfg = cfg

    @torch.no_grad()
    def cache(self, keys16, W32, device):
        """Derived inference cache, rebuilt at every boundary: (W on device, mW = keys32 @ W)."""
        Wd = W32.to(device)
        return Wd, keys16.to(device).float() @ Wd

    def energies(self, q16, Wd, mw):
        return whitened_min_energy(q16, mw, Wd, self.cfg.query_chunk)
