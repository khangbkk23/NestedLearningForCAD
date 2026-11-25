"""Residual Normality Geometry Estimator (slow level, closed form, once per task boundary).

S <- S + sum_{x in X_t \\ M_t} r r^T, r = x - NN^E_{M_t}(x) (own-task bank, uncentred, fp64 accumulation);
n <- n + |X_t \\ M_t|;  W = (0.5 S/n + 0.5 tr(S/n)/d I)^{-1/2}.
Oracle: exps/nanm_r2b/run.py::residual_stats (mem_idx path), ::whiten; exps/nanm_r2/run.py::shrink, ::inv_sqrt_psd.
"""
from __future__ import annotations

import torch

from nanm.retrieval import sqdist


@torch.no_grad()
def residual_moment(fit, mem_idx, chunk):
    """fit [N, D] fp32 on device; mem_idx on device. Returns (S fp64 CPU, residual count)."""
    n, D = fit.shape
    S = torch.zeros(D, D, dtype=torch.float64, device=fit.device); cnt = 0
    member = torch.zeros(n, dtype=torch.bool, device=fit.device); member[mem_idx] = True
    mem = fit[mem_idx]
    for i in range(0, n, chunk):
        x = fit[i:i + chunk]
        x = x[~member[i:i + chunk]]
        j = sqdist(x, mem).argmin(1); r = x - mem[j]
        S += (r.T @ r).double(); cnt += len(r)
    return S.cpu(), cnt


def shrink(C, alpha):
    return (1 - alpha) * C + alpha * (torch.trace(C) / C.shape[0]) * torch.eye(C.shape[0], dtype=C.dtype)


def inv_sqrt_psd(S, floor):
    w, V = torch.linalg.eigh(S.double())
    if not float(w.min()) > floor:
        raise RuntimeError('metric not positive definite')
    return (V * w.rsqrt()) @ V.T, float(w.max() / w.min())


def whiten(Ssum, n, alpha, floor):
    return inv_sqrt_psd(shrink(Ssum / n, alpha), floor)


class ResidualNormalityGeometryEstimator:
    def __init__(self, cfg):
        self.cfg = cfg

    def task_statistic(self, fit16, mem_idx):
        return residual_moment(fit16.float(), mem_idx, self.cfg.residual_chunk)

    def geometry(self, S, n):
        """fp64 W, cond(T). Callers store W.float() for reading (as the oracle stage matrices)."""
        return whiten(S, n, self.cfg.shrink_alpha, self.cfg.eig_floor)
