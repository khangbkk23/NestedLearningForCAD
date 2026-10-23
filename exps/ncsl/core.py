"""Math and deterministic mechanics for NCSL-01.

This module deliberately keeps FPS discrete.  Gradients are taken only through
the metric for a fixed selected identity set; identities are reselected by an
explicit alternating step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from exps.csg01.select import select_fps


DIM = 768
RANK = 32
GAMMA = 0.25
K_TOTAL = 2500
BINS = 49
MIN_QUOTA = 16


class LowRankMetric(torch.nn.Module):
    """W = I + U V^T, with deterministic nonzero U and V=0 at W=I."""

    def __init__(self, dim: int = DIM, rank: int = RANK, seed: int = 20261022):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        # QR gives orthonormal columns; small U is nonzero while V=0 preserves W=I.
        q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=g, dtype=torch.float64), mode="reduced")
        self.U = torch.nn.Parameter((0.1 * q).to(torch.float32))
        self.V = torch.nn.Parameter(torch.zeros(dim, rank, dtype=torch.float32))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.V) @ self.U.transpose(0, 1)

    def update_matrix(self) -> torch.Tensor:
        return self.U @ self.V.transpose(0, 1)

    def spectral_norm(self) -> torch.Tensor:
        return torch.linalg.svdvals(self.update_matrix())[0]

    def project(self, gamma: float = GAMMA) -> float:
        with torch.no_grad():
            n = float(self.spectral_norm().item())
            if n > gamma and n > 0.0:
                self.U.mul_(gamma / n)
                n = float(self.spectral_norm().item())
            return n

    def state_digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for p in (self.U, self.V):
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def transformed(metric: LowRankMetric, x: torch.Tensor) -> torch.Tensor:
    return metric.apply(x)


def squared_distances(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    q2 = (q * q).sum(1, keepdim=True)
    x2 = (x * x).sum(1).unsqueeze(0)
    return (q2 + x2 - 2.0 * q @ x.transpose(0, 1)).clamp_min(0.0)


def nearest(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if q.numel() == 0:
        return q.new_empty((0,))
    if x.numel() == 0:
        raise ValueError("empty support")
    return squared_distances(q, x).min(dim=1).values.sqrt()


def deterministic_uniform_quotas(n_bins: int = BINS, total: int = K_TOTAL) -> np.ndarray:
    if total < n_bins * MIN_QUOTA:
        raise ValueError("budget below minimum quota")
    q = np.full(n_bins, MIN_QUOTA, dtype=np.int64)
    rem = total - int(q.sum())
    q += rem // n_bins
    q[: rem % n_bins] += 1
    return q


def proportional_quotas(bins: Sequence[torch.Tensor], total: int = K_TOTAL,
                        min_quota: int = MIN_QUOTA) -> tuple[np.ndarray, np.ndarray]:
    if len(bins) != BINS:
        raise ValueError("expected 49 bins")
    weights = []
    for x in bins:
        y = x.to(torch.float64)
        pilot = select_fps(y, min_quota)
        s = y[pilot]
        rad = nearest(y, s).max().item()
        weights.append(max(float(rad), 1e-12))
    w = np.asarray(weights, dtype=np.float64)
    q = np.full(len(bins), min_quota, dtype=np.int64)
    rem = total - int(q.sum())
    raw = rem * w / w.sum()
    base = np.floor(raw).astype(np.int64)
    q += base
    leftover = int(rem - int(base.sum()))
    frac = raw - base
    order = sorted(range(len(bins)), key=lambda i: (-float(frac[i]), i))
    for i in order[:leftover]:
        q[i] += 1
    if int(q.sum()) != total:
        raise AssertionError("quota arithmetic")
    return q, w


def select_support(bins: Sequence[torch.Tensor], metric: LowRankMetric,
                   quotas: np.ndarray) -> tuple[list[torch.Tensor], list[np.ndarray]]:
    if len(bins) != BINS or int(np.asarray(quotas).sum()) != K_TOTAL:
        raise ValueError("invalid support specification")
    out, ids = [], []
    with torch.no_grad():
        for b, x in enumerate(bins):
            tx = transformed(metric, x)
            idx = select_fps(tx.to(torch.float64), int(quotas[b]))
            ids.append(idx)
            out.append(x[idx].contiguous())
    return out, ids


def subset_excess(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    e = student - teacher
    if bool((e < -1e-5).any()):
        raise AssertionError("subset monotonicity violated")
    return e.clamp_min(0.0)


def byte_ledger(k: int = K_TOTAL, dim: int = DIM, rank: int = RANK) -> dict:
    support = k * dim * 4
    metric = 2 * dim * rank * 4
    optimizer = 2 * metric
    return {"support_bytes": support, "metric_bytes": metric,
            "optimizer_bytes": optimizer,
            "persistent_detector_bytes": support + metric,
            "training_bytes": support + metric + optimizer,
            "under_10mb": support + metric + optimizer < 10_000_000}

