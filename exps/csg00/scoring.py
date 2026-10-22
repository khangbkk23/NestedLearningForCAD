"""Exact distance functions for CSG-00 arms A, B and C.

Arm A and C use an exact Euclidean nearest-neighbour distance inside the query
patch's spatial bin.  Arm B uses the closed-form distance to the bounded support
``{mu + U a + v : |a_j| <= R_j, U^T v = 0, ||v|| <= rho}``:

    dist(z, S) = sqrt( sum_j max(|a_j| - R_j, 0)^2 + max(||e|| - rho, 0)^2 )

with ``a = U^T (z - mu)`` and ``e = (z - mu) - U a``.

CPU canonical scores are computed in float64.  The CUDA path computes candidate
neighbours in float32 and recomputes the selected candidates in float64, which
keeps scores numerically accurate while bounding FP32 cost.  A CPU/CUDA parity
check is part of the run evidence.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import contract as C

EXACT_CANDIDATES = 4


def _as_device_dtype(tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype)


def squared_distances(Q: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """[nq, m] squared distances via the norm-dot identity, clamped at zero."""
    q2 = (Q * Q).sum(dim=1, keepdim=True)
    x2 = (X * X).sum(dim=1).unsqueeze(0)
    d2 = q2 + x2 - 2.0 * (Q @ X.transpose(0, 1))
    return d2.clamp_min_(0.0)


def nn_distance(Q: torch.Tensor, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Exact nearest-neighbour Euclidean distance for each row of Q.

    CPU: float64 norm-dot, exact for the frozen feature scale.
    CUDA: float32 candidate selection, then float64 recomputation of the four
    best candidates; the minimum of those is the returned distance.
    """
    if X.shape[0] == 0:
        raise RuntimeError("empty normal support")
    if not torch.isfinite(X).all() or not torch.isfinite(Q).all():
        raise RuntimeError("non-finite descriptor")

    if device == "cpu":
        Qd = _as_device_dtype(Q, device, torch.float64)
        Xd = _as_device_dtype(X, device, torch.float64)
        return squared_distances(Qd, Xd).min(dim=1).values.sqrt()

    Qf = _as_device_dtype(Q, device, torch.float32)
    Xf = _as_device_dtype(X, device, torch.float32)
    d2 = squared_distances(Qf, Xf)
    kk = min(EXACT_CANDIDATES, Xf.shape[0])
    _, idx = torch.topk(d2, kk, dim=1, largest=False, sorted=True)
    Qd = Q.to(device=device, dtype=torch.float64)
    Xd = X.to(device=device, dtype=torch.float64)
    best = torch.full((Q.shape[0],), float("inf"), dtype=torch.float64, device=device)
    for j in range(kk):
        diff = Qd - Xd[idx[:, j]]
        best = torch.minimum(best, diff.norm(dim=1))
    return best


def nn_distance_with_index(Q: torch.Tensor, X: torch.Tensor, device: str = "cpu"
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (distance, index) with deterministic first-index tie handling."""
    if X.shape[0] == 0:
        raise RuntimeError("empty normal support")
    if device == "cpu":
        Qd = _as_device_dtype(Q, device, torch.float64)
        Xd = _as_device_dtype(X, device, torch.float64)
        vals, idx = squared_distances(Qd, Xd).min(dim=1)
        return vals.sqrt(), idx
    Qf = _as_device_dtype(Q, device, torch.float32)
    Xf = _as_device_dtype(X, device, torch.float32)
    d2 = squared_distances(Qf, Xf)
    _, idx = d2.min(dim=1)
    Qd = Q.to(device=device, dtype=torch.float64)
    Xd = X.to(device=device, dtype=torch.float64)
    diff = Qd - Xd[idx]
    return diff.norm(dim=1), idx


def mode_distance(Q: torch.Tensor, mu: torch.Tensor, U: torch.Tensor,
                  R: torch.Tensor, rho: float, device: str = "cpu") -> torch.Tensor:
    """Closed-form distance to one bounded tangent-box x orthogonal-ball mode."""
    Qd = _as_device_dtype(Q, device, torch.float64)
    mud = _as_device_dtype(mu, device, torch.float64)
    Ud = _as_device_dtype(U, device, torch.float64)
    Rd = _as_device_dtype(R, device, torch.float64)
    d = Qd - mud
    if Ud.shape[1] == 0:
        et = torch.zeros((Qd.shape[0],), dtype=torch.float64, device=device)
        e = d
    else:
        a = d @ Ud
        e = d - a @ Ud.transpose(0, 1)
        et = torch.clamp(a.abs() - Rd, min=0.0)
        et = (et * et).sum(dim=1)
    eo = torch.clamp(e.norm(dim=1) - float(rho), min=0.0)
    out = torch.sqrt(et + eo * eo)
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("non-finite mode distance")
    return out


def score_binned(queries_by_bin: Sequence[torch.Tensor],
                 descriptors_by_bin: Sequence[torch.Tensor],
                 device: str = "cpu") -> List[torch.Tensor]:
    """Arm A / C: nearest distance restricted to the query patch's own bin."""
    if len(queries_by_bin) != C.BINS or len(descriptors_by_bin) != C.BINS:
        raise ValueError("expected 49 bins")
    out: List[torch.Tensor] = []
    for b in range(C.BINS):
        out.append(nn_distance(queries_by_bin[b], descriptors_by_bin[b], device=device))
    return out


def score_global(queries_by_bin: Sequence[torch.Tensor],
                 descriptors_by_bin: Sequence[torch.Tensor],
                 device: str = "cpu") -> List[torch.Tensor]:
    """Report-only global unrestricted NN: min over all 49 descriptor bins.

    The 49 bins partition the 784 token positions, so the global minimum is the
    minimum of the 49 within-bin minima.  This is a diagnostic, never a primary
    result.
    """
    if len(queries_by_bin) != C.BINS or len(descriptors_by_bin) != C.BINS:
        raise ValueError("expected 49 bins")
    # Diagnostic only: float32 candidate selection with exact float64 winner
    # recomputation keeps memory bounded on CPU and CUDA.
    work_dtype = torch.float32
    Xd = [x.to(device=device, dtype=work_dtype) for x in descriptors_by_bin]
    out: List[torch.Tensor] = []
    for b in range(C.BINS):
        Q = queries_by_bin[b]
        Qw = Q.to(device=device, dtype=work_dtype)
        q2 = (Qw * Qw).sum(dim=1, keepdim=True)
        nq = Qw.shape[0]
        best = torch.full((nq,), float("inf"), dtype=torch.float64, device=device)
        best_bin = torch.full((nq,), -1, dtype=torch.long, device=device)
        best_idx = torch.zeros((nq,), dtype=torch.long, device=device)
        for b2 in range(C.BINS):
            X = Xd[b2]
            x2 = (X * X).sum(dim=1).unsqueeze(0)
            d2 = (q2 + x2 - 2.0 * (Qw @ X.transpose(0, 1))).clamp_min_(0.0)
            vals, idx = d2.min(dim=1)
            vals64 = vals.to(torch.float64)
            upd = vals64 < best
            if bool(upd.any()):
                best = torch.where(upd, vals64, best)
                best_bin = torch.where(upd, torch.full_like(best_bin, b2), best_bin)
                best_idx = torch.where(upd, idx, best_idx)
        # exact float64 recomputation of the winning candidate
        Qe = Q.to(device=device, dtype=torch.float64)
        exact = torch.empty((nq,), dtype=torch.float64, device=device)
        for b2 in torch.unique(best_bin).tolist():
            if b2 < 0:
                raise RuntimeError("global diagnostic produced no neighbour")
            mask = best_bin == b2
            Xe = descriptors_by_bin[b2].to(device=device, dtype=torch.float64)
            diff = Qe[mask] - Xe[best_idx[mask]]
            exact[mask] = diff.norm(dim=1)
        out.append(exact)
    return out


def score_modes(queries_by_bin: Sequence[torch.Tensor], state: Dict[str, np.ndarray],
                device: str = "cpu") -> List[torch.Tensor]:
    """Arm B: min over ready modes per bin, using the serialized state arrays."""
    ready = state["ready"]
    mu, U, R, rho = state["mu"], state["U"], state["R"], state["rho"]
    out: List[torch.Tensor] = []
    for b in range(C.BINS):
        Q = queries_by_bin[b]
        best = torch.full((Q.shape[0],), float("inf"), dtype=torch.float64, device=device)
        n_ready = 0
        for k in range(C.K_MODES):
            if int(ready[b, k]) == 0:
                continue
            n_ready += 1
            dist = mode_distance(Q, torch.as_tensor(mu[b, k]), torch.as_tensor(U[b, k]),
                                 torch.as_tensor(R[b, k]), float(rho[b, k]), device=device)
            best = torch.minimum(best, dist)
        if n_ready == 0:
            raise RuntimeError(f"bin {b} has no ready mode (pre-registered BLOCKED condition)")
        if not bool(torch.isfinite(best).all()):
            raise RuntimeError(f"bin {b} produced non-finite mode scores")
        out.append(best)
    return out


def score_reservoir(queries_by_bin: Sequence[torch.Tensor], state: Dict[str, np.ndarray],
                    device: str = "cpu") -> List[torch.Tensor]:
    """Arm C: exact NN inside the bin's byte-matched reservoir."""
    res, count = state["res"], state["count"]
    out: List[torch.Tensor] = []
    for b in range(C.BINS):
        n = int(count[b])
        if n <= 0:
            raise RuntimeError(f"bin {b} has an empty reservoir")
        out.append(nn_distance(queries_by_bin[b], torch.as_tensor(res[b, :n]), device=device))
    return out


def bin_queries(features: torch.Tensor, b: int, positions: Sequence[np.ndarray]) -> torch.Tensor:
    pos = positions[b]
    view = features[:, torch.as_tensor(pos, dtype=torch.long), :]
    return view.reshape(-1, view.shape[-1])


def per_bin_queries(features: torch.Tensor, positions: Sequence[np.ndarray]) -> List[torch.Tensor]:
    """[n_img, 784, 768] -> list of 49 [n_img*16, 768] query tensors."""
    return [bin_queries(features, b, positions) for b in range(C.BINS)]


def assemble(per_bin: Sequence[torch.Tensor], n_images: int,
             positions: Sequence[np.ndarray]) -> torch.Tensor:
    """Inverse of per_bin_queries: [n_img, 784] float64 scores on CPU.

    Device-agnostic scoring tensors are copied back to CPU here so that score
    aggregation, metric computation and JSON evidence are device-independent.
    """
    out = torch.empty((n_images, C.PATCHES), dtype=torch.float64)
    for b in range(C.BINS):
        pos = positions[b]
        block = per_bin[b].reshape(n_images, len(pos)).detach().cpu().to(torch.float64)
        out[:, torch.as_tensor(pos, dtype=torch.long)] = block
    return out
