"""Spatially eligible scoring and covering-radius diagnostics for CSG-01."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from exps.csg00 import scoring as S0

from . import contract as C1
from . import quantize


def support_bins_from_state(state: Dict[str, np.ndarray], arm: str) -> List[torch.Tensor]:
    counts = np.asarray(state["counts"], dtype=np.int64)
    offsets = np.asarray(state["offsets"], dtype=np.int64)
    if arm in {"B", "E"}:
        return [torch.as_tensor(np.asarray(state["desc"])[offsets[b]:offsets[b] + counts[b]],
                                dtype=torch.float32) for b in range(C1.BINS)]
    if arm in {"C", "D"}:
        return quantize.quantized_bin_support(state)
    raise ValueError(f"unknown arm {arm}")


def score_bins(queries_by_bin: Sequence[torch.Tensor],
               support_bins: Sequence[torch.Tensor],
               device: str = "cpu") -> List[torch.Tensor]:
    if len(queries_by_bin) != C1.BINS or len(support_bins) != C1.BINS:
        raise ValueError("expected 49 bins")
    out = []
    for b in range(C1.BINS):
        if support_bins[b].shape[0] == 0:
            raise RuntimeError(f"bin {b}: empty support")
        out.append(S0.nn_distance(queries_by_bin[b], support_bins[b], device=device))
    return out


def score_arm(queries_by_bin: Sequence[torch.Tensor],
              support_bins: Sequence[torch.Tensor],
              n_images: int, positions: Sequence[np.ndarray],
              device: str = "cpu") -> torch.Tensor:
    return S0.assemble(score_bins(queries_by_bin, support_bins, device=device),
                       n_images, positions)


def covering_radius(full_bins: Sequence[torch.Tensor],
                    support_bins: Sequence[torch.Tensor]) -> np.ndarray:
    """epsilon_b = max_{x in D_b} min_{c in C_b} ||x-c||_2, per bin (float64 CPU)."""
    if len(full_bins) != C1.BINS or len(support_bins) != C1.BINS:
        raise ValueError("expected 49 bins")
    out = np.zeros((C1.BINS,), dtype=np.float64)
    for b in range(C1.BINS):
        D = full_bins[b].to(torch.float64)
        S = support_bins[b].to(torch.float64)
        if S.shape[0] == 0:
            raise RuntimeError(f"bin {b}: empty support")
        d2 = (D * D).sum(dim=1).unsqueeze(1) + (S * S).sum(dim=1).unsqueeze(0) \
            - 2.0 * (D @ S.transpose(0, 1))
        d2.clamp_min_(0.0)
        out[b] = float(torch.sqrt(d2.min(dim=1).values.max()).item())
    return out


def radius_summary(values: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)
    return {
        "n_bins": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
        "min": float(v.min()),
    }


def image_max(scores: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    return np.asarray(scores, dtype=np.float64).max(axis=1)


def image_max_bounds(epsilon: np.ndarray, delta: np.ndarray,
                     scores_D: np.ndarray, scores_X: np.ndarray,
                     positions: Sequence[np.ndarray]) -> Dict[str, Any]:
    """Check the image-max two-sided bounds for subset (B/E/D) and quantized (C) arms."""
    sD = np.asarray(scores_D, dtype=np.float64).max(axis=1)
    sX = np.asarray(scores_X, dtype=np.float64).max(axis=1)
    eps_max = float(np.max(epsilon))
    delta_max = float(np.max(delta))
    return {
        "epsilon_max": eps_max,
        "delta_max": delta_max,
        "lower_gap": float(np.min(sX - sD)),
        "upper_gap": float(np.max(sX - sD)),
        "subset_monotone_all": bool(np.all(sX >= sD - C1.QUANT_TOL)),
        "upper_bound": float(sD.max() + eps_max + delta_max),
        "lower_bound": float(sD.max() - delta_max),
    }
