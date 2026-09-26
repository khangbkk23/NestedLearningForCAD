"""Distance helpers shared by the legacy and patch-vector memory banks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


SUPPORTED_DISTANCE_METRICS = {"euclidean", "cosine"}


def validate_distance_metric(metric: str) -> str:
    metric = str(metric).strip().lower()
    if metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(
            f"distance_metric must be one of {sorted(SUPPORTED_DISTANCE_METRICS)}, got {metric!r}."
        )
    return metric


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Return pairwise distances for two [N,D] and [M,D] feature matrices."""
    metric = validate_distance_metric(metric)
    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError(f"Expected compatible [N,D]/[M,D] tensors, got {tuple(x.shape)} and {tuple(y.shape)}.")
    if metric == "euclidean":
        return torch.cdist(x, y, p=2)
    # Cosine distance is implemented directly. Euclidean mode deliberately does
    # not normalize the features.
    return 1.0 - F.normalize(x, p=2, dim=-1) @ F.normalize(y, p=2, dim=-1).T
