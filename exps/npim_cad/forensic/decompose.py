# exps/npim_cad/forensic/decompose.py
"""Exact NPIM mixture/residual decomposition for EXP-NPIM-02.

For ready levels l with fixed weights omega_l summing to one:

    yhat = sum_l omega_l W_l k
    r    = y - yhat = sum_l omega_l r_l
    s    = ||r||^2/d
         = sum_l omega_l^2 s_l
           + (2/d) sum_{l<m} omega_l omega_m <r_l, r_m>

This module computes the identity, per-level scores, pairwise residual dot
products/cosines, and the effective linear map sum_l omega_l W_l.  It is
read-only and never updates a state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch

LEVEL_NAMES = ("fast", "mid", "slow")
PAIR_INDEX = ((0, 1), (0, 2), (1, 2))


@dataclass
class MixtureDecomposition:
    level_scores: torch.Tensor       # [L, N]
    fused_scores: torch.Tensor       # [N]
    self_terms: torch.Tensor         # [L, N] = omega_l^2 s_l
    cross_terms: torch.Tensor        # [P, N] = (2/d) omega_i omega_j <r_i,r_j>
    dot_products: torch.Tensor       # [P, N] = <r_i,r_j>
    cosine: torch.Tensor             # [P, N]
    residual_l2: torch.Tensor        # [L, N] = ||r_l||
    fused_residual_l2: torch.Tensor  # [N]
    identity_max_abs: float
    effective_weight_error: float


def _check_inputs(K: torch.Tensor, Y: torch.Tensor, weights: torch.Tensor, Ws: Sequence[torch.Tensor]) -> None:
    if K.ndim != 2 or Y.ndim != 2 or K.shape[0] != Y.shape[0]:
        raise ValueError("K and Y must share the patch dimension")
    if len(Ws) != len(weights) or len(Ws) == 0:
        raise ValueError("one W per weight is required")
    if not torch.isfinite(K).all() or not torch.isfinite(Y).all():
        raise ValueError("non-finite K or Y")
    if not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
        raise ValueError("invalid mixture weights")
    for W in Ws:
        if W.ndim != 2 or W.shape[1] != K.shape[1] or W.shape[0] != Y.shape[1]:
            raise ValueError("W geometry mismatch")
        if not torch.isfinite(W).all():
            raise ValueError("non-finite W")


@torch.no_grad()
def decompose_mixture(
    Y: torch.Tensor,
    K: torch.Tensor,
    Ws: Sequence[torch.Tensor],
    weights: torch.Tensor,
) -> MixtureDecomposition:
    """Compute the exact per-level/residual decomposition for one entry."""

    weights = weights.to(device=K.device, dtype=K.dtype)
    weights = weights / weights.sum()
    _check_inputs(K, Y, weights, Ws)
    d = Y.shape[1]
    level_count = len(Ws)

    residuals: List[torch.Tensor] = []
    level_scores: List[torch.Tensor] = []
    residual_l2: List[torch.Tensor] = []
    yhats: List[torch.Tensor] = []
    for W in Ws:
        yhat = K @ W.to(device=K.device, dtype=K.dtype).T
        residual = Y - yhat
        residuals.append(residual)
        level_scores.append(residual.square().mean(dim=-1))
        residual_l2.append(residual.norm(dim=-1))
        yhats.append(yhat)
    level_scores_t = torch.stack(level_scores, dim=0)
    residual_l2_t = torch.stack(residual_l2, dim=0)

    fused_yhat = torch.zeros_like(Y)
    for index, yhat in enumerate(yhats):
        fused_yhat = fused_yhat + weights[index] * yhat
    fused_residual = Y - fused_yhat
    fused_scores = fused_residual.square().mean(dim=-1)

    self_terms = torch.stack(
        [weights[index].square() * level_scores_t[index] for index in range(level_count)], dim=0
    )
    pair_dots: List[torch.Tensor] = []
    pair_cross: List[torch.Tensor] = []
    pair_cosine: List[torch.Tensor] = []
    for left, right in PAIR_INDEX:
        if left >= level_count or right >= level_count:
            continue
        dot = (residuals[left] * residuals[right]).sum(dim=-1)
        pair_dots.append(dot)
        pair_cross.append((2.0 / d) * weights[left] * weights[right] * dot)
        denominator = residual_l2_t[left] * residual_l2_t[right]
        pair_cosine.append(dot / denominator.clamp_min(1e-12))
    dot_products = torch.stack(pair_dots, dim=0) if pair_dots else torch.zeros((0, Y.shape[0]), device=Y.device)
    cross_terms = torch.stack(pair_cross, dim=0) if pair_cross else torch.zeros((0, Y.shape[0]), device=Y.device)
    cosine = torch.stack(pair_cosine, dim=0) if pair_cosine else torch.zeros((0, Y.shape[0]), device=Y.device)

    reconstructed = self_terms.sum(dim=0) + cross_terms.sum(dim=0)
    identity_max_abs = float((reconstructed - fused_scores).abs().max().item())

    effective_W = torch.zeros_like(Ws[0])
    for index, W in enumerate(Ws):
        effective_W = effective_W + weights[index] * W
    effective_yhat = K @ effective_W.T
    effective_weight_error = float((effective_yhat - fused_yhat).abs().max().item())

    return MixtureDecomposition(
        level_scores=level_scores_t,
        fused_scores=fused_scores,
        self_terms=self_terms,
        cross_terms=cross_terms,
        dot_products=dot_products,
        cosine=cosine,
        residual_l2=residual_l2_t,
        fused_residual_l2=fused_residual.norm(dim=-1),
        identity_max_abs=identity_max_abs,
        effective_weight_error=effective_weight_error,
    )


def mix_from_statistics(
    level_scores: torch.Tensor,
    dot_products: torch.Tensor,
    weights: torch.Tensor,
    d: int = 768,
) -> torch.Tensor:
    """Reconstruct any weighted mixture score from stored s_l and pairwise dots.

    ``level_scores`` is [L,N]; ``dot_products`` is [P,N] in PAIR_INDEX order.
    Weights may be zero for excluded levels.  This is exact algebra, not a model.
    """

    if level_scores.ndim != 2:
        raise ValueError("level_scores must be [L,N]")
    weights = weights.to(device=level_scores.device, dtype=level_scores.dtype)
    if weights.numel() != level_scores.shape[0]:
        raise ValueError("weight/level count mismatch")
    weights = weights / weights.sum()
    self_terms = torch.zeros_like(level_scores[0])
    for index in range(weights.numel()):
        if float(weights[index]) != 0.0:
            if not torch.isfinite(level_scores[index]).all():
                raise ValueError("non-finite level score for an active weight")
            self_terms = self_terms + weights[index].square() * level_scores[index]
    total = self_terms
    for pair_index, (left, right) in enumerate(PAIR_INDEX):
        if left >= weights.numel() or right >= weights.numel() or pair_index >= dot_products.shape[0]:
            continue
        if float(weights[left]) == 0.0 or float(weights[right]) == 0.0:
            continue
        if not torch.isfinite(dot_products[pair_index]).all():
            raise ValueError("non-finite dot product for an active pair")
        total = total + (2.0 / d) * weights[left] * weights[right] * dot_products[pair_index]
    return total


def effective_W(Ws: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """The frozen mixture is exactly this single linear map on the shared key."""

    if len(Ws) != weights.numel():
        raise ValueError("W/weight count mismatch")
    output = torch.zeros_like(Ws[0])
    for index, W in enumerate(Ws):
        output = output + weights[index].to(device=W.device, dtype=W.dtype) * W
    return output


def ready_weights(detector) -> tuple[List[int], torch.Tensor]:
    """Return ready level indices and the detector's frozen mixture weights."""

    mixture = detector._mixture()
    indices = [detector.levels.index(level) for level, _weight in mixture]
    weights = torch.tensor([weight for _level, weight in mixture], dtype=torch.float32)
    return indices, weights


def level_weights_for(detector, arm: str, counterfactual: str | None = None) -> tuple[List[int], torch.Tensor]:
    """Weights for the frozen mixture or a post-hoc isolated/counterfactual arm."""

    if counterfactual is None:
        return ready_weights(detector)
    if counterfactual == "fast":
        return [0], torch.tensor([1.0])
    if counterfactual == "mid":
        return [1], torch.tensor([1.0])
    if counterfactual == "slow":
        return [2], torch.tensor([1.0])
    if counterfactual == "fast_mid":
        return [0, 1], torch.tensor([0.5, 0.5])
    raise ValueError(f"unknown counterfactual: {counterfactual}")
