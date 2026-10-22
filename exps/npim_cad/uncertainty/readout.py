# exps/npim_cad/uncertainty/readout.py
"""Frozen residual readouts A0/A1/A2 for EXP-NPIM-03.

A0 is the original NPIM L2 innovation and must match
``NPIMDetector.score_keys`` exactly (up to float roundoff).  A1 adds the frozen
normal residual mean; A2 divides by the frozen per-channel safe variance.  The
functions are read-only and receive no category/label/task input.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

D = 768


def _as_vectors(mu: torch.Tensor | None, var_safe: torch.Tensor | None, d: int, device, dtype):
    if mu is None or var_safe is None:
        raise ValueError("A1/A2 require mu and var_safe")
    mu = torch.as_tensor(mu, dtype=dtype, device=device).reshape(-1)
    var_safe = torch.as_tensor(var_safe, dtype=dtype, device=device).reshape(-1)
    if mu.numel() != d or var_safe.numel() != d:
        raise ValueError("moment geometry mismatch")
    if not torch.isfinite(mu).all() or not torch.isfinite(var_safe).all():
        raise ValueError("non-finite moments")
    if bool((var_safe <= 0).any()):
        raise ValueError("non-positive safe variance")
    return mu, var_safe


def score_patch_residual(
    residual: torch.Tensor,
    *,
    kind: str,
    mu: torch.Tensor | None = None,
    var_safe: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score [N,d] patch residuals with A0, A1 or A2."""

    residual = torch.as_tensor(residual)
    if residual.ndim != 2 or residual.shape[1] != D:
        raise ValueError(f"expected [N,{D}] residuals, got {tuple(residual.shape)}")
    if not torch.isfinite(residual).all():
        raise ValueError("non-finite residual")
    if kind == "A0":
        return residual.square().mean(dim=-1)
    if kind not in ("A1", "A2"):
        raise ValueError(f"unknown readout kind {kind}")
    mu, var_safe = _as_vectors(mu, var_safe, D, residual.device, residual.dtype)
    centered = residual - mu
    if kind == "A1":
        return centered.square().mean(dim=-1)
    return (centered.square() / var_safe).mean(dim=-1)


def score_keys(detector, K: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
    """A0 from the frozen detector, plus the residual for A1/A2."""

    K = torch.as_tensor(K)
    Y = torch.as_tensor(Y)
    with torch.no_grad():
        prediction = detector.predict(K)
        if prediction is None:
            raise RuntimeError("detector is not calibrated")
        residual = Y - prediction
        patch_scores = residual.square().mean(dim=-1)
    return {
        "residual": residual,
        "patch_scores": patch_scores,
        "image_score": float(patch_scores.max().item()),
    }


def a0_parity(detector, K: torch.Tensor, Y: torch.Tensor, *, atol: float = 1e-6) -> Dict[str, Any]:
    """Compare the explicit A0 with the original detector readout."""

    with torch.no_grad():
        explicit = score_keys(detector, K, Y)["patch_scores"]
        reference = detector.score_keys(K, Y)["patch_scores"]
    max_abs = float((explicit - reference).abs().max().item())
    relative = float((explicit - reference).norm() / reference.norm().clamp_min(1e-12))
    return {
        "max_abs": max_abs,
        "relative_l2": relative,
        "pass": bool(max_abs <= atol),
        "atol": float(atol),
    }


def image_max(patch_scores: torch.Tensor) -> float:
    patch_scores = torch.as_tensor(patch_scores)
    if patch_scores.ndim != 1 or not torch.isfinite(patch_scores).all():
        raise ValueError("invalid patch scores")
    return float(patch_scores.max().item())
