# models/hope_cad/npim_proximal.py
"""Rank-compressed, image-clock proximal associative predictors for NPIM-CAD.

This is a NEW, isolated EXP-NPIM-00 module.  It does not reuse the MLP
objective of :mod:`models.hope_cad.continuum_memory`; it implements the
closed-form ridge proximal update of handoff section 4.

For a level with persistent weight ``W0`` and a chunk of ``n`` patch pairs
``(k_i, y_i)`` observed since that level's last commit::

    A = K K^T / n,   B = V K^T / n
    F(W) = ||W K - V||_F^2 / (2n) + (lambda/2) ||W - W0||_F^2
    W_new = (B + lambda W0) (A + lambda I_r)^{-1}

The solve uses Cholesky (never an explicit inverse) and only sufficient
statistics ``S_kk = sum k k^T``, ``S_yk = sum y k^T`` and ``N_pairs`` are
persisted between commits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from models.hope_cad.npim_context import sha256_tensor


@dataclass(frozen=True)
class ProximalConfig:
    d: int = 768
    r: int = 64
    period: int = 1
    lam: float = 0.1
    dtype: torch.dtype = torch.float32

    def validate(self) -> None:
        if self.d <= 0 or self.r <= 0:
            raise ValueError("d and r must be positive")
        if self.period <= 0:
            raise ValueError("period must be a positive number of completed images")
        if not (self.lam > 0.0 and torch.isfinite(torch.tensor(self.lam))):
            raise ValueError("lambda must be finite and strictly positive")


class ProximalLevel:
    """One scheduled closed-form ridge memory with bounded persistent state."""

    def __init__(self, config: ProximalConfig, device: str | torch.device = "cpu") -> None:
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.W = torch.zeros(config.d, config.r, dtype=config.dtype, device=self.device)
        self.Skk = torch.zeros(config.r, config.r, dtype=config.dtype, device=self.device)
        self.Syk = torch.zeros(config.d, config.r, dtype=config.dtype, device=self.device)
        self.pair_count = 0
        self.pending_images = 0
        self.ready = False
        self.update_count = 0

    # -- staging -------------------------------------------------------------
    def stage_image(self, K: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Sufficient-statistic increments for one normal image (no raw replay)."""

        _check_pairs(K, Y, self.config.d, self.config.r)
        return K.T @ K, Y.T @ K, int(K.shape[0])

    # -- boundary solve ------------------------------------------------------
    def candidate_on_boundary(
        self,
        Skk: torch.Tensor,
        Syk: torch.Tensor,
        pair_count: int,
        W0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Closed-form proximal candidate; raises on non-SPD / non-finite input."""

        if pair_count <= 0:
            raise ValueError("cannot solve a level with zero staged pairs")
        _check_statistics(Skk, Syk, self.config.d, self.config.r)
        base = self.W if W0 is None else W0
        A = Skk / float(pair_count)
        B = Syk / float(pair_count)
        identity = torch.eye(self.config.r, dtype=self.config.dtype, device=self.device)
        system = A + self.config.lam * identity
        if not torch.isfinite(system).all():
            raise ValueError("non-finite normal equations")
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info.item()) != 0:
            raise RuntimeError("normal equations are not positive definite; refusing to commit")
        rhs = B + self.config.lam * base
        candidate = torch.cholesky_solve(rhs.T, factor).T
        if not torch.isfinite(candidate).all():
            raise RuntimeError("proximal candidate is non-finite; refusing to commit")
        return candidate

    # -- commit --------------------------------------------------------------
    def apply_commit(self, candidate: torch.Tensor) -> None:
        if candidate.shape != self.W.shape or not torch.isfinite(candidate).all():
            raise ValueError("invalid candidate tensor")
        self.W.copy_(candidate)
        self.Skk.zero_()
        self.Syk.zero_()
        self.pair_count = 0
        self.pending_images = 0
        self.ready = True
        self.update_count += 1

    def stage_pending(
        self,
        accumulated_kk: torch.Tensor,
        accumulated_yk: torch.Tensor,
        pair_count: int,
    ) -> None:
        self.Skk.copy_(accumulated_kk)
        self.Syk.copy_(accumulated_yk)
        self.pair_count = int(pair_count)
        self.pending_images += 1

    # -- state ---------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "W": self.W.detach().to("cpu", copy=True),
            "Skk": self.Skk.detach().to("cpu", copy=True),
            "Syk": self.Syk.detach().to("cpu", copy=True),
            "pair_count": int(self.pair_count),
            "pending_images": int(self.pending_images),
            "ready": bool(self.ready),
            "update_count": int(self.update_count),
            "period": int(self.config.period),
            "lam": float(self.config.lam),
            "d": int(self.config.d),
            "r": int(self.config.r),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        required = {"W", "Skk", "Syk", "pair_count", "pending_images", "ready", "update_count"}
        missing = required - set(state)
        if missing:
            raise KeyError(f"missing proximal-level state keys: {sorted(missing)}")
        if tuple(state["W"].shape) != (self.config.d, self.config.r):
            raise ValueError("W geometry mismatch")
        if tuple(state["Skk"].shape) != (self.config.r, self.config.r):
            raise ValueError("Skk geometry mismatch")
        if tuple(state["Syk"].shape) != (self.config.d, self.config.r):
            raise ValueError("Syk geometry mismatch")
        for name in ("W", "Skk", "Syk"):
            if not torch.isfinite(state[name]).all():
                raise ValueError(f"non-finite persisted {name}")
        if int(state["period"]) != self.config.period or float(state["lam"]) != self.config.lam:
            raise ValueError("persisted level configuration differs from the target level")
        self.W.copy_(state["W"].to(device=self.device, dtype=self.config.dtype))
        self.Skk.copy_(state["Skk"].to(device=self.device, dtype=self.config.dtype))
        self.Syk.copy_(state["Syk"].to(device=self.device, dtype=self.config.dtype))
        self.pair_count = int(state["pair_count"])
        self.pending_images = int(state["pending_images"])
        self.ready = bool(state["ready"])
        self.update_count = int(state["update_count"])

    def state_hash(self) -> str:
        header = (
            f"{self.pair_count}|{self.pending_images}|{int(self.ready)}|{self.update_count}|"
            f"{self.config.period}|{self.config.lam}"
        )
        digest = sha256_tensor(self.W)[:32]
        return f"{header}|{digest}"

    # -- accounting ----------------------------------------------------------
    def tensor_bytes(self) -> Dict[str, int]:
        return {
            "W": int(self.W.numel() * self.W.element_size()),
            "Skk": int(self.Skk.numel() * self.Skk.element_size()),
            "Syk": int(self.Syk.numel() * self.Syk.element_size()),
        }


def _check_pairs(K: torch.Tensor, Y: torch.Tensor, d: int, r: int) -> None:
    if K.ndim != 2 or K.shape[1] != r:
        raise ValueError(f"K must be [n,{r}], got {tuple(K.shape)}")
    if Y.ndim != 2 or Y.shape[0] != K.shape[0] or Y.shape[1] != d:
        raise ValueError(f"Y must be [{K.shape[0]},{d}], got {tuple(Y.shape)}")
    if not torch.isfinite(K).all() or not torch.isfinite(Y).all():
        raise ValueError("non-finite patch pairs")


def _check_statistics(Skk: torch.Tensor, Syk: torch.Tensor, d: int, r: int) -> None:
    if tuple(Skk.shape) != (r, r) or tuple(Syk.shape) != (d, r):
        raise ValueError("sufficient-statistic geometry mismatch")
    if not torch.isfinite(Skk).all() or not torch.isfinite(Syk).all():
        raise ValueError("non-finite sufficient statistics")


# ---------------------------------------------------------------------------
# Independent CPU numerical oracles (FP64)
# ---------------------------------------------------------------------------


def proximal_solve_oracle(
    Skk: torch.Tensor,
    Syk: torch.Tensor,
    pair_count: int,
    W0: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """FP64 direct ``torch.linalg.solve`` oracle for ``(B + lam W0)(A + lam I)^-1``."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    A = Skk.to(torch.float64) / float(pair_count)
    B = Syk.to(torch.float64) / float(pair_count)
    r = A.shape[0]
    system = A + float(lam) * torch.eye(r, dtype=torch.float64)
    rhs = (B + float(lam) * W0.to(torch.float64)).T
    return torch.linalg.solve(system, rhs).T


def proximal_lstsq_oracle(
    Skk: torch.Tensor,
    Syk: torch.Tensor,
    pair_count: int,
    W0: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Second FP64 oracle via an SVD-based least-squares solve."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    A = Skk.to(torch.float64) / float(pair_count)
    B = Syk.to(torch.float64) / float(pair_count)
    r = A.shape[0]
    system = A + float(lam) * torch.eye(r, dtype=torch.float64)
    rhs = (B + float(lam) * W0.to(torch.float64)).T
    solution, *_ = torch.linalg.lstsq(system, rhs)
    return solution.T


def strong_convexity_floor(Skk: torch.Tensor, pair_count: int, lam: float) -> float:
    """Smallest Hessian eigenvalue of the averaged objective (should be >= lambda)."""

    A = Skk.to(torch.float64) / float(pair_count)
    system = A + float(lam) * torch.eye(A.shape[0], dtype=torch.float64)
    return float(torch.linalg.eigvalsh(system)[0].item())


def drift_bound(
    Skk: torch.Tensor,
    Syk: torch.Tensor,
    pair_count: int,
    W0: torch.Tensor,
    lam: float,
) -> float:
    """Proved one-update bound ``||W_new - W0||_F <= ||B - W0 A||_F / lambda``."""

    A = Skk.to(torch.float64) / float(pair_count)
    B = Syk.to(torch.float64) / float(pair_count)
    residual = B - W0.to(torch.float64) @ A
    return float(residual.norm().item() / float(lam))


# ---------------------------------------------------------------------------
# Corrected self-modifying recurrence algebra (theory/tests only; C4 deferred)
# ---------------------------------------------------------------------------


def smt_original(
    W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float, eta: float
) -> torch.Tensor:
    """Original sketch recurrence ``W(alpha I - eta k k^T) + eta v k^T``."""

    identity = torch.eye(k.shape[0], dtype=W.dtype, device=W.device)
    return W @ (alpha * identity - eta * torch.outer(k, k)) + eta * torch.outer(v, k)


def smt_form_alpha_minus_kk(
    W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float, eta: float
) -> torch.Tensor:
    """Correct choice 1: ``A = alpha I - eta k k^T``, ``B = eta v k^T``."""

    identity = torch.eye(k.shape[0], dtype=W.dtype, device=W.device)
    return W @ (alpha * identity - eta * torch.outer(k, k)) + eta * torch.outer(v, k)


def smt_form_residual(
    W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float, eta: float
) -> torch.Tensor:
    """Correct choice 2: ``A = alpha I``, ``B = eta (v - W k) k^T``."""

    delta = v - W @ k
    return alpha * W + eta * torch.outer(delta, k)


def smt_mixed_invalid(
    W: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: float, eta: float
) -> torch.Tensor:
    """Erroneous sketch mix ``A = alpha I - eta k k^T`` with ``B = eta delta k^T``.

    It subtracts the ``W k k^T`` term twice; it is NOT equal to the recurrence.
    """

    delta = v - W @ k
    identity = torch.eye(k.shape[0], dtype=W.dtype, device=W.device)
    return W @ (alpha * identity - eta * torch.outer(k, k)) + eta * torch.outer(delta, k)
