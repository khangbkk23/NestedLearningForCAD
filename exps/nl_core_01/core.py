"""Corrected two-clock self-generated-target DGD core.

This module implements the locked specification in agents/plans/nl_core_01_spec.md.
It is intentionally independent of exps.nl_core_00 and never edits that oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StepResult:
    event: int
    slow_update: bool
    pre_loss: float
    post_loss: float
    target_norm: float
    generator_norm: float
    memory_norm: float


def normalize_key(key: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(key, dim=0, eps=eps)


def corrected_dgd_transition(
    memory: torch.Tensor,
    key: torch.Tensor,
    target: torch.Tensor,
    *,
    eta: float,
    alpha: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply W <- alpha W - eta (Wk-target) k^T and return residual."""
    k = normalize_key(key, eps)
    residual = memory @ k - target
    updated = alpha * memory - eta * torch.outer(residual, k)
    return updated, residual, k


def standard_delta_transition(
    memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, eta: float, eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return corrected_dgd_transition(memory, key, value, eta=eta, alpha=1.0, eps=eps)


class CorrectedSelfGeneratedDGD(nn.Module):
    """Fast content DGD plus a lower-frequency self-target generator."""

    schema_version = 1

    def __init__(
        self,
        dim: int,
        *,
        eta_w: float = 0.10,
        alpha_w: float = 0.99,
        eta_g: float = 0.05,
        alpha_g: float = 0.995,
        slow_period: int = 4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not (eta_w > 0 and eta_g > 0 and 0 < alpha_w <= 1 and 0 < alpha_g <= 1):
            raise ValueError("invalid DGD controls")
        if not isinstance(slow_period, int) or isinstance(slow_period, bool) or slow_period < 1:
            raise ValueError("slow_period must be positive")
        self.dim = dim
        self.eta_w = float(eta_w)
        self.alpha_w = float(alpha_w)
        self.eta_g = float(eta_g)
        self.alpha_g = float(alpha_g)
        self.slow_period = int(slow_period)
        self.eps = float(eps)
        self.register_buffer("memory", torch.zeros(dim, dim))
        self.register_buffer("generator", torch.eye(dim))
        self.register_buffer("event_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("slow_update_count", torch.zeros((), dtype=torch.int64))

    def get_extra_state(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "dim": self.dim,
                "eta_w": self.eta_w, "alpha_w": self.alpha_w,
                "eta_g": self.eta_g, "alpha_g": self.alpha_g,
                "slow_period": self.slow_period, "eps": self.eps}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        expected = self.get_extra_state()
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"checkpoint configuration differs at {key}")

    def reset_state(self) -> None:
        with torch.no_grad():
            self.memory.zero_()
            self.generator.copy_(torch.eye(self.dim, dtype=self.generator.dtype, device=self.generator.device))
            self.event_count.zero_()
            self.slow_update_count.zero_()

    def _validate(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape != (self.dim,) or value.shape != (self.dim,):
            raise ValueError(f"key/value must have shape ({self.dim},)")
        if not torch.is_floating_point(key) or not torch.is_floating_point(value):
            raise TypeError("key/value must be floating point")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError("key/value must be finite")

    def generator_transition(self, generator: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Self-generated objective with target stopgrad(G_(t-1)v)."""
        k = normalize_key(key, self.eps)
        target = (generator @ value).detach()
        residual = generator @ k - target
        updated = self.alpha_g * generator - self.eta_g * torch.outer(residual, k)
        return updated, residual

    def evaluate(self, key: torch.Tensor, value: torch.Tensor) -> float:
        self._validate(key, value)
        k = normalize_key(key, self.eps)
        target = self.generator @ value
        return float(0.5 * (self.memory @ k - target).square().sum().item())

    def step(self, key: torch.Tensor, value: torch.Tensor) -> StepResult:
        self._validate(key, value)
        memory = self.memory.detach().clone()
        generator = self.generator.detach().clone()
        target = generator @ value
        updated_memory, residual, _ = corrected_dgd_transition(
            memory, key, target, eta=self.eta_w, alpha=self.alpha_w, eps=self.eps,
        )
        event = int(self.event_count.item()) + 1
        slow = event % self.slow_period == 0
        if slow:
            updated_generator, _ = self.generator_transition(generator, key, value)
        else:
            updated_generator = generator
        post_target = updated_generator @ value
        post_loss = 0.5 * (updated_memory @ normalize_key(key, self.eps) - post_target).square().sum()
        if not torch.isfinite(updated_memory).all() or not torch.isfinite(updated_generator).all():
            raise FloatingPointError("non-finite corrected DGD state")
        with torch.no_grad():
            self.memory.copy_(updated_memory)
            self.generator.copy_(updated_generator)
            self.event_count.fill_(event)
            if slow:
                self.slow_update_count.add_(1)
        return StepResult(
            event=event, slow_update=slow,
            pre_loss=float(0.5 * residual.square().sum().item()),
            post_loss=float(post_loss.item()),
            target_norm=float(target.norm().item()),
            generator_norm=float(self.generator.norm().item()),
            memory_norm=float(self.memory.norm().item()),
        )

    def persistent_bytes(self) -> int:
        return int((self.memory.numel() + self.generator.numel()) * self.memory.element_size()
                   + self.event_count.numel() * self.event_count.element_size()
                   + self.slow_update_count.numel() * self.slow_update_count.element_size())
