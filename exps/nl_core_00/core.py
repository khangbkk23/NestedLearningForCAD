"""A minimal, auditable self-modifying Nested Learning memory.

The fast state is a linear associative memory W.  The slow state C is itself
an associative memory which maps the current key and the fast residual norm to
the learning-rate and retention logits.  C is updated through the consequence
of the differentiable fast update, so the slow state is causally connected to
the fast trajectory.  No history is retained.

This is a project mapping of the linear DGD/self-referential recurrence in
Nested Learning Eq. 7 and the control-generating memories in VisionHOPE; it is
not a reproduction of the full HOPE language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StepResult:
    """Detached evidence from one causal update."""

    pre_loss: float
    post_loss: float
    eta: float
    alpha: float
    residual_norm: float
    controller_grad_norm: float
    update_index: int


class CanonicalNLCore(nn.Module):
    """Smallest NL core with bounded fast and slow state.

    For a key/value pair (k_t, v_t), let k be unit-normalized and

        e_t = W_{t-1} k_t - v_t
        z_t = [k_t, ||e_t||_2]
        [a_t, b_t] = C_{t-1} z_t
        eta_t = eta_max sigmoid(a_t)
        alpha_t = alpha_min + (alpha_max-alpha_min) sigmoid(b_t)

    The fast retained DGD update is

        W_t = W_{t-1}(alpha I - eta k k^T) - eta e k^T.

    The controller is updated by differentiating the post-update local
    associative loss through W_t:

        J_t(C) = 1/2 ||W_t(C) k_t - v_t||^2
        C_t = C_{t-1} - beta clip(grad_C J_t).

    This is self-modifying because C generates the update controls and is
    changed by the consequence of the fast update it generated.  State is
    detached after each commit, so the online graph is bounded.
    """

    schema_version = 1

    def __init__(
        self,
        dim: int,
        *,
        eta_max: float = 0.20,
        alpha_min: float = 0.90,
        alpha_max: float = 0.999,
        controller_lr: float = 0.02,
        controller_grad_clip: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not (0.0 < eta_max and 0.0 < alpha_min < alpha_max <= 1.0):
            raise ValueError("invalid bounded control ranges")
        if controller_lr <= 0 or controller_grad_clip <= 0 or eps <= 0:
            raise ValueError("controller_lr, controller_grad_clip and eps must be positive")
        self.dim = dim
        self.eta_max = float(eta_max)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.controller_lr = float(controller_lr)
        self.controller_grad_clip = float(controller_grad_clip)
        self.eps = float(eps)
        self.register_buffer("memory", torch.zeros(dim, dim))
        self.register_buffer("controller", torch.zeros(2, dim + 1))
        self.register_buffer("update_count", torch.zeros((), dtype=torch.int64))

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dim": self.dim,
            "eta_max": self.eta_max,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "controller_lr": self.controller_lr,
            "controller_grad_clip": self.controller_grad_clip,
            "eps": self.eps,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        expected = self.get_extra_state()
        for key in expected:
            if key not in state or state[key] != expected[key]:
                raise ValueError(f"checkpoint configuration differs at {key}")

    def reset_state(self) -> None:
        with torch.no_grad():
            self.memory.zero_()
            self.controller.zero_()
            self.update_count.zero_()

    def _validate_pair(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.ndim != 1 or value.ndim != 1 or key.shape != (self.dim,) or value.shape != (self.dim,):
            raise ValueError(f"key/value must both have shape ({self.dim},)")
        if not (torch.is_floating_point(key) and torch.is_floating_point(value)):
            raise TypeError("key/value must be floating point")
        if not (torch.isfinite(key).all() and torch.isfinite(value).all()):
            raise ValueError("key/value must be finite")
        return key, value

    def _controls(self, controller: torch.Tensor, key: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unit_key = F.normalize(key, dim=0, eps=self.eps)
        z = torch.cat((unit_key, residual.norm().reshape(1)))
        logits = controller @ z
        eta = self.eta_max * torch.sigmoid(logits[0])
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(logits[1])
        return eta, alpha

    def transition(
        self, memory: torch.Tensor, controller: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure differentiable transition used by the online step and tests."""
        key, value = self._validate_pair(key, value)
        unit_key = F.normalize(key, dim=0, eps=self.eps)
        residual = memory @ unit_key - value
        eta, alpha = self._controls(controller, unit_key, residual)
        fast = memory @ (alpha * torch.eye(self.dim, dtype=memory.dtype, device=memory.device)
                         - eta * torch.outer(unit_key, unit_key)) - eta * torch.outer(residual, unit_key)
        post_residual = fast @ unit_key - value
        post_loss = 0.5 * post_residual.square().sum()
        return fast, post_loss, eta, alpha, residual

    @torch.no_grad()
    def evaluate(self, key: torch.Tensor, value: torch.Tensor) -> float:
        """Read-only pre-update associative loss."""
        key, value = self._validate_pair(key, value)
        unit_key = F.normalize(key, dim=0, eps=self.eps)
        return float(0.5 * (self.memory @ unit_key - value).square().sum().item())

    def step(self, key: torch.Tensor, value: torch.Tensor) -> StepResult:
        """Perform exactly one causal fast update and self-modifying slow update."""
        key, value = self._validate_pair(key, value)
        memory = self.memory.detach().clone()
        controller = self.controller.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            fast, post_loss, eta, alpha, residual = self.transition(memory, controller, key, value)
            gradient = torch.autograd.grad(post_loss, controller, allow_unused=False)[0]
        grad_norm = gradient.norm()
        if not torch.isfinite(gradient).all() or not torch.isfinite(fast).all():
            raise FloatingPointError("non-finite NL transition")
        scale = torch.clamp(self.controller_grad_clip / (grad_norm + self.eps), max=1.0)
        clipped = gradient * scale
        with torch.no_grad():
            self.memory.copy_(fast.detach())
            self.controller.copy_((controller - self.controller_lr * clipped).detach())
            self.update_count.add_(1)
        if not torch.isfinite(self.controller).all():
            raise FloatingPointError("non-finite controller state")
        pre_loss = 0.5 * residual.square().sum()
        return StepResult(
            pre_loss=float(pre_loss.detach().item()),
            post_loss=float(post_loss.detach().item()),
            eta=float(eta.detach().item()),
            alpha=float(alpha.detach().item()),
            residual_norm=float(residual.norm().detach().item()),
            controller_grad_norm=float(gradient.norm().detach().item()),
            update_index=int(self.update_count.item()),
        )

    def persistent_bytes(self) -> int:
        """Bytes of the online state, excluding transient autograd tensors."""
        return int(self.memory.numel() * self.memory.element_size()
                   + self.controller.numel() * self.controller.element_size()
                   + self.update_count.numel() * self.update_count.element_size())
