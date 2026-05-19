"""
titans_memory.py
----------------
TITANS Fast Memory for Phase 1 of Meta-NATH CAD.

Design note: this is not an MLP or a trainable neural network.
The TITANS memory is a single matrix M [d x d] updated by the
Delta Rule entirely under torch.no_grad().

Math (instruction_CAD.md section 2):
    M_t = (1 - alpha) * M_{t-1} + eta_t * (v_t - M_{t-1} k_t) k_t^T
    eta_t = eta0 / (1 + surprise_t)
    clamp M to [-5.0, 5.0]
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. TITANSMemory - stores only the memory matrix M.
# ---------------------------------------------------------------------------

class TITANSMemory:
    """
    Associative memory represented by a single [d, d] matrix.

    This class is not an nn.Module because M is updated manually without
    autograd. Use state_dict() / load_state_dict() for checkpoints.
    """

    def __init__(self, d: int = 768, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d = d
        self.device = device
        # Main buffer; never participates in autograd.
        self.M: torch.Tensor = torch.zeros(d, d, device=device)

    # ------------------------------------------------------------------
    def retrieve(self, k: torch.Tensor) -> torch.Tensor:
        """
        Retrieve memory: z = M @ k^T, returned as [batch, d].

        Args:
            k: [batch, d] key vector, usually z_cls from the backbone

        Returns:
            pred: [batch, d]
        """
        return (self.M @ k.T).T   # [batch, d]

    # ------------------------------------------------------------------
    def surprise(self, v: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the surprise scalar as the mean residual norm over the batch.

        Returns:
            Scalar tensor without gradients.
        """
        residual = v - pred                          # [batch, d]
        return residual.norm(dim=-1).mean()          # scalar

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: float = 0.9,
        eta0: float = 0.01,
        clamp: float = 5.0,
    ) -> float:
        """
        Update M with the Delta Rule.

        Args:
            k:     [batch, d] key
            v:     [batch, d] value, usually k.detach() for self-supervision
            alpha: decay rate (default 0.9)
            eta0:  base learning rate (default 0.01)
            clamp: absolute clamp value for M (default 5.0)

        Returns:
            surprise_scalar (float), used by TTTEngine for logging and eta scaling.
        """
        pred            = self.retrieve(k)                      # [batch, d]
        surprise_scalar = self.surprise(v, pred).item()

        eta_t  = eta0 / (1.0 + surprise_scalar)
        delta  = v - pred                                        # [batch, d]
        # Mean outer product over the batch: [d, d].
        update = torch.einsum("bi,bj->ij", delta, k) / k.shape[0]

        self.M.data = (1.0 - alpha) * self.M.data + eta_t * update
        self.M.data.clamp_(-clamp, clamp)

        return surprise_scalar

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"M": self.M.cpu().clone(), "d": self.d}

    def load_state_dict(self, sd: dict) -> None:
        self.M = sd["M"].to(self.device)
        self.d = sd["d"]

    def reset(self) -> None:
        """Reset M to zero, useful before a fresh experiment."""
        self.M.zero_()


# ---------------------------------------------------------------------------
# 2. TTTEngine - Phase 1 orchestrator.
# ---------------------------------------------------------------------------

class TTTEngine:
    """
    Test-Time Training Engine for Phase 1.

    Receives z_cls from the backbone, updates TITANSMemory, and returns
    z_updated. Phase 2 approval is handled by ACCGating.

    This class is not an nn.Module; all updates run under no_grad.
    """

    def __init__(
        self,
        d: int = 768,
        alpha: float = 0.9,
        eta0: float = 0.01,
        clamp: float = 5.0,
        device: str | None = None,
    ):
        self.alpha  = alpha
        self.eta0   = eta0
        self.clamp  = clamp
        self.memory = TITANSMemory(d=d, device=device)
        self._step  = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def process(self, z_cls: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Run one TTT step for a z_cls batch.

        Pipeline:
            1. Retrieve:  pred = M @ z_cls^T
            2. Compute surprise
            3. Update M (Delta Rule)
            4. Produce z_updated = M_updated @ z_cls^T

        Args:
            z_cls: [batch, d] detached CLS token from the backbone

        Returns:
            z_updated:       [batch, d]
            surprise_scalar: float
        """
        self._step += 1

        # k = v = z_cls for self-supervised associative memory.
        k = z_cls
        v = z_cls

        surprise_scalar = self.memory.update(
            k=k, v=v,
            alpha=self.alpha,
            eta0=self.eta0,
            clamp=self.clamp,
        )

        # Retrieve after the update to produce z_updated.
        z_updated = self.memory.retrieve(z_cls)    # [batch, d]

        if self._step % 100 == 0:
            logger.debug(
                f"[TTTEngine] step={self._step} "
                f"surprise={surprise_scalar:.4f} "
                f"|M|_F={self.memory.M.norm().item():.4f}"
            )

        return z_updated, surprise_scalar

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "memory": self.memory.state_dict(),
            "alpha":  self.alpha,
            "eta0":   self.eta0,
            "step":   self._step,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.memory.load_state_dict(sd["memory"])
        self.alpha  = sd["alpha"]
        self.eta0   = sd["eta0"]
        self._step  = sd["step"]

    def reset_memory(self) -> None:
        """Clear Fast Memory before a fresh experiment."""
        self.memory.reset()
        self._step = 0
