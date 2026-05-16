"""
null_space_proj.py
------------------
NSP2 projector for Phase 3 consolidation.

The projector is intentionally small: fit a task subspace from anchor
embeddings, then project eligible gradients away from that subspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NSP2Config:
    enabled: bool = False
    energy_threshold: float = 0.99
    min_null_dim: int = 64


class NullSpaceProjector:
    def __init__(self, d: int, config: NSP2Config, device: str | torch.device):
        self.d = int(d)
        self.config = config
        self.device = torch.device(device)
        self.basis: Optional[torch.Tensor] = None
        self.projector: Optional[torch.Tensor] = None
        self.rank = 0
        self.null_dim = self.d

    @torch.no_grad()
    def fit(self, embeddings: torch.Tensor) -> dict:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(f"Expected embeddings [N,{self.d}], got {tuple(embeddings.shape)}.")

        x = embeddings.detach().to(self.device, dtype=torch.float32)
        if x.shape[0] < 2:
            self._set_empty_projector()
            return self.stats()

        x = x - x.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        if s.numel() == 0 or torch.all(s <= 0):
            self._set_empty_projector()
            return self.stats()

        energy = s.square()
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-12)
        rank = int(torch.searchsorted(cumulative, self.config.energy_threshold).item() + 1)
        rank = min(rank, vh.shape[0], self.d)

        max_rank_for_null = max(0, self.d - int(self.config.min_null_dim))
        if rank > max_rank_for_null:
            rank = max_rank_for_null

        self.rank = rank
        self.null_dim = self.d - rank
        self.basis = vh[:rank].contiguous() if rank > 0 else None
        self.projector = self._build_projector()
        return self.stats()

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.config.enabled or self.projector is None:
            return grad

        p = self.projector.to(device=grad.device, dtype=grad.dtype)
        if grad.ndim == 1 and grad.shape[0] == self.d:
            return p @ grad
        if grad.ndim >= 2 and grad.shape[-1] == self.d:
            return grad @ p
        if grad.ndim >= 2 and grad.shape[0] == self.d:
            flat = grad.reshape(self.d, -1)
            return (p @ flat).reshape_as(grad)
        return grad

    def stats(self) -> dict:
        return {
            "enabled": bool(self.config.enabled),
            "rank": int(self.rank),
            "null_dim": int(self.null_dim),
            "energy_threshold": float(self.config.energy_threshold),
            "min_null_dim": int(self.config.min_null_dim),
        }

    def _set_empty_projector(self) -> None:
        self.rank = 0
        self.null_dim = self.d
        self.basis = None
        self.projector = torch.eye(self.d, device=self.device)

    def _build_projector(self) -> torch.Tensor:
        eye = torch.eye(self.d, device=self.device)
        if self.basis is None or self.basis.numel() == 0:
            return eye
        return eye - self.basis.T @ self.basis
