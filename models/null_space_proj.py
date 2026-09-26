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
    recycling_enabled: bool = True
    fallback_null_dims: tuple[int, ...] = (64, 32, 16)


class NullSpaceProjector:
    def __init__(self, d: int, config: NSP2Config, device: str | torch.device):
        self.d = int(d)
        self.config = config
        self.device = torch.device(device)
        self.basis: Optional[torch.Tensor] = None
        self.projector: Optional[torch.Tensor] = None
        self.rank = 0
        self.null_dim = self.d
        self.energy_rank = 0
        self.natural_null_dim = self.d
        self.target_null_dim = int(config.min_null_dim)
        self.recycled = False

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
        energy_rank = int(torch.searchsorted(cumulative, self.config.energy_threshold).item() + 1)
        energy_rank = min(energy_rank, vh.shape[0], self.d)
        natural_null_dim = self.d - energy_rank

        target_null_dim = self._select_target_null_dim(natural_null_dim)
        max_rank_for_null = max(0, self.d - target_null_dim)
        rank = min(energy_rank, max_rank_for_null)

        self.energy_rank = int(energy_rank)
        self.natural_null_dim = int(natural_null_dim)
        self.target_null_dim = int(target_null_dim)
        self.rank = rank
        self.null_dim = self.d - rank
        self.recycled = bool(self.null_dim > self.natural_null_dim)
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
            "energy_rank": int(self.energy_rank),
            "natural_null_dim": int(self.natural_null_dim),
            "target_null_dim": int(self.target_null_dim),
            "recycling_enabled": bool(self.config.recycling_enabled),
            "recycled": bool(self.recycled),
            "fallback_null_dims": list(self.config.fallback_null_dims),
            "energy_threshold": float(self.config.energy_threshold),
            "min_null_dim": int(self.config.min_null_dim),
        }

    def _set_empty_projector(self) -> None:
        self.rank = 0
        self.null_dim = self.d
        self.energy_rank = 0
        self.natural_null_dim = self.d
        self.target_null_dim = int(self.config.min_null_dim)
        self.recycled = False
        self.basis = None
        self.projector = torch.eye(self.d, device=self.device)

    def _build_projector(self) -> torch.Tensor:
        eye = torch.eye(self.d, device=self.device)
        if self.basis is None or self.basis.numel() == 0:
            return eye
        return eye - self.basis.T @ self.basis

    def _select_target_null_dim(self, natural_null_dim: int) -> int:
        min_null_dim = max(0, min(int(self.config.min_null_dim), self.d))
        if natural_null_dim >= min_null_dim:
            return min_null_dim
        if not self.config.recycling_enabled:
            return max(0, min(natural_null_dim, self.d))

        fallbacks = [
            max(0, min(int(dim), self.d))
            for dim in self.config.fallback_null_dims
            if int(dim) > 0
        ]
        if min_null_dim not in fallbacks:
            fallbacks.insert(0, min_null_dim)

        for dim in sorted(set(fallbacks), reverse=True):
            if dim <= self.d:
                return dim
        return max(0, min(natural_null_dim, self.d))
