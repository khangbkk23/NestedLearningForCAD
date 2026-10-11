# exps/cms_nm01/gaussian_control.py
"""Streaming Gaussian control for CMS-NM-01."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .cms_nd import DIM, FixedProjection, LATENT_DIM


class StreamingGaussian64(nn.Module):
    """Diagonal normal-density state on the locked CMS-NM projection."""

    def __init__(self, *, projection_seed: int = 2401, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.projection = FixedProjection(DIM, latent_dim, seed=projection_seed)
        self.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("mean", torch.zeros(latent_dim))
        self.register_buffer("m2", torch.zeros(latent_dim))

    def project(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.shape[-1] != DIM:
            raise ValueError("patches must have shape [B,N,768]")
        return self.projection(patches)

    @torch.no_grad()
    def update_image(self, patches: torch.Tensor) -> None:
        z = self.project(patches).reshape(-1, self.mean.numel()).to(torch.float64)
        n_batch = int(z.shape[0])
        if n_batch == 0:
            return
        batch_mean = z.mean(dim=0)
        batch_m2 = ((z - batch_mean) ** 2).sum(dim=0)
        n_old = int(self.count.item())
        if n_old == 0:
            self.mean.copy_(batch_mean.to(self.mean.dtype))
            self.m2.copy_(batch_m2.to(self.m2.dtype))
            self.count.fill_(n_batch)
            return
        n_total = n_old + n_batch
        delta = batch_mean - self.mean.to(torch.float64)
        new_mean = self.mean.to(torch.float64) + delta * (n_batch / n_total)
        new_m2 = self.m2.to(torch.float64) + batch_m2 + delta.square() * (n_old * n_batch / n_total)
        self.mean.copy_(new_mean.to(self.mean.dtype))
        self.m2.copy_(new_m2.to(self.m2.dtype))
        self.count.fill_(n_total)

    @torch.no_grad()
    def score_patches(self, patches: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.project(patches)
        n = max(int(self.count.item()), 1)
        variance = (self.m2 / max(n - 1, 1)).clamp_min(1e-4)
        logp = -0.5 * (((z - self.mean) ** 2) / variance + variance.log() + math.log(2.0 * math.pi)).sum(dim=-1)
        return {"mixture_nll": -logp, "level_nll": (-logp,), "log_probs": (logp,)}

    def persistent_bytes(self) -> int:
        return sum(int(value.numel() * value.element_size()) for value in self.state_dict().values())
