"""
cbp.py
------
Conservative Continual Backprop helpers for Phase 3.

The first Phase 3 path uses this as monitor-only by default. Reset can be
enabled later once N2B-NC drift behavior is understood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .null_space_proj import NullSpaceProjector


@dataclass
class CBPConfig:
    enabled: bool = False
    monitor_only: bool = True
    threshold: float = 0.01
    reinit_std: float = 0.02


class CBPMonitor:
    def __init__(self, config: CBPConfig):
        self.config = config

    @torch.no_grad()
    def scan_and_maybe_reset(
        self,
        module: nn.Module,
        projector: Optional[NullSpaceProjector] = None,
    ) -> dict:
        total_units = 0
        dead_units = 0
        reset_units = 0
        layer_stats = []

        for name, child in module.named_modules():
            if not isinstance(child, nn.Linear) or child.weight.grad is None:
                continue

            weight = child.weight
            utility = weight.detach().norm(dim=1)
            mean_utility = utility.mean().clamp_min(1e-12)
            normalized = utility / mean_utility
            dead_mask = normalized < float(self.config.threshold)

            n_units = int(dead_mask.numel())
            n_dead = int(dead_mask.sum().item())
            total_units += n_units
            dead_units += n_dead

            if self.config.enabled and not self.config.monitor_only and n_dead > 0:
                fresh = torch.randn_like(weight[dead_mask]) * float(self.config.reinit_std)
                if projector is not None:
                    fresh = projector.project(fresh)
                weight[dead_mask] = fresh
                if child.bias is not None:
                    child.bias[dead_mask] = 0.0
                reset_units += n_dead

            layer_stats.append(
                {
                    "name": name,
                    "units": n_units,
                    "dead_units": n_dead,
                    "dead_ratio": float(n_dead / max(n_units, 1)),
                }
            )

        return {
            "enabled": bool(self.config.enabled),
            "monitor_only": bool(self.config.monitor_only),
            "threshold": float(self.config.threshold),
            "total_units": total_units,
            "dead_units": dead_units,
            "reset_units": reset_units,
            "dead_neuron_ratio": float(dead_units / max(total_units, 1)),
            "layers": layer_stats,
        }
