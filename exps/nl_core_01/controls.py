"""Independent controls for the NL-CORE-01 synthetic study."""

from __future__ import annotations

import torch

from .core import corrected_dgd_transition, normalize_key, standard_delta_transition


def fixed_target_step(memory: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, eta: float, alpha: float):
    return corrected_dgd_transition(memory, key, value, eta=eta, alpha=alpha)


def frozen_generator_step(memory: torch.Tensor, generator: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          *, eta: float, alpha: float):
    target = generator @ value
    updated, residual, unit_key = corrected_dgd_transition(memory, key, target, eta=eta, alpha=alpha)
    return updated, generator, residual, unit_key


def learned_generator_step(memory: torch.Tensor, generator: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           *, eta_w: float, alpha_w: float, eta_g: float, alpha_g: float,
                           slow: bool):
    target = generator @ value
    updated, residual, unit_key = corrected_dgd_transition(memory, key, target, eta=eta_w, alpha=alpha_w)
    if slow:
        g_target = (generator @ value).detach()
        g_residual = generator @ unit_key - g_target
        generator = alpha_g * generator - eta_g * torch.outer(g_residual, unit_key)
    return updated, generator, residual, unit_key


def nonnested_generator_step(memory: torch.Tensor, generator: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             *, eta_w: float, alpha_w: float, eta_g: float, slow: bool):
    target = generator @ value
    updated, residual, unit_key = corrected_dgd_transition(memory, key, target, eta=eta_w, alpha=alpha_w)
    if slow:
        # Same generator state, clock and matrix update cost; fixed target v.
        generator, _, _ = standard_delta_transition(generator, key, value, eta=eta_g)
    return updated, generator, residual, unit_key
