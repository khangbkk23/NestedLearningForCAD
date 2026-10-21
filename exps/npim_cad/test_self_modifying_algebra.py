# exps/npim_cad/test_self_modifying_algebra.py
"""Handoff oracle 10: corrected self-modifying algebra (C4 is NOT implemented).

EXP-NPIM-00 only fixes the documented algebra error and demonstrates the
invalid mixed form on a counterexample.  No self-modifying memory is used by the
detector.
"""

from __future__ import annotations

import torch

from models.hope_cad.npim_proximal import (
    smt_form_alpha_minus_kk,
    smt_form_residual,
    smt_mixed_invalid,
    smt_original,
)


def data(seed: int = 0, d: int = 6, r: int = 4):
    generator = torch.Generator().manual_seed(seed)
    W = torch.randn(d, r, generator=generator)
    k = torch.randn(r, generator=generator)
    v = torch.randn(d, generator=generator)
    return W, k, v


def unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def test_both_corrected_forms_exactly_match_the_recurrence():
    W, k, v = data(0)
    alpha, eta = 0.9, 0.03
    original = smt_original(W, k, v, alpha, eta)
    form1 = smt_form_alpha_minus_kk(W, k, v, alpha, eta)
    form2 = smt_form_residual(W, k, v, alpha, eta)
    assert torch.equal(form1, original)
    # The residual form is algebraically identical but reassociates the FP32
    # products, so allow ordinary float32 rounding (not a mathematical gap).
    assert torch.allclose(form2, original, atol=1e-6, rtol=1e-6)


def test_erroneous_mixed_form_fails_on_a_counterexample():
    W, k, v = data(1)
    alpha, eta = 0.9, 0.03
    original = smt_original(W, k, v, alpha, eta)
    mixed = smt_mixed_invalid(W, k, v, alpha, eta)
    difference = (mixed - original).abs().max().item()
    assert difference > 1e-6
    # The mix subtracts the W k k^T term twice.
    expected_error = -eta * (W @ torch.outer(k, k))
    assert torch.allclose(mixed - original, expected_error, atol=1e-6, rtol=1e-6)


def test_spectral_norm_condition_gives_contraction():
    generator = torch.Generator().manual_seed(2)
    for _ in range(20):
        k = unit(torch.randn(4, generator=generator))
        alpha = float(torch.rand(1, generator=generator).item()) * 0.99 + 0.005
        eta = float(torch.rand(1, generator=generator).item()) * (2.0 * alpha)
        A = alpha * torch.eye(4) - eta * torch.outer(k, k)
        assert float(torch.linalg.matrix_norm(A, ord=2)) <= alpha + 1e-6


def test_invariant_radius_under_first_form_injection_cap():
    M, alpha, eta, d, r = 2.0, 0.9, 0.05, 8, 4
    generator = torch.Generator().manual_seed(3)
    k = unit(torch.randn(r, generator=generator))
    v = unit(torch.randn(d, generator=generator)) * 3.5  # eta*||v||*||k|| = 0.175 <= 0.2
    W = torch.randn(d, r, generator=generator)
    W = W / W.norm() * (0.9 * M)
    assert eta * v.norm() * k.norm() <= (1 - alpha) * M + 1e-12
    for _ in range(100):
        W = smt_form_alpha_minus_kk(W, k, v, alpha, eta)
        assert W.norm().item() <= M + 1e-6


def test_residual_form_one_step_injection_cap():
    M, alpha, eta, d, r = 2.0, 0.9, 0.05, 8, 4
    generator = torch.Generator().manual_seed(4)
    k = unit(torch.randn(r, generator=generator))
    W = torch.randn(d, r, generator=generator)
    W = W / W.norm() * (0.5 * M)
    delta = unit(torch.randn(d, generator=generator)) * 3.5  # eta*||delta|| <= 0.175 <= 0.2
    v = W @ k + delta
    assert eta * delta.norm() * k.norm() <= (1 - alpha) * M + 1e-12
    updated = smt_form_residual(W, k, v, alpha, eta)
    assert updated.norm().item() <= M + 1e-6


def test_alpha_one_requires_zero_injection():
    """At alpha=1 the contraction-and-injection proof needs zero injection."""

    M, eta, d, r = 1.0, 0.05, 6, 3
    generator = torch.Generator().manual_seed(5)
    k = unit(torch.randn(r, generator=generator))
    W = torch.randn(d, r, generator=generator)
    W = W / W.norm() * M
    # A non-zero innovation aligned with W k adds rank-one energy; at alpha=1
    # there is no contraction to absorb it.
    v = W @ k + unit(W @ k)
    updated = smt_original(W, k, v, 1.0, eta)
    assert updated.norm().item() > M
