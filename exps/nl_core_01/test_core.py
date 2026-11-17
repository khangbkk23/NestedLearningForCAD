from __future__ import annotations

import io

import torch

from .core import CorrectedSelfGeneratedDGD, corrected_dgd_transition
from .controls import fixed_target_step


def test_fixed_target_matches_registered_recurrence():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(4, 4, generator=g, dtype=torch.float64)
    k = torch.randn(4, generator=g, dtype=torch.float64)
    v = torch.randn(4, generator=g, dtype=torch.float64)
    actual, residual, unit = corrected_dgd_transition(W, k, v, eta=0.1, alpha=0.99)
    expected = 0.99 * W - 0.1 * torch.outer(W @ unit - v, unit)
    assert torch.equal(actual, expected)
    assert torch.equal(residual, W @ unit - v)


def test_generator_gradient_matches_finite_difference():
    g = torch.Generator().manual_seed(4)
    G = (torch.randn(3, 3, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
    k = torch.randn(3, generator=g, dtype=torch.float64)
    v = torch.randn(3, generator=g, dtype=torch.float64)
    unit = k / k.norm()
    target = (G.detach() @ v).detach()
    objective = 0.5 * (G @ unit - target).square().sum()
    grad = torch.autograd.grad(objective, G)[0]
    idx, eps = (1, 2), 1e-6
    plus, minus = G.detach().clone(), G.detach().clone()
    plus[idx] += eps
    minus[idx] -= eps
    fd = (0.5 * (plus @ unit - target).square().sum() - 0.5 * (minus @ unit - target).square().sum()) / (2 * eps)
    assert abs(float(grad[idx] - fd)) < 2e-6


def test_fixed_point_and_perfect_association():
    W = torch.zeros(3, 3, dtype=torch.float64)
    k = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v = k.clone()
    for _ in range(250):
        W, _, _ = corrected_dgd_transition(W, k, v, eta=0.1, alpha=1.0)
    assert abs(float((W @ k)[0] - 1.0)) < 1e-8
    W_perfect = torch.eye(3, dtype=torch.float64)
    W_next, _, _ = corrected_dgd_transition(W_perfect, k, v, eta=0.1, alpha=0.99)
    assert abs(float((W_next @ k)[0] - 0.99)) < 1e-8


def test_slow_clock_and_checkpoint_equivalence():
    a = CorrectedSelfGeneratedDGD(4).double()
    b = CorrectedSelfGeneratedDGD(4).double()
    b.load_state_dict(a.state_dict())
    g = torch.Generator().manual_seed(8)
    events = [(torch.randn(4, generator=g, dtype=torch.float64), torch.randn(4, generator=g, dtype=torch.float64)) for _ in range(8)]
    slow = []
    for key, value in events:
        slow.append(a.step(key, value).slow_update)
        b.step(key, value)
    assert slow == [False, False, False, True, False, False, False, True]
    assert torch.equal(a.memory, b.memory)
    assert torch.equal(a.generator, b.generator)
    payload = io.BytesIO()
    torch.save(a.state_dict(), payload)
    c = CorrectedSelfGeneratedDGD(4).double()
    c.load_state_dict(torch.load(io.BytesIO(payload.getvalue()), weights_only=True))
    for key, value in events:
        a.step(key, value)
        c.step(key, value)
    assert torch.equal(a.memory, c.memory)
    assert torch.equal(a.generator, c.generator)


def test_state_is_finite_and_bounded():
    core = CorrectedSelfGeneratedDGD(6)
    for i in range(32):
        core.step(torch.randn(6), torch.randn(6))
    assert torch.isfinite(core.memory).all()
    assert torch.isfinite(core.generator).all()
    assert int(core.slow_update_count) == 8
    assert core.persistent_bytes() == (36 + 36) * 4 + 16
