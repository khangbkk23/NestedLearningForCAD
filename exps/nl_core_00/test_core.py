from __future__ import annotations

import io

import pytest
import torch

from .benchmark import run_benchmark
from .core import CanonicalNLCore


def pair(dim=4, dtype=torch.float64):
    key = torch.tensor([1.0, -2.0, 0.5, 0.25], dtype=dtype)[:dim]
    value = torch.tensor([0.3, -0.1, 0.8, -0.2], dtype=dtype)[:dim]
    return key, value


def test_equation_transition_and_bounds():
    core = CanonicalNLCore(4).double()
    key, value = pair()
    before = core.memory.clone()
    result = core.step(key, value)
    assert result.update_index == 1
    assert 0.0 < result.eta < core.eta_max
    assert core.alpha_min < result.alpha < core.alpha_max
    assert not torch.equal(before, core.memory)
    assert core.persistent_bytes() == (16 + 10) * 8 + 8


def test_controller_gradient_matches_finite_difference():
    core = CanonicalNLCore(3).double()
    key = torch.tensor([0.4, -1.2, 0.7], dtype=torch.float64)
    value = torch.tensor([0.2, 0.5, -0.8], dtype=torch.float64)
    memory = torch.randn(3, 3, dtype=torch.float64) * 0.1
    controller = torch.randn(2, 4, dtype=torch.float64) * 0.1
    c = controller.clone().requires_grad_(True)
    loss = core.transition(memory, c, key, value)[1]
    grad = torch.autograd.grad(loss, c)[0]
    idx = (1, 2)
    eps = 1e-6
    plus, minus = controller.clone(), controller.clone()
    plus[idx] += eps
    minus[idx] -= eps
    fd = (core.transition(memory, plus, key, value)[1] - core.transition(memory, minus, key, value)[1]) / (2 * eps)
    assert abs(float(grad[idx] - fd)) < 2e-6


def test_self_modifying_state_is_causal_and_detached():
    core = CanonicalNLCore(4).double()
    key, value = pair()
    before = core.controller.clone()
    result = core.step(key, value)
    assert result.controller_grad_norm > 0.0
    assert not torch.equal(before, core.controller)
    assert core.memory.grad_fn is None and core.controller.grad_fn is None


def test_checkpoint_equivalence_and_determinism():
    torch.manual_seed(5)
    a = CanonicalNLCore(4).double()
    b = CanonicalNLCore(4).double()
    b.load_state_dict(a.state_dict())
    stream = [pair() for _ in range(4)]
    for key, value in stream[:2]:
        a.step(key, value)
        b.step(key, value)
    assert torch.equal(a.memory, b.memory)
    assert torch.equal(a.controller, b.controller)
    buffer = io.BytesIO()
    torch.save(a.state_dict(), buffer)
    c = CanonicalNLCore(4).double()
    c.load_state_dict(torch.load(io.BytesIO(buffer.getvalue()), weights_only=True))
    for key, value in stream[2:]:
        a.step(key, value)
        c.step(key, value)
    assert torch.equal(a.memory, c.memory)
    assert torch.equal(a.controller, c.controller)
    assert int(a.update_count) == 4


def test_invalid_inputs_and_bounded_state():
    core = CanonicalNLCore(3)
    with pytest.raises(ValueError):
        core.step(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError):
        core.step(torch.tensor([float("nan"), 0.0, 1.0]), torch.ones(3))
    for _ in range(30):
        key = torch.randn(3)
        core.step(key, torch.randn(3))
    assert torch.isfinite(core.memory).all()
    assert torch.isfinite(core.controller).all()
    assert core.persistent_bytes() == (9 + 8) * 4 + 8


def test_synthetic_benchmark_acquisition_retention_and_adaptation():
    result = run_benchmark(seed=17, dim=12, repeats=18)
    assert result["finite"]
    assert result["controller_adapted"]
    assert result["persistent_bytes"] == (144 + 26) * 8 + 8
    assert all(v < 1e6 for v in result["final_probe_loss"])
    # At least the first task must improve from the all-zero memory state.
    assert result["final_probe_loss"][0] < result["initial_probe_loss"][0]
