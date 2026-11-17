"""Dependency-light verification runner for NL-CORE-00.

This is intentionally runnable without pytest.  It exercises the same core
invariants as ``test_core.py`` and exits non-zero on any failure.
"""

from __future__ import annotations

import io

import torch

from .benchmark import run_benchmark
from .core import CanonicalNLCore


def main() -> None:
    torch.manual_seed(5)
    a = CanonicalNLCore(4).double()
    b = CanonicalNLCore(4).double()
    b.load_state_dict(a.state_dict())
    stream = [(torch.randn(4, dtype=torch.float64), torch.randn(4, dtype=torch.float64)) for _ in range(4)]
    for key, value in stream[:2]:
        a.step(key, value)
        b.step(key, value)
    assert torch.equal(a.memory, b.memory)
    assert torch.equal(a.controller, b.controller)

    payload = io.BytesIO()
    torch.save(a.state_dict(), payload)
    c = CanonicalNLCore(4).double()
    c.load_state_dict(torch.load(io.BytesIO(payload.getvalue()), weights_only=True))
    for key, value in stream[2:]:
        a.step(key, value)
        c.step(key, value)
    assert torch.equal(a.memory, c.memory)
    assert torch.equal(a.controller, c.controller)

    core = CanonicalNLCore(3).double()
    key = torch.tensor([0.4, -1.2, 0.7], dtype=torch.float64)
    value = torch.tensor([0.2, 0.5, -0.8], dtype=torch.float64)
    memory = torch.randn(3, 3, dtype=torch.float64) * 0.1
    controller = (torch.randn(2, 4, dtype=torch.float64) * 0.1).requires_grad_()
    loss = core.transition(memory, controller, key, value)[1]
    gradient = torch.autograd.grad(loss, controller)[0]
    i, eps = (1, 2), 1e-6
    plus, minus = controller.detach().clone(), controller.detach().clone()
    plus[i] += eps
    minus[i] -= eps
    finite_difference = (core.transition(memory, plus, key, value)[1]
                         - core.transition(memory, minus, key, value)[1]) / (2 * eps)
    assert abs(float(gradient[i] - finite_difference)) < 2e-6

    result = run_benchmark(seed=17, dim=12, repeats=18)
    assert result["finite"] and result["controller_adapted"]
    assert result["final_probe_loss"][0] < result["initial_probe_loss"][0]
    assert result["persistent_bytes"] == (144 + 26) * 8 + 8
    bounded = CanonicalNLCore(3)
    for _ in range(40):
        event = bounded.step(torch.randn(3), torch.randn(3))
        assert 0.0 < event.eta < bounded.eta_max
        assert bounded.alpha_min < event.alpha < bounded.alpha_max
    assert torch.isfinite(bounded.memory).all() and torch.isfinite(bounded.controller).all()
    assert bounded.persistent_bytes() == (9 + 8) * 4 + 8
    print("NL-CORE-00 verification: PASS")
    print("checkpoint_equivalence=PASS finite_difference=PASS bounded_state=PASS stability_bounds=PASS synthetic_benchmark=PASS")


if __name__ == "__main__":
    main()
