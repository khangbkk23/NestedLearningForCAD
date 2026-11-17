"""Dependency-light Phase B verification for NL-CORE-01."""

from __future__ import annotations

import torch

from .core import CorrectedSelfGeneratedDGD, corrected_dgd_transition


def main() -> None:
    W = torch.zeros(3, 3, dtype=torch.float64)
    k = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    v = k.clone()
    for _ in range(250):
        W, _, _ = corrected_dgd_transition(W, k, v, eta=0.1, alpha=1.0)
    assert abs(float((W @ k)[0] - 1.0)) < 1e-8

    core = CorrectedSelfGeneratedDGD(4).double()
    g = torch.Generator().manual_seed(8)
    slow = []
    for _ in range(8):
        result = core.step(torch.randn(4, generator=g, dtype=torch.float64), torch.randn(4, generator=g, dtype=torch.float64))
        slow.append(result.slow_update)
    assert slow == [False, False, False, True, False, False, False, True]
    assert torch.isfinite(core.memory).all() and torch.isfinite(core.generator).all()
    assert int(core.slow_update_count) == 2
    print("NL-CORE-01 Phase B verification: PASS")
    print("fixed_point=PASS generator_update=PASS slow_clock=PASS finite_state=PASS")


if __name__ == "__main__":
    main()
