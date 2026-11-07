"""CPU checks for NL-ALIGN-01 G2 (slow operator)."""
import torch

from exps.nl_align_01 import g2


def test_rho_identity_and_derivative():
    s = g2.Slow(3)
    r, d = s.rho([1.0, 0.3, 0.5])
    assert abs(r - 0.25) < 1e-12                      # psi = 0 reproduces the G1 FS_A coefficient
    s.psi.data = torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64)
    h, e = [1.0, 0.4, 0.5], 1e-6
    r, d = s.rho(h)
    for k in range(3):
        s.psi.data[k] += e
        r2, _ = s.rho(h)
        s.psi.data[k] -= e
        assert abs((r2 - r) / e - d[k]) < 1e-5
    assert 0 < r < g2.RHO_MAX


def test_clo_is_scalar_and_state_independent():
    s = g2.Slow(g2.DIM["CLO"])
    s.psi.data = torch.tensor([0.7], dtype=torch.float64)
    assert s.rho([1.0, -0.9, 0.1])[0] == s.rho([1.0, 0.9, 0.8])[0]


def test_slow_step_moves_against_hypergradient():
    s = g2.Slow(2)
    assert s.step([1.0, -1.0])
    assert s.psi[0] < 0 < s.psi[1]
