# exps/test_hope_outer_learning_horizon_audit_v1.py
"""Synthetic arithmetic and lifecycle checks for the OL-04D diagnostic."""

from __future__ import annotations

import numpy as np
import torch

from exps.ol04d.hope_outer_learning_horizon_audit_v1 import EPS, HORIZONS, decomposition, loss_per_image, teacher_panels
from exps.ol04.hope_outer_learning_pilot_v1 import Parameters, event, initial_state, state_fingerprint


def synthetic_parameters(dim=4, seed=3):
    generator = torch.Generator().manual_seed(seed)
    def make(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
    return Parameters(make((dim, dim)), make((dim, dim)),
                      {name: make(((1 if name in ('eta', 'alpha') else dim), dim)) for name in ('k', 'v', 'eta', 'alpha')},
                      make((dim, dim, 4)), make((dim,)))


def test_loss_change_decomposition_closes():
    torch.manual_seed(0)
    u = torch.randn(2, 4, 3, dtype=torch.float64)
    p0 = torch.randn_like(u)
    p1 = torch.randn_like(u)
    d = p1 - p0
    direct = 0.5 * ((p1-u).square().sum(-1).mean(-1) - (p0-u).square().sum(-1).mean(-1))
    linear = ((p0-u)*d).sum(-1).mean(-1)
    quadratic = 0.5*d.square().sum(-1).mean(-1)
    torch.testing.assert_close(direct, linear + quadratic, rtol=1e-12, atol=1e-12)


def test_residual_identity_closes():
    torch.manual_seed(1)
    u = torch.randn(3, 4, 5, dtype=torch.float64)
    mu = torch.randn(4, 5, dtype=torch.float64)
    p = torch.randn_like(u)
    lhs = 0.5*(p-u).square().sum(-1).mean()
    mean_loss = 0.5*(mu.unsqueeze(0)-u).square().sum(-1).mean()
    rhs = 0.5*(p-mu.unsqueeze(0)).square().sum(-1).mean() - ((p-mu.unsqueeze(0))*(u-mu.unsqueeze(0))).sum(-1).mean()
    torch.testing.assert_close(lhs-mean_loss, rhs, rtol=1e-12, atol=1e-12)


def test_magnitude_identity_and_zero_handling():
    torch.manual_seed(2)
    u = torch.nn.functional.normalize(torch.randn(10, 6, dtype=torch.float64), dim=-1)
    p = torch.randn_like(u)
    rho = p.norm(dim=-1)
    cos = (p*u).sum(-1)/(rho+EPS)
    radial = 0.5*(rho-1).square()
    angular = rho*(1-cos)
    torch.testing.assert_close(0.5*(p-u).square().sum(-1), radial+angular, rtol=1e-12, atol=1e-12)
    zero = torch.zeros(1, 6, dtype=torch.float64)
    assert float(zero.norm()) < 1e-8


def test_event_counter_and_read_fingerprint():
    p = synthetic_parameters(dim=4, seed=3)
    state = initial_state(p)
    image = torch.randn(1, 7, 4)
    before = state_fingerprint(state)
    next_state, proposal = event(p, state, image)
    assert proposal.counters.completed == 1
    assert next_state.counters.completed == 1
    assert state_fingerprint(state) == before
    assert next_state.counters.online == 2


def test_horizon_panel_is_declared():
    assert HORIZONS == (0, 1, 2, 4, 8, 16, 32, 50)
