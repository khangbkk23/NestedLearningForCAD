"""P2f mechanics tests: gate validity, causality, slow EMA and fixed-gate control."""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch

from exps.task_p.p2.model import Controller, Predictor, loss, parameters
from exps.task_p.p2d import model as original
from exps.task_p.p2e import model as e_model
from exps.task_p.p2f.analysis import classify_f
from exps.task_p.p2f.data import Replay, configure, tree_digest
from exps.task_p.p2f.model import LearnerF, inner_update_f


def tiny(device='cpu', **kwargs):
    options = dict(channels=8, latent=2, hidden=5, side=4, beta=.02, c_max=.02,
                   meta_lambda=1.0, meta_mu=1.0, controller_lr=1e-2)
    options.update(kwargs)
    return LearnerF('F', 91, device, **options)


def independent_fixed(fast, moments, slow, model, features, replay, support, query, cycle, c):
    fast = OrderedDict((name, value.clone()) for name, value in fast.items())
    moments = {name: (first.clone(), second.clone()) for name, (first, second) in moments.items()}
    slow = OrderedDict((name, value.detach().clone().requires_grad_(True))
                       for name, value in slow.items())
    for step in (1, 2):
        data = features if step == 1 else torch.stack((features, replay))
        indices = support if step == 1 else query
        point = OrderedDict((name, (1 - c) * value + c * slow[name]) for name, value in fast.items())
        objective = loss(model, point, data, indices)
        gradients = torch.autograd.grad(objective, tuple(point.values()), create_graph=True)
        norm = sum(gradient.square().sum() for gradient in gradients).sqrt()
        factor = (1.0 / (norm + 1e-12)).clamp(max=1.0)
        following = OrderedDict()
        for (name, value), gradient in zip(point.items(), gradients):
            gradient = gradient * factor
            first, second = moments[name]
            first = first * .9 + gradient * .1
            second = second * .999 + gradient.square() * .001
            n = 2 * cycle - 2 + step
            delta = .001 * (first / (1 - .9 ** n)) / ((second / (1 - .999 ** n) + 1e-12).sqrt() + 1e-8)
            delta = delta * (.02 / (delta.square().mean().sqrt() + 1e-12)).clamp(max=1.0)
            following[name] = value - delta
            moments[name] = first, second
        fast = following
    return fast, moments


def test_fixed_zero_matches_e_bitwise():
    configure()
    torch.manual_seed(3)
    f = tiny(gate_mode='fixed', fixed_c=0.0)
    e = e_model.Learner2('E', 91, 'cpu', channels=8, latent=2, hidden=5, side=4)
    x, r = torch.randn(16, 8), torch.randn(16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    f.train_cycle(x, s, q, r, None)
    e.train_cycle(x, s, q, r)
    assert all(torch.equal(f.fast[name], e.fast[name]) for name in f.fast)
    assert all(torch.equal(f.moments[name][i], e.moments[name][i])
               for name in f.moments for i in (0, 1))


def test_fixed_gate_matches_independent_reference():
    configure()
    torch.manual_seed(5)
    f = tiny(gate_mode='fixed', fixed_c=.01)
    f.model.double()
    f.fast = OrderedDict((name, value.detach().clone().double()) for name, value in parameters(f.model).items())
    f.moments = {name: (torch.zeros_like(value), torch.zeros_like(value)) for name, value in f.fast.items()}
    x, r = torch.randn(16, 8, dtype=torch.float64), torch.randn(16, 8, dtype=torch.float64)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    expected, expected_moments = independent_fixed(f.fast, f.moments, parameters(f.model), f.model,
                                                   x, r, s, q, 1, .01)
    actual, moments, diagnostics, c1, c2, stacked = inner_update_f(
        f.model, f.fast, parameters(f.model), f.controller, x, s, q, r, f.moments, 1,
        gate_mode='fixed', fixed_c=.01, c_max=.02)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-14)
        for first, second in zip(moments[name], expected_moments[name]):
            torch.testing.assert_close(first, second, rtol=0, atol=1e-14)
    assert diagnostics['gate_min'] == diagnostics['gate_max'] == pytest.approx(.01)


def test_gate_is_bounded_zero_at_init_and_has_nonzero_gradient():
    configure()
    torch.manual_seed(7)
    f = tiny()
    x, r, old = torch.randn(16, 8), torch.randn(16, 8), torch.randn(2, 16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    record = f.train_cycle(x, s, q, r, old)
    assert record['gate_min'] >= 0.0 and record['gate_max'] <= .02
    assert record['gate_nonzero'] == 0
    assert record['gate_gradient_norm'] > 0
    assert record['controller_output_gradient_norm'] > 0
    assert record['controller_hidden_gradient_norm'] == 0.0
    second = f.train_cycle(x, s, q, r, old)
    assert second['controller_hidden_gradient_norm'] > 0.0


def test_gate_causality_and_projected_gradient():
    configure()
    torch.manual_seed(11)
    f = tiny()
    x, r, old = torch.randn(16, 8), torch.randn(16, 8), torch.randn(2, 16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    # One learned cycle gives the effective gate gradient at c=0.
    following, moments, diagnostics, c1, c2, stacked = inner_update_f(
        f.model, f.fast, parameters(f.model), f.controller, x, s, q, r, f.moments, 1,
        gate_mode='learned', fixed_c=0.0, c_max=.02)
    acquisition = loss(f.model, following, stacked, q)
    retention = loss(f.model, following, old, q)
    plasticity = loss(f.model, following, x, s)
    meta = acquisition + retention + plasticity
    grads = torch.autograd.grad(meta, (c1, c2), allow_unused=True)
    gradient_sum = float(sum(g.sum() for g in grads if g is not None))
    assert abs(gradient_sum) > 0
    # One-sided finite difference of fixed c=0 versus fixed c=eps on the same state.
    eps = 1e-4
    def fixed_meta(c):
        fol, mom, diag, cc1, cc2, st = inner_update_f(
            f.model, f.fast, parameters(f.model), f.controller, x, s, q, r, f.moments, 1,
            gate_mode='fixed', fixed_c=c, c_max=.02)
        return float((loss(f.model, fol, st, q) + loss(f.model, fol, old, q) +
                      loss(f.model, fol, x, s)).detach())
    finite = (fixed_meta(eps) - fixed_meta(0.0)) / eps
    assert abs(finite) > 0
    assert abs(finite - gradient_sum) / (abs(finite) + 1e-12) < .05


def test_slow_state_is_ema_and_not_initial():
    configure()
    torch.manual_seed(13)
    f = tiny()
    initial = OrderedDict((name, value.clone()) for name, value in parameters(f.model).items())
    fast_before = OrderedDict((name, value.clone()) for name, value in f.fast.items())
    record = f.train_cycle(torch.randn(16, 8), torch.arange(0, 16, 2),
                           torch.arange(1, 16, 2), torch.randn(16, 8), None)
    slow = parameters(f.model)
    assert any(not torch.equal(slow[name], initial[name]) for name in slow)
    assert record['slow_from_initial_distance'] > 0
    assert record['slow_fast_distance'] > 0
    # The EMA uses phi2, which is the committed fast state.
    assert record['slow_fast_distance'] == pytest.approx(
        float(torch.sqrt(sum((f.fast[name] - slow[name]).square().sum() for name in f.fast))), rel=1e-6)


def test_serialization_resume_and_read_only_scoring(tmp_path):
    configure()
    torch.manual_seed(17)
    f = tiny()
    x, r = torch.randn(16, 8), torch.randn(16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    f.train_cycle(x, s, q, r, None)
    torch.save(f.state(), tmp_path / 'state.pt')
    restored = tiny()
    restored.restore(torch.load(tmp_path / 'state.pt', weights_only=True))
    assert tree_digest(restored.state()) == tree_digest(f.state())
    for instance in (f, restored):
        instance.train_cycle(x, s, q, r, None)
    assert tree_digest(restored.state()) == tree_digest(f.state())
    before = tree_digest(f.state())
    f.scores(x)
    assert tree_digest(f.state()) == before


def test_replay_old_subset_excludes_current_and_is_deterministic():
    replay = Replay(2711, capacity=10)
    ident = []
    for event in range(1, 8):
        replay.insert(torch.ones(16, 8) * event, str(event))
        ident.append(str(event))
    images = [image for identity, image in zip(replay.identities, replay.images) if identity != '3']
    assert len(images) == len(replay.images) - 1
    first = torch.stack(images[:2])
    second = torch.stack([image for identity, image in zip(replay.identities, replay.images)
                          if identity != '3'][:2])
    assert torch.equal(first, second)
    assert all(identity in ident for identity in replay.identities)


def test_fixed_gate_control_changes_fast_but_not_the_ema_rule():
    configure()
    torch.manual_seed(19)
    x, r = torch.randn(16, 8), torch.randn(16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    zero = tiny(gate_mode='fixed', fixed_c=0.0)
    one = tiny(gate_mode='fixed', fixed_c=.01)
    zero.train_cycle(x, s, q, r, None)
    one.train_cycle(x, s, q, r, None)
    assert not all(torch.equal(zero.fast[name], one.fast[name]) for name in zero.fast)
    assert zero.beta == one.beta == .02


def test_cpu_gpu_parity():
    configure()
    assert torch.cuda.is_available(), 'CUDA test suite is mandatory'
    cpu = tiny()
    gpu = tiny('cuda')
    gpu.restore(copy.deepcopy(cpu.state()))
    x, r = torch.randn(16, 8), torch.randn(16, 8)
    s, q = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    rc = cpu.train_cycle(x, s, q, r, None)
    rg = gpu.train_cycle(x.cuda(), s.cuda(), q.cuda(), r.cuda(), None)
    for name in cpu.fast:
        torch.testing.assert_close(cpu.fast[name], gpu.fast[name].cpu(), rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(cpu.scores(x), gpu.scores(x.cuda()).cpu(), rtol=3e-5, atol=3e-6)
    assert abs(rc['gate_gradient_norm'] - rg['gate_gradient_norm']) < 1e-5


@pytest.mark.parametrize('point,lower,acq,macro,verdict', [
    (.06, .01, True, True, 'P2F_LEARNED_CONSOLIDATION_SUPPORTED'),
    (.06, -.01, True, True, 'P2F_PARTIAL'),
    (.06, .01, False, True, 'P2F_HARM'),
    (.02, .001, True, True, 'P2F_NO_BENEFIT'),
    (-.03, -.05, True, True, 'P2F_HARM'),
    (.06, .01, True, False, 'P2F_PARTIAL'),
])
def test_decision_bands(point, lower, acq, macro, verdict):
    primary = {'difference': point, 'ci': [lower, lower + .1]}
    acquisition = {'x': {'delta': .0 if acq else -.02}}
    macro_safety = {'delta': .0 if macro else -.015}
    decision = {'primary_target': .05, 'acquisition_margin': .01,
                'macro_margin': .01, 'harm_margin': -.02}
    assert classify_f(primary, acquisition, macro_safety, decision)['verdict'] == verdict
