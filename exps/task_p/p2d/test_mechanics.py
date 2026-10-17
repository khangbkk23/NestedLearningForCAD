"""Synthetic tests of the P2d two-step nested update: contract, causality, weights, resume, parity.

These tests are written before exps/task_p/p2d/model.py exists (TDD). They pin:
- the frozen two-step update contract against an independently written reference
- D2 (initial learned) == C2 (fixed) for the first cycle, bitwise
- the outer controller derivative via central finite differences (CPU and GPU, double precision)
- controller intervention changing deployed parameters and patch scores
- state persistence/resume equality, read-only evaluation, detached deployed state
- objective weights: S/Q disjoint and covering; stacked replay loss == mean of per-image losses
- replay determinism and normal-only/no-future gating
- CPU/GPU parity
"""
import copy
from collections import OrderedDict

import pytest
import torch

from exps.task_p.p2.data import NormalReader, Replay, tree_digest
from exps.task_p.p2.model import loss, parameters
from exps.task_p.p2d.model import Learner2, inner_update2


def tiny(arm="D2", device="cpu"):
    return Learner2(arm, 7, device, channels=8, latent=2, hidden=5, side=4)


def statistics(fast, slow, moments, names):
    return torch.stack([
        torch.stack([
            fast[name].square().mean().sqrt(),
            (fast[name] - slow[name]).square().mean().sqrt(),
            moments[name][0].square().mean().sqrt(),
            moments[name][1].mean().sqrt(),
        ])
        for name in names
    ]).detach()


def reference_update(model, fast, slow, controller, features, support, query, replay, moments, cycle, learned=True):
    """Independent reference implementation of the frozen P2d contract."""
    names = list(slow)

    def gates(sample):
        if learned:
            return controller(sample)
        rate = sample.new_full((len(names),), 0.001)
        mix = sample.new_full((len(names),), 0.02)
        return rate, mix

    def direction(first, second, step):
        return (first / (1 - 0.9 ** step)) / ((second / (1 - 0.999 ** step) + 1e-12).sqrt() + 1e-8)

    def rms_clip(delta):
        return delta * torch.clamp(0.02 / (delta.square().mean().sqrt() + 1e-12), max=1)

    def bounded(gradients):
        norms = torch.sqrt(sum(g.square().sum() for g in gradients))
        scale = torch.clamp(1.0 / (norms + 1e-12), max=1.0)
        return [g * scale for g in gradients]

    # step 1: direct support update
    sample1 = statistics(fast, slow, moments, names)
    rate1, mix1 = gates(sample1)
    w1 = OrderedDict((name, (1 - mix1[i]) * fast[name] + mix1[i] * slow[name]) for i, name in enumerate(names))
    loss1 = loss(model, w1, features, support)
    raw1 = bounded(torch.autograd.grad(loss1, tuple(w1.values()), create_graph=True))
    phi1, first1, second1 = OrderedDict(), {}, {}
    for i, (name, gradient) in enumerate(zip(names, raw1)):
        first = 0.9 * moments[name][0] + 0.1 * gradient
        second = 0.999 * moments[name][1] + 0.001 * gradient.square()
        phi1[name] = w1[name] - rms_clip(rate1[i] * direction(first, second, 2 * cycle - 1))
        first1[name], second1[name] = first, second

    # step 2: direct current+replay query update
    sample2 = statistics(phi1, slow, {n: (first1[n], second1[n]) for n in names}, names)
    rate2, mix2 = gates(sample2)
    w2 = OrderedDict((name, (1 - mix2[i]) * phi1[name] + mix2[i] * slow[name]) for i, name in enumerate(names))
    stacked = features if replay is None else torch.stack((features, replay))
    loss2 = loss(model, w2, stacked, query)
    raw2 = bounded(torch.autograd.grad(loss2, tuple(w2.values()), create_graph=True))
    following, moments_after = OrderedDict(), {}
    for i, (name, gradient) in enumerate(zip(names, raw2)):
        first = 0.9 * first1[name] + 0.1 * gradient
        second = 0.999 * second1[name] + 0.001 * gradient.square()
        following[name] = w2[name] - rms_clip(rate2[i] * direction(first, second, 2 * cycle))
        moments_after[name] = (first.detach(), second.detach())
    return following, moments_after


@pytest.mark.parametrize("learned,arm", [(False, "C2"), (True, "D2")])
def test_two_step_update_matches_independent_reference(learned, arm):
    learner = tiny(arm)
    learner.model.double()
    learner.controller.double()
    slow = parameters(learner.model)
    fast = OrderedDict((n, v.detach().clone().double()) for n, v in slow.items())
    moments = {n: (torch.zeros_like(v), torch.zeros_like(v)) for n, v in fast.items()}
    features = torch.randn(16, 8, dtype=torch.float64)
    replay = torch.randn(16, 8, dtype=torch.float64)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    following, moments_after, record = inner_update2(learner.model, fast, slow, learner.controller,
                                                     features, support, query, replay, moments, 1, learned=learned)
    expected, expected_moments = reference_update(learner.model, fast, slow, learner.controller,
                                                  features, support, query, replay, moments, 1, learned=learned)
    for name in following:
        assert torch.equal(following[name], expected[name]), name
        assert torch.equal(moments_after[name][0], expected_moments[name][0]), name
        assert torch.equal(moments_after[name][1], expected_moments[name][1]), name
    assert record["query_loss"] > 0 and record["support_loss"] > 0


def test_first_cycle_learned_equals_fixed_and_controller_moves():
    fixed, learned = tiny("C2"), tiny("D2")
    features = torch.randn(16, 8)
    replay = torch.randn(16, 8)
    support, query = torch.arange(8), torch.arange(8, 16)
    fixed.train_cycle(features, support, query, replay)
    learned.train_cycle(features, support, query, replay)
    for name in fixed.fast:
        assert torch.equal(fixed.fast[name], learned.fast[name]), name
    changed = any(not torch.equal(a, b) for a, b in zip(learned.controller.state_dict().values(),
                                                       tiny("D2").controller.state_dict().values()))
    assert changed, "learned controller did not receive a gradient update"
    for a, b in zip(fixed.controller.state_dict().values(), tiny("C2").controller.state_dict().values()):
        assert torch.equal(a, b), "C2 controller must remain unused/frozen"


def test_outer_controller_derivative_finite_difference(device="cpu"):
    learner = tiny("D2", device)
    learner.model.double()
    learner.controller.double()
    slow = parameters(learner.model)
    fast = OrderedDict((n, v.detach().clone().double()) for n, v in slow.items())
    moments = {n: (torch.zeros_like(v), torch.zeros_like(v)) for n, v in fast.items()}
    features = torch.randn(16, 8, dtype=torch.float64, device=device)
    replay = torch.randn(16, 8, dtype=torch.float64, device=device)
    support, query = torch.arange(0, 16, 2, device=device), torch.arange(1, 16, 2, device=device)
    stacked = torch.stack((features, replay))

    def objective():
        following, _, _ = inner_update2(learner.model, fast, slow, learner.controller,
                                        features, support, query, replay, moments, 3)
        return loss(learner.model, following, stacked, query)

    value = objective()
    derivative = torch.autograd.grad(value, learner.controller.output.bias)[0][0]
    assert abs(float(derivative)) > 1e-9
    eps = 1e-5
    with torch.no_grad():
        learner.controller.output.bias[0] += eps
    positive = objective()
    with torch.no_grad():
        learner.controller.output.bias[0] -= 2 * eps
    negative = objective()
    with torch.no_grad():
        learner.controller.output.bias[0] += eps
    finite_difference = (positive - negative) / (2 * eps)
    relative = abs(float(derivative.detach()) - float(finite_difference.detach())) / abs(float(finite_difference.detach()))
    assert relative < 1e-4, (float(derivative), float(finite_difference))
    # intervention changes deployed parameters and scores
    base, _, _ = inner_update2(learner.model, fast, slow, learner.controller, features, support, query,
                               replay, moments, 3)
    with torch.no_grad():
        learner.controller.output.bias[0] += 0.5
    changed, _, _ = inner_update2(learner.model, fast, slow, learner.controller, features, support, query,
                                  replay, moments, 3)
    with torch.no_grad():
        learner.controller.output.bias[0] -= 0.5
    with torch.no_grad():
        shift = float(torch.sqrt(sum((changed[n] - base[n]).square().sum() for n in base)))
        assert shift > 0
        before = torch.func.functional_call(learner.model, base, (features,))
        after = torch.func.functional_call(learner.model, changed, (features,))
        score_shift = float((before - after).square().sum(-1).sqrt().max())
    assert score_shift > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_finite_difference_and_parity():
    test_outer_controller_derivative_finite_difference("cuda")
    cpu, gpu = tiny("D2"), tiny("D2", "cuda")
    gpu.restore(copy.deepcopy(cpu.state()))
    features = torch.randn(16, 8)
    torch.testing.assert_close(cpu.scores(features), gpu.scores(features.cuda()).cpu(),
                               rtol=2e-5, atol=2e-6)
    record_cpu = cpu.train_cycle(features, torch.arange(8), torch.arange(8, 16))
    record_gpu = gpu.train_cycle(features.cuda(), torch.arange(8, device="cuda"),
                                 torch.arange(8, 16, device="cuda"))
    assert record_gpu["controller_gradient_norm"] > 0
    assert record_cpu["controller_gradient_norm"] > 0


def test_state_persistence_resume_read_only_and_detached():
    learner = tiny("D2")
    features = torch.randn(16, 8)
    support, query = torch.arange(8), torch.arange(8, 16)
    replay = torch.randn(16, 8)
    learner.train_cycle(features, support, query, replay)
    for value in learner.fast.values():
        assert not value.requires_grad
    for pair in learner.moments.values():
        assert not pair[0].requires_grad and not pair[1].requires_grad
    before = learner.scores(features).clone()
    state = copy.deepcopy(learner.state())
    digest = tree_digest(state)
    assert torch.equal(learner.scores(features), before)
    assert tree_digest(learner.state()) == digest
    restored = tiny("D2")
    restored.restore(state)
    assert torch.equal(restored.scores(features), learner.scores(features))
    learner.train_cycle(features, support, query, replay)
    restored.train_cycle(features, support, query, replay)
    assert tree_digest(learner.state()) == tree_digest(restored.state())
    snapshot = learner.scores(features).clone()
    learner.scores(features)
    assert torch.equal(learner.scores(features), snapshot)


def test_objective_weights_positions_and_outer_reach():
    learner = tiny("D2")
    slow = parameters(learner.model)
    features = torch.randn(16, 8)
    replay = torch.randn(16, 8)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    assert set(support.tolist()).isdisjoint(query.tolist())
    assert len(set(support.tolist()) | set(query.tolist())) == 16
    stacked = torch.stack((features, replay))
    weights = OrderedDict((n, v.detach().clone()) for n, v in slow.items())
    pooled = loss(learner.model, weights, stacked, query)
    averaged = 0.5 * (loss(learner.model, weights, features, query)
                      + loss(learner.model, weights, replay, query))
    torch.testing.assert_close(pooled, averaged, rtol=0, atol=1e-15)
    # outer backward must not create gradients on the committed deployed state
    learner.train_cycle(features, support, query, replay)
    assert all(not v.requires_grad for v in learner.fast.values())
    assert all(p.grad is not None for p in learner.model.parameters())
    assert all(p.grad is not None for p in learner.controller.parameters())


def test_replay_determinism_and_normal_gating():
    first, second = Replay(42, capacity=3), Replay(42, capacity=3)
    seen = []
    for index in range(12):
        image = torch.full((4, 2), float(index))
        for replay in (first, second):
            identity, = [replay.sample()[1]]
            seen.append(identity)
            replay.insert(image, str(index))
    assert tree_digest(first.state()) == tree_digest(second.state())
    assert seen[:1] == [None]
    reader = NormalReader([{"role": "normal_train", "relative_path": "bottle/train/good/000.png"}])
    with pytest.raises(RuntimeError, match="future"):
        reader.get({"role": "normal_train", "relative_path": "carpet/train/good/000.png"})


def test_gradient_flow_both_steps_and_conflict_metric():
    learner = tiny("D2")
    slow = parameters(learner.model)
    fast = OrderedDict((n, v.detach().clone()) for n, v in slow.items())
    moments = {n: (torch.zeros_like(v), torch.zeros_like(v)) for n, v in fast.items()}
    features = torch.randn(16, 8)
    replay = torch.randn(16, 8)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    following, _, record = inner_update2(learner.model, fast, slow, learner.controller,
                                         features, support, query, replay, moments, 1)
    assert record["support_gradient_norm"] > 0
    assert record["query_gradient_norm"] > 0
    assert record["inner_delta_norm"] > 0
    assert record["rate_min"] == record["rate_max"]
    assert record["rate_min"] == pytest.approx(0.001, rel=1e-6)  # zero-initialized gates
    assert record["mix_min"] == pytest.approx(0.02, rel=1e-6)
    # exact gradient conflict at a shared point (diagnostic used by the probe)
    weights = OrderedDict((n, v.detach().clone().requires_grad_(True)) for n, v in slow.items())
    g_support = torch.autograd.grad(loss(learner.model, weights, features, support),
                                    tuple(weights.values()), create_graph=False)
    g_query = torch.autograd.grad(loss(learner.model, weights, stacked := torch.stack((features, replay)), query),
                                  tuple(weights.values()), create_graph=False)
    flat_s = torch.cat([g.flatten() for g in g_support])
    flat_q = torch.cat([g.flatten() for g in g_query])
    cosine = float(torch.dot(flat_s, flat_q) / (flat_s.norm() * flat_q.norm() + 1e-12))
    assert -1.0 <= cosine <= 1.0
