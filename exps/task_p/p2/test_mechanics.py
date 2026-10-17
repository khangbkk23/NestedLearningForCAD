"""Synthetic tests of controller causality, resume and scientific invariants."""
import copy
from collections import OrderedDict

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from exps.task_p.p1_bootstrap_bounds import CorrectedBoundedPixelBootstrap
from exps.task_p.p1_readout import PixelRanking, score_histogram
from exps.task_p.p2.data import NormalReader, Replay, tree_digest
from exps.task_p.p2.model import Learner, Predictor, inner_update, loss, parameters
from exps.task_p.p2.metrics import bootstrap_bounds, exact_ranking


def tiny(arm="D", device="cpu"):
    return Learner(arm, 7, device, channels=8, latent=2, hidden=5, side=4)


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "D_noinner"])
def test_writes_resume_and_read_only(arm):
    learner = tiny(arm)
    features = torch.randn(16, 8)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
    before = learner.scores(features).clone()
    record = learner.train_cycle(features, support, query, features * 0.7)
    assert all(np.isfinite(value) for value in record.values())
    if arm == "D_noinner":
        torch.testing.assert_close(before, learner.scores(features), rtol=0, atol=0)
        learner.train_cycle(features, support, query, features * 0.7)
    assert not torch.equal(before, learner.scores(features))
    state = copy.deepcopy(learner.state())
    digest = tree_digest(state)
    assert torch.equal(learner.scores(features), learner.scores(features))
    assert tree_digest(learner.state()) == digest
    restored = tiny(arm)
    restored.restore(state)
    assert torch.equal(restored.scores(features), learner.scores(features))
    learner.train_cycle(features, support, query)
    restored.train_cycle(features, support, query)
    assert tree_digest(learner.state()) == tree_digest(restored.state())


def test_outer_controller_derivative_finite_difference_and_intervention():
    learner = tiny()
    learner.model.double()
    learner.controller.double()
    slow = parameters(learner.model)
    fast = OrderedDict((name, value.detach().clone()) for name, value in slow.items())
    moments = {name: (torch.zeros_like(value), torch.zeros_like(value)) for name, value in fast.items()}
    features = torch.randn(16, 8, dtype=torch.float64)
    support, query = torch.arange(0, 16, 2), torch.arange(1, 16, 2)

    def objective():
        following, _, _ = inner_update(learner.model, fast, slow, learner.controller, features, support, moments, 1)
        return loss(learner.model, following, features, query), following

    original, following = objective()
    derivative = torch.autograd.grad(original, learner.controller.output.bias)[0][0]
    assert abs(float(derivative)) > 1e-9
    epsilon = 1e-4
    with torch.no_grad():
        learner.controller.output.bias[0] += epsilon
    positive, _ = objective()
    with torch.no_grad():
        learner.controller.output.bias[0] -= 2 * epsilon
    negative, _ = objective()
    finite_difference = (positive - negative) / (2 * epsilon)
    torch.testing.assert_close(derivative, finite_difference, rtol=1e-4, atol=1e-9)
    with torch.no_grad():
        learner.controller.output.bias[0] += epsilon + 0.5
    changed_loss, changed = objective()
    assert abs(float((changed_loss - original).detach())) > 1e-8
    assert any(not torch.equal(following[name], changed[name]) for name in following)


def test_fixed_control_equals_initial_learned_rule():
    fixed, learned = tiny("C"), tiny("D")
    features = torch.randn(16, 8)
    fixed.train_cycle(features, torch.arange(8), torch.arange(8, 16))
    learned.train_cycle(features, torch.arange(8), torch.arange(8, 16))
    for name in fixed.fast:
        torch.testing.assert_close(fixed.fast[name], learned.fast[name], rtol=0, atol=0)


def test_bottleneck_nullspace_blocks_identity():
    model = Predictor(channels=8, latent=2, hidden=5, side=4).double()
    _, _, right = torch.linalg.svd(model.encode.weight, full_matrices=True)
    perturbation = right[-1]
    features = torch.randn(16, 8, dtype=torch.float64)
    changed = features.clone()
    changed[5] += perturbation
    torch.testing.assert_close(model(features), model(changed), rtol=1e-10, atol=1e-10)
    assert not torch.equal(features, changed)


def test_replay_bounded_deterministic_and_source_gating():
    first, second = Replay(42, capacity=3), Replay(42, capacity=3)
    for index in range(12):
        image = torch.full((4, 2), float(index))
        for replay in (first, second):
            replay.insert(image, str(index))
            replay.sample()
    assert len(first.images) == 3 and tree_digest(first.state()) == tree_digest(second.state())
    reader = NormalReader([{"role": "normal_train", "relative_path": "bottle/train/good/000.png"}])
    with pytest.raises(RuntimeError, match="future"):
        reader.get({"role": "normal_train", "relative_path": "carpet/train/good/000.png"})


def test_native_ap_and_bootstrap_bounds_with_ties():
    scores = [np.array([[0.1, 0.9], [0.9, 0.3]], dtype=np.float32),
              np.array([[0.6, 0.6], [0.2, 0.2]], dtype=np.float32)]
    masks = [np.array([[False, True], [False, True]]), np.array([[True, False], [False, False]])]
    histograms = [score_histogram(score, mask) for score, mask in zip(scores, masks)]
    ranking = PixelRanking(histograms)
    bounds = CorrectedBoundedPixelBootstrap(histograms, bins=4)
    for weights in ([1, 1], [2, 1], [1, 3]):
        repeated_scores = np.concatenate([np.tile(score.ravel(), count) for score, count in zip(scores, weights)])
        repeated_masks = np.concatenate([np.tile(mask.ravel(), count) for mask, count in zip(masks, weights)])
        direct = average_precision_score(repeated_masks, repeated_scores)
        assert abs(direct - ranking.metrics(np.array(weights))["ap"]) < 1e-12
        lower, upper = bounds.bounds(np.array(weights))
        assert lower[0] <= direct <= upper[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requires host GPU access")
def test_cpu_gpu_scores_and_nested_gradient():
    cpu, gpu = tiny(), tiny(device="cuda")
    gpu.restore(copy.deepcopy(cpu.state()))
    features = torch.randn(16, 8)
    torch.testing.assert_close(cpu.scores(features), gpu.scores(features.cuda()).cpu(), rtol=2e-5, atol=2e-6)
    record = gpu.train_cycle(features.cuda(), torch.arange(8, device="cuda"), torch.arange(8, 16, device="cuda"))
    assert record["controller_gradient_norm"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA requires host GPU access")
def test_gpu_exact_native_metric_and_bootstrap_cpu_parity():
    generator = np.random.default_rng(71)
    scores = [generator.integers(0, 7, (9, 11)).astype(np.float32) for _ in range(4)]
    masks = [generator.random((9, 11)) > .7 for _ in range(4)]
    histograms = [score_histogram(score, mask) for score, mask in zip(scores, masks)]
    weights = generator.integers(1, 5, (17, 4))
    cpu_bounds = CorrectedBoundedPixelBootstrap(histograms, bins=16).bounds(weights)
    gpu_scores = [torch.from_numpy(score).cuda() for score in scores]
    gpu_masks = [torch.from_numpy(mask).cuda() for mask in masks]
    gpu_bounds = bootstrap_bounds(gpu_scores, gpu_masks, weights, bins=16, chunk=5)
    np.testing.assert_allclose(cpu_bounds, gpu_bounds, rtol=0, atol=1e-12)
    metric = exact_ranking(torch.cat([score.flatten() for score in gpu_scores]),
                           torch.cat([mask.flatten() for mask in gpu_masks]))
    assert abs(metric["pixel_ap"] - PixelRanking(histograms).metrics()["ap"]) < 1e-12
