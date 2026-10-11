# exps/cms_nm01/test_cms_nd.py
"""Focused CMS-NM-01 correctness checks; no dataset or anomaly access."""

import math

import torch

from exps.cms_nm01.cms_nd import CMSNormalDensityMemory, FixedProjection, _AffineFlow, mixture_nll


def test_fixed_projection_is_orthonormal_and_deterministic():
    first = FixedProjection(seed=22)
    second = FixedProjection(seed=22)
    assert torch.equal(first.matrix, second.matrix)
    assert first.orthonormal_error() < 2e-5


def test_affine_flow_log_likelihood_matches_manual_base_and_logdet():
    torch.manual_seed(3)
    flow = _AffineFlow(latent_dim=4, hidden_dim=8, blocks=1, scale_limit=2.0)
    x = torch.randn(5, 4)
    logp, _, transformed = flow.log_prob(x)
    logstd = flow.base_logstd.clamp(-5.0, 5.0)
    manual = -0.5 * (((transformed - flow.base_mean) * torch.exp(-logstd)).square() + 2 * logstd + math.log(2 * math.pi)).sum(-1)
    left, right = x.chunk(2, dim=-1)
    raw_scale, shift = flow.blocks[0].net(left).chunk(2, dim=-1)
    scale = flow.blocks[0].scale_limit * torch.tanh(raw_scale / flow.blocks[0].scale_limit)
    assert torch.allclose(logp, manual + scale.sum(-1), atol=1e-6, rtol=1e-6)


def test_multiblock_log_likelihood_matches_autograd_jacobian():
    torch.manual_seed(30)
    flow = _AffineFlow(latent_dim=4, hidden_dim=8, blocks=4).double()
    x = torch.randn(4, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(lambda value: flow.log_prob(value)[2], x)
    _, log_absolute_determinant = torch.linalg.slogdet(jacobian)
    logp, _, transformed = flow.log_prob(x)
    expected = torch.distributions.Independent(
        torch.distributions.Normal(flow.base_mean, flow.base_logstd.exp()), 1
    ).log_prob(transformed) + log_absolute_determinant
    assert torch.allclose(logp, expected, atol=1e-10, rtol=1e-10)


def test_mixture_nll_is_exact_equal_weight_logmeanexp():
    values = (torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]]), torch.tensor([[3.0, 4.0]]))
    expected = -torch.logsumexp(-torch.stack(values), dim=0) + math.log(3.0)
    assert torch.allclose(mixture_nll(tuple(-value for value in values)), expected)


def test_clocks_accumulators_and_anchor_propagation():
    torch.manual_seed(7)
    model = CMSNormalDensityMemory(levels=(1, 2, 4), seed=7)
    image = torch.randn(1, 3, 768)
    for event in range(1, 5):
        result = model.commit_image(image)
        assert result.event_id == event
        expected = tuple(index for index, period in enumerate((1, 2, 4)) if event % period == 0)
        assert result.due_levels == expected
        assert tuple(int(level.update_count) for level in model.levels) == (event, event // 2, event // 4)
        assert tuple(int(level.pending_count) for level in model.levels) == (0, event % 2, event % 4)
    assert torch.equal(model.levels[0].anchor_2, model.levels[1].flow.blocks[0].net[0].weight)
    assert torch.equal(model.levels[1].anchor_2, model.levels[2].flow.blocks[0].net[0].weight)


def test_serialization_resume_and_read_only_inference():
    torch.manual_seed(8)
    model = CMSNormalDensityMemory(seed=8)
    image = torch.randn(1, 3, 768)
    model.commit_image(image)
    snapshot = model.state_snapshot()
    resumed = CMSNormalDensityMemory.from_snapshot(snapshot)
    test_input = torch.randn(1, 3, 768)
    before = model.state_snapshot()
    score_a = model.score_patches(test_input)["mixture_nll"]
    score_b = resumed.score_patches(test_input)["mixture_nll"]
    after = model.state_snapshot()
    assert torch.equal(score_a, score_b)
    for key in before["state_dict"]:
        assert torch.equal(before["state_dict"][key], after["state_dict"][key])
    rng_before = torch.random.get_rng_state().clone()
    model.score_patches(test_input)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for _ in range(64):
        model.commit_image(image)
        resumed.commit_image(image)
    for key, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    assert tuple(int(level.update_count) for level in model.levels) == (65, 8, 1)
    assert tuple(int(level.pending_count) for level in model.levels) == (0, 1, 1)


def test_accumulated_mean_gradient_and_retention_step():
    model = CMSNormalDensityMemory(levels=(2,), learning_rates=(0.01,), retention=(0.2,), max_step_norm=1e6)
    level = model.levels[0]
    parameter = level.current_parameters()[0]
    anchor = level.anchors()[0]
    with torch.no_grad():
        anchor.copy_(parameter - 0.5)
    before = parameter.detach().clone()
    for coefficient in (2.0, 4.0):
        level.accumulate(coefficient * parameter.sum())
    assert parameter.grad is None
    assert torch.equal(level.accumulators()[0], torch.full_like(parameter, 6.0))
    step = level.propose_step()
    assert torch.allclose(step[0], torch.full_like(parameter, 0.031), atol=1e-8)
    level.commit_step(step)
    assert torch.allclose(parameter, before - 0.031, atol=1e-8)
    assert int(level.pending_count) == 0
    assert all(torch.count_nonzero(value) == 0 for value in level.accumulators())


def test_finite_gradients_and_memory_accounting():
    torch.manual_seed(9)
    model = CMSNormalDensityMemory(seed=9)
    image = torch.randn(1, 3, 768)
    losses = model._losses_with_graph(image)
    for loss in losses:
        loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert model.persistent_bytes() > 1_900_000
