# tests/test_continuum_memory.py
"""Numerical and state-contract checks for sequential scheduled memories."""

import copy
import inspect

import pytest
import torch
from torch.nn import functional as F

from models.hope_cad import ContinuumMemorySystem
from models.hope_cad import state
from models.hope_cad import continuum_memory


def make_cms(periods=(1, 8), *, dim=3, hidden_dim=None, rates=None):
    if rates is None:
        rates = [1e-3] * len(periods)
    return ContinuumMemorySystem(dim, periods, rates, hidden_dim=hidden_dim)


def quadratic(index, level_input, level_output, metadata):
    del index, level_input, metadata
    return 0.5 * level_output.square().mean()


def objectives(count, callback=quadratic):
    return [callback] * count


def snapshot(module):
    return state.snapshot_persistent_state(module.state_dict())


def manual_level(x, level):
    w1, b1, w2, b2 = (getattr(level, name) for name in
                       ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"))
    return x + F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)


def test_generic_k_geometry_and_state_inventory():
    for periods in ([1], [1, 8], [1, 8, 64], [1, 4, 16, 64]):
        module = make_cms(periods, dim=5, hidden_dim=7)
        assert len(module.levels) == len(periods)
        assert module.update_periods == tuple(periods)
        assert module(torch.randn(2, 9, 5)).shape == (2, 9, 5)
        for level in module.levels:
            assert level.fc1_weight.shape == (7, 5)
            assert level.fc1_bias.shape == (7,)
            assert level.fc2_weight.shape == (5, 7)
            assert level.fc2_bias.shape == (5,)
            for name in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
                assert isinstance(getattr(level, f"initial_{name}"), torch.nn.Parameter)
                assert name in dict(level.named_buffers())
                assert f"grad_accum_{name}" in dict(level.named_buffers())
            assert level.pending_count.item() == level.update_count.item() == 0
        assert module.completed_events.item() == 0


@pytest.mark.parametrize("periods", [[], [0], [-1], [True], [1.0], [2, 1], [1, "8"]])
def test_invalid_periods_rejected_without_sorting(periods):
    with pytest.raises(ValueError, match="update_periods"):
        make_cms(periods)


@pytest.mark.parametrize("rates", [[0.0], [-0.1], [float("inf")], [float("nan")], [True], []])
def test_invalid_learning_rates_rejected(rates):
    with pytest.raises(ValueError, match="learning_rates"):
        make_cms([1], rates=rates)


def test_input_validation_and_one_image_commit_boundary():
    module = make_cms([1])
    before = snapshot(module)
    for x in (torch.randn(3), torch.randn(1, 3, 4), torch.randn(0, 3, 3),
              torch.randn(1, 0, 3), torch.full((1, 2, 3), float("nan"))):
        with pytest.raises(ValueError):
            module(x)
    with pytest.raises(ValueError, match="B == 1"):
        module.commit_image(torch.randn(2, 3, 3), objectives(1))
    assert state.persistent_states_equal(before, module.state_dict())


def test_exact_sequential_composition_and_noncommutative_order():
    module = make_cms([1, 2], dim=2)
    x = torch.tensor([[[0.3, -0.4], [0.7, 0.2]]])
    with torch.no_grad():
        for index, level in enumerate(module.levels):
            level.fc1_weight.copy_(torch.tensor([[1.0, 0.5], [-0.2, 0.9]]) * (index + 1))
            level.fc2_weight.copy_(torch.tensor([[0.4, -0.7], [0.6, 0.2]]) * (index + 1))
            level.fc1_bias.copy_(torch.tensor([0.3, -0.1]) * (index + 1))
            level.fc2_bias.copy_(torch.tensor([-0.2, 0.5]) * (index + 1))
    expected = manual_level(manual_level(x, module.levels[0]), module.levels[1])
    reversed_output = manual_level(manual_level(x, module.levels[1]), module.levels[0])
    assert torch.allclose(module(x), expected, atol=1e-7)
    assert not torch.allclose(expected, reversed_output)


def test_forward_is_exactly_read_only_and_deterministic():
    module = make_cms([1, 3])
    x = torch.randn(2, 4, 3)
    before = snapshot(module)
    first, second = module(x), module(x)
    assert torch.equal(first, second)
    assert state.persistent_states_equal(before, module.state_dict())
    assert all(param.grad is None for param in module.parameters())


@pytest.mark.parametrize(
    ("periods", "expected"),
    [
        ([1, 8], [((0, 1) if event in (8, 16) else (0,)) for event in range(1, 18)]),
        ([2, 5], [tuple(index for index, period in enumerate((2, 5)) if event % period == 0)
                  for event in range(1, 18)]),
    ],
)
def test_event_schedule_exact_through_seventeen(periods, expected):
    module = make_cms(periods)
    x = torch.randn(1, 3, 3) * 0.2
    for event, due in enumerate(expected, start=1):
        result = module.commit_image(x, objectives(2))
        assert result.event_id == event
        assert result.due_levels == due
        assert result.pending_counts == tuple(event % period for period in periods)
        assert result.update_counts == tuple(event // period for period in periods)
        assert result.output.shape == (1, 3, 3)


def test_equal_periods_remain_distinct_levels():
    module = make_cms([2, 2], rates=[0.01, 0.02])
    x = torch.randn(1, 4, 3)
    first = module.commit_image(x, objectives(2))
    assert first.due_levels == ()
    assert first.pending_counts == (1, 1)
    second = module.commit_image(x, objectives(2))
    assert second.due_levels == (0, 1)
    assert second.update_counts == (1, 1)
    assert module.levels[0].fc1_weight.data_ptr() != module.levels[1].fc1_weight.data_ptr()


def test_objective_receives_pre_event_inputs_outputs_and_read_only_metadata():
    module = make_cms([1, 2])
    x = torch.randn(1, 4, 3)
    expected_l0 = manual_level(x, module.levels[0])
    expected_l1 = manual_level(expected_l0, module.levels[1])
    seen = []

    def objective(index, level_input, level_output, metadata):
        seen.append((index, level_input.detach().clone(), level_output.detach().clone(), metadata))
        return (index + 1) * level_output.square().mean()

    result = module.commit_image(x, [objective, objective], {"source": "normal"})
    assert [item[0] for item in seen] == [0, 1]
    assert torch.allclose(seen[0][1], x)
    assert torch.allclose(seen[0][2], expected_l0)
    assert torch.allclose(seen[1][1], expected_l0)
    assert torch.allclose(seen[1][2], expected_l1)
    assert seen[0][3]["source"] == seen[1][3]["source"] == "normal"
    with pytest.raises(TypeError):
        seen[0][3]["source"] = "changed"
    assert len(result.objective_values) == 2


@pytest.mark.parametrize("failure", ["raise", "nonscalar", "nonfinite", "nan_gradient"])
def test_failed_objective_is_transactional(failure):
    module = make_cms([1, 2])
    x = torch.randn(1, 2, 3)
    before = snapshot(module)

    def bad(index, level_input, level_output, metadata):
        del index, level_input, metadata
        if failure == "raise":
            raise RuntimeError("bad objective")
        if failure == "nonscalar":
            return level_output
        if failure == "nonfinite":
            return level_output.sum() * float("nan")

        class NaNGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return tensor.sum()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * torch.full_like(level_output, float("nan"))

        return NaNGradient.apply(level_output)

    with pytest.raises((RuntimeError, ValueError)):
        module.commit_image(x, [quadratic, bad])
    assert state.persistent_states_equal(before, module.state_dict())


def test_independent_manual_gradient_sum_and_one_step_update():
    module = make_cms([2], dim=2, hidden_dim=3, rates=[0.03])
    level = module.levels[0]
    names = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
    original = [getattr(level, name).detach().clone() for name in names]
    xs = (torch.tensor([[[0.1, -0.3], [0.4, 0.2]]]),
          torch.tensor([[[-0.2, 0.5], [0.3, -0.1]]]))
    expected_sum = [torch.zeros_like(tensor) for tensor in original]
    for event, x in enumerate(xs, start=1):
        manual = [tensor.clone().requires_grad_(True) for tensor in original]
        y = x + F.linear(F.gelu(F.linear(x, manual[0], manual[1])), manual[2], manual[3])
        grads = torch.autograd.grad(0.5 * y.square().sum(), manual)
        expected_sum = [left + right.detach() for left, right in zip(expected_sum, grads)]

        def loss(index, level_input, level_output, metadata):
            del index, level_input, metadata
            return 0.5 * level_output.square().sum()

        module.commit_image(x, [loss])
        if event == 1:
            for name, initial, expected in zip(names, original, expected_sum):
                assert torch.equal(getattr(level, name), initial)
                assert torch.allclose(getattr(level, f"grad_accum_{name}"), expected, atol=1e-7)
        else:
            for name, initial, expected in zip(names, original, expected_sum):
                assert torch.allclose(getattr(level, name), initial - 0.03 * expected, atol=1e-7)
                assert torch.count_nonzero(getattr(level, f"grad_accum_{name}")).item() == 0
    assert level.pending_count.item() == 0
    assert level.update_count.item() == 1
    assert all(parameter.grad is None for parameter in module.parameters())


def test_later_objective_does_not_update_earlier_level():
    module = make_cms([1, 1], rates=[0.02, 0.02])
    first_before = [tensor.clone() for tensor in module.levels[0].current_tensors()]

    def zero_objective(index, level_input, level_output, metadata):
        del index, level_input, metadata
        return 0.0 * level_output.sum()

    module.commit_image(torch.randn(1, 4, 3), [zero_objective, quadratic])
    assert all(torch.equal(before, after) for before, after in
               zip(first_before, module.levels[0].current_tensors()))
    assert all(torch.count_nonzero(acc).item() == 0 for acc in
               module.levels[0].gradient_accumulators())


@pytest.mark.parametrize("batch_size", [2, 3])
def test_transport_batch_matches_ordered_singletons(batch_size):
    torch.manual_seed(123)
    batched = make_cms([1, 2], rates=[0.05, 0.05])
    singles = copy.deepcopy(batched)
    x = torch.randn(batch_size, 3, 3)
    stale_second = batched.forward(x[1:2]).detach().clone()
    batch_results = batched.commit_batch(x, objectives(2))
    single_results = [singles.commit_image(x[i:i + 1], objectives(2)) for i in range(batch_size)]
    for batch_result, single_result in zip(batch_results, single_results):
        assert batch_result.event_id == single_result.event_id
        assert batch_result.due_levels == single_result.due_levels
        assert torch.equal(batch_result.output, single_result.output)
    assert not torch.equal(batch_results[1].output, stale_second)
    assert state.persistent_states_equal(batched.state_dict(), singles.state_dict())


def test_simultaneously_due_levels_use_entire_pre_event_chain():
    module = make_cms([1, 1], rates=[0.05, 0.05])
    x = torch.randn(1, 5, 3)
    pre_output = module(x).detach().clone()
    pre_second_input = manual_level(x, module.levels[0]).detach().clone()
    seen_second = []

    def second(index, level_input, level_output, metadata):
        del index, metadata
        seen_second.append(level_input.detach().clone())
        return level_output.square().sum()

    result = module.commit_image(x, [quadratic, second])
    assert result.due_levels == (0, 1)
    assert torch.equal(result.output, pre_output)
    assert torch.equal(seen_second[0], pre_second_input)
    assert not torch.equal(module(x), pre_output)


def test_partial_period_serialization_continues_exactly():
    torch.manual_seed(31)
    uninterrupted = make_cms([1, 4], rates=[0.002, 0.002])
    images = [torch.randn(1, 3, 3) for _ in range(4)]
    for x in images[:3]:
        uninterrupted.commit_image(x, objectives(2))
    assert uninterrupted.levels[1].pending_count.item() == 3
    saved = snapshot(uninterrupted)
    next_read = uninterrupted(images[3]).detach().clone()
    next_result = uninterrupted.commit_image(images[3], objectives(2))
    torch.manual_seed(999)
    resumed = make_cms([1, 4], rates=[0.002, 0.002])
    resumed.load_state_dict(saved)
    assert torch.equal(resumed(images[3]), next_read)
    resumed_result = resumed.commit_image(images[3], objectives(2))
    assert next_result.due_levels == resumed_result.due_levels == (0, 1)
    assert next_result.objective_values == resumed_result.objective_values
    assert torch.equal(next_result.output, resumed_result.output)
    assert state.persistent_states_equal(uninterrupted.state_dict(), resumed.state_dict())


def test_incompatible_checkpoint_rejected_before_state_mutation():
    source = make_cms([1, 4])
    target = make_cms([1, 5])
    before = snapshot(target)
    with pytest.raises(ValueError, match="configuration or schema"):
        target.load_state_dict(source.state_dict())
    assert state.persistent_states_equal(before, target.state_dict())


def test_reset_restores_initial_state_without_rng_use():
    torch.manual_seed(121)
    module = make_cms([1, 3])
    fresh = copy.deepcopy(module)
    x = torch.randn(1, 4, 3)
    for _ in range(5):
        module.commit_image(x, objectives(2))
    rng_before = torch.random.get_rng_state().clone()
    module.reset_state()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert state.persistent_states_equal(module.state_dict(), fresh.state_dict())


def test_evaluation_clone_is_isolated_frozen_and_order_invariant():
    module = make_cms([1, 4])
    for _ in range(3):
        module.commit_image(torch.randn(1, 3, 3), objectives(2))
    before = snapshot(module)
    a, b = torch.randn(1, 3, 3), torch.randn(1, 3, 3)
    clone_ab = module.clone_for_evaluation()
    clone_ba = module.clone_for_evaluation()
    assert clone_ab.levels[0].fc1_weight.data_ptr() != module.levels[0].fc1_weight.data_ptr()
    clone_before = snapshot(clone_ab)
    a1, b1 = clone_ab(a), clone_ab(b)
    b2, a2 = clone_ba(b), clone_ba(a)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert torch.equal(a1, clone_ab(a))
    with pytest.raises(RuntimeError, match="evaluation clone"):
        clone_ab.commit_image(a, objectives(2))
    assert state.persistent_states_equal(before, module.state_dict())
    assert state.persistent_states_equal(clone_before, clone_ab.state_dict())


def test_no_persistent_graph_or_unbounded_state_after_many_events():
    module = make_cms([1, 2, 4])
    before = state.persistent_tensor_bytes(module.state_dict())
    tensor_keys = set(key for key, value in module.state_dict().items() if isinstance(value, torch.Tensor))
    for _ in range(12):
        module.commit_image(torch.randn(1, 7, 3, requires_grad=True), objectives(3))
    after = state.persistent_tensor_bytes(module.state_dict())
    assert before == after
    assert tensor_keys == set(key for key, value in module.state_dict().items()
                              if isinstance(value, torch.Tensor))
    assert state.validate_finite_state(module.state_dict())
    assert all(value.grad_fn is None for value in module.state_dict().values()
               if isinstance(value, torch.Tensor))
    assert all(not buffer.requires_grad for buffer in module.buffers())
    assert all(parameter.grad is None for parameter in module.parameters())
    assert not any("window" in key or "context" in key for key in tensor_keys)


def test_state_bytes_independent_of_numeric_periods():
    one = make_cms([1, 8])
    two = make_cms([1, 8000])
    assert state.persistent_tensor_bytes(one.state_dict()) == state.persistent_tensor_bytes(two.state_dict())


def test_no_legacy_or_anomaly_imports():
    source = inspect.getsource(continuum_memory).lower()
    for forbidden in ("cadic", "meta_nath", "acc_gating", "titans_memory", "anomaly"):
        assert forbidden not in source
