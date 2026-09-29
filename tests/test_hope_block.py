# tests/test_hope_block.py
"""Integration checks for the canonical SMT-to-CMS HOPE core."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from models.hope_cad import HopeBlock
from models.hope_cad import state


def make_hope(
    *,
    dim: int = 4,
    periods=(1, 2),
    rates=None,
    adaptive_q: bool = False,
    memory_chunk_size: int = 2,
    auxiliary_memory_chunk_size: int = 3,
) -> HopeBlock:
    if rates is None:
        rates = tuple(1e-3 for _ in periods)
    return HopeBlock(
        dim,
        adaptive_q=adaptive_q,
        memory_chunk_size=memory_chunk_size,
        auxiliary_memory_chunk_size=auxiliary_memory_chunk_size,
        cms_update_periods=periods,
        cms_learning_rates=rates,
    )


def quadratic(index, level_input, level_output, metadata):
    del index, level_input, metadata
    return 0.5 * level_output.square().mean()


def objectives(count, callback=quadratic):
    return [callback] * count


def full_snapshot(module):
    values = module.state_dict() if hasattr(module, "state_dict") else module
    return state.snapshot_persistent_state(values)


def test_read_only_composition_matches_independent_smt_then_cms() -> None:
    torch.manual_seed(11)
    module = make_hope(dim=4)
    x = torch.randn(2, 5, 4)
    before = full_snapshot(module)
    smt_copy = deepcopy(module.smt)
    cms_copy = deepcopy(module.cms)
    expected_smt = smt_copy(x, update=False).memory_prediction
    expected = cms_copy(expected_smt)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.shape == x.shape
    assert state.persistent_states_equal(before, module.state_dict())


def test_read_only_batch_equals_singletons_and_is_deterministic() -> None:
    torch.manual_seed(12)
    module = make_hope(dim=5, periods=(1, 8))
    x = torch.randn(2, 6, 5)
    before = full_snapshot(module)
    batch = module(x)
    singleton = torch.cat([module(x[index:index + 1]) for index in range(2)], dim=0)
    assert torch.allclose(batch, singleton, atol=1e-5, rtol=1e-5)
    assert torch.equal(module(x), batch)
    assert state.persistent_states_equal(before, module.state_dict())
    assert module.cms.completed_events.item() == 0
    assert all(value.item() == 0 for level in module.cms.levels for value in
               (level.pending_count, level.update_count))


def test_training_commit_uses_exact_one_causal_smt_output_for_cms() -> None:
    torch.manual_seed(13)
    module = make_hope(dim=4, periods=(1,))
    x = torch.randn(1, 7, 4)
    smt_probe = deepcopy(module.smt)
    cms_probe = deepcopy(module.cms)
    expected = smt_probe(x, update=True).memory_prediction.detach()
    captured = {}
    original = module.cms.commit_image

    def spy(representation, supplied_objectives, metadata=None):
        captured["input"] = representation.detach().clone()
        return original(representation, supplied_objectives, metadata)

    with patch.object(module.cms, "commit_image", side_effect=spy) as commit_spy:
        result = module.commit_image(x, objectives(1), {"source": "synthetic"})
    assert commit_spy.call_count == 1
    assert torch.equal(captured["input"], expected)
    assert torch.equal(result.smt_representation, expected)
    assert torch.allclose(result.output, cms_probe(result.smt_representation), atol=1e-6)
    assert result.event_id == 1
    assert module.cms.completed_events.item() == 1


def test_training_commit_is_one_smt_pass_and_returns_cms_output() -> None:
    module = make_hope(dim=3, periods=(1,))
    x = torch.randn(1, 4, 3)
    cms_probe = deepcopy(module.cms)
    calls = {"count": 0}
    original = module.smt.forward

    def spy(value, update=False):
        calls["count"] += 1
        return original(value, update=update)

    with patch.object(module.smt, "forward", side_effect=spy):
        result = module.commit_image(x, objectives(1))
    assert calls["count"] == 1
    expected = cms_probe(result.smt_representation)
    assert torch.allclose(result.output, expected, atol=1e-6)


def test_objectives_are_owned_by_caller_and_metadata_is_forwarded() -> None:
    module = make_hope(dim=3, periods=(1, 2))
    x = torch.randn(1, 3, 3)
    seen = []

    def objective(index, level_input, level_output, metadata):
        seen.append((index, tuple(level_input.shape), tuple(level_output.shape), dict(metadata)))
        return level_output.square().mean()

    module.commit_image(x, [objective, objective], {"image": 4})
    assert seen == [(0, (1, 3, 3), (1, 3, 3), {"image": 4}),
                    (1, (1, 3, 3), (1, 3, 3), {"image": 4})]


def test_commit_failure_rolls_back_only_mutable_state() -> None:
    torch.manual_seed(14)
    module = make_hope(dim=4, periods=(1, 2))
    x = torch.randn(1, 6, 4)
    online_before = module.snapshot_online_state()
    static_before = full_snapshot(module)

    def failing(index, level_input, level_output, metadata):
        del index, level_input, level_output, metadata
        raise RuntimeError("synthetic objective failure")

    with pytest.raises(RuntimeError, match="synthetic objective failure"):
        module.commit_image(x, [failing, quadratic])
    assert state.persistent_states_equal(online_before, module.snapshot_online_state())
    assert state.persistent_states_equal(static_before, module.state_dict())


@pytest.mark.parametrize(
    "bad_objective",
    [
        lambda i, a, b, m: torch.ones(1),
        lambda i, a, b, m: torch.tensor(float("nan")),
    ],
)
def test_invalid_objective_is_transactional(bad_objective) -> None:
    module = make_hope(dim=3, periods=(1,))
    before = full_snapshot(module)
    with pytest.raises(ValueError):
        module.commit_image(torch.randn(1, 3, 3), [bad_objective])
    assert state.persistent_states_equal(before, module.state_dict())


def test_commit_batch_is_ordered_singleton_transport_with_metadata() -> None:
    torch.manual_seed(15)
    batch_module = make_hope(dim=3, periods=(1, 2))
    singleton_module = deepcopy(batch_module)
    x = torch.randn(3, 4, 3)
    metadata = [{"index": index} for index in range(3)]
    batch_results = batch_module.commit_batch(x, objectives(2), metadata)
    singleton_results = tuple(
        singleton_module.commit_image(x[index:index + 1], objectives(2), metadata[index])
        for index in range(3)
    )
    assert len(batch_results) == 3
    for left, right in zip(batch_results, singleton_results):
        assert torch.allclose(left.output, right.output, atol=1e-6)
        assert left.event_id == right.event_id
        assert left.due_levels == right.due_levels
        assert left.pending_counts == right.pending_counts
    assert state.persistent_states_equal(batch_module.state_dict(), singleton_module.state_dict())


def test_evaluation_isolated_clone_adapts_smt_but_freezes_cms() -> None:
    torch.manual_seed(16)
    module = make_hope(dim=4, periods=(1, 8))
    x = torch.randn(1, 7, 4)
    source_before = full_snapshot(module)
    clone = module.clone_for_evaluation()
    assert clone is not module
    for (name, left), (_, right) in zip(module.state_dict().items(), clone.state_dict().items()):
        if isinstance(left, torch.Tensor):
            assert left.data_ptr() != right.data_ptr()
    clone_smt_before = state.snapshot_persistent_state(clone.smt.online_state())
    clone_cms_before = full_snapshot(clone.cms)
    with pytest.raises(RuntimeError):
        clone.cms.commit_image(x, objectives(2))
    output = module.evaluate_image(x)
    assert output.shape == x.shape
    assert state.persistent_states_equal(source_before, module.state_dict())
    clone.smt(x, update=True)
    assert not state.persistent_states_equal(clone_smt_before, clone.smt.online_state())
    assert state.persistent_states_equal(clone_cms_before, clone.cms.state_dict())


def test_evaluation_is_deterministic_and_order_invariant() -> None:
    torch.manual_seed(17)
    source = make_hope(dim=3, periods=(1, 8))
    first, second = torch.randn(1, 5, 3), torch.randn(1, 5, 3)
    ab = (source.evaluate_image(first), source.evaluate_image(second))
    torch.manual_seed(17)
    other = make_hope(dim=3, periods=(1, 8))
    ba = (other.evaluate_image(second), other.evaluate_image(first))
    assert torch.allclose(ab[0], ba[1], atol=1e-6)
    assert torch.allclose(ab[1], ba[0], atol=1e-6)
    assert torch.equal(source.evaluate_image(first), source.evaluate_image(first))


def test_reset_matches_fresh_module_without_reinitializing_static_parameters() -> None:
    torch.manual_seed(18)
    fresh = make_hope(dim=4, periods=(1, 4))
    torch.manual_seed(18)
    evolved = make_hope(dim=4, periods=(1, 4))
    x = torch.randn(1, 5, 4)
    for _ in range(3):
        evolved.commit_image(x, objectives(2))
    static_before = {
        name: value.detach().clone()
        for name, value in evolved.named_parameters()
    }
    evolved.reset_state()
    assert state.persistent_states_equal(fresh.state_dict(), evolved.state_dict())
    for name, value in evolved.named_parameters():
        assert torch.equal(value, static_before[name])


def test_state_dict_continuation_includes_partial_cms_period() -> None:
    torch.manual_seed(19)
    left = make_hope(dim=4, periods=(1, 4))
    x = torch.randn(4, 5, 4)
    for index in range(3):
        left.commit_image(x[index:index + 1], objectives(2))
    saved = full_snapshot(left.state_dict())
    right = make_hope(dim=4, periods=(1, 4))
    right.load_state_dict(saved)
    left_result = left.commit_image(x[3:4], objectives(2))
    right_result = right.commit_image(x[3:4], objectives(2))
    assert torch.allclose(left_result.output, right_result.output, atol=1e-6)
    assert left_result.due_levels == right_result.due_levels
    assert left_result.pending_counts == right_result.pending_counts
    assert state.persistent_states_equal(left.state_dict(), right.state_dict(), atol=1e-6)


def test_invalid_input_is_rejected_before_any_state_change() -> None:
    module = make_hope(dim=4)
    before = full_snapshot(module)
    for invalid in (torch.randn(4), torch.randn(2, 3, 5), torch.randn(0, 3, 4),
                    torch.full((1, 3, 4), float("inf")), torch.ones(1, 3, 4, dtype=torch.int64)):
        with pytest.raises((ValueError, TypeError)):
            module(invalid)
    assert state.persistent_states_equal(before, module.state_dict())


def test_generic_k_and_duplicate_periods_are_preserved() -> None:
    for periods in ((1,), (1, 8), (1, 8, 64), (1, 4, 16, 64), (2, 2)):
        module = make_hope(dim=3, periods=periods)
        assert module.cms.K == len(periods)
        assert module.cms.update_periods == periods
        assert module(torch.randn(1, 4, 3)).shape == (1, 4, 3)


def test_state_bytes_are_bounded_and_finite_after_many_commits() -> None:
    module = make_hope(dim=4, periods=(1, 8))
    x = torch.randn(1, 4, 4)
    initial_bytes = module.memory_stats()["hope_full_tensor_bytes"]
    for _ in range(20):
        result = module.commit_image(x, objectives(2))
        assert torch.isfinite(result.output).all()
    stats = module.memory_stats()
    assert stats["hope_full_tensor_bytes"] == initial_bytes
    assert validate_state(module)


def validate_state(module):
    for value in module.state_dict().values():
        if isinstance(value, torch.Tensor) and value.grad_fn is not None:
            return False
    for value in module.buffers():
        if not torch.isfinite(value).all() or value.grad_fn is not None or value.requires_grad:
            return False
    return True


def test_adaptive_query_path_integrates_without_fixed_query_state() -> None:
    module = make_hope(dim=3, periods=(1,), adaptive_q=True)
    x = torch.randn(1, 4, 3)
    before = module.smt.memories["q"].weight.detach().clone()
    module.commit_image(x, objectives(1))
    assert not torch.equal(before, module.smt.memories["q"].weight)
