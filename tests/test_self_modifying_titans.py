# tests/test_self_modifying_titans.py
import inspect

import pytest
import torch
from torch import nn

from models.hope_cad import state
from models.hope_cad.self_modifying_titans import SMTProjectionResult, SelfModifyingTitans


def make_smt(*, dim: int = 4, adaptive_q: bool = False, **kwargs: int) -> SelfModifyingTitans:
    return SelfModifyingTitans(dim=dim, adaptive_q=adaptive_q, **kwargs)


def test_snapshot_is_independent_and_detects_later_mutation() -> None:
    current = {"memory": torch.tensor([1.0, 2.0]), "nested": {"step": torch.tensor(3)}}
    saved = state.snapshot_persistent_state(current)
    current["memory"][0] = 99.0
    current["nested"]["step"].fill_(7)
    assert saved["memory"][0].item() == 1.0
    assert saved["nested"]["step"].item() == 3
    assert not state.persistent_states_equal(current, saved)


def test_nested_snapshot_comparison_supports_exact_and_tolerant_modes() -> None:
    left = {"a": [torch.ones(2), (torch.tensor(2.0),)], "meta": "ok"}
    right = state.snapshot_persistent_state(left)
    right["a"][0][0] += 1e-5
    assert not state.persistent_states_equal(left, right)
    assert state.persistent_states_equal(left, right, atol=1e-4)


def test_finite_validator_accepts_and_rejects_nested_state() -> None:
    assert state.validate_finite_state({"a": torch.zeros(2), "b": [torch.ones(1)]})
    with pytest.raises(ValueError, match="non-finite persistent state tensor"):
        state.validate_finite_state({"nested": {"value": torch.tensor([float("nan")])}})


def test_persistent_tensor_bytes_on_known_nested_tensors() -> None:
    values = {"float": torch.zeros(2, 3), "half": [torch.zeros(5, dtype=torch.float16)]}
    assert state.persistent_tensor_bytes(values) == 2 * 3 * 4 + 5 * 2


def test_restore_snapshot_supports_deterministic_reset() -> None:
    fresh = {"weight": torch.tensor([1.0, 2.0]), "nested": [torch.tensor(4.0)]}
    current = state.snapshot_persistent_state(fresh)
    initial = state.snapshot_persistent_state(fresh)
    current["weight"].add_(10)
    current["nested"][0].zero_()
    state.restore_persistent_state(current, initial)
    assert state.persistent_states_equal(current, fresh)


def test_state_module_has_no_legacy_runtime_imports() -> None:
    source = inspect.getsource(state)
    for forbidden in ("cadic", "meta_nath", "acc_gating", "titans_memory"):
        assert forbidden not in source.lower()


def test_construction_shapes_and_canonical_defaults() -> None:
    module = make_smt(dim=8)
    assert module.adaptive_q is False
    result = module(torch.randn(2, 7, 8))
    assert isinstance(result, SMTProjectionResult)
    assert result.q.shape == result.k.shape == result.v.shape == (2, 7, 8)
    assert result.eta.shape == result.alpha.shape == (2, 7, 1)
    assert module.local_conv.kernel_size == (4,)


def test_input_validation_and_row_major_conv1d() -> None:
    module = make_smt(dim=8)
    assert isinstance(module.local_conv, nn.Conv1d)
    assert not any(isinstance(child, nn.Conv2d) for child in module.modules())
    assert module.local_conv_padding == (1, 2)
    assert module.preprocess(torch.randn(2, 9, 8)).shape == (2, 9, 8)
    with pytest.raises(ValueError, match="rank 3"):
        module(torch.randn(7, 8))
    with pytest.raises(ValueError, match="final dimension 8"):
        module(torch.randn(1, 7, 9))


def test_q_and_k_are_l2_normalized_and_outputs_finite() -> None:
    result = make_smt(dim=8)(torch.randn(2, 7, 8))
    assert torch.allclose(result.q.norm(dim=-1), torch.ones(2, 7), atol=1e-5)
    assert torch.allclose(result.k.norm(dim=-1), torch.ones(2, 7), atol=1e-5)
    for value in result.values():
        assert torch.isfinite(value).all()


def test_fixed_and_adaptive_query_paths_have_distinct_state_contracts() -> None:
    fixed = make_smt(dim=4, adaptive_q=False)
    adaptive = make_smt(dim=4, adaptive_q=True)
    assert not hasattr(fixed.memories, "q")
    assert hasattr(adaptive.memories, "q")
    q_before = fixed.base_q.weight.detach().clone()
    fixed(torch.randn(1, 5, 4), update=True)
    assert torch.equal(fixed.base_q.weight, q_before)
    adaptive_before = adaptive.memories["q"].weight.detach().clone()
    adaptive(torch.randn(1, 5, 4), update=True)
    assert not torch.equal(adaptive.memories["q"].weight, adaptive_before)


def test_chunk_spans_cover_every_token_once() -> None:
    for length, chunk in ((3, 7), (7, 7), (8, 7), (20, 16), (20, 7), (32, 16)):
        spans = SelfModifyingTitans.chunk_spans(length, chunk)
        covered = [position for start, end in spans for position in range(start, end)]
        assert covered == list(range(length))
        assert all(end > start for start, end in spans)


def test_unequal_chunk_streams_are_independent_and_have_remainders() -> None:
    module = make_smt(dim=4, memory_chunk_size=16, auxiliary_memory_chunk_size=7)
    inspection = module.inspect(torch.randn(1, 20, 4), update=True)
    memory = [trace.positions for trace in inspection.updates if trace.name == "memory"]
    auxiliary = [trace.positions for trace in inspection.updates if trace.name == "k"]
    assert memory == [tuple(range(16)), tuple(range(16, 20))]
    assert auxiliary == [tuple(range(7)), tuple(range(7, 14)), tuple(range(14, 20))]
    assert module.memory_update_count.item() == 2
    assert module.auxiliary_update_count.item() == 3


def test_equal_small_boundary_and_batch_streams() -> None:
    module = make_smt(dim=4, memory_chunk_size=4, auxiliary_memory_chunk_size=4)
    inspection = module.inspect(torch.randn(2, 5, 4), update=True)
    memory = [trace.positions for trace in inspection.updates if trace.name == "memory"]
    assert memory == [tuple(range(4)), tuple(range(4, 5)), tuple(range(4)), tuple(range(4, 5))]
    assert module.memory_update_count.item() == 4
    assert module.auxiliary_update_count.item() == 4


def test_associative_raw_residual_matches_manual_reference() -> None:
    module = make_smt(dim=3)
    with torch.no_grad():
        module.memories["memory"].weight.copy_(torch.tensor([[1., 2., 0.], [0., 1., 1.], [1., 0., 1.]]))
    key = torch.tensor([[[1., 2., 3.], [0., 1., 2.]]])
    value = torch.tensor([[[1., 0., 2.], [2., 1., 0.]]])
    prediction, residual, scalar = module.associative_memory_loss(key, value)
    expected = torch.matmul(key, module.memories["memory"].weight.t()) - value
    assert torch.allclose(residual, expected)
    assert torch.allclose(scalar, expected.square().sum(dim=-1).mean())


def test_surprise_matches_direct_autograd_without_parameter_grads() -> None:
    weight = torch.tensor([[1., 2.], [0.5, -1.]])
    key = torch.tensor([[[2., -1.]]])
    target = torch.tensor([[[0.5, 1.5]]])
    surprise = SelfModifyingTitans.compute_instantaneous_surprise(key, target, weight=weight)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = 0.5 * (torch.nn.functional.linear(key, reference_weight) - target).square().sum()
    expected = torch.autograd.grad(reference, reference_weight)[0]
    assert torch.allclose(surprise, expected)


def _manual_eq93(weight: torch.Tensor, key: torch.Tensor, target: torch.Tensor, eta: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    gradient = (weight @ key - target).outer(key)
    return weight @ (alpha * torch.eye(weight.shape[1]) - eta * torch.outer(key, key)) - eta * gradient


def test_each_linear_memory_matches_manual_eq93_one_step() -> None:
    module = make_smt(dim=3, memory_chunk_size=1, auxiliary_memory_chunk_size=1)
    with torch.no_grad():
        for memory in module.memories.values():
            memory.weight.copy_(torch.eye(memory.weight.shape[0], module.dim))
        module.local_conv.weight.zero_()
        module.local_conv.bias.fill_(1.0)
    inspection = module.inspect(torch.ones(1, 1, 3), update=True)
    for trace in inspection.updates:
        prior = trace.prior_weight
        key = trace.key[0]
        value = trace.value[0]
        target = trace.target[0]
        eta = trace.eta[0, 0]
        alpha = trace.alpha[0, 0]
        expected = _manual_eq93(prior, key, target, eta, alpha)
        assert torch.allclose(target, prior @ value)
        assert torch.allclose(trace.post_weight, expected, atol=1e-6, rtol=1e-6)
        assert torch.allclose(trace.surprise[0], (prior @ key - target).outer(key))


def test_self_targets_are_pre_update_memory_outputs() -> None:
    module = make_smt(dim=3, memory_chunk_size=1, auxiliary_memory_chunk_size=1)
    inspection = module.inspect(torch.randn(1, 2, 3), update=False)
    for trace in inspection.updates:
        expected = (trace.prior_weight @ trace.key[0]).unsqueeze(0)
        assert trace.target.shape == expected.shape


def test_causal_pre_update_retrieval_is_observable() -> None:
    torch.manual_seed(10)
    first = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=2)
    second = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=2)
    second.load_state_dict(first.state_dict())
    x = torch.randn(1, 4, 3)
    before = first(x, update=False)
    during = second(x, update=True)
    assert torch.allclose(before.memory_prediction[:, :2], during.memory_prediction[:, :2])
    after = second(x, update=False)
    assert not torch.allclose(during.memory_prediction, after.memory_prediction)


def test_update_false_is_strictly_read_only_and_deterministic() -> None:
    module = make_smt(dim=4)
    x = torch.randn(2, 6, 4)
    before = state.snapshot_persistent_state(module.state_dict())
    first = module(x, update=False)
    middle = state.snapshot_persistent_state(module.state_dict())
    second = module(x, update=False)
    after = state.snapshot_persistent_state(module.state_dict())
    assert state.persistent_states_equal(before, middle)
    assert state.persistent_states_equal(before, after)
    assert torch.equal(first.memory_prediction, second.memory_prediction)


def test_reset_and_state_dict_continuation() -> None:
    torch.manual_seed(7)
    module = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=3)
    initial = state.snapshot_persistent_state(module.state_dict())
    x = torch.randn(1, 5, 3)
    module(x, update=True)
    module.reset_state()
    assert state.persistent_states_equal(initial, module.state_dict())
    uninterrupted = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=3)
    uninterrupted.load_state_dict(initial)
    checkpoint = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=3)
    checkpoint.load_state_dict(initial)
    uninterrupted(x, update=True)
    checkpoint(x, update=True)
    saved = state.snapshot_persistent_state(checkpoint.state_dict())
    resumed = make_smt(dim=3, memory_chunk_size=2, auxiliary_memory_chunk_size=3)
    resumed.load_state_dict(saved)
    y_a = uninterrupted(x, update=True)
    y_b = resumed(x, update=True)
    assert torch.allclose(y_a.memory_prediction, y_b.memory_prediction)
    assert state.persistent_states_equal(uninterrupted.state_dict(), resumed.state_dict())


def test_online_state_is_finite_bounded_and_detached() -> None:
    module = make_smt(dim=4, memory_chunk_size=3, auxiliary_memory_chunk_size=2)
    bytes_before = state.persistent_tensor_bytes(module.online_state())
    x = torch.randn(1, 8, 4)
    for _ in range(4):
        module(x, update=True)
    bytes_after = state.persistent_tensor_bytes(module.online_state())
    assert bytes_before == bytes_after
    assert state.validate_finite_state(module.online_state())
    assert all(not value.requires_grad and value.grad_fn is None for value in module.online_state().values())


def test_fixed_query_and_local_conv_receive_external_gradients() -> None:
    module = make_smt(dim=4)
    result = module(torch.randn(1, 5, 4), update=False)
    result.memory_prediction.square().mean().backward()
    assert module.base_q.weight.grad is not None
    assert module.local_conv.weight.grad is not None
    assert torch.isfinite(module.base_q.weight.grad).all()
    assert torch.isfinite(module.local_conv.weight.grad).all()


def test_no_legacy_runtime_imports() -> None:
    source = inspect.getsource(SelfModifyingTitans)
    for forbidden in ("cadic", "meta_nath", "acc_gating", "titans_memory"):
        assert forbidden not in source.lower()
