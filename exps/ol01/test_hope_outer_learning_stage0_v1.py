# exps/test_hope_outer_learning_stage0_v1.py
"""Synthetic CPU algebra, derivatives, state integrity and attribution tests."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import resource
import time

import pytest
import torch
from torch.nn import functional as F

from exps.ol03.hope_outer_learning_p1_v1 import (
    ARMS, CENTERS, Configuration, Counters, FunctionalState, MEMORIES,
    checkpoint_payload, commit_functional_event, detached_parameters, evaluate_arm,
    functional_image_event, functional_state_to_detached_snapshot,
    functional_support_sequence, initial_functional_state, load_checkpoint,
    masked_synthetic_rgb, mock_patch_backbone, parameter_fingerprint,
    project_from_snapshot, propose_functional_event, read_query_without_update,
    read_with_transition, restore_checkpoint, save_checkpoint,
    state_fingerprint, synthetic_parameters, tensor_inventory,
)
from scripts.exps.ol01.hope_outer_learning_stage0_v1 import (
    check_forward_parity, check_gradients, check_graph_lifecycle,
    check_restart, install_synthetic_io_guard, run_validation,
    source_hashes, synthetic_inputs,
)


_REPORTS = {"passed": 0, "failed": 0, "skipped": 0}


def pytest_sessionstart(session):
    session.config._synthetic_started = time.perf_counter()
    session.config._synthetic_access = install_synthetic_io_guard()
    session.config._synthetic_hashes = source_hashes()
    session.config._synthetic_deselected = []
    torch.set_num_threads(1)


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.skipped):
        _REPORTS[report.outcome] += 1


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    result = yield
    if not hasattr(session.config, "_synthetic_access"):
        return result
    data = {"exitstatus": int(exitstatus), "tests": dict(_REPORTS),
            "deselected": len(session.config._synthetic_deselected),
            "wall_seconds": time.perf_counter() - session.config._synthetic_started,
            "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "io_guard": session.config._synthetic_access,
            "source_hashes_unchanged": source_hashes() == session.config._synthetic_hashes}
    if not data["source_hashes_unchanged"] or data["io_guard"]["forbidden_access_attempts"]:
        session.exitstatus = 1
    if os.environ.get("HOPE_SYNTHETIC_TEST_AUDIT"):
        Path(os.environ["HOPE_SYNTHETIC_TEST_AUDIT"]).write_text(json.dumps(data, indent=2) + "\n")
    return result


def pytest_deselected(items):
    # pytest's terminal summary remains the authoritative deselection count.
    for item in items:
        item.config._synthetic_deselected.append(item.nodeid)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("n", [1, 2, 7, 17])
def test_exact_fp64_all_five_map_algebra(n):
    parameters = synthetic_parameters(dim=3)
    state = initial_functional_state(parameters)
    support, _, _ = synthetic_inputs(parameters, (n,))
    projection = project_from_snapshot(parameters, state, support[0])
    event = propose_functional_event(parameters, state, support[0])
    keys, values, gates = projection.keys, projection.values, projection.gates[:, 0]
    C = keys.T @ torch.diag(gates / n) @ keys
    D = (keys - values).T @ torch.diag(gates / n) @ keys
    transition = torch.eye(3, dtype=torch.float64) - 0.02 * (C + D)
    torch.testing.assert_close(event.C, C, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(event.D, D, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(event.transition, transition, rtol=1e-11, atol=1e-12)
    for name in MEMORIES:
        before = state.weights[name]
        reference = before.clone()
        for k, v, g in zip(keys, values, gates):
            reference = reference - 0.02 * g / n * (
                torch.outer(before @ k, k) + torch.outer(before @ (k - v), k)
            )
        torch.testing.assert_close(event.weights[name], reference, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_original_p1_forward_oracle(dtype):
    evidence = check_forward_parity(dtype=dtype)
    assert evidence["passed"] and evidence["max_abs"] <= 1e-5
    assert evidence["counters"] == {"memory": 4, "auxiliary": 4, "online": 8, "completed": 4}


def test_a0_wq_finite_difference_and_product_gradients():
    evidence = check_gradients()
    assert evidence["passed"]
    for name in ("A0", "Wq"):
        assert evidence["parameters"][name]["max_relative_error"] <= 1e-4
        assert evidence["parameters"][name]["max_absolute_error"] <= 1e-7
        assert evidence["parameters"][name]["gradient_norm"] > 1e-8


def test_frozen_self_target_surrogate_gradient_all_five_maps():
    parameters = synthetic_parameters(dim=3)
    state = initial_functional_state(parameters)
    support, _, _ = synthetic_inputs(parameters, (7,))
    event = propose_functional_event(parameters, state, support[0])
    keys, values = event.projection.keys, event.projection.values
    sqrt_weights = (event.projection.gates / keys.shape[0]).sqrt()
    before_hash = state_fingerprint(state)
    for name in MEMORIES:
        before = state.weights[name]
        candidate = before.detach().clone().requires_grad_()
        target = F.linear(values, before).detach()
        prediction = F.linear(keys, candidate)
        loss = 0.5 * ((prediction - target) * sqrt_weights).square().sum()
        loss = loss + 0.5 * (prediction * sqrt_weights).square().sum()
        gradient, = torch.autograd.grad(loss, candidate)
        expected = before.detach() @ (event.C + event.D)
        torch.testing.assert_close(gradient, expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(candidate - 0.02 * gradient, event.weights[name],
                                   rtol=1e-11, atol=1e-12)
    assert state_fingerprint(state) == before_hash


def test_independent_normalized_query_gradient_and_frozen_ownership():
    parameters = synthetic_parameters(dim=3)
    support, query, target = synthetic_inputs(parameters, (7, 4, 3))
    result = functional_support_sequence(parameters, support)
    prediction = read_query_without_update(parameters, result.state, query)
    loss = 0.5 * (prediction - target).square().sum(-1).mean()
    loss.backward()
    spatial = project_from_snapshot(parameters, result.state, query).spatial
    raw = F.linear(spatial, parameters.wq)
    q = F.normalize(raw, dim=-1, eps=1e-8)
    read_gradient = (prediction.detach() - target) / query.shape[1]
    q_gradient = read_gradient @ result.state.weights["memory"].detach()
    raw_gradient = (q_gradient - (q_gradient * q.detach()).sum(-1, keepdim=True) * q.detach()) / raw.detach().norm(dim=-1, keepdim=True)
    expected_wq = raw_gradient.T @ spatial
    torch.testing.assert_close(parameters.wq.grad, expected_wq, rtol=1e-11, atol=1e-12)
    assert parameters.a0.grad is not None and parameters.wq.grad is not None
    assert all(value.grad is None and not value.requires_grad for value in parameters.auxiliary.values())
    assert parameters.conv_weight.grad is parameters.conv_bias.grad is None
    assert target.grad is None and not target.requires_grad
    assert not result.product.requires_grad


def test_support_transition_independent_of_selected_parameters():
    parameters = synthetic_parameters()
    support, _, _ = synthetic_inputs(parameters)
    original = functional_support_sequence(parameters, support)
    changed = replace(parameters, a0=(parameters.a0.detach() + 0.7).requires_grad_(),
                      wq=(parameters.wq.detach() - 0.3).requires_grad_())
    other = functional_support_sequence(changed, support)
    assert torch.equal(original.product, other.product)
    for a, b in zip(original.events, other.events):
        assert torch.equal(a.transition, b.transition)
        assert not a.transition.requires_grad and a.transition.grad_fn is None
    for name in ("k", "v", "eta", "alpha"):
        assert torch.equal(original.state.weights[name], other.state.weights[name])


@pytest.mark.parametrize("part", ["k", "eta", "conv_weight", "conv_bias"])
def test_unapproved_parameter_gradients_rejected(part):
    parameters = synthetic_parameters()
    if part in parameters.auxiliary:
        auxiliary = {**parameters.auxiliary, part: parameters.auxiliary[part].clone().requires_grad_()}
        invalid = replace(parameters, auxiliary=auxiliary)
    else:
        invalid = replace(parameters, **{part: getattr(parameters, part).clone().requires_grad_()})
    with pytest.raises(ValueError, match="frozen"):
        initial_functional_state(invalid)


@pytest.mark.parametrize("n", [1, 2, 15, 17, 31, 784])
def test_image_counter_schema_reset_and_query_read_only(n):
    parameters = synthetic_parameters()
    state = initial_functional_state(parameters)
    support, query, _ = synthetic_inputs(parameters, (n, 3))
    initial_hash = state_fingerprint(state)
    slow_hash = parameter_fingerprint(parameters)
    inventory = tensor_inventory(state)
    versions = {name: value._version for name, value in state.weights.items()}
    rng = torch.get_rng_state().clone()
    result = functional_support_sequence(parameters, support)
    before = state_fingerprint(result.state)
    output = read_query_without_update(parameters, result.state, query)
    output = output.detach()
    output.zero_()
    assert state_fingerprint(result.state) == before
    assert state_fingerprint(state) == initial_hash
    assert parameter_fingerprint(parameters) == slow_hash
    assert torch.equal(torch.get_rng_state(), rng)
    assert versions == {name: value._version for name, value in state.weights.items()}
    assert result.state.counters == Counters(2, 2, 4, 2)
    assert tensor_inventory(result.state)["tensor_bytes"] == inventory["tensor_bytes"]
    assert tensor_inventory(result.state)["schema"] == inventory["schema"]
    reset = functional_support_sequence(parameters, support)
    assert state_fingerprint(reset.state) == before


def test_different_synthetic_image_orders_are_distinct_and_deterministic():
    parameters = synthetic_parameters()
    support, _, _ = synthetic_inputs(parameters, (4, 5, 6))
    normal = functional_support_sequence(parameters, support)
    reversed_order = functional_support_sequence(parameters, tuple(reversed(support)))
    repeated = functional_support_sequence(parameters, support)
    assert state_fingerprint(normal.state) == state_fingerprint(repeated.state)
    assert not torch.equal(normal.state.weights["memory"], reversed_order.state.weights["memory"])
    assert normal.state.counters == reversed_order.state.counters


@pytest.mark.parametrize("fault", ["nan", "shape", "schema", "counter", "finite_change"])
def test_invalid_proposal_is_atomic_and_stale_snapshot_rejected(fault):
    parameters = synthetic_parameters()
    state = initial_functional_state(parameters)
    support, _, _ = synthetic_inputs(parameters, (7,))
    proposal = propose_functional_event(parameters, state, support[0])
    original = state_fingerprint(state)
    weights = dict(proposal.weights)
    if fault == "nan":
        weights["alpha"] = torch.full_like(weights["alpha"], float("nan"))
    elif fault == "shape":
        weights["v"] = weights["v"][:1]
    elif fault == "schema":
        del weights["eta"]
    elif fault == "finite_change":
        weights["memory"] = weights["memory"] + 0.01
    invalid = replace(proposal, weights=weights)
    if fault == "counter":
        invalid = replace(invalid, counters=Counters(2, 2, 4, 2))
    with pytest.raises(ValueError):
        commit_functional_event(parameters, state, invalid)
    assert state_fingerprint(state) == original
    committed = commit_functional_event(parameters, state, proposal)
    with pytest.raises(ValueError, match="stale"):
        commit_functional_event(parameters, committed, proposal)
    assert state_fingerprint(state) == original


def test_snapshot_conversion_is_detached_disjoint_and_prediction_equivalent():
    parameters = synthetic_parameters()
    support, query, _ = synthetic_inputs(parameters)
    result = functional_support_sequence(parameters, support)
    before = state_fingerprint(result.state)
    snapshot = functional_state_to_detached_snapshot(result.state)
    frozen = detached_parameters(parameters)
    expected = read_query_without_update(parameters, result.state, query)
    actual = read_query_without_update(frozen, snapshot, query)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not actual.requires_grad
    assert not tensor_inventory(snapshot)["graph_bearing"]
    for name in MEMORIES:
        assert snapshot.weights[name].data_ptr() != result.state.weights[name].data_ptr()
        snapshot.weights[name].zero_()
    assert state_fingerprint(result.state) == before
    assert set(asdict(snapshot.counters)) == {"memory", "auxiliary", "online", "completed"}


def test_synthetic_checkpoint_restart_exact(tmp_path):
    assert check_restart(tmp_path)["forward_bitwise_equal"]


def test_checkpoint_identity_integrity_rng_and_new_episode_gradients(tmp_path):
    parameters = synthetic_parameters()
    support, query, target = synthetic_inputs(parameters)
    state = functional_support_sequence(parameters, (support[2],)).state
    payload = checkpoint_payload(parameters, state, input_identities=("a", "b", "c"), order=(2, 0, 1))
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    rng = torch.get_rng_state().clone()
    loaded, detached = load_checkpoint(path, expected_identities=("a", "b", "c"),
                                       expected_order=(2, 0, 1), requires_grad=True)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not tensor_inventory(detached)["graph_bearing"]
    fresh = functional_support_sequence(loaded, support)
    loss = (read_query_without_update(loaded, fresh.state, query) - target).square().mean()
    gradients = torch.autograd.grad(loss, (loaded.a0, loaded.wq))
    assert all(float(gradient.norm()) > 0 for gradient in gradients)
    with pytest.raises(ValueError, match="identities"):
        restore_checkpoint(payload, expected_identities=("b", "a", "c"), expected_order=(2, 0, 1))
    corrupted = {**payload, "weights": {**payload["weights"], "v": payload["weights"]["v"] + 1}}
    with pytest.raises(ValueError, match="integrity"):
        restore_checkpoint(corrupted, expected_identities=("a", "b", "c"), expected_order=(2, 0, 1))
    with torch.random.fork_rng(devices=[]):
        torch.rand(3)
        restore_checkpoint(payload, expected_identities=("a", "b", "c"),
                           expected_order=(2, 0, 1), restore_rng=True)
        assert torch.equal(torch.get_rng_state(), payload["cpu_rng_state"])


def test_checkpoint_rejects_identity_metadata_shorter_than_completed_events():
    parameters = synthetic_parameters()
    support, _, _ = synthetic_inputs(parameters)
    state = functional_support_sequence(parameters, support).state
    payload = checkpoint_payload(parameters, state,
                                 input_identities=("a", "b", "c"), order=(0, 1, 2))
    inconsistent = {**payload, "input_identities": ("a",), "order": (0,)}
    with pytest.raises(ValueError, match="inconsistent"):
        restore_checkpoint(inconsistent, expected_identities=("a",), expected_order=(0,))


def test_graph_references_are_disposed():
    evidence = check_graph_lifecycle(repeats=12)
    assert evidence["passed"] and evidence["all_weak_references_released"]


def test_shared_mask_geometry_normalization_and_no_direct_pixel_copy():
    patch_ids = torch.arange(784, dtype=torch.float32).reshape(28, 28)
    clean = patch_ids.repeat_interleave(8, 0).repeat_interleave(8, 1)[None, None].repeat(1, 3, 1, 1)
    teacher = mock_patch_backbone(clean).clone()
    student, mask = masked_synthetic_rgb(clean)
    assert student.shape == clean.shape == (1, 3, 224, 224)
    assert mask.sum() == 144 * 8 * 8
    for row, col in CENTERS:
        assert bool(mask[(row - 1) * 8:(row + 2) * 8, (col - 1) * 8:(col + 2) * 8].all())
        assert teacher[0, row * 28 + col, 0] == row * 28 + col
    changed_center = torch.where(mask[None, None], clean + 999, clean)
    changed_view, same_mask = masked_synthetic_rgb(changed_center)
    assert torch.equal(student, changed_view) and torch.equal(mask, same_mask)
    assert torch.equal(clean, patch_ids.repeat_interleave(8, 0).repeat_interleave(8, 1)[None, None].repeat(1, 3, 1, 1))
    assert not torch.equal(mock_patch_backbone(changed_center), teacher)
    assert student.data_ptr() != clean.data_ptr()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    assert torch.equal((student * std + mean)[:, :, mask], mean.expand_as(student)[:, :, mask])


def test_all_six_arm_interfaces_and_history_substitution_are_pure():
    parameters = synthetic_parameters()
    static_parameters = synthetic_parameters(seed=19)
    support, query, _ = synthetic_inputs(parameters)
    result = functional_support_sequence(parameters, support)
    wrong = functional_support_sequence(parameters, tuple(reversed(support)))
    fixed = torch.eye(parameters.config.dim, dtype=parameters.a0.dtype)
    before = state_fingerprint(result.state)
    parameter_before = parameter_fingerprint(parameters)
    for arm in ARMS:
        kwargs = {"adapted_state": result.state} if arm.endswith("P1") else {}
        if arm == "STATIC_META":
            kwargs = {"static_parameters": static_parameters}
        if arm == "STATIC_NORMAL_CONTROL":
            kwargs = {"static_predictor": lambda image: image[0].clone()}
        output = evaluate_arm(arm, parameters, query, **kwargs)
        assert output.shape == (5, 4) and bool(torch.isfinite(output).all())
    for product in (result.product, wrong.product, fixed):
        assert read_with_transition(parameters, query, product).shape == (5, 4)
    assert state_fingerprint(result.state) == before
    assert parameter_fingerprint(parameters) == parameter_before
    assert parameters.a0.data_ptr() != static_parameters.a0.data_ptr()


def test_constant_transform_can_fake_paired_improvement_and_be_absorbed():
    parameters = synthetic_parameters()
    _, query, _ = synthetic_inputs(parameters)
    frozen = evaluate_arm("META_FROZEN", parameters, query)
    constant = 0.7 * torch.eye(4, dtype=parameters.a0.dtype)
    actual = evaluate_arm("META_P1", parameters, query, support_transition=constant)
    wrong = read_with_transition(parameters, query, constant)
    static = replace(parameters, a0=parameters.a0 @ constant)
    equivalent = evaluate_arm("STATIC_META", parameters, query, static_parameters=static)
    target = torch.zeros_like(actual)
    assert (actual - target).square().mean() < (frozen - target).square().mean()
    assert torch.equal(actual, wrong) and torch.equal(actual, equivalent)
    # Output sensitivity alone is also insufficient: an orthogonal sign change
    # changes the read but has identical error against this fixed zero target.
    changed = read_with_transition(parameters, query, -constant)
    assert not torch.equal(actual, changed)
    assert torch.equal(actual.square(), changed.square())


def test_bounded_validation_artifacts_reopen_without_parameter_fitting(tmp_path):
    summary = run_validation(tmp_path / "synthetic", seed=0)
    reopened = json.loads((tmp_path / "synthetic/summary.json").read_text())
    assert summary == reopened
    assert all(summary["checks"].values())
    assert summary["outer_optimizer_steps"] == 0
    assert summary["real_data_accessed"] is summary["gpu_used"] is False
    assert summary["source_hashes_before"] == summary["source_hashes_after"]


@pytest.mark.parametrize("h", [0.01, 0.03])
def test_update_constant_cannot_be_changed(h):
    with pytest.raises(ValueError, match="configuration"):
        Configuration(4, h=h)
