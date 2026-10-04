# tests/test_hope_retention_stabilization.py
"""Focused tests for the isolated retention/control experiments."""

from pathlib import Path

import pytest
import torch

from exps.hope_retention_stabilization import (
    RetentionMapping,
    canonical_initial_states,
    canonical_oracle,
    clone_smt_from_state,
    clone_state_dict,
    effective_rank,
    GATE2_CANDIDATE_NAMES,
    GATE2B_CANDIDATE_NAMES,
    HORIZON_TOKENS,
    build_read_only_rms_reference,
    gate2b_mappings,
    gate2_mappings,
    load_bottle_event,
    load_cached_stream,
    run_candidate,
    run_stream_candidate,
    run_experimental_smt,
    write_table,
    read_table,
)
from models.hope_cad.self_modifying_titans import SelfModifyingTitans


def test_r1_r2_and_eta_mapping_formulas_are_exact():
    raw_eta = torch.tensor([[[-1.0], [0.0], [1.0]]])
    raw_alpha = torch.tensor([[[-2.0], [0.0], [2.0]]])
    eta, alpha = RetentionMapping("r1", "near_one", 0.002).controls(raw_eta, raw_alpha)
    assert torch.equal(eta, torch.sigmoid(raw_eta))
    assert torch.equal(alpha, 1.0 - 0.002 * torch.sigmoid(raw_alpha))
    eta, alpha = RetentionMapping("r2", "residual_one", 0.001).controls(raw_eta, raw_alpha)
    assert torch.equal(eta, torch.sigmoid(raw_eta))
    assert torch.equal(alpha, 1.0 + 0.001 * torch.tanh(raw_alpha))
    eta, _ = RetentionMapping("scaled", "pm0", eta_kind="scaled_sigmoid", eta_scale=0.1).controls(raw_eta, raw_alpha)
    assert torch.equal(eta, 0.1 * torch.sigmoid(raw_eta))


def test_pm0_small_experimental_path_matches_canonical_oracle():
    torch.manual_seed(12)
    canonical = SelfModifyingTitans(4, memory_chunk_size=16, auxiliary_memory_chunk_size=16)
    state = clone_state_dict(canonical.state_dict())
    x = torch.randn(1, 9, 4)
    experiment = clone_smt_from_state(state, 4)
    output, _ = run_experimental_smt(experiment, x, RetentionMapping("PM0"))
    oracle = clone_smt_from_state(state, 4)
    expected = oracle(x, update=True).memory_prediction
    assert torch.allclose(output, expected, rtol=3e-5, atol=3e-5)
    for name in oracle.memories:
        assert torch.allclose(oracle.memories[name].weight, experiment.memories[name].weight, rtol=3e-5, atol=3e-5)
    assert torch.equal(oracle.memory_update_count, experiment.memory_update_count)
    assert torch.equal(oracle.auxiliary_update_count, experiment.auxiliary_update_count)


def test_candidates_are_isolated_and_do_not_mutate_source():
    torch.manual_seed(4)
    smt_state, cms_state = canonical_initial_states(4, 4)
    source = clone_smt_from_state(smt_state, 4)
    x = torch.randn(1, 7, 4)
    first = run_candidate(x, RetentionMapping("PM0"), smt_state, cms_state)
    second = run_candidate(x, RetentionMapping("R1", "near_one", 0.001), smt_state, cms_state)
    assert not torch.equal(first["smt_state"].memories["memory"].weight, second["smt_state"].memories["memory"].weight)
    for name in source.memories:
        assert torch.equal(source.memories[name].weight, smt_state[f"memories.{name}.weight"])


def test_chunk_boundaries_and_causal_output_are_preserved():
    torch.manual_seed(5)
    smt_state, _ = canonical_initial_states(5, 4)
    module = clone_smt_from_state(smt_state, 4)
    output, trace = run_experimental_smt(module, torch.randn(1, 20, 4), RetentionMapping("PM0"))
    assert output.shape == (1, 20, 4)
    assert [row["point"] for row in trace] == [0, 1, 2, 4, 8, 16, 20]
    assert all(torch.isfinite(value).all().item() for value in module.memory_state().values())


def test_effective_rank_uses_all_rows_and_is_deterministic():
    rows = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    assert effective_rank(rows) == pytest.approx(2.0, rel=1e-5)
    assert effective_rank(rows) == effective_rank(rows)


def test_no_persistent_autograd_graph_after_experiment():
    torch.manual_seed(9)
    smt_state, _ = canonical_initial_states(9, 4)
    module = clone_smt_from_state(smt_state, 4)
    run_experimental_smt(module, torch.randn(1, 8, 4), RetentionMapping("PM0"))
    for value in module.state_dict().values():
        if isinstance(value, torch.Tensor):
            assert value.grad_fn is None
            assert not value.requires_grad


def test_pm0_real_bottle_event_matches_canonical_when_cache_exists():
    path = Path("results/hope_cad/real_feature_probe/real_cpu_seed0/features/class_bottle.pt")
    if not path.is_file():
        pytest.skip("Task-3A bottle feature cache is unavailable")
    x, _ = load_bottle_event(path)
    smt_state, cms_state = canonical_initial_states(0, 768)
    result = run_candidate(x, RetentionMapping("PM0"), smt_state, cms_state)
    oracle = canonical_oracle(x, smt_state, result)
    assert oracle["output_equal"]
    assert oracle["state_equal"]
    assert oracle["counters_equal"]


def test_artifact_table_can_be_reopened(tmp_path: Path):
    path = tmp_path / "trace.parquet"
    write_table(path, [{"candidate": "PM0", "point": 0, "finite": True}])
    assert read_table(path) == [{"candidate": "PM0", "finite": True, "point": 0}]


def test_gate2_candidate_set_is_exact_and_ordered():
    assert tuple(mapping.name for mapping in gate2_mappings()) == GATE2_CANDIDATE_NAMES
    assert len(gate2_mappings()) == 4


def test_gate2b_candidate_set_and_horizon_are_exact():
    assert tuple(mapping.name for mapping in gate2b_mappings()) == GATE2B_CANDIDATE_NAMES
    assert all(mapping.horizon == HORIZON_TOKENS for mapping in gate2b_mappings())
    assert len(gate2b_mappings()) == 5


def test_horizon_normalized_and_c0_controls_are_exact():
    raw_eta = torch.tensor([[[-1.0], [0.0], [1.0]]])
    raw_alpha = torch.tensor([[[-2.0], [0.0], [2.0]]])
    c0 = gate2b_mappings()[1]
    eta, alpha = c0.controls(raw_eta, raw_alpha)
    assert torch.equal(eta, 0.02 * torch.sigmoid(raw_eta))
    assert torch.equal(alpha, torch.ones_like(raw_alpha))
    h2 = gate2b_mappings()[3]
    eta, alpha = h2.controls(raw_eta, raw_alpha)
    expected_eta = 0.02 * torch.sigmoid(raw_eta)
    expected_alpha = 1.0 - (0.005 / 784.0) * torch.sigmoid(raw_alpha)
    assert torch.equal(eta, expected_eta)
    assert torch.equal(alpha, expected_alpha)


def test_cumulative_retention_and_rms_reference_are_deterministic():
    torch.manual_seed(23)
    smt_state, cms_state = canonical_initial_states(23, 4)
    records = [
        {"patches": torch.randn(1, 7, 4), "class_name": "a", "relative_path": f"a/{index}.png"}
        for index in range(2)
    ]
    references = build_read_only_rms_reference(records, smt_state, cms_state)
    result = run_stream_candidate(records, gate2b_mappings()[1], smt_state, cms_state, read_only_references=references)
    assert [row["alpha_product_observed"] for row in result["rows"]] == [1.0, 1.0]
    assert [row["alpha_product_cumulative"] for row in result["rows"]] == [1.0, 1.0]
    for row, reference in zip(result["rows"], references):
        assert row["smt_rms_fresh_read_only"] == pytest.approx(reference["smt_rms"])
        assert row["hope_rms_fresh_read_only"] == pytest.approx(reference["hope_rms"])
        assert row["smt_rms_vs_fresh_read_only"] >= 0.0
        assert row["hope_rms_vs_fresh_read_only"] >= 0.0


def test_cached_gate2_stream_has_deterministic_50_event_order_when_available():
    root = Path("results/hope_cad/real_feature_probe/real_cpu_seed0/features")
    if not (root / "class_bottle.pt").is_file() or not (root / "class_carpet.pt").is_file():
        pytest.skip("Task-3A feature shards are unavailable")
    records = load_cached_stream(root, count=50)
    assert len(records) == 50
    assert records[0]["class_name"] == "bottle"
    assert records[39]["class_name"] == "bottle"
    assert records[40]["class_name"] == "carpet"
    assert records[0]["relative_path"] == "bottle/train/good/000.png"
    assert records[40]["relative_path"] == "carpet/train/good/000.png"


def test_stream_candidate_advances_one_cms_event_per_image_without_reset():
    torch.manual_seed(22)
    smt_state, cms_state = canonical_initial_states(22, 4)
    records = [
        {"patches": torch.randn(1, 7, 4), "class_name": "a", "relative_path": f"a/{index}.png"}
        for index in range(3)
    ]
    result = run_stream_candidate(records, gate2_mappings()[0], smt_state, cms_state)
    assert result["event_count"] == 3
    assert [row["cms_completed_events"] for row in result["rows"]] == [1, 2, 3]
    assert result["state_bytes_constant"]
    assert result["state_schema_constant"]
