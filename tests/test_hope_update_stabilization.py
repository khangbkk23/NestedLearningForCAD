# tests/test_hope_update_stabilization.py
"""Focused tests for the isolated update-stabilization experiment."""

from pathlib import Path

import pytest
import torch

from exps.hope_retention_stabilization import canonical_initial_states, clone_smt_from_state, read_table, write_table
from exps.hope_update_stabilization import (
    HORIZON_TOKENS,
    UpdateMapping,
    phase_a_mappings,
    phase_c_mapping,
    pm0,
    pm0_oracle_check,
    run_update_smt,
)


def test_hnp_and_fixed_eta_formulas_are_exact():
    raw_eta = torch.tensor([[[-1.0], [0.0], [1.0]]])
    raw_alpha = torch.tensor([[[-2.0], [0.0], [2.0]]])
    mapping = UpdateMapping("hnp", eta_kind="horizon_sigmoid", eta_h=0.10)
    eta, alpha = mapping.controls(raw_eta, raw_alpha)
    assert torch.equal(eta, (0.10 / HORIZON_TOKENS) * torch.sigmoid(raw_eta))
    assert torch.equal(alpha, torch.ones_like(raw_alpha))
    fixed = phase_c_mapping(0.10, "fixed")
    fixed_eta, _ = fixed.controls(raw_eta, raw_alpha)
    assert torch.equal(fixed_eta, torch.full_like(raw_eta, 0.10 / HORIZON_TOKENS * 0.5))


def test_hnr_alpha_formula_and_alpha_one_are_exact():
    raw_eta = torch.zeros(1, 2, 1)
    raw_alpha = torch.tensor([[[-1.0], [1.0]]])
    hnr = UpdateMapping("hnr", alpha_kind="horizon_near_one", lambda_h=0.005)
    _, alpha = hnr.controls(raw_eta, raw_alpha)
    assert torch.equal(alpha, 1.0 - (0.005 / 784.0) * torch.sigmoid(raw_alpha))
    _, one = UpdateMapping("one").controls(raw_eta, raw_alpha)
    assert torch.equal(one, torch.ones_like(raw_alpha))


def test_pm0_oracle_matches_production_on_tiny_input():
    state, _ = canonical_initial_states(91, 4)
    result = pm0_oracle_check(torch.randn(1, 20, 4), state)
    assert result["passed"]


def test_term_isolation_disables_only_selected_term_and_keeps_schema():
    state, _ = canonical_initial_states(13, 4)
    x = torch.randn(1, 20, 4)
    outputs = {}
    for mapping in phase_a_mappings()[:3]:
        module = clone_smt_from_state(state, 4)
        output, _, terms = run_update_smt(module, x, mapping, capture_trace=True, capture_terms=True)
        outputs[mapping.name] = output
        assert terms
        assert all(torch.isfinite(value).all().item() for value in module.state_dict().values() if isinstance(value, torch.Tensor))
    assert not torch.equal(outputs["RANK_ONLY_alpha1_eta0.02"], outputs["FULL_alpha1_eta0.02"])
    assert not torch.equal(outputs["SURPRISE_ONLY_alpha1_eta0.02"], outputs["FULL_alpha1_eta0.02"])


def test_frozen_m_eta_does_not_mutate_eta_memory():
    state, _ = canonical_initial_states(17, 4)
    module = clone_smt_from_state(state, 4)
    initial = module.memories["eta"].weight.detach().clone()
    mapping = phase_c_mapping(0.02, "frozen")
    run_update_smt(module, torch.randn(1, 20, 4), mapping)
    assert torch.equal(module.memories["eta"].weight, initial)


def test_experiment_artifact_round_trip(tmp_path: Path):
    path = tmp_path / "rows.parquet"
    write_table(path, [{"candidate": "HNP", "finite": True, "event": 1}])
    assert read_table(path) == [{"candidate": "HNP", "event": 1, "finite": True}]

