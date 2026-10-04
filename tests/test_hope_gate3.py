# tests/test_hope_gate3.py
"""Focused tests for long-horizon Gate 3 diagnostics."""

from pathlib import Path

import pytest
import torch

from exps.hope_gate3 import build_retention_matrix, fit_slopes, full_geometry, load_200_stream
from exps.hope_retention_stabilization import canonical_initial_states, clone_smt_from_state, write_table, read_table
from exps.hope_update_stabilization import UpdateMapping, run_update_smt


def test_full_geometry_top1_energy_matches_independent_reference():
    torch.manual_seed(3)
    value = torch.randn(1, 784, 4)
    result = full_geometry(value)
    centered = value[0].double() - value[0].double().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    expected = float((singular[0].square() / singular.square().sum()).item())
    assert result["top1_energy_fraction"] == pytest.approx(expected, rel=1e-10)


def test_state_slope_and_retention_matrix_are_deterministic():
    rows = [{"event_id": index, "memory_norm": float(index), "smt_rms": float(index), "hope_rms": float(index), "smt_effective_rank": float(index), "hope_effective_rank": float(index), "smt_top1_energy_fraction": 0.5, "hope_top1_energy_fraction": 0.5, "complete_event": True} for index in range(1, 11)]
    first = fit_slopes(rows, windows=((1, 10),))
    second = fit_slopes(rows, windows=((1, 10),))
    assert first == second
    anchors = [{"checkpoint_event": 40, "class_name": "bottle", "space": "SMT", "cosine": 1.0, "relative_l2": 0.0}]
    assert build_retention_matrix(anchors, "cosine")[0]["mean_cosine"] == 1.0


def test_dualrate_changes_only_selected_update_coefficient():
    state, _ = canonical_initial_states(14, 4)
    x = torch.randn(1, 20, 4)
    uniform = UpdateMapping("uniform", eta_kind="horizon_sigmoid", eta_h=.1)
    dual = UpdateMapping("dual", eta_kind="horizon_sigmoid", eta_h=.1, surprise_eta_multiplier=.25)
    left = clone_smt_from_state(state, 4)
    right = clone_smt_from_state(state, 4)
    _, _, uniform_terms = run_update_smt(left, x, uniform, capture_terms=True, capture_term_names=frozenset({"memory"}))
    _, _, dual_terms = run_update_smt(right, x, dual, capture_terms=True, capture_term_names=frozenset({"memory"}))
    assert uniform_terms[0]["delta_rank_norm"] == pytest.approx(dual_terms[0]["delta_rank_norm"], rel=1e-6)
    assert dual_terms[0]["delta_surprise_norm"] == pytest.approx(uniform_terms[0]["delta_surprise_norm"] * .25, rel=1e-5)


def test_200_stream_order_and_boundaries_when_cache_exists():
    root = Path("results/hope_cad/real_feature_probe/real_cpu_seed0/features")
    if not (root / "class_transistor.pt").is_file():
        pytest.skip("full Task-3A cache unavailable")
    records = load_200_stream(root)
    assert len(records) == 200
    assert [records[index]["class_name"] for index in (0, 39, 40, 79, 80, 119, 120, 159, 160, 199)] == ["bottle", "bottle", "carpet", "carpet", "grid", "grid", "toothbrush", "toothbrush", "transistor", "transistor"]
    assert records[40]["relative_path"].startswith("carpet/train/good/")


def test_gate3_artifact_round_trip(tmp_path: Path):
    path = tmp_path / "gate3.parquet"
    write_table(path, [{"event_id": 1, "candidate": "HNP", "finite": True}])
    assert read_table(path) == [{"candidate": "HNP", "event_id": 1, "finite": True}]

