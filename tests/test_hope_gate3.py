# tests/test_hope_gate3.py
"""Focused tests for long-horizon Gate 3 diagnostics."""

from pathlib import Path

import pytest
import torch

from exps.task3r.hope_gate3 import (
    anchor_rows_at_checkpoint, anchor_stream, build_retention_matrix,
    check_event_counters, evaluate_anchor, fit_slopes, full_geometry,
    load_200_stream, load_checkpoint, neutral_retention, reference_outputs,
    read_only_oracle,
    replay_candidate, save_checkpoint, self_reference_check, snapshot_module,
    tensor_rms, tensor_state_equal, run_long_candidate,
)
from exps.hope_retention_stabilization import canonical_initial_states, clone_smt_from_state, clone_cms_from_state, probe_objective, write_table, read_table
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
    anchors = [{"candidate": "one", "evaluation_checkpoint": 40, "anchor_position": 1, "class_name": "bottle", "space": "SMT", "cosine": 1.0, "relative_l2": 0.0}]
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


def tiny_anchors():
    generator = torch.Generator().manual_seed(27)
    records = [{"patches": torch.randn(1, 784, 4, generator=generator), "class_name": "bottle", "relative_path": f"bottle/train/good/{index:03d}.png", "class_index": index} for index in range(40)]
    return records, anchor_stream(records)


def test_four_anchors_preserve_identity_and_own_reference():
    state, cms_state = canonical_initial_states(0, 4)
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    _, anchors = tiny_anchors()
    references = reference_outputs(smt, cms, anchors["bottle"], "cpu")
    assert set(references) == {("bottle", pos) for pos in (1, 10, 20, 40)}
    assert not torch.equal(references[("bottle", 1)]["SMT"], references[("bottle", 10)]["SMT"])
    store = {}
    rows = anchor_rows_at_checkpoint("candidate", 40, smt, cms, anchors, store, "cpu")
    assert len(rows) == 8
    for row in rows:
        assert row["cosine"] == pytest.approx(1, abs=3e-5)
        assert row["relative_l2"] == pytest.approx(0, abs=3e-5)
        assert row["rms_ratio"] == pytest.approx(1, abs=3e-5)
        self_reference_check(row)
    # Reproducing the old class-only mixup must fail the hard invariant.
    store[("bottle", 10)] = store[("bottle", 1)]
    with pytest.raises(AssertionError, match="anchor path mismatch"):
        anchor_rows_at_checkpoint("candidate", 50, smt, cms, anchors, store, "cpu")


def test_self_reference_check_reports_exact_offending_identity():
    row = {"candidate": "candidate", "class_name": "bottle", "anchor_position": 10, "evaluation_checkpoint": 40, "reference_checkpoint": 40, "space": "SMT", "cosine": .7, "relative_l2": .3, "rms_ratio": 1}
    with pytest.raises(AssertionError, match="anchor_position.*10.*cosine=0.7"):
        self_reference_check(row)


def test_retention_matrices_cannot_collide_across_candidates():
    rows = [{"candidate": candidate, "evaluation_checkpoint": 80, "class_name": "bottle", "anchor_position": pos, "space": space, "cosine": value} for candidate, value in (("A", .99), ("B", .9)) for space in ("SMT", "HOPE") for pos in (1, 10, 20, 40)]
    matrix = build_retention_matrix(rows, "cosine")
    assert len(matrix) == 4
    keys = {(row["candidate"], row["checkpoint"], row["class_name"], row["space"], row["metric"]) for row in matrix}
    assert len(keys) == 4
    assert all(row["anchor_count"] == 4 for row in matrix)
    with pytest.raises(ValueError, match="duplicate matrix source"):
        build_retention_matrix(rows + [rows[0]], "cosine")


def test_read_only_anchor_cannot_mutate_state_or_expose_mutable_storage():
    state, cms_state = canonical_initial_states(0, 4)
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    before_smt, before_cms = snapshot_module(smt), snapshot_module(cms)
    smt_out, hope_out = evaluate_anchor(smt, cms, torch.randn(1, 784, 4))
    smt_out.zero_()
    hope_out.zero_()
    assert tensor_state_equal(before_smt, snapshot_module(smt))
    assert tensor_state_equal(before_cms, snapshot_module(cms))


def test_fresh_initial_reference_is_not_evolving_pre_event():
    records, _ = tiny_anchors()
    state, cms_state = canonical_initial_states(0, 4)
    before = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in state.items()}
    mapping = UpdateMapping("candidate", alpha_kind="horizon_near_one", lambda_h=.5, eta_kind="horizon_sigmoid", eta_h=.1)
    result = run_long_candidate(records[:2], mapping, state, cms_state, "cpu", max_events=2)
    row = result["rows"][1]
    fresh_smt, fresh_cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    smt_out, hope_out = evaluate_anchor(fresh_smt, fresh_cms, records[1]["patches"])
    assert row["fresh_initial_read_only_smt_rms"] == pytest.approx(tensor_rms(smt_out), rel=3e-5)
    assert row["fresh_initial_read_only_hope_rms"] == pytest.approx(tensor_rms(hope_out), rel=3e-5)
    assert row["pre_event_read_only_smt_rms"] != row["fresh_initial_read_only_smt_rms"]
    assert row["post_event_smt_rms"] == row["smt_rms"]
    assert "smt_rms_fresh_read_only" not in row
    assert tensor_state_equal(before, state)


def test_checkpoint_exact_continuation_and_counters(tmp_path: Path):
    state, cms_state = canonical_initial_states(0, 4)
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    mapping = UpdateMapping("candidate", eta_kind="horizon_sigmoid", eta_h=.02)
    image = torch.randn(1, 20, 4)
    for event in (1, 2, 3):
        out, _, _ = run_update_smt(smt, image, mapping, event_id=event, capture_trace=False)
        cms.commit_image(out, [probe_objective] * cms.K)
    path = tmp_path / "candidate__event_003.pt"
    metadata = save_checkpoint(path, smt, cms, mapping, 3, 20)
    other_smt, other_cms, _ = load_checkpoint(path, mapping, "cpu")
    assert metadata["reopened_exactly"]
    assert tensor_state_equal(snapshot_module(smt), snapshot_module(other_smt))
    assert tensor_state_equal(snapshot_module(cms), snapshot_module(other_cms))
    for module, continuum in ((smt, cms), (other_smt, other_cms)):
        out, _, _ = run_update_smt(module, image, mapping, event_id=4, capture_trace=False)
        continuum.commit_image(out, [probe_objective] * continuum.K)
        check_event_counters(module, continuum, 4, 20)
    assert tensor_state_equal(snapshot_module(smt), snapshot_module(other_smt))
    assert tensor_state_equal(snapshot_module(cms), snapshot_module(other_cms))
    with pytest.raises(ValueError, match="incompatible checkpoint mapping"):
        load_checkpoint(path, UpdateMapping("other"), "cpu")


def test_replay_does_not_recompute_svd_and_matches_counters(tmp_path: Path, monkeypatch):
    state, cms_state = canonical_initial_states(0, 4)
    records = [{"patches": torch.randn(1, 20, 4), "class_name": "bottle", "relative_path": "bottle/train/good/000.png"}]
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    mapping = UpdateMapping("candidate", eta_kind="horizon_sigmoid", eta_h=.02)
    out, _, _ = run_update_smt(smt, records[0]["patches"], mapping, capture_trace=False)
    hope = cms.commit_image(out, [probe_objective] * cms.K).output
    from exps.hope_retention_stabilization import _stream_state_summary
    original = {"candidate": mapping.name, "event_id": 1, "memory_norm": float(smt.memories["memory"].weight.double().norm()), "smt_rms": tensor_rms(out), "hope_rms": tensor_rms(hope), **_stream_state_summary(smt, cms)}
    def forbidden(*args, **kwargs):
        raise AssertionError("state replay called SVD")
    monkeypatch.setattr(torch.linalg, "svdvals", forbidden)
    result = replay_candidate(records, mapping, state, cms_state, "cpu", tmp_path, [original])
    assert result["passed"] and all(check["passed"] for check in result["checks"])
    assert tensor_state_equal(snapshot_module(clone_smt_from_state(state, 4)), state)


def test_neutral_retention_uses_whole_image_horizon():
    mapping = UpdateMapping("candidate", alpha_kind="horizon_near_one", lambda_h=.002)
    expected = (1 - .002 / (2 * 784)) ** (784 * 200)
    assert neutral_retention(mapping, 200) == pytest.approx(expected, rel=1e-10)
    assert neutral_retention(UpdateMapping("no_forgetting"), 200) == 1


def test_old_artifact_reference_fields_are_renamed_without_mutating_source():
    from scripts.exps.task3r.hope_gate3 import correct_reference_semantics
    original = {"event_id": 1, "pre_smt_rms": .1, "pre_hope_rms": .2, "smt_rms_fresh_read_only": .1, "hope_rms_fresh_read_only": .2, "smt_rms": .05, "hope_rms": .1}
    snapshot = dict(original)
    corrected = correct_reference_semantics([original], [{"smt_rms": .4, "hope_rms": .5}])[0]
    assert original == snapshot
    assert corrected["pre_event_read_only_smt_rms"] == .1
    assert corrected["fresh_initial_read_only_smt_rms"] == .4
    assert corrected["post_event_smt_rms"] == .05
    assert corrected["smt_rms_to_fresh_initial"] == pytest.approx(.125)
    assert "smt_rms_fresh_read_only" not in corrected


def test_pareto_dominance_keeps_adaptation_and_history_separate():
    from scripts.exps.task3r.hope_gate3 import pareto_frontier
    def point(name, stability, plasticity):
        return {"candidate": name, "classification": "STABLE", **{f"mean_corrected_{space}_relative_l2": stability for space in ("smt", "hope")}, **{f"mean_corrected_{space}_anchor_cosine": 1 - stability for space in ("smt", "hope")}, **{f"mean_{space}_anchor_amplitude_change": stability for space in ("smt", "hope")}, **{f"mean_current_image_{space}_relative_l2": plasticity for space in ("smt", "hope")}}
    low = point("low", .01, .01)
    high = point("high", .02, .03)
    dominated = point("dominated", .03, .005)
    assert pareto_frontier([low, high, dominated]) == ["low", "high"]


def test_counter_validation_rejects_a_replay_reset():
    state, cms_state = canonical_initial_states(0, 4)
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    with pytest.raises(AssertionError, match="continuation counter"):
        check_event_counters(smt, cms, 1, 784)


def test_public_projection_read_only_oracle_matches_initial_and_evolved_state():
    state, cms_state = canonical_initial_states(0, 4)
    smt, cms = clone_smt_from_state(state, 4), clone_cms_from_state(cms_state, 4)
    image = torch.randn(1, 20, 4)
    assert read_only_oracle(smt, cms, image)["passed"]
    mapping = UpdateMapping("candidate", eta_kind="horizon_sigmoid", eta_h=.02)
    output, _, _ = run_update_smt(smt, image, mapping, capture_trace=False)
    cms.commit_image(output, [probe_objective] * cms.K)
    assert read_only_oracle(smt, cms, image)["passed"]
