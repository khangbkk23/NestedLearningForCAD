# exps/test_hope_anomaly_score_ablation.py
"""Independent angular-score, decomposition and immutable-evidence checks."""

import os

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from exps.hope_anomaly_score_ablation import (
    COSINE_EPS, affinity_components, affinity_scores, angular_components, angular_scores, assert_files_unchanged,
    file_manifest, original_process_status, verify_unit_arrays,
)
from exps.hope_image_synchronous_memory import ImageSynchronousMemory, fingerprint
from models.hope_cad.self_modifying_titans import SelfModifyingTitans
from scripts.exps.hope_anomaly_signal import write_json, read_json


def memory():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        state = SelfModifyingTitans(4).state_dict()
    return ImageSynchronousMemory(state, "P1")


@pytest.mark.parametrize("sign,expected", [(1, 0), (-1, 2)])
def test_identical_and_opposite_vectors(sign, expected):
    a = torch.tensor([[1., 2., 3.], [4., -1., .5]])
    result = angular_components(a, a * sign)
    torch.testing.assert_close(result["angle"], torch.full((2,), float(expected)), atol=3e-7, rtol=0)


def test_positive_rescaling_keeps_angle_but_changes_raw_l2():
    a = torch.tensor([[1., 2., 3.], [.2, -.1, .5]])
    b = a.flip(1)
    first = angular_components(a, b)
    second = angular_components(a * 7, b * .3)
    torch.testing.assert_close(first["angle"], second["angle"], atol=3e-7, rtol=0)
    assert not torch.allclose(first["raw"], second["raw"])


def test_zero_and_near_zero_vectors_are_finite_and_flagged():
    a = torch.tensor([[0., 0.], [1e-10, 0.], [1., 0.]])
    b = torch.tensor([[0., 0.], [2e-10, 0.], [1., 0.]])
    result = angular_components(a, b)
    assert COSINE_EPS == 1e-8
    torch.testing.assert_close(result["angle"], 1 - F.cosine_similarity(a, b, dim=-1, eps=1e-8))
    assert result["well_defined"].tolist() == [False, False, True]
    assert all(torch.isfinite(t).all() for t in result.values())


def test_l2_radial_angular_decomposition_matches_independent_reference():
    generator = torch.Generator().manual_seed(2)
    a = torch.randn(23, 7, generator=generator)
    b = 30 * torch.randn(23, 7, generator=generator)
    result = angular_components(a, b)
    torch.testing.assert_close(result["raw"], (a.double() - b.double()).square().sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(result["radial"] + result["angular"], result["raw"], rtol=2e-6, atol=1e-5)
    assert result["well_defined"].all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_vectors_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        angular_components(torch.tensor([[bad, 0.]]), torch.ones(1, 2))


def test_geometry_and_dtype_rejected():
    with pytest.raises(ValueError):
        angular_components(torch.zeros(1, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError):
        angular_components(torch.zeros(1, 3), torch.zeros(1, 3).double())


def test_scoring_is_read_only_graph_free_and_uses_current_key_value_pair():
    model = memory()
    initial = memory()
    features = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4) / 10
    model.commit_event(model.propose_event(features))
    before = fingerprint(model.smt.state_dict())
    counters = model.state_fingerprint()
    initial_before = initial.state_fingerprint()
    rng = torch.get_rng_state().clone()
    model.commit_event = lambda *_: pytest.fail("unexpected write")
    snapshot = model.snapshot_state()
    q = model.generate_update_quantities(features, snapshot)
    a = F.linear(q.keys, snapshot.weights["memory"])
    expected = 1 - F.cosine_similarity(a, q.values, dim=-1, eps=1e-8)
    result = angular_scores(model, features, initial_model=initial, capture_components=True)
    torch.testing.assert_close(result["patch_scores"], expected, atol=0, rtol=0)
    assert result["image_score"] == float(expected.max())
    assert result["patch_scores"].grad_fn is None and not result["patch_scores"].requires_grad
    assert result["history_angle_max_abs_delta"] > 0
    result["patch_scores"].zero_()
    result["patch_components"]["norm_a"].fill(0)
    assert fingerprint(model.smt.state_dict()) == before
    assert model.state_fingerprint() == counters
    assert initial.state_fingerprint() == initial_before
    assert torch.equal(rng, torch.get_rng_state())


def test_identical_initial_state_has_zero_reset_effect():
    model, initial = memory(), memory()
    image = torch.arange(28).float().reshape(1, 7, 4)
    result = angular_scores(model, image, initial_model=initial)
    assert result["reset_score_relative_l2"] == 0
    assert result["history_angle_max_abs_delta"] == 0


def test_bilinear_geometry_is_the_existing_unscaled_pixel_path():
    from exps.hope_anomaly_signal import pixel_map
    scores = torch.linspace(0, 2, 784)
    result = pixel_map(scores, (53, 61))
    reference = F.interpolate(scores.reshape(1, 1, 28, 28), size=(53, 61), mode="bilinear", align_corners=False)
    np.testing.assert_array_equal(result, reference[0, 0].numpy())


def test_original_evidence_guard_detects_changed_or_added_files(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    (source / "result.json").write_text('{"original": true}')
    records = file_manifest(source)
    assert_files_unchanged(source, records)
    (source / "new.json").write_text('{}')
    with pytest.raises(ValueError, match="original artifacts"):
        assert_files_unchanged(source, records)


def test_reused_pid_is_not_treated_as_original_runner(tmp_path):
    path = tmp_path / "job.pid"
    path.write_text(str(os.getpid()))
    assert not original_process_status(path)["alive"]


def test_score_artifacts_round_trip_without_live_state(tmp_path):
    payload = {"passed": True, "eps": COSINE_EPS, "memory_updates": False}
    write_json(tmp_path / "score_checks.json", payload)
    assert read_json(tmp_path / "score_checks.json") == payload
    path = tmp_path / "patches.npz"
    np.savez_compressed(path, angle=np.zeros((20, 784), dtype=np.float32))
    with np.load(path, allow_pickle=False) as result:
        assert result["angle"].shape == (20, 784)


def test_wrong_label_and_nonfinite_artifacts_rejected():
    rows = [{"image_id": "normal", "label": 0}, {"image_id": "defect", "label": 1}]
    payload = {"context": {"seed": 0, "category": "bottle"}, "images": [dict(rows[0], image_score=0.), dict(rows[1], image_score=1.)],
               "metrics": {"pixel_bootstrap_point_error": 0.}}
    arrays = {"image_ids": np.array(["normal", "defect"]), "patch_scores": np.array([[0.] * 784, [1.] * 784]),
              "image_scores": np.array([0., 1.]), "bootstrap_image_AUROC": np.ones(400), "bootstrap_pixel_AUPR": np.ones(400)}
    verify_unit_arrays(payload, arrays, rows)
    payload["images"][1]["label"] = 0
    with pytest.raises(ValueError, match="label"):
        verify_unit_arrays(payload, arrays, rows)
    payload["images"][1]["label"] = 1
    arrays["patch_scores"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        verify_unit_arrays(payload, arrays, rows)


def test_better_frozen_readout_is_not_reported_as_memory_learning(tmp_path):
    import pandas as pd
    from exps.hope_anomaly_signal import CATEGORIES, CHECKPOINTS, METHOD_NAMES
    from scripts.exps.hope_anomaly_score_ablation import analyze_readout

    source, output = tmp_path / "raw", tmp_path / "angular"
    source.mkdir()
    output.mkdir()
    old_rows, new_rows, distributions, deltas = [], [], [], []
    for seed in range(3):
        for method in METHOD_NAMES.values():
            for event in CHECKPOINTS:
                for category in CATEGORIES:
                    context = {"seed": seed, "method": method, "checkpoint": event, "category": category}
                    raw = .2 + (.01 if method != "FROZEN" and event else 0)
                    old_rows.append({**context, "image_AUROC": raw, "pixel_AUPR": raw})
                    new_rows.append({**context, "image_AUROC": .8, "pixel_AUPR": .8,
                                     "image_AP": .8, "pixel_AUROC": .8, "persistent_bytes": 64})
                    deltas.append({**context, "delta_image_AUROC_vs_frozen": 0., "delta_pixel_AUPR_vs_frozen": 0.})
                    distributions.append({**context, "normal_pixel_mean": 1., "anomaly_score_mean": 1.,
                                          "defect_score_mean": 1., "anomaly_background_score_mean": 1.})
                    for folder, value in ((source, raw), (output, .8)):
                        stem = folder / f"seed{seed}/units/{method}_event{event}_{category}"
                        stem.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(stem.with_suffix(".npz"), image_ids=np.array(["n", "a"]),
                                 bootstrap_image_AUROC=np.full(400, value), bootstrap_pixel_AUPR=np.full(400, value))
                        if folder == output:
                            write_json(stem.with_suffix(".json"), {"scoring_seconds": 0., "metrics_seconds": 0.})
    pd.DataFrame(old_rows).to_parquet(source / "metrics_by_checkpoint.parquet")
    pd.DataFrame(new_rows).to_parquet(output / "metrics_by_checkpoint.parquet")
    pd.DataFrame(distributions).to_parquet(output / "score_distributions.parquet")
    pd.DataFrame(deltas).to_parquet(output / "method_deltas.parquet")
    pd.DataFrame({"dummy": [0]}).to_parquet(output / "score_decomposition.parquet")
    analyze_readout(source, output)
    result = pd.read_parquet(output / "readout_comparison.parquet")
    assert result[result.effect == "MEMORY_NEW_SCORE"].delta.eq(0).all()
    assert result[result.effect == "READOUT_AT_FIXED_STATE"].delta.gt(.5).all()
    mutable = result[(result.effect == "MEMORY_RAW") & (result.method != "FROZEN") & (result.checkpoint > 0)]
    np.testing.assert_allclose(mutable.delta, .01, rtol=0, atol=1e-12)
    assert read_json(output / "storage.json")["new_memory_state_bytes"] == 0


def test_spatial_affinity_matches_independent_coordinate_neighbor_loop():
    generator = torch.Generator().manual_seed(7)
    x, m = (torch.randn(9, 4, generator=generator, dtype=torch.float64) for _ in range(2))
    result = affinity_components(x, m, (3, 3))
    expected = []
    for index in range(9):
        row, col = divmod(index, 3)
        values = []
        for other in range(9):
            y, z = divmod(other, 3)
            if other != index and max(abs(y - row), abs(z - col)) == 1:
                cx = torch.dot(x[index], x[other]) / (x[index].norm() * x[other].norm())
                cm = torch.dot(m[index], m[other]) / (m[index].norm() * m[other].norm())
                values.append((cx - cm).square())
        expected.append(torch.stack(values).mean())
    torch.testing.assert_close(result["affinity"], torch.stack(expected), atol=1e-14, rtol=1e-14)
    assert result["neighbor_count"].tolist() == [3, 5, 3, 5, 8, 5, 3, 5, 3]


def test_equal_spatial_geometry_has_zero_score_and_positive_scaling_is_invariant():
    x = torch.arange(36).float().reshape(9, 4) + 1
    result = affinity_components(x, 5 * x, (3, 3))
    torch.testing.assert_close(result["affinity"], torch.zeros(9), atol=1e-12, rtol=0)
    m = x.flip(1) * torch.tensor([1., 2., 3., 4.])
    first = affinity_components(x, m, (3, 3))["affinity"]
    second = affinity_components(7 * x, .3 * m, (3, 3))["affinity"]
    torch.testing.assert_close(first, second, rtol=2e-5, atol=1e-8)


def test_spatial_zero_rows_are_finite_and_invalid_geometry_is_rejected():
    x = torch.zeros(9, 4)
    result = affinity_components(x, torch.ones_like(x), (3, 3))
    assert all(torch.isfinite(value).all() for value in result.values())
    with pytest.raises(ValueError):
        affinity_components(x, x, (2, 4))
    x[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        affinity_components(x, x, (3, 3))


def test_spatial_scoring_is_read_only_uses_query_readout_and_has_reset_effect():
    model, initial = memory(), memory()
    image = torch.arange(36).float().reshape(1, 9, 4) / 10 + .1
    model.commit_event(model.propose_event(image))
    before, initial_before = fingerprint(model.smt.state_dict()), initial.state_fingerprint()
    rng = torch.get_rng_state().clone()
    model.commit_event = lambda *_: pytest.fail("unexpected write")
    snapshot = model.snapshot_state()
    quantities = model.generate_update_quantities(image, snapshot)
    readout = model.read_from_snapshot(quantities, snapshot)
    expected = affinity_components(image[0], readout, (3, 3))["affinity"]
    result = affinity_scores(model, image, initial_model=initial, capture_components=True, grid_shape=(3, 3))
    torch.testing.assert_close(result["patch_scores"], expected, atol=0, rtol=0)
    assert result["image_score"] == float(expected.max())
    assert result["memory_reset_relative_l2"] > 0
    assert result["patch_scores"].grad_fn is None
    result["patch_scores"].zero_()
    result["patch_components"]["norm_m"].fill(0)
    assert fingerprint(model.smt.state_dict()) == before
    assert initial.state_fingerprint() == initial_before
    assert torch.equal(rng, torch.get_rng_state())


def test_spatial_identical_initial_state_has_zero_reset_effect():
    model, initial = memory(), memory()
    image = torch.arange(36).float().reshape(1, 9, 4) / 10 + .1
    result = affinity_scores(model, image, initial_model=initial, grid_shape=(3, 3))
    assert result["memory_reset_relative_l2"] == 0
    assert result["history_affinity_relative_l2"] == 0


def test_parallel_score_progress_cannot_collide_between_seeds(tmp_path, monkeypatch):
    from scripts.exps import hope_anomaly_score_ablation as runner
    original_progress = runner.original.progress

    def fake_seed(source, output, device, score, seed):
        runner.original.progress(output, "evaluated", seed=seed)
        return seed

    monkeypatch.setattr(runner, "score_seed", fake_seed)
    for seed in (0, 1):
        assert runner._score_worker((tmp_path, tmp_path, "cpu", "AFFINITY", seed)) == seed
        assert runner.original.progress is original_progress
        assert read_json(tmp_path / f"seed{seed}/progress.json")["seed"] == seed
    assert not (tmp_path / "progress.json").exists()
