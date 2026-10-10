# exps/test_hope_anomaly_signal.py
"""Independent score, split, statistical-memory and uncertainty checks."""

from copy import deepcopy

import numpy as np
from PIL import Image
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from exps.hope_anomaly_signal import (
    CATEGORIES, PooledCovariance, bootstrap_image_counts, development_manifests,
    evaluate_metrics, histogram_ap, native_mask, pixel_map, residual_scores,
)
from exps.hope_image_synchronous_memory import ImageSynchronousMemory, fingerprint
from models.hope_cad.self_modifying_titans import SelfModifyingTitans
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1
from scripts.exps.hope_anomaly_signal import validate_dev_cache, manifest_identity, write_json, read_json, save_table, table_records


def memory():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        initial = SelfModifyingTitans(4).state_dict()
    return ImageSynchronousMemory(initial, "P1")


def test_snapshot_residual_uses_current_values_not_self_target_or_query():
    model = memory()
    features = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4) / 8
    state = fingerprint(model.smt.state_dict())
    rng = torch.get_rng_state().clone()
    model.commit_event = lambda *_: pytest.fail("evaluation attempted a write")
    snapshot = model.snapshot_state()
    q = model.generate_update_quantities(features, snapshot)
    prediction = F.linear(q.keys, snapshot.weights["memory"])
    expected = (prediction - q.values).square().sum(-1)
    wrong_target = (prediction - F.linear(q.values, snapshot.weights["memory"])).square().sum(-1)
    wrong_query = (F.linear(q.queries, snapshot.weights["memory"]) - q.values).square().sum(-1)
    actual = residual_scores(model, features)
    torch.testing.assert_close(actual["patch_scores"], expected, rtol=0, atol=0)
    assert not torch.allclose(expected, wrong_target)
    assert not torch.allclose(expected, wrong_query)
    assert actual["image_score"] == float(expected.max())
    assert fingerprint(model.smt.state_dict()) == state
    assert torch.equal(rng, torch.get_rng_state())
    actual["patch_scores"].zero_()
    assert fingerprint(model.smt.state_dict()) == state


def test_pixel_map_exact_bilinear_no_normalization():
    patches = torch.arange(784, dtype=torch.float32)
    actual = pixel_map(patches, (49, 67))
    expected = F.interpolate(patches.reshape(1, 1, 28, 28), size=(49, 67), mode="bilinear", align_corners=False)
    np.testing.assert_array_equal(actual, expected[0, 0].numpy())
    assert actual.max() > 1
    with pytest.raises(ValueError):
        pixel_map(torch.zeros(7), (20, 20))


def test_manifests_round_robin_disjoint_and_fixed_before_scoring(tmp_path):
    for category in CATEGORIES:
        for defect in ("good", "a", "b"):
            folder = tmp_path / category / "test" / defect
            folder.mkdir(parents=True)
            for index in range(12):
                Image.new("RGB", (8, 9), color=(index, 2, 3)).save(folder / f"{index:03}.png")
                if defect != "good":
                    gt = tmp_path / category / "ground_truth" / defect
                    gt.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (8, 9), color=255).save(gt / f"{index:03}_mask.png")
    dev, confirmation = development_manifests(tmp_path, limit=4)
    assert len(dev) == len(confirmation) == 24
    assert not {r["image_id"] for r in dev} & {r["image_id"] for r in confirmation}
    assert development_manifests(tmp_path, limit=4) == (dev, confirmation)
    anomaly = [r for r in dev if r["category"] == "bottle" and r["label"]]
    assert [r["defect_type"] for r in anomaly] == ["a", "b", "a", "b"]
    assert native_mask(tmp_path, anomaly[0]).shape == (9, 8)
    path = tmp_path / "manifest.parquet"
    save_table(path, dev)
    assert table_records(path) == dev


def test_cache_contract_rejects_wrong_layer_checkpoint_order_and_dtype():
    rows = [{"relative_path": "bottle/test/good/001.png"}]
    metadata = {"block_index": 8, "checkpoint_sha256": "fixed", "preprocessing": "fixed"}
    cache = {"patches": torch.zeros(1, 784, 768), "relative_paths": [rows[0]["relative_path"]],
             "manifest_identity": manifest_identity(rows), "metadata": metadata}
    validate_dev_cache(cache, rows, metadata)
    for key in metadata:
        wrong = deepcopy(cache)
        wrong["metadata"][key] = "wrong"
        with pytest.raises(ValueError):
            validate_dev_cache(wrong, rows, metadata)
    wrong = deepcopy(cache)
    wrong["relative_paths"] = ["other"]
    with pytest.raises(ValueError):
        validate_dev_cache(wrong, rows, metadata)
    wrong = deepcopy(cache)
    wrong["patches"] = wrong["patches"].double()
    with pytest.raises(ValueError):
        validate_dev_cache(wrong, rows, metadata)


def test_covariance_matches_independent_pooled_and_shrinkage_reference():
    generator = torch.Generator().manual_seed(4)
    training = F.normalize(torch.randn(17, 4, generator=generator, dtype=torch.float64), dim=-1)
    model = PooledCovariance(4)
    model.update(training[:3])
    model.update(training[3:10])
    model.update(training[10:])
    torch.testing.assert_close(model.mean, training.mean(0), rtol=1e-12, atol=1e-12)
    covariance = torch.cov(training.T)
    identity = torch.eye(4, dtype=torch.float64)
    expected = .9 * covariance + (.1 * covariance.trace() / 4 + 1e-6) * identity
    model.factorize()
    torch.testing.assert_close(model.factor @ model.factor.T, expected, rtol=1e-12, atol=1e-12)
    queries = training[:6]
    centered = queries - training.mean(0)
    distances = (centered * torch.linalg.solve(expected, centered.T).T).sum(-1)
    before = fingerprint(model.state_dict())
    torch.testing.assert_close(model.score(queries), distances, rtol=1e-12, atol=1e-12)
    assert fingerprint(model.state_dict()) == before
    assert model.storage()["persistent_bytes"] == 16 + 4 * 8 + 2 * 4 * 4 * 8
    assert model.storage()["factorization_cache_bytes"] == 4 * 4 * 8
    restored = PooledCovariance(4)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.score(queries), distances, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError):
        PooledCovariance(4).score(queries)
    assert all(t.grad_fn is None for t in model.state_dict().values())


def test_bootstrap_resamples_paired_whole_images_and_both_strata():
    labels = np.array([0, 0, 1, 1])
    counts = bootstrap_image_counts(labels, repetitions=20, seed=7)
    np.testing.assert_array_equal(counts, bootstrap_image_counts(labels, repetitions=20, seed=7))
    assert np.all(counts[:, :2].sum(1) == 2)
    assert np.all(counts[:, 2:].sum(1) == 2)
    assert np.all(counts.sum(1) == len(labels))
    assert np.unique(counts, axis=0).shape[0] > 1


def test_exact_pixel_points_and_image_block_intervals():
    labels = np.array([0, 0, 1, 1])
    scores = [0.1, .3, .5, .4]
    maps = [np.array([[.1, .1], [.2, .2]]), np.array([[.3, .1], [.1, .1]]),
            np.array([[.5, .1], [.5, .2]]), np.array([[.2, .4], [.1, .4]])]
    masks = [np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool),
             np.array([[True, False], [True, False]]), np.array([[False, True], [False, True]])]
    counts = bootstrap_image_counts(labels, repetitions=12, seed=8)
    result = evaluate_metrics(scores, labels, maps, masks, counts)
    target = np.concatenate([gt.ravel() for gt in masks])
    values = np.concatenate([x.ravel() for x in maps]).astype(np.float32)
    assert result.metrics["pixel_AUPR"] == average_precision_score(target, values)
    assert result.metrics["image_AUROC"] == roc_auc_score(labels, scores)
    assert result.metrics["pixel_bootstrap_point_error"] <= .0002
    repeated = evaluate_metrics(scores, labels, maps, masks, counts)
    np.testing.assert_array_equal(result.bootstrap_pixel_aupr, repeated.bootstrap_pixel_aupr)
    np.testing.assert_array_equal(result.bootstrap_image_auroc - repeated.bootstrap_image_auroc, np.zeros(12))
    # One high-score tied block contains one positive and one negative.
    assert histogram_ap(np.array([0, 1]), np.array([2, 1]))[0] == .5


def test_cadic_native_score_is_reused_and_memory_is_read_only():
    config = CADICPatchCoresetConfig(budget=3, dim=2, image_neighbors=2, chunk_size=2,
                                    query_chunk_size=2, pair_chunk_size=2)
    model = CADICPatchCoresetV1(config)
    model.update(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
    before = fingerprint(model.state_dict())
    queries = torch.tensor([[[.2, .2], [.8, .2]]])
    image, pixels = model.score(queries)
    torch.testing.assert_close(pixels[0], torch.cdist(queries[0], model.features).min(1).values)
    assert torch.isfinite(image).all()
    assert fingerprint(model.state_dict()) == before


def test_artifact_round_trip_and_failure_is_not_success(tmp_path):
    path = tmp_path / "summary.json"
    payload = {"status": "FAILED", "ANOMALY_SIGNAL_GATE_COMPLETE": "NO", "failure_reason": "nonfinite"}
    write_json(path, payload)
    assert read_json(path) == payload
    assert not path.with_suffix(".json.tmp").exists()
