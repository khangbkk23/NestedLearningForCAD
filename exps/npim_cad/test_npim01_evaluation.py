# exps/npim_cad/test_npim01_evaluation.py
"""G1/G2 evaluation-mechanics tests: read-only scoring, chunking, CUDA, metrics, seal, masks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from exps.npim_cad import npim01_evaluate as evaluate_mod
from exps.npim_cad import npim01_metrics as metrics_lib
from exps.npim_cad.npim01_contract import DevRecord, PILOT, arm_config, sha256_file
from models.hope_cad.npim_cad import NPIMDetector
from models.hope_cad.npim_state import load_state_file, rng_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def load_arm(arm: str = "C3", boundary: int = 350) -> NPIMDetector:
    detector = NPIMDetector(arm_config(arm), device="cpu")
    detector.load_state_dict(load_state_file(PILOT / "states" / f"{arm}_{boundary}.pt"))
    return detector


def test_scoring_is_read_only_and_chunk_equivalent():
    detector = load_arm("C3", 350)
    generator = torch.Generator().manual_seed(0)
    K = torch.randn(784, 64, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(784, 768, generator=generator)
    before_state = detector.state_hash()
    before_rng = rng_fingerprint()
    full = detector.score_keys(K, Y)["patch_scores"]
    chunks = [
        detector.score_keys(K[start : start + 97], Y[start : start + 97])["patch_scores"]
        for start in range(0, 784, 97)
    ]
    chunked = torch.cat(chunks)
    assert torch.allclose(full, chunked, atol=1e-5, rtol=1e-5)
    assert detector.state_hash() == before_state
    assert rng_fingerprint() == before_rng


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_encoder_and_scorer_cpu_cuda_agreement():
    from models.hope_cad.npim_context import NPIMContextConfig, NPIMFeatureExtractor

    cfg = NPIMContextConfig(
        checkpoint_path=str(ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"),
        checkpoint_sha256="755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415",
    )
    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 224)
    cpu_extractor = NPIMFeatureExtractor(cfg, device="cpu")
    cuda_extractor = NPIMFeatureExtractor(cfg, device="cuda")
    try:
        z_cpu, y_cpu = cpu_extractor.encode(image)
        torch.cuda.reset_peak_memory_stats()
        z_gpu, y_gpu = cuda_extractor.encode(image.to("cuda"))
        assert torch.allclose(z_cpu, z_gpu.cpu(), atol=3e-4, rtol=3e-4)
        assert torch.allclose(y_cpu, y_gpu.cpu(), atol=3e-4, rtol=3e-4)
        assert int(torch.cuda.max_memory_allocated()) < int(4.5 * 1024**3)

        detector = load_arm("C3", 350)
        cpu_score = detector.score_keys(detector.build_keys(z_cpu[0]), y_cpu[0])["patch_scores"]
        gpu_detector = NPIMDetector(arm_config("C3"), device="cuda")
        gpu_detector.load_state_dict(load_state_file(PILOT / "states" / "C3_350.pt"))
        gpu_score = gpu_detector.score_keys(
            gpu_detector.build_keys(z_gpu[0]), y_gpu[0]
        )["patch_scores"].cpu()
        assert torch.allclose(cpu_score, gpu_score, atol=1e-4, rtol=1e-4)
    finally:
        cpu_extractor.close()
        cuda_extractor.close()


def brute_force_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    positives = labels.sum()
    true_positive = false_positive = 0
    total = 0.0
    index = 0
    while index < len(sorted_scores):
        end = index
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group_positive = int(sorted_labels[index:end].sum())
        true_positive += group_positive
        false_positive += (end - index) - group_positive
        if group_positive:
            total += (group_positive / positives) * (true_positive / (true_positive + false_positive))
        index = end
    return total


def test_native_metric_parity_and_tie_handling():
    rng = np.random.default_rng(0)
    maps = [rng.integers(0, 6, size=(16, 20)).astype(np.float64) for _ in range(4)]
    masks = [rng.random((16, 20)) < 0.1 for _ in range(4)]
    masks[0][0, 0] = True
    masks[1][0, 0] = True
    masks[3][0, 0] = True
    labels = [0, 1, 0, 1]
    image_scores = [float(m.max()) for m in maps]
    metrics = metrics_lib.group_point_metrics(maps, masks, labels, image_scores)
    expected = brute_force_ap(
        np.concatenate([m.ravel() for m in maps]),
        np.concatenate([k.ravel() for k in masks]),
    )
    assert abs(metrics["pixel_ap"] - expected) < 1e-12
    assert metrics["positive_pixels"] == int(sum(k.sum() for k in masks))
    # Native map parity with the existing P1 geometry.
    from exps.task_p.p1_readout import native_map as p1_native_map

    patch = torch.rand(784, generator=torch.Generator().manual_seed(1))
    ours = metrics_lib.native_scores(patch.numpy(), (48, 64))
    reference = p1_native_map(patch, (48, 64))
    assert np.array_equal(ours, reference)


def test_seal_is_required_before_masks(tmp_path: Path):
    from types import SimpleNamespace

    contract = SimpleNamespace(
        cohort_digest="x",
        protocol_sha256="y",
        checkpoint_sha256="z",
        records=[],
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="seal is missing"):
        evaluate_mod._load_sealed_predictions(run_dir, contract)

    torch.save({"schema": evaluate_mod.PREDICTION_SCHEMA}, run_dir / "predictions.pt")
    (run_dir / "predictions_seal.json").write_text(
        json.dumps({"schema": evaluate_mod.PREDICTION_SCHEMA, "sha256": "0" * 64})
    )
    with pytest.raises(RuntimeError, match="seal mismatch"):
        evaluate_mod._load_sealed_predictions(run_dir, contract)


def test_paired_image_auroc_bootstrap_is_paired_and_finite():
    labels = [0, 0, 1, 1]
    scores = {
        "C1": [0.1, 0.2, 0.8, 0.9],
        "C2": [0.1, 0.3, 0.7, 0.8],
        "C3": [0.05, 0.25, 0.75, 0.95],
    }
    result = metrics_lib.paired_image_auroc_bootstrap(labels, scores, reps=50, seed=0)
    assert set(result["auroc"]) == {"C1", "C2", "C3"}
    for row in result["auroc"].values():
        assert np.isfinite(row["mean"]) and len(row["interval"]) == 2
    assert "C3-C2" in result and "C3-C1" in result
    assert np.isfinite(result["C3-C2"]["mean"])


def test_bounded_pixel_bootstrap_small_is_finite():
    rng = np.random.default_rng(1)
    maps = [rng.random((16, 16)) for _ in range(4)]
    masks = [rng.random((16, 16)) < 0.15 for _ in range(4)]
    masks[0][0, 0] = True
    masks[1][0, 0] = True
    masks[2][0, 0] = True
    masks[3][0, 0] = True
    result = metrics_lib._binned_bootstrap(
        maps, masks, [0, 0, 1, 1], reps=20, bins=256, seed=0
    )
    assert np.isfinite(result["lower"]) and np.isfinite(result["upper"])
    assert result["lower"] <= result["upper"]


def _fake_mask(path: Path, values: np.ndarray) -> str:
    Image.fromarray(values.astype(np.uint8)).save(path)
    return sha256_file(path)


def test_native_mask_rejects_missing_changed_empty_and_bad_values(tmp_path: Path):
    record = DevRecord(
        index=0, relative_path="synthetic/test/anomaly/000.png", sha256="0" * 64,
        category="bottle", label=1, hw=(16, 16), mask_path="mask.png", mask_sha256=None,
    )
    with pytest.raises(ValueError, match="lacks a pinned mask"):
        metrics_lib.load_native_mask(tmp_path, record)
    record = DevRecord(
        index=0, relative_path="synthetic/test/anomaly/000.png", sha256="0" * 64,
        category="bottle", label=1, hw=(16, 16), mask_path="mask.png", mask_sha256="0" * 64,
    )
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    _fake_mask(tmp_path / "mask.png", mask)
    with pytest.raises(ValueError, match="content changed"):
        metrics_lib.load_native_mask(tmp_path, record)
    digest = sha256_file(tmp_path / "mask.png")
    good = DevRecord(0, record.relative_path, record.sha256, "bottle", 1, (16, 16), "mask.png", digest)
    loaded = metrics_lib.load_native_mask(tmp_path, good)
    assert loaded.dtype == bool and loaded.sum() == 9
    empty = np.zeros((16, 16), dtype=np.uint8)
    digest = _fake_mask(tmp_path / "empty.png", empty)
    with pytest.raises(ValueError, match="empty"):
        metrics_lib.load_native_mask(tmp_path, DevRecord(0, record.relative_path, record.sha256, "bottle", 1, (16, 16), "empty.png", digest))
    invalid = np.zeros((16, 16), dtype=np.uint8)
    invalid[2:5, 2:5] = 7
    digest = _fake_mask(tmp_path / "invalid.png", invalid)
    with pytest.raises(ValueError, match="unresolved values"):
        metrics_lib.load_native_mask(tmp_path, DevRecord(0, record.relative_path, record.sha256, "bottle", 1, (16, 16), "invalid.png", digest))
    small = np.zeros((8, 8), dtype=np.uint8)
    small[1:2, 1:2] = 255
    digest = _fake_mask(tmp_path / "small.png", small)
    with pytest.raises(ValueError, match="geometry"):
        metrics_lib.load_native_mask(tmp_path, DevRecord(0, record.relative_path, record.sha256, "bottle", 1, (16, 16), "small.png", digest))


def test_metrics_default_deny_exits_before_reading_anything():
    result = subprocess.run(
        [sys.executable, "-m", "exps.npim_cad.npim01_evaluate", "metrics", "--run-id", "npim01_deny_metrics_pytest"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED_CONSUMED_DEV_NOT_AUTHORIZED" in result.stdout
