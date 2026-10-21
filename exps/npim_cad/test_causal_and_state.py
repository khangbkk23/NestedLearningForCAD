# exps/npim_cad/test_causal_and_state.py
"""G4/G6/G7 oracles: image clocks, atomicity, deterministic resume, read-only scoring."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from models.hope_cad.npim_cad import NPIMConfig, NPIMDetector, UncalibratedDetectorError
from models.hope_cad.npim_state import load_state_file, rng_fingerprint

ROOT = Path(__file__).resolve().parents[2]


def make_detector(r: int = 4, use_diag: bool = False) -> NPIMDetector:
    return NPIMDetector(
        NPIMConfig(
            r=r,
            periods=(1, 16, 128),
            lambdas=(0.1, 1.0, 10.0),
            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            use_diag=use_diag,
        ),
        device="cpu",
    )


def pairs(seed: int = 0, r: int = 4):
    generator = torch.Generator().manual_seed(seed)
    K = torch.randn(784, r, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(784, 768, generator=generator)
    return K, Y


def stream(count: int, r: int = 4):
    generator = torch.Generator().manual_seed(1234)
    out = []
    for _ in range(count):
        K = torch.randn(784, r, generator=generator)
        K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
        Y = torch.randn(784, 768, generator=generator)
        out.append((K, Y))
    return out


def advance(detector: NPIMDetector, count: int, K: torch.Tensor, Y: torch.Tensor) -> None:
    for _ in range(count):
        detector.commit_keys(K, Y)


def summary_at(detector: NPIMDetector) -> dict:
    return {
        "ready": [level.ready for level in detector.levels],
        "updates": [level.update_count for level in detector.levels],
        "pending": [level.pending_images for level in detector.levels],
        "pairs": [level.pair_count for level in detector.levels],
    }


def test_exact_clock_schedule_and_ready_flags():
    detector = make_detector()
    K, Y = pairs()
    observed = {}
    for event in range(1, 351):
        detector.commit_keys(K, Y)
        if event in {1, 15, 16, 100, 128, 200, 256, 300, 350}:
            observed[event] = summary_at(detector)

    assert observed[1]["ready"] == [True, False, False]
    assert observed[1]["updates"] == [1, 0, 0]
    assert observed[15]["updates"] == [15, 0, 0]
    assert observed[16]["ready"] == [True, True, False]
    assert observed[16]["updates"] == [16, 1, 0]

    assert observed[100]["ready"] == [True, True, False]
    assert observed[100]["updates"] == [100, 6, 0]
    assert observed[100]["pairs"] == [0, 4 * 784, 100 * 784]
    assert observed[100]["pending"] == [0, 4, 100]

    assert observed[128]["ready"] == [True, True, True]
    assert observed[128]["updates"] == [128, 8, 1]
    assert observed[128]["pairs"] == [0, 0, 0]
    assert observed[128]["pending"] == [0, 0, 0]

    assert observed[200]["updates"] == [200, 12, 1]
    assert observed[200]["pending"] == [0, 8, 72]
    assert observed[256]["updates"] == [256, 16, 2]
    assert observed[256]["pending"] == [0, 0, 0]
    assert observed[300]["updates"] == [300, 18, 2]
    assert observed[300]["pending"] == [0, 12, 44]
    assert observed[350]["updates"] == [350, 21, 2]
    assert observed[350]["pending"] == [0, 14, 94]
    assert observed[350]["pairs"] == [0, 14 * 784, 94 * 784]


def test_slow_statistics_accumulate_before_first_commit():
    detector = make_detector()
    K, Y = pairs(seed=5)
    advance(detector, 100, K, Y)
    slow = detector.levels[2]
    assert slow.ready is False
    assert slow.update_count == 0
    assert slow.pair_count == 100 * 784
    assert torch.count_nonzero(slow.W).item() == 0
    assert torch.isfinite(slow.Skk).all() and slow.Skk.abs().sum().item() > 0.0
    assert torch.isfinite(slow.Syk).all() and slow.Syk.abs().sum().item() > 0.0


def test_no_flush_at_boundaries_or_stream_end():
    detector = make_detector()
    K, Y = pairs(seed=6)
    advance(detector, 350, K, Y)
    assert detector.completed_events == 350
    assert [level.update_count for level in detector.levels] == [350, 21, 2]
    assert [level.pending_images for level in detector.levels] == [0, 14, 94]


def test_atomic_rollback_on_candidate_exception():
    detector = make_detector()
    K, Y = pairs(seed=7)
    advance(detector, 5, K, Y)
    before_hash = detector.state_hash()
    before_summary = summary_at(detector)

    def boom(*_args, **_kwargs):
        raise RuntimeError("synthetic boundary failure")

    detector.levels[0].candidate_on_boundary = boom
    with pytest.raises(RuntimeError):
        detector.commit_keys(K, Y)
    assert detector.state_hash() == before_hash
    assert summary_at(detector) == before_summary
    assert detector.completed_events == 5


def test_atomic_rollback_on_nonfinite_candidate():
    detector = make_detector()
    K, Y = pairs(seed=8)
    advance(detector, 5, K, Y)
    before_hash = detector.state_hash()
    detector.levels[0].candidate_on_boundary = lambda *_a, **_k: torch.full(
        (768, 4), float("nan")
    )
    with pytest.raises(ValueError):
        detector.commit_keys(K, Y)
    assert detector.state_hash() == before_hash
    assert detector.completed_events == 5


def test_scoring_is_read_only_including_rng():
    detector = make_detector()
    K, Y = pairs(seed=9)
    advance(detector, 20, K, Y)
    before_hash = detector.state_hash()
    before_rng = rng_fingerprint()
    Z = torch.randn(784, 768, generator=torch.Generator().manual_seed(10))
    for _ in range(5):
        out = detector.score(Z, Y)
        assert out["score_type"] == "l2"
    assert detector.state_hash() == before_hash
    assert rng_fingerprint() == before_rng


def test_save_load_resume_is_deterministic(tmp_path: Path):
    items = stream(130)
    detector = make_detector()
    for index, (K, Y) in enumerate(items[:120]):
        detector.commit_keys(K, Y, {"event": index + 1})
    state_path = tmp_path / "state_120.pt"
    detector.save(str(state_path))
    reloaded = make_detector()
    reloaded.load_state_dict(load_state_file(state_path))
    assert reloaded.state_hash() == detector.state_hash()
    for index, (K, Y) in enumerate(items[120:], start=121):
        detector.commit_keys(K, Y, {"event": index})
        reloaded.commit_keys(K, Y, {"event": index})
    assert reloaded.state_hash() == detector.state_hash()
    assert summary_at(reloaded) == summary_at(detector)


def test_transport_batching_preserves_event_commits():
    items = stream(12)
    sequential = make_detector()
    for K, Y in items:
        sequential.commit_keys(K, Y)
    batched = make_detector()
    for start in range(0, len(items), 4):
        batch = items[start : start + 4]
        for K, Y in batch:
            batched.commit_keys(K, Y)
    assert sequential.state_hash() == batched.state_hash()


def test_state_ledger_is_constant_in_stream_length():
    detector = make_detector()
    K, Y = pairs(seed=11)
    advance(detector, 5, K, Y)
    first = detector.ledger()
    advance(detector, 45, K, Y)
    second = detector.ledger()
    assert first["tensor_subtotal_bytes"] == second["tensor_subtotal_bytes"]
    assert first["tensor_bytes_by_path"] == second["tensor_bytes_by_path"]


def test_uncalibrated_detector_is_explicit():
    detector = make_detector()
    K, Y = pairs(seed=12)
    with pytest.raises(UncalibratedDetectorError):
        detector.score_keys(K, Y)


def test_mixture_uses_ready_levels_only_with_normalised_weights():
    detector = make_detector()
    K, Y = pairs(seed=13)
    detector.commit_keys(K, Y)
    mixture = detector._mixture()
    assert len(mixture) == 1
    assert abs(mixture[0][1] - 1.0) < 1e-12
    prediction = detector.predict(K)
    assert torch.allclose(prediction, K @ detector.levels[0].W.T, atol=0.0)
    advance(detector, 15, K, Y)
    mixture = detector._mixture()
    assert len(mixture) == 2
    assert abs(sum(weight for _, weight in mixture) - 1.0) < 1e-12
    assert all(abs(weight - 0.5) < 1e-12 for _, weight in mixture)
    advance(detector, 112, K, Y)
    mixture = detector._mixture()
    assert len(mixture) == 3
    assert all(abs(weight - 1.0 / 3.0) < 1e-12 for _, weight in mixture)


def test_diagonal_readout_is_disabled_by_default_and_gated_when_enabled():
    detector = make_detector()
    K, Y = pairs(seed=14)
    assert detector.config.use_diag is False
    advance(detector, 2, K, Y)
    assert detector.score_keys(K, Y)["score_type"] == "l2"

    gated = make_detector(use_diag=True)
    with pytest.raises(UncalibratedDetectorError):
        gated.score_keys(K, Y)
    advance(gated, 3, K, Y)
    assert gated.var_count == 2
    assert gated.score_keys(K, Y)["score_type"] == "diag"


def test_no_label_category_or_task_argument_reaches_the_learner():
    detector = make_detector()
    for method in (
        detector.commit_keys,
        detector.commit_normal_image,
        detector.score,
        detector.score_keys,
        detector.build_keys,
        detector.predict,
    ):
        names = list(inspect.signature(method).parameters)
        lowered = " ".join(names).lower()
        for forbidden in ("label", "category", "task", "mask", "class"):
            assert forbidden not in lowered, (method.__name__, names)
    state = detector.state_dict()
    assert "category" not in repr(state.keys())
    assert "label" not in repr(state.keys())


def test_stage_e_gate_refuses_without_authorization():
    result = subprocess.run(
        [sys.executable, str(ROOT / "exps/npim_cad/evaluate_dev.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED_STAGE_E_NOT_AUTHORIZED" in result.stdout
    assert '"opened_anomaly_data": false' in result.stdout
