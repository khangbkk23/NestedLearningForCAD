# exps/npim_cad/test_resource_ledger.py
"""G9 oracle: exact persistent detector byte accounting and immutable persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from models.hope_cad.npim_cad import NPIMConfig, NPIMDetector
from models.hope_cad.npim_state import (
    deterministic_state_hash,
    ledger,
    load_state_file,
    persistent_tensor_bytes,
    save_state_atomic,
    validate_finite_state,
)


def make_detector(r: int = 64, levels: int = 3) -> NPIMDetector:
    return NPIMDetector(
        NPIMConfig(
            r=r,
            periods=tuple(1 if i == 0 else 16 * i for i in range(levels)),
            lambdas=tuple(0.1 * (10**i) for i in range(levels)),
            weights=tuple(1.0 / levels for _ in range(levels)),
        ),
        device="cpu",
    )


def expected_tensor_bytes(r: int, levels: int, d: int = 768) -> int:
    projection = r * d
    weights = levels * d * r
    skk = levels * r * r
    syk = levels * d * r
    variance = d
    return 4 * (projection + weights + skk + syk + variance)


def test_analytic_byte_ledger_matches_implementation():
    detector = make_detector(r=64, levels=3)
    report = detector.ledger()
    assert report["tensor_subtotal_bytes"] == expected_tensor_bytes(64, 3)
    assert report["tensor_subtotal_bytes"] == 1_428_480
    assert report["metadata_bytes"] > 0
    assert report["total_persistent_detector_bytes"] == (
        report["tensor_subtotal_bytes"] + report["metadata_bytes"]
    )
    assert report["tensor_bytes_by_group"]["projection"] == 196_608
    assert report["tensor_bytes_by_group"]["levels"] == 589_824 + 49_152 + 589_824
    assert report["serialized_state_bytes"] > report["tensor_subtotal_bytes"]


def test_ledger_is_independent_of_stream_length():
    detector = make_detector()
    generator = torch.Generator().manual_seed(0)
    K = torch.randn(784, 64, generator=generator)
    Y = torch.randn(784, 768, generator=generator)
    first = detector.ledger()
    for _ in range(30):
        detector.commit_keys(K, Y)
    second = detector.ledger()
    assert first["tensor_bytes_by_path"] == second["tensor_bytes_by_path"]
    assert first["tensor_subtotal_bytes"] == second["tensor_subtotal_bytes"]


def test_ledger_does_not_mutate_state():
    detector = make_detector()
    before = detector.state_hash()
    _ = detector.ledger()
    assert detector.state_hash() == before


def test_state_hash_changes_after_commit_and_reload_is_exact(tmp_path: Path):
    detector = make_detector()
    empty_hash = detector.state_hash()
    generator = torch.Generator().manual_seed(1)
    K = torch.randn(784, 64, generator=generator)
    Y = torch.randn(784, 768, generator=generator)
    detector.commit_keys(K, Y)
    assert detector.state_hash() != empty_hash
    path = tmp_path / "state.pt"
    record = detector.save(str(path))
    assert record["bytes"] == path.stat().st_size
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    fresh = make_detector()
    fresh.load_state_dict(load_state_file(path))
    assert fresh.state_hash() == detector.state_hash()
    assert deterministic_state_hash(fresh.state_dict()) == fresh.state_hash()


def test_immutable_save_refuses_overwrite(tmp_path: Path):
    detector = make_detector()
    path = tmp_path / "state.pt"
    save_state_atomic(detector.state_dict(), path)
    with pytest.raises(FileExistsError):
        save_state_atomic(detector.state_dict(), path)


def test_finite_state_validation_rejects_nan():
    with pytest.raises(ValueError):
        validate_finite_state({"bad": torch.tensor([float("nan")])})


def test_byte_matched_single_level_control_is_close_to_three_level_hypothesis():
    single = make_detector(r=145, levels=1).ledger()["tensor_subtotal_bytes"]
    triple = make_detector(r=64, levels=3).ledger()["tensor_subtotal_bytes"]
    relative = abs(single - triple) / triple
    assert relative <= 0.01, (single, triple, relative)
    assert single <= triple  # conservative: the control is not given extra bytes


def test_tensor_byte_helper_agrees_with_ledger():
    detector = make_detector()
    assert persistent_tensor_bytes(detector.state_dict()) == detector.ledger()["tensor_subtotal_bytes"]
