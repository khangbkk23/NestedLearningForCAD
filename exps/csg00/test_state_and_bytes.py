"""G1: state serialization, byte accounting, save/reload parity and guards."""

from __future__ import annotations

import numpy as np
import pytest

from exps.csg00 import contract as C


def _synthetic_b_state(rng) -> dict:
    state = {
        "ready": np.ones((C.BINS, C.K_MODES), dtype=np.uint8),
        "count": np.full((C.BINS, C.K_MODES), 32, dtype=np.int64),
        "n_total": np.full((C.BINS,), 1600, dtype=np.int64),
        "mu": rng.normal(size=(C.BINS, C.K_MODES, C.DIM)).astype(np.float32),
        "U": np.zeros((C.BINS, C.K_MODES, C.DIM, C.Q_TANGENT), dtype=np.float32),
        "lam": rng.uniform(1, 10, size=(C.BINS, C.K_MODES, C.Q_TANGENT)).astype(np.float32),
        "R": rng.uniform(1, 5, size=(C.BINS, C.K_MODES, C.Q_TANGENT)).astype(np.float32),
        "rho": rng.uniform(1, 5, size=(C.BINS, C.K_MODES)).astype(np.float32),
    }
    for b in range(C.BINS):
        for k in range(C.K_MODES):
            Q, _ = np.linalg.qr(rng.normal(size=(C.DIM, C.Q_TANGENT)))
            state["U"][b, k] = Q.astype(np.float32)
    return state


def _synthetic_c_state(rng) -> dict:
    return {
        "res": rng.normal(size=(C.BINS, C.RESERVOIR_CAPACITY, C.DIM)).astype(np.float32),
        "count": np.full((C.BINS,), C.RESERVOIR_CAPACITY, dtype=np.int64),
        "seen": np.full((C.BINS,), 5600, dtype=np.int64),
    }


def test_declared_byte_budget_and_actual_ratio(tmp_path):
    rng = np.random.default_rng(3)
    b = _synthetic_b_state(rng)
    c = _synthetic_c_state(rng)
    b_info = C.write_state(tmp_path / "b.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)
    c_info = C.write_state(tmp_path / "c.csg00", "C", 350, {"capacity": 10}, c, C.C_STATE_LAYOUT)
    assert b_info["payload_bytes"] == C.b_payload_bytes()
    assert c_info["payload_bytes"] == C.c_payload_bytes()
    ratio = c_info["bytes"] / b_info["bytes"]
    assert 0.95 <= ratio <= 1.05, ratio
    assert b_info["bytes"] < 1_600_000
    assert c_info["bytes"] < 1_600_000


def test_save_reload_parity_and_deterministic_bytes(tmp_path):
    rng = np.random.default_rng(5)
    b = _synthetic_b_state(rng)
    p1 = C.write_state(tmp_path / "b1.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)
    p2 = C.write_state(tmp_path / "b2.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)
    assert p1["sha256"] == p2["sha256"]
    header, reloaded = C.read_state(tmp_path / "b1.csg00")
    assert header["arm"] == "B" and header["boundary"] == 350
    for key in b:
        assert np.array_equal(reloaded[key], b[key]), key
        assert reloaded[key].dtype == b[key].dtype


def test_tamper_detection(tmp_path):
    rng = np.random.default_rng(7)
    b = _synthetic_b_state(rng)
    info = C.write_state(tmp_path / "b.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)
    raw = bytearray((tmp_path / "b.csg00").read_bytes())
    raw[-1] ^= 0xFF
    (tmp_path / "tampered.csg00").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="payload hash"):
        C.read_state(tmp_path / "tampered.csg00")
    assert info["sha256"] != C.sha256_file(tmp_path / "tampered.csg00")


def test_state_write_is_create_exclusive(tmp_path):
    rng = np.random.default_rng(9)
    b = _synthetic_b_state(rng)
    C.write_state(tmp_path / "b.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)
    with pytest.raises(FileExistsError):
        C.write_state(tmp_path / "b.csg00", "B", 350, {"k": 2}, b, C.B_STATE_LAYOUT)


def test_normal_only_guard_is_default_deny():
    whitelist = {"bottle/train/good/000.png"}
    C.assert_normal_only("bottle/train/good/000.png", whitelist)
    for bad in (
        "bottle/test/crack/000.png",
        "bottle/ground_truth/crack/000_mask.png",
        "carpet/train/good/000.png",
        "bottle/train/bad/000.png",
        "sealed/bottle/000.png",
    ):
        with pytest.raises(PermissionError):
            C.assert_normal_only(bad, whitelist)


def test_reservoir_capacity_is_schema_derived():
    info = C.reservoir_capacity_for_byte_match()
    assert info["chosen_capacity"] == C.RESERVOIR_CAPACITY
    assert abs(info["payload_ratio"] - 1.0) < 0.05


def test_splitmix_and_counter_uniform_are_deterministic():
    values = [C.counter_uniform(C.RESERVOIR_SEED, 3, i) for i in range(50)]
    assert values == [C.counter_uniform(C.RESERVOIR_SEED, 3, i) for i in range(50)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert len(set(values)) > 40
