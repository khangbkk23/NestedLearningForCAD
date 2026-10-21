# exps/npim_cad/test_npim01_contract.py
"""G1 contract tests: 12 pinned states, 60-image whitelist, default-deny, no mask reads."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exps.npim_cad import npim01_contract as contract_mod
from exps.npim_cad import npim01_evaluate as evaluate_mod
from exps.npim_cad.npim01_contract import (
    BOUNDARIES,
    COHORT_DIGEST,
    P1_PROTOCOL,
    PILOT,
    PROTOCOL,
    STATE_HASH,
    STATE_SHA256,
    ContractError,
    admitted_categories,
    load_contract,
    load_state_detectors,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]


def test_contract_pins_and_cohort():
    contract = load_contract()
    assert len(contract.records) == 60
    assert contract.protocol_sha256 == sha256_file(PROTOCOL)
    assert contract.cohort_digest == COHORT_DIGEST
    assert contract.writer_digest == "9549f24aae822ea8cccdbe63dd9731790633cf5eba80b37506c11c2944df3c24"
    assert sum(1 for record in contract.records if record.label == 0) == 30
    assert sum(1 for record in contract.records if record.label == 1) == 30
    for category in ("bottle", "carpet", "hazelnut"):
        rows = [record for record in contract.records if record.category == category]
        assert len(rows) == 20
        assert sum(1 for record in rows if record.label == 0) == 10
        assert sum(1 for record in rows if record.label == 1) == 10
    assert all(record.relative_path.startswith(record.category + "/test/") for record in contract.records)
    assert contract.source_sha256


def test_authoritative_writer_order_is_100_100_100_50():
    p1 = json.loads(P1_PROTOCOL.read_text())
    runs, previous = [], None
    for row in p1["updates"]:
        if row["category"] != previous:
            runs.append([row["event"], row["category"]])
            previous = row["category"]
    assert runs == [[1, "bottle"], [101, "carpet"], [201, "hazelnut"], [301, "bottle"]]
    counts = {category: sum(1 for row in p1["updates"] if row["category"] == category)
              for category in ("bottle", "carpet", "hazelnut")}
    assert counts == {"bottle": 150, "carpet": 100, "hazelnut": 100}


def test_all_12_states_restore_with_expected_ready_flags():
    detectors = load_state_detectors("cpu")
    assert len(detectors) == 12
    for (arm, boundary), detector in detectors.items():
        assert detector.state_hash() == STATE_HASH[(arm, boundary)]
    assert [level.ready for level in detectors[("C3", 100)].levels] == [True, True, False]
    assert [level.ready for level in detectors[("C3", 200)].levels] == [True, True, True]
    assert detectors[("C3", 100)].levels[2].pair_count == 100 * 784
    assert detectors[("C3", 350)].levels[2].update_count == 2
    assert detectors[("C3", 350)].levels[2].pending_images == 94


def test_contract_does_not_hash_or_open_annotation_masks(monkeypatch):
    original = contract_mod.sha256_file

    def guarded(path):
        if "ground_truth" in str(path):
            raise AssertionError(f"contract attempted to hash a mask: {path}")
        return original(path)

    monkeypatch.setattr(contract_mod, "sha256_file", guarded)
    contract = load_contract()
    assert len(contract.records) == 60


def test_tampered_state_pin_is_rejected(monkeypatch):
    monkeypatch.setitem(STATE_SHA256, ("C1", 100), "0" * 64)
    with pytest.raises(ContractError):
        load_contract()


def test_unknown_boundary_is_rejected():
    with pytest.raises(ContractError):
        admitted_categories(999)


def test_default_deny_exits_before_creating_a_run(monkeypatch, tmp_path):
    run_id = "npim01_default_deny_pytest"
    run_dir = evaluate_mod.RESULTS / run_id
    assert not run_dir.exists()
    result = subprocess.run(
        [sys.executable, "-m", "exps.npim_cad.npim01_evaluate", "predict", "--run-id", run_id],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "BLOCKED_CONSUMED_DEV_NOT_AUTHORIZED" in result.stdout
    assert '"opened_anomaly_data": false' in result.stdout
    assert not run_dir.exists()


def test_e1_source_has_no_mask_call_but_e2_does():
    predict_source = inspect.getsource(evaluate_mod.run_predict)
    metrics_source = inspect.getsource(evaluate_mod.run_metrics)
    assert "load_native_mask" not in predict_source
    assert "load_native_mask" in metrics_source
    assert metrics_source.index("_load_sealed_predictions") < metrics_source.index("load_native_mask")
