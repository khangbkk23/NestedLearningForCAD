"""G1: G3 design fixtures — default deny, seal-before-mask and synthetic AP.

No real anomaly image, mask, label or sealed identity is opened.  The fixture
uses synthetic images in a temporary directory and a synthetic feature function;
the real cohort path guard is exercised against the known exposed roots only by
string/path resolution, never by opening them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from exps.csg00 import contract as C
from exps.csg00 import g3_design as G3


def _write_png(path: Path, size=(32, 32)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _synthetic_state(path: Path, arm: str) -> None:
    if arm == "B":
        state = {
            "ready": np.ones((C.BINS, C.K_MODES), dtype=np.uint8),
            "count": np.full((C.BINS, C.K_MODES), 100, dtype=np.int64),
            "n_total": np.full((C.BINS,), 1600, dtype=np.int64),
            "mu": np.zeros((C.BINS, C.K_MODES, C.DIM), dtype=np.float32),
            "U": np.zeros((C.BINS, C.K_MODES, C.DIM, C.Q_TANGENT), dtype=np.float32),
            "lam": np.ones((C.BINS, C.K_MODES, C.Q_TANGENT), dtype=np.float32),
            "R": np.zeros((C.BINS, C.K_MODES, C.Q_TANGENT), dtype=np.float32),
            "rho": np.zeros((C.BINS, C.K_MODES), dtype=np.float32),
        }
        for b in range(C.BINS):
            for k in range(C.K_MODES):
                state["U"][b, k, : C.Q_TANGENT, :] = np.eye(C.Q_TANGENT, dtype=np.float32)
        C.write_state(path, "B", 350, {"k": 2}, state, C.B_STATE_LAYOUT)
    else:
        state = {
            "res": np.zeros((C.BINS, C.RESERVOIR_CAPACITY, C.DIM), dtype=np.float32),
            "count": np.full((C.BINS,), C.RESERVOIR_CAPACITY, dtype=np.int64),
            "seen": np.full((C.BINS,), 5600, dtype=np.int64),
        }
        C.write_state(path, "C", 350, {"capacity": 10}, state, C.C_STATE_LAYOUT)


def _build_fixture(tmp_path: Path):
    root = tmp_path / "fresh_cohort"
    rows = []
    labels = []
    for category, kind in (("widget", "object"), ("fabric", "texture")):
        for i in range(2):
            rel = f"{category}/normal_{i}.png"
            _write_png(root / rel)
            rows.append({"relative_path": rel, "category": category, "type": kind,
                         "label": 0, "mask_relative_path": None})
            labels.append(0)
        for i in range(2):
            rel = f"{category}/anomaly_{i}.png"
            _write_png(root / rel)
            rows.append({"relative_path": rel, "category": category, "type": kind,
                         "label": 1,
                         "mask_relative_path": f"{category}/anomaly_{i}_mask.png"})
            labels.append(1)
    manifest = {
        "schema": "csg00_g3_cohort_v1",
        "cohort_id": "SYNTHETIC_FIXTURE_ONLY",
        "root": str(root),
        "fresh": True,
        "images": rows,
    }
    manifest_path = tmp_path / "cohort.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    auth = {
        "schema": "csg00_g3_authorization_v1",
        "authorized": True,
        "cohort_manifest_sha256": C.sha256_file(manifest_path),
        "fresh": True,
        "one_shot": True,
        "mask_access": "after_prediction_seal_only",
        "target_categories": [{"name": "widget", "type": "object"},
                              {"name": "fabric", "type": "texture"}],
        "note": "synthetic fixture only",
    }
    auth_path = tmp_path / "authorization.json"
    auth_path.write_text(json.dumps(auth, indent=2))
    state_b = tmp_path / "b.csg00"
    state_c = tmp_path / "c.csg00"
    _synthetic_state(state_b, "B")
    _synthetic_state(state_c, "C")
    return manifest_path, auth_path, state_b, state_c, root, rows


def _synthetic_features(image_paths):
    features = torch.zeros((len(image_paths), C.PATCHES, C.DIM), dtype=torch.float32)
    for i, path in enumerate(image_paths):
        if "anomaly" in str(path):
            features[i] = 5.0
    return features


def _synthetic_masks(path: Path) -> np.ndarray:
    return np.ones((32, 32), dtype=bool)


def test_fresh_path_guard_denies_exposed_roots():
    for rel in ("data/mvtec/bottle/test/crack/000.png",
                "data/mvtec/bottle/ground_truth/crack/000_mask.png",
                "data/visa/candle/test/000.png"):
        with pytest.raises(G3.G3Refused):
            G3.assert_fresh_cohort_path(C.ROOT / rel)


def test_authorization_is_mandatory_and_complete(tmp_path):
    with pytest.raises(G3.G3Refused):
        G3.load_authorization(None)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "csg00_g3_authorization_v1",
                               "authorized": True, "fresh": True, "one_shot": True,
                               "cohort_manifest_sha256": "x",
                               "mask_access": "after_prediction_seal_only",
                               "target_categories": [{"name": "a", "type": "object"}]}))
    with pytest.raises(G3.G3Refused, match="object and one texture"):
        G3.load_authorization(bad)


def test_prediction_seal_before_mask_and_synthetic_ap(tmp_path):
    manifest_path, auth_path, state_b, state_c, root, rows = _build_fixture(tmp_path)
    # Deliberately do not create any mask file: prediction must not need or open one.
    out = tmp_path / "g3out"
    info = G3.predict_cohort(manifest_path, auth_path, state_b, state_c, out,
                             feature_fn=_synthetic_features)
    assert (out / "PREDICTION_SEAL.json").is_file()
    seal = json.loads((out / "PREDICTION_SEAL.json").read_text())
    assert seal["mask_bytes_opened"] is False

    # One-shot: a second prediction is refused.
    with pytest.raises(G3.G3Refused, match="one-shot"):
        G3.predict_cohort(manifest_path, auth_path, state_b, state_c, out,
                          feature_fn=_synthetic_features)

    # Mask scoring without explicit unlock is refused.
    with pytest.raises(G3.G3Refused, match="unlock"):
        G3.score_cohort(manifest_path, auth_path, out / "PREDICTION_SEAL.json",
                        tmp_path / "result.json", unlock_masks=False,
                        mask_loader=_synthetic_masks)

    # Scoring with a bogus seal is refused.
    bogus = tmp_path / "bogus_seal.json"
    bogus.write_text(json.dumps({"schema": "csg00_g3_prediction_seal_v1",
                                 "cohort_manifest_sha256": "x"}))
    with pytest.raises(G3.G3Refused):
        G3.score_cohort(manifest_path, auth_path, bogus, tmp_path / "result.json",
                        unlock_masks=True, mask_loader=_synthetic_masks)

    summary = G3.score_cohort(manifest_path, auth_path, out / "PREDICTION_SEAL.json",
                              tmp_path / "result.json", unlock_masks=True,
                              mask_loader=_synthetic_masks, bootstrap_replicates=8)
    assert summary["schema"] == "csg00_g3_result_v1"
    assert summary["macro"]["B"]["native_pixel_ap"] == pytest.approx(1.0)
    assert summary["macro"]["B"]["image_auroc"] == pytest.approx(1.0)
    assert summary["per_category"]["B"]["widget"]["type"] == "object"
    assert "object" in summary["negative_gap_safeguards"]["B"]
    assert "texture" in summary["negative_gap_safeguards"]["B"]
    assert summary["bootstrap"]["replicates"] == 8


def test_native_bilinear_mapping_and_ap_helpers():
    patch = np.zeros((C.GRID, C.GRID))
    patch[3, 4] = 5.0
    native = G3._native_scores(patch, 64, 80)
    assert native.shape == (64, 80)
    assert float(native.max()) > 0.0
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert G3.average_precision(scores, labels) == pytest.approx(1.0)
    assert G3.image_auroc(scores, labels) == pytest.approx(1.0)
    assert G3.average_precision(np.array([0.1, 0.2]), np.array([0, 0])) is None
