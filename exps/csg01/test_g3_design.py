"""G1: CSG-01 G3 design fixtures (default deny, seal-before-mask, synthetic AP)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from exps.csg00 import g3_design as G3
from exps.csg01 import contract as C1
from exps.csg01 import g3_design as G31


def _png(path: Path, size=(24, 32)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)


def _state(path: Path, arm: str):
    if arm in {"B", "E"}:
        n = C1.SELECTION_COUNT if arm == "B" else C1.SIZE_MATCHED_COUNT
        q = C1.quota_vector(n)
        state = {
            "counts": q.astype(np.int32),
            "offsets": np.concatenate([[0], np.cumsum(q)[:-1]]).astype(np.int32),
            "desc": np.zeros((n, C1.DIM), dtype=np.float32),
        }
        C1.write_state(path, arm, 350, C1.arms_params()[arm], state)
    else:
        q = C1.quota_vector(C1.SELECTION_COUNT)
        state = {
            "counts": q.astype(np.int32),
            "offsets": np.concatenate([[0], np.cumsum(q)[:-1]]).astype(np.int32),
            "q": np.zeros((C1.SELECTION_COUNT, C1.DIM), dtype=np.uint8),
            "xmin": np.zeros((C1.SELECTION_COUNT,), dtype=np.float32),
            "step": np.zeros((C1.SELECTION_COUNT,), dtype=np.float32),
        }
        C1.write_state(path, arm, 350, C1.arms_params()[arm], state)


def _fixture(tmp_path: Path):
    root = tmp_path / "fresh"
    rows = []
    for cat, kind in (("widget", "object"), ("fabric", "texture")):
        for i in range(2):
            rel = f"{cat}/normal_{i}.png"
            _png(root / rel)
            rows.append({"relative_path": rel, "category": cat, "label": 0,
                         "mask_relative_path": None})
        for i in range(2):
            rel = f"{cat}/anomaly_{i}.png"
            _png(root / rel)
            rows.append({"relative_path": rel, "category": cat, "label": 1,
                         "mask_relative_path": f"{cat}/anomaly_{i}_mask.png"})
    manifest = {"schema": "csg01_g3_cohort_v1", "root": str(root), "fresh": True,
                "images": rows}
    mpath = tmp_path / "cohort.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    auth = {"schema": "csg00_g3_authorization_v1", "authorized": True,
            "cohort_manifest_sha256": C1.C0.sha256_file(mpath), "fresh": True,
            "one_shot": True, "mask_access": "after_prediction_seal_only",
            "target_categories": [{"name": "widget", "type": "object"},
                                  {"name": "fabric", "type": "texture"}]}
    apath = tmp_path / "authorization.json"
    apath.write_text(json.dumps(auth, indent=2))
    states = {}
    for arm in "BCDE":
        p = tmp_path / f"{arm}.csg00"
        _state(p, arm)
        states[arm] = p
    return mpath, apath, states


def _features(paths):
    out = torch.zeros((len(paths), C1.PATCHES, C1.DIM), dtype=torch.float32)
    for i, p in enumerate(paths):
        if "anomaly" in str(p):
            out[i] = 5.0
    return out


def _masks(path):
    return np.ones((32, 24), dtype=bool)


def test_g3_default_deny_and_forbidden_paths():
    for rel in ("data/mvtec/bottle/test/crack/000.png",
                "data/mvtec/bottle/ground_truth/crack/000_mask.png"):
        with pytest.raises(G3.G3Refused):
            G3.assert_fresh_cohort_path(C1.ROOT / rel)
    with pytest.raises(G3.G3Refused):
        G3.load_authorization(None)


def test_g3_predict_seal_then_score(tmp_path):
    mpath, apath, states = _fixture(tmp_path)
    out = tmp_path / "out"
    info = G31.predict_cohort(mpath, apath, states, out, feature_fn=_features)
    assert (out / "PREDICTION_SEAL.json").is_file()
    seal = json.loads((out / "PREDICTION_SEAL.json").read_text())
    assert seal["mask_bytes_opened"] is False
    with pytest.raises(G3.G3Refused, match="one-shot"):
        G31.predict_cohort(mpath, apath, states, out, feature_fn=_features)
    with pytest.raises(G3.G3Refused, match="unlock"):
        G31.score_cohort(mpath, apath, out / "PREDICTION_SEAL.json",
                         tmp_path / "result.json", unlock_masks=False,
                         mask_loader=_masks)
    result = G31.score_cohort(mpath, apath, out / "PREDICTION_SEAL.json",
                              tmp_path / "result.json", unlock_masks=True,
                              mask_loader=_masks, bootstrap_replicates=4)
    assert result["schema"] == "csg01_g3_result_v1"
    for arm in "BCDE":
        assert result["macro"][arm]["native_pixel_ap"] == pytest.approx(1.0)
        assert result["macro"][arm]["image_auroc"] == pytest.approx(1.0)
    assert result["bootstrap"]["replicates"] == 4
