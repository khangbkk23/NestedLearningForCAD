"""Identity-only MVTec AD v1 preparation and arrival-gated frozen extraction.

Private splits are never traversed. Mask pixels are opened only by native_mask,
after the resolved protocol lock and only at an admitted evaluation boundary.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset.benchmark_protocol_v1 import _ImageDataset
from exps.task_p.p0_normal_support import require
from exps.task_p.p1_readout import features_ok

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CP_SHA = "755e9f5f74fcb6d0d8f23228d9a36328bffe183b192004f5e9ef7b3f0a2ea415"
CATEGORIES = ("bottle", "carpet", "hazelnut")
BOUNDARIES = (100, 200, 300, 350)
MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: neither protocol nor results are silently overwritten.
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_path(root, relative):
    root = Path(root).resolve()
    p = root / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe source path")
    require(p.resolve() == p and p.is_file(), "missing source or symlink")
    require(p.is_relative_to(root), "source outside dataset")
    return p


def identity(root, path, category, role):
    relative = path.relative_to(root).as_posix()
    path = source_path(root, relative)
    # Header inspection establishes geometry; no image array or mask pixels here.
    with Image.open(path) as im:
        w, h, mode = im.width, im.height, im.mode
    require(mode in ("RGB", "L"), f"unresolved image mode {mode}")
    return {"relative_path": relative, "sha256": sha(path), "bytes": path.stat().st_size,
            "hw": [h, w], "mode": mode, "category": category, "role": role}


def discover(root):
    root = Path(root).resolve()
    updates, evaluations = [], []
    sources = json.loads((ROOT / "results/hope_cad/frozen_features_v1/source_manifest.json").read_text())
    by_path = {r["relative_path"]: r for r in sources["entries"]}
    old = json.loads((ROOT / "results/hope_cad/task_p/p0c/protocol_resolved.json").read_text())
    for r in old["updates"]:
        s = by_path[r["relative_path"]]
        updates.append({"category": r["category"], "relative_path": r["relative_path"],
                        "sha256": s["raw_png_sha256"], "bytes": s["raw_png_bytes"],
                        "hw": [s["original_height"], s["original_width"]], "mode": s["original_mode"],
                        "event": r["event"], "cache_index": r["cache_index"], "role": "normal_train"})
    sealed_paths = []
    for c in CATEGORIES:
        folder = root / c / "test"
        normal = sorted((folder / "good").glob("*.png"))
        groups = [sorted(p.glob("*.png")) for p in sorted(folder.iterdir())
                  if p.is_dir() and p.name != "good"]
        anomaly = [paths[i] for i in range(max(map(len, groups), default=0))
                   for paths in groups if i < len(paths)]
        # Reconstruct the historical first-10 / next-10 identity rule without
        # invoking its helper, which also opens confirmation images and masks.
        # Filename uniqueness makes the helper's secondary content-hash sort moot.
        for label, files in ((0, normal), (1, anomaly)):
            require(len(files) >= 20, f"BLOCKED_DATA: {c} needs dev and sealed identity ranges")
            sealed_paths.extend(p.relative_to(root).as_posix() for p in files[10:20])
            for p in files[:10]:
                row = identity(root, p, c, "development_eval")
                row.update(label=label, mask_path=None, mask_sha256=None)
                if row["label"]:
                    mask = root / c / "ground_truth" / p.parent.name / (p.stem + "_mask.png")
                    row["mask_path"] = mask.relative_to(root).as_posix()
                    row["mask_sha256"] = sha(source_path(root, row["mask_path"]))
                evaluations.append(row)
    validate_roles(updates, evaluations)
    require(not {r["relative_path"] for r in evaluations}.intersection(sealed_paths), "sealed overlap")
    return updates, evaluations, sealed_paths


def validate_roles(updates, evaluations):
    paths, contents = set(), set()
    for role, rows in (("normal_train", updates), ("development_eval", evaluations)):
        for row in rows:
            parts = Path(row["relative_path"]).parts
            require(row["role"] == role and row["category"] in CATEGORIES, "invalid role/category")
            require(parts[0] == row["category"] and len(parts) == 4, "invalid source layout")
            require(parts[1:3] == ("train", "good") if role == "normal_train" else
                    parts[1] == "test", "forbidden partition")
            if role == "development_eval":
                require(row["label"] == int(parts[2] != "good"), "image label/layout mismatch")
                require(bool(row["mask_path"]) == bool(row["label"]), "image mask/label mismatch")
            require(row["relative_path"] not in paths and row["sha256"] not in contents,
                    "duplicate source identity/content or train/eval overlap")
            paths.add(row["relative_path"])
            contents.add(row["sha256"])


def native_mask(root, row):
    h, w = row["hw"]
    if row["label"] == 0:
        require(row["mask_path"] is None, "normal image has nonempty mask identity")
        return np.zeros((h, w), dtype=bool)
    require(row["label"] == 1 and row["mask_path"], "invalid anomaly annotation")
    path = source_path(root, row["mask_path"])
    require(sha(path) == row["mask_sha256"], "mask source changed")
    with Image.open(path) as im:
        mask = np.array(im)
    require(mask.ndim == 2 and mask.shape == (h, w), "native annotation geometry mismatch")
    require(np.isin(mask, [0, 1, 255]).all(), "unresolved mask polarity/values")
    mask = mask > 0
    require(bool(mask.any()), "defective image has empty annotation")
    return mask


class FeatureCache:
    """One immutable FP32 file per admitted source, never an old-task replay bank."""
    def __init__(self, root, directory, protocol_sha, device="cpu"):
        self.root, self.directory = Path(root), Path(directory)
        self.protocol_sha, self.device = protocol_sha, device
        self.extractor = None
        self.timings = []

    def get(self, row):
        if row["role"] == "normal_train":
            # Reuse P0c's frozen clean tensor; never regenerate original normals.
            c = row["category"]
            obj = torch.load(ROOT / f"results/hope_cad/frozen_features_v1/clean/{c}_train.pt",
                             mmap=True, weights_only=True)
            i = row["cache_index"]
            require(obj["relative_paths"][i] == row["relative_path"] and
                    obj["per_image_source_sha256"][i] == row["sha256"] and
                    obj["metadata"]["checkpoint_sha256"] == CP_SHA and obj["view"] == "clean",
                    "normal cache/source mismatch")
            features = obj["patches"][i].clone()
            features_ok(features)
            h = hashlib.sha256(str((tuple(features.shape), features.dtype)).encode() +
                               features.numpy().tobytes()).hexdigest()
            require(h == obj["per_image_feature_sha256"][i], "normal row digest mismatch")
            return features
        path = source_path(self.root, row["relative_path"])
        require(sha(path) == row["sha256"], "image source changed")
        key = hashlib.sha256(row["relative_path"].encode()).hexdigest()
        target = self.directory / (key + ".pt")
        marker = target.with_suffix(".json")
        if target.exists() or marker.exists():
            require(target.is_file() and marker.is_file(), "incomplete feature cache; refuse overwrite")
            meta = json.loads(marker.read_text())
            require(meta["protocol_sha256"] == self.protocol_sha and meta["source"] == row
                    and meta["checkpoint_sha256"] == CP_SHA and sha(target) == meta["file_sha256"],
                    "cache provenance mismatch")
            features = torch.load(target, map_location="cpu", weights_only=True)
            features_ok(features)
            return features
        if self.extractor is None:
            from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
            require(sha(CP) == CP_SHA, "checkpoint mismatch")
            self.extractor = CADICViTFeatureExtractor(CADICViTConfig(
                checkpoint_path=str(CP), checkpoint_identity="sha256:" + CP_SHA), self.device)
        t = time.perf_counter()
        # Canonical image preprocessing is reused unchanged, training=True only
        # selects its image-only return path; its constant label is never used.
        data = _ImageDataset(self.root.resolve(), [row], 224, MEAN, STD, training=True)
        image = data[0]["images"][None]
        with torch.no_grad():
            features = self.extractor(image)[0].detach().cpu().contiguous()
        features_ok(features)
        self.directory.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "cache collision")
        partial = target.with_suffix(".part")
        require(not partial.exists(), "incomplete cache write exists")
        torch.save(features, partial)
        partial.replace(target)
        elapsed = time.perf_counter() - t
        write_json(marker, {"source": row, "protocol_sha256": self.protocol_sha,
                           "checkpoint_sha256": CP_SHA, "file_sha256": sha(target),
                           "feature_sha256": hashlib.sha256(features.numpy().tobytes()).hexdigest(),
                           "extraction_seconds": elapsed, "device": self.device})
        self.timings.append({"path": row["relative_path"], "seconds": elapsed})
        return features


class ArrivalReader:
    def __init__(self, protocol, cache):
        self.protocol, self.cache = protocol, cache
        self.completed, self.admitted, self.access = 0, set(), []

    def update(self, row):
        require(self.completed < len(self.protocol["updates"]) and
                row == self.protocol["updates"][self.completed] and row["role"] == "normal_train",
                "future, replayed or anomalous update forbidden")
        features = self.cache.get(row)
        self.completed += 1
        self.admitted.add(row["category"])
        self.access.append({"role": "update", "event": self.completed, "path": row["relative_path"]})
        return features

    def evaluate(self, row, boundary):
        require(boundary == self.completed and boundary in self.protocol["boundaries"] and
                row in self.protocol["evaluations"] and row["category"] in self.admitted,
                "future or unauthorized evaluation")
        self.access.append({"role": "evaluation", "event": boundary, "path": row["relative_path"]})
        return self.cache.get(row), native_mask(self.cache.root, row)
