"""Frozen normal-feature adaptor for CSG-00.

The experiment reuses the immutable ``frozen_features_v1`` clean caches, whose
rows are the raw block-8 (layer 9) ViT-B/8 patch descriptors already used by the
historical normal-support work.  No new feature is extracted for the 350 writer
events or the 59 witnesses; a small preregistered GPU parity check re-encodes a
few normal images to confirm the current environment reproduces the cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import contract as C


@dataclass
class FeatureCache:
    patches: Dict[str, torch.Tensor]
    relative_paths: Dict[str, List[str]]
    source_sha256: Dict[str, List[str]]
    files: Dict[str, Dict[str, Any]]

    def row(self, category: str, index: int) -> torch.Tensor:
        return self.patches[category][index]


def _verify_cache_file(path: Path, expected_sha: str) -> Dict[str, Any]:
    actual = C.sha256_file(path)
    return {
        "path": str(path.relative_to(C.ROOT)),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "expected_sha256": expected_sha,
        "match": actual == expected_sha,
    }


def load_feature_cache(verify_hashes: bool = True) -> FeatureCache:
    patches, relative_paths, source_sha256, files = {}, {}, {}, {}
    for category in C.CATEGORIES:
        path = C.CACHE_DIR / f"{category}_train.pt"
        info = _verify_cache_file(path, C.CACHE_SHA256[category]) if verify_hashes else {
            "path": str(path.relative_to(C.ROOT)), "bytes": path.stat().st_size,
            "sha256": C.sha256_file(path),
        }
        if verify_hashes and not info["match"]:
            raise RuntimeError(f"feature cache digest mismatch: {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        meta = payload["metadata"]
        checks = {
            "model_name": meta.get("model_name") == "vit_base_patch8_224",
            "layer_number": meta.get("layer_number") == 9,
            "block_index": meta.get("block_index") == 8,
            "cls_removed": meta.get("cls_removed") is True,
            "feature_normalization": str(meta.get("feature_normalization")).lower().startswith("none"),
            "checkpoint_sha256": meta.get("checkpoint_sha256") == C.CHECKPOINT_SHA256,
            "source_manifest_digest": payload.get("source_manifest_digest") == C.SOURCE_MANIFEST_DIGEST,
            "dtype": payload.get("dtype") == "float32",
            "view": payload.get("view") == "clean",
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise RuntimeError(f"feature cache metadata mismatch for {category}: {failed}")
        tensor = payload["patches"]
        if tuple(tensor.shape)[1:] != (C.PATCHES, C.DIM) or tensor.dtype != torch.float32:
            raise RuntimeError(f"feature cache shape/dtype mismatch for {category}: {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"non-finite feature cache values for {category}")
        patches[category] = tensor
        relative_paths[category] = list(payload["relative_paths"])
        source_sha256[category] = list(payload["per_image_source_sha256"])
        info["metadata"] = dict(meta)
        info["rows"] = int(tensor.shape[0])
        info["row_index"] = {p: i for i, p in enumerate(relative_paths[category])}
        files[category] = info
    return FeatureCache(patches, relative_paths, source_sha256, files)


def verify_cache_rows(cache: FeatureCache, updates: Sequence[Dict[str, Any]],
                      probes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Prove that every protocol identity maps to the frozen cache row it names."""
    whitelist = {row["relative_path"] for row in updates} | {row["relative_path"] for row in probes}
    mismatches = []
    checked = 0
    for row in list(updates) + list(probes):
        C.assert_normal_only(row["relative_path"], whitelist)
        category, index = row["category"], int(row["cache_index"])
        if cache.relative_paths[category][index] != row["relative_path"]:
            mismatches.append((row["relative_path"], "path"))
        if cache.source_sha256[category][index] != row["raw_png_sha256"]:
            mismatches.append((row["relative_path"], "content_sha"))
        if not cache.patches[category][index].is_contiguous():
            mismatches.append((row["relative_path"], "contiguity"))
        checked += 1
    if mismatches:
        raise RuntimeError(f"cache row identity failures: {mismatches[:5]}")
    return {
        "checked_rows": checked,
        "writer_rows": len(updates),
        "witness_rows": len(probes),
        "path_and_content_match": True,
        "cache_files": {
            category: {k: v for k, v in info.items() if k in ("path", "bytes", "sha256", "rows")}
            for category, info in cache.files.items()
        },
    }


def writer_tensor(cache: FeatureCache, updates: Sequence[Dict[str, Any]], t: int) -> torch.Tensor:
    """Return the causal writer prefix as [t, 784, 768] float32 in arrival order."""
    if not 1 <= t <= len(updates):
        raise ValueError(f"invalid boundary {t}")
    rows = [cache.row(updates[e]["category"], int(updates[e]["cache_index"])) for e in range(t)]
    return torch.stack(rows, dim=0)


def witness_tensor(cache: FeatureCache, probes: Sequence[Dict[str, Any]]) -> torch.Tensor:
    rows = [cache.row(row["category"], int(row["cache_index"])) for row in probes]
    return torch.stack(rows, dim=0)


def bin_view(features: torch.Tensor, bin_index: int, positions: Sequence[np.ndarray]) -> torch.Tensor:
    """Flatten the 16 positions of one spatial bin, keeping leading index order.

    ``features`` is [n, 784, 768]; output is [n*16, 768] in (leading, position)
    order, which is the canonical arrival order inside a bin.
    """
    pos = positions[bin_index]
    view = features[:, torch.as_tensor(pos, dtype=torch.long), :]
    return view.reshape(-1, view.shape[-1])


def descriptor_bins(writer: torch.Tensor, positions: Sequence[np.ndarray]) -> List[torch.Tensor]:
    """Per-bin writer descriptors [t*16, 768] in event order, then position order."""
    return [bin_view(writer, b, positions) for b in range(C.BINS)]


# ---------------------------------------------------------------------------
# Preregistered GPU feature parity check (normal-only)
# ---------------------------------------------------------------------------

# The four normal writer events are fixed before execution: one early bottle,
# one late bottle, the first carpet event, and one hazelnut event.
PARITY_EVENTS = (1, 50, 101, 250)


def _image_batch(relative_paths: Sequence[str]) -> torch.Tensor:
    from PIL import Image

    root = C.ROOT / "data" / "mvtec"
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    images = []
    for relative in relative_paths:
        path = root / relative
        with Image.open(path) as source:
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            array = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255.0
        images.append((array - mean) / std)
    return torch.stack(images, dim=0)


def feature_parity_check(cache: FeatureCache, updates: Sequence[Dict[str, Any]],
                         device: str = "cuda") -> Dict[str, Any]:
    """Re-encode four preregistered normal images and compare with the frozen cache."""
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor

    events = [e for e in PARITY_EVENTS if e <= len(updates)]
    rows = [updates[e - 1] for e in events]
    rels = [row["relative_path"] for row in rows]
    config = CADICViTConfig(
        checkpoint_path=str(C.CHECKPOINT),
        checkpoint_identity=f"sha256:{C.CHECKPOINT_SHA256}",
        layer_number=9,
        layer_indexing="one_based",
    )
    extractor = CADICViTFeatureExtractor(config, device=device)
    tensors = _image_batch(rels).to(device)
    with torch.no_grad():
        features = extractor.extract_patch_features(tensors).detach().cpu().float()
    extractor.close()

    records = []
    ok = True
    for event, row, produced in zip(events, rows, features):
        reference = cache.row(row["category"], int(row["cache_index"])).to(torch.float32)
        diff = (produced - reference).to(torch.float64)
        denom = reference.to(torch.float64).norm().item()
        relative_l2 = float(diff.norm().item() / denom) if denom else float("inf")
        scaled_max_abs = float(diff.abs().max().item() / denom) if denom else float("inf")
        row_ok = relative_l2 <= 3.0517578125e-05 and scaled_max_abs <= 1.220703125e-04
        bitwise = bool(torch.equal(produced, reference))
        ok = ok and row_ok
        records.append({
            "event": event,
            "relative_path": row["relative_path"],
            "bitwise_equal": bitwise,
            "max_abs": float(diff.abs().max().item()),
            "relative_l2": relative_l2,
            "scaled_max_abs": scaled_max_abs,
            "row_ok": row_ok,
        })
    return {
        "device": device,
        "events": events,
        "tolerance": {"relative_l2": 3.0517578125e-05, "scaled_max_abs": 1.220703125e-04,
                      "bitwise_required": False, "source": "repository cache parity convention"},
        "records": records,
        "all_ok": ok,
        "note": "tolerance-defined parity; block-8 output, CLS removed, no normalization",
    }
