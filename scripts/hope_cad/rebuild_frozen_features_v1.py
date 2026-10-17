# scripts/hope_cad/rebuild_frozen_features_v1.py
"""Rebuild canonical clean and RGB-masked ViT-B/8 feature caches for all 15 MVTec AD classes.

Scope of this tool (one-time, reusable feature preparation):

* source images: ``data/mvtec/<category>/train/good/*.png`` only, canonical
  protocol order from ``conf/benchmarks/protocols/mvtec_1x15_v1.yaml``;
* representation: the frozen CADIC-compatible ViT-B/8 extractor in
  :mod:`models.feature_extractors.cadic_vit_v1` (block 8 capture, CLS removed,
  FP32, no post-hoc normalization);
* two views per source image: ``clean`` and ``masked_rgb16`` where the shared
  16-region RGB mask of ``exps.ol04.hope_outer_learning_pilot_v1.mask_rgb`` is
  applied to the *normalized input image* before ViT attention;
* outputs: ``results/hope_cad/frozen_features_v1/{clean,masked_rgb16}/<category>_train.pt``
  with per-image integrity records, atomic writes and completion markers.

No training, no feature statistics, no dimensionality reduction, no precision
change and no evaluation data are involved.  This module is additive and does
not modify any existing scientific core file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FEATURES_ROOT = REPO_ROOT / "results/hope_cad/frozen_features_v1"
PROTOCOL_PATH = REPO_ROOT / "conf/benchmarks/protocols/mvtec_1x15_v1.yaml"
DATASET_ROOT = REPO_ROOT / "data/mvtec"
CHECKPOINT = REPO_ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
CHECKPOINT_PROVENANCE = FEATURES_ROOT / "checkpoint_provenance.json"

VIEWS = ("clean", "masked_rgb16")
FEATURE_SHAPE = (784, 768)
VIEW_CONTRACT = {
    "clean": {"masked": False, "contract": "unmodified normalized input image"},
    "masked_rgb16": {
        "masked": True,
        "mask_centers": [[3, 3], [3, 10], [3, 17], [3, 24], [10, 3], [10, 10], [10, 17], [10, 24],
                         [17, 3], [17, 10], [17, 17], [17, 24], [24, 3], [24, 10], [24, 17], [24, 24]],
        "region_pixels": [24, 24],
        "masked_pixel_area": 9216,
        "fill_value": "normalized zero (ImageNet mean RGB)",
        "oracle": "exps.ol04.hope_outer_learning_pilot_v1.mask_rgb",
        "view_contract": "ONE_SHARED_16_REGION_VIEW",
        "applied_before": "ViT feature extraction",
    },
}
SELECTION_RULE = "all train/good images in canonical protocol order (task_order then lexicographic relative path)"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    """Identity hash of one feature tensor (same convention as the OL-04 tooling)."""
    plain = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(plain.shape), plain.dtype)).encode() + plain.numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def code_hashes() -> dict:
    files = {
        "this_script": Path(__file__),
        "cad ic_vit_v1": REPO_ROOT / "models/feature_extractors/cadic_vit_v1.py",
        "mask_oracle": REPO_ROOT / "exps/ol04/hope_outer_learning_pilot_v1.py",
        "centers_oracle": REPO_ROOT / "exps/ol03/hope_outer_learning_p1_v1.py",
        "image_dataset": REPO_ROOT / "dataset/benchmark_protocol_v1.py",
        "manifest_builder": REPO_ROOT / "dataset/benchmark_manifest_v1.py",
        "preprocessing_reference": REPO_ROOT / "scripts/hope_cad/probe_real_features.py",
        "protocol": PROTOCOL_PATH,
    }
    return {name: sha256_file(path) for name, path in files.items()}


# --------------------------------------------------------------------------------------
# source manifest
# --------------------------------------------------------------------------------------

def build_source_manifest(categories: list[str] | None = None) -> dict:
    from PIL import Image

    from dataset.benchmark_manifest_v1 import build_training_manifest

    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    protocol["dataset"]["root"] = str(DATASET_ROOT)
    canonical_order = list(protocol["task_order"])
    selected = list(categories or canonical_order)
    for category in selected:
        if category not in canonical_order:
            raise ValueError(f"category outside the canonical protocol order: {category}")
    manifest = build_training_manifest(protocol, 0, "feature_rebuild_v1")

    entries = []
    for row in manifest["entries"]:
        if row["task_name"] not in selected:
            continue
        relative = row["relative_path"]
        parts = Path(relative).parts
        if len(parts) != 4 or parts[1:3] != ("train", "good") or parts[0] not in canonical_order:
            raise ValueError(f"non train/good source rejected: {relative}")
        if "test" in parts or "ground_truth" in parts:
            raise ValueError(f"non training source rejected: {relative}")
        path = DATASET_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"source identity rejected: {relative}")
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        entries.append({
            "category_index": row["task_id"],
            "category": row["task_name"],
            "relative_path": relative,
            "raw_png_sha256": sha256_file(path),
            "raw_png_bytes": path.stat().st_size,
            "original_width": width,
            "original_height": height,
            "original_mode": mode,
            "source_identity": "mvtec_train_good_png",
            "source_root": "data/mvtec",
        })

    by_digest: dict[str, list[str]] = {}
    for entry in entries:
        by_digest.setdefault(entry["raw_png_sha256"], []).append(entry["relative_path"])
    duplicate_groups = [
        {"sha256": digest, "relative_paths": sorted(paths), "count": len(paths)}
        for digest, paths in sorted(by_digest.items()) if len(paths) > 1
    ]
    per_category = {}
    for category in selected:
        rows = [entry for entry in entries if entry["category"] == category]
        per_category[category] = {
            "count": len(rows),
            "first": rows[0]["relative_path"],
            "last": rows[-1]["relative_path"],
            "canonical_category_index": rows[0]["category_index"],
        }

    payload = {
        "schema": "hope_cad_frozen_features_v1_source_manifest",
        "scope": "r1_inputs_three_categories" if categories else "all_protocol_categories",
        "categories_in_scope": selected,
        "protocol_id": protocol["id"],
        "protocol_manifest_digest": manifest["digest"],
        "canonical_task_order": canonical_order,
        "ordering_rule": manifest["ordering_rule"],
        "total_source_images": len(entries),
        "per_category": per_category,
        "duplicate_content_check": {
            "duplicate_groups": duplicate_groups,
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_image_count": sum(group["count"] for group in duplicate_groups),
            "distinct_content_count": len(by_digest),
        },
        "entries": entries,
        "restrictions": {
            "train_good_only": True,
            "test_split_accessed": False,
            "ground_truth_accessed": False,
            "sealed_confirmation_accessed": False,
            "extracted_categories": selected,
            "categories_not_extracted": [category for category in canonical_order if category not in selected],
        },
    }
    payload["manifest_digest"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "manifest_digest"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def load_source_manifest() -> dict:
    path = FEATURES_ROOT / "source_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"source manifest missing: {path}; run the manifest step first")
    return load_json(path)


# --------------------------------------------------------------------------------------
# checkpoint / extractor
# --------------------------------------------------------------------------------------

def pinned_checkpoint_sha() -> str:
    provenance = load_json(CHECKPOINT_PROVENANCE)
    if provenance["upstream_files"]["pytorch_model.bin"]["sha256_matches"] is not True:
        raise RuntimeError("checkpoint provenance does not record an upstream integrity match")
    return provenance["canonical_sha256"]


def build_extractor(device: str):
    import torch

    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor

    sha = pinned_checkpoint_sha()
    actual = sha256_file(CHECKPOINT)
    if actual != sha:
        raise RuntimeError(f"checkpoint identity differs from pinned provenance: {actual} != {sha}")
    config = CADICViTConfig(checkpoint_path=str(CHECKPOINT), checkpoint_identity=f"sha256:{sha}")
    extractor = CADICViTFeatureExtractor(config, device)
    if extractor.block_index != 8 or extractor.config.layer_number != 9:
        raise RuntimeError("extractor block selection differs from the frozen contract")
    if any(parameter.requires_grad for parameter in extractor.parameters()):
        raise RuntimeError("backbone parameters are not frozen")
    if extractor.training or extractor.model.training:
        raise RuntimeError("extractor is not in eval mode")
    return extractor


def extractor_contract(extractor, batch_size: int, threads: int, device: str) -> dict:
    import torch

    metadata = extractor.protocol_metadata()
    upstream_config = None
    config_path = Path(FEATURES_ROOT) / "upstream_config.json"
    if config_path.is_file():
        raw = load_json(config_path)
        upstream_config = {
            "path": str(config_path),
            "architecture": raw.get("architecture"),
            "tag": raw.get("pretrained_cfg", {}).get("tag"),
            "num_classes": raw.get("num_classes"),
            "num_features": raw.get("num_features"),
            "upstream_default_preprocessing_used": False,
            "upstream_default_mean": raw.get("pretrained_cfg", {}).get("mean"),
            "upstream_default_std": raw.get("pretrained_cfg", {}).get("std"),
            "upstream_default_interpolation": raw.get("pretrained_cfg", {}).get("interpolation"),
            "upstream_default_crop_pct": raw.get("pretrained_cfg", {}).get("crop_pct"),
            "note": ("the upstream weights metadata advertises generic timm defaults (0.5/0.5 statistics, "
                     "bicubic, center-crop 0.9); this project's frozen contract uses "
                     "mean (0.485,0.456,0.406), std (0.229,0.224,0.225) and a direct square PIL bilinear "
                     "resize instead. The defaults were deliberately not substituted."),
        }
    return {
        "schema": "hope_cad_frozen_features_v1_extractor_contract",
        "architecture": "vit_base_patch8_224",
        "pretraining": "augreg_in21k",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": pinned_checkpoint_sha(),
        "input": {
            "layout": "RGB",
            "size": [224, 224],
            "resize": "PIL bilinear direct square resize from the original PNG",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "normalized_range_scale": "uint8/255 then (x-mean)/std",
        },
        "capture": {
            "layer_number_one_based": 9,
            "block_index_zero_based": 8,
            "capture_point": "ViT block output before final backbone normalization",
            "cls_token": "removed",
            "output_shape_per_image": list(FEATURE_SHAPE),
        },
        "dtype": "float32",
        "post_extraction_normalization": "none",
        "module_mode": "frozen, eval, torch.no_grad",
        "batch_size": batch_size,
        "torch_num_threads": threads,
        "device": device,
        "environment": {
            "torch": torch.__version__,
            "timm": __import__("timm").__version__,
            "pillow": __import__("PIL").__version__,
            "numpy": __import__("numpy").__version__,
            "python": __import__("sys").version.split()[0],
            "cuda_available": torch.cuda.is_available(),
            "platform": __import__("platform").platform(),
        },
        "extractor_protocol_metadata": metadata,
        "upstream_weights_metadata": upstream_config,
        "code_hashes": code_hashes(),
        "views": {name: VIEW_CONTRACT[name] for name in VIEWS},
    }


# --------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------

def dataset_rows_for_category(manifest: dict, category: str) -> list[dict]:
    rows = [entry for entry in manifest["entries"] if entry["category"] == category]
    if not rows:
        raise ValueError(f"no source entries for category {category}")
    return rows


def make_image_dataset(rows: list[dict]):
    from dataset.benchmark_protocol_v1 import _ImageDataset

    return _ImageDataset(DATASET_ROOT, [{"relative_path": row["relative_path"]} for row in rows],
                         224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True)


def category_paths(view: str, category: str) -> tuple[Path, Path]:
    directory = FEATURES_ROOT / view
    return directory / f"{category}_train.pt", directory / f"{category}_train.done.json"


def validate_payload(payload: dict, category: str, view: str, rows: list[dict], metadata: dict) -> dict:
    import torch

    expected_paths = [row["relative_path"] for row in rows]
    if payload.get("view") != view or payload.get("class_name") != category:
        raise ValueError(f"cache view/category mismatch for {category}/{view}")
    if payload.get("relative_paths") != expected_paths:
        raise ValueError(f"cache path order mismatch for {category}/{view}")
    if payload.get("metadata") != metadata:
        raise ValueError(f"cache feature identity/preprocessing mismatch for {category}/{view}")
    if payload.get("dtype") != "float32" or payload.get("selection_rule") != SELECTION_RULE:
        raise ValueError(f"cache declaration mismatch for {category}/{view}")
    patches = payload["patches"]
    if not isinstance(patches, torch.Tensor) or patches.dtype != torch.float32:
        raise ValueError(f"cache tensor type mismatch for {category}/{view}")
    if tuple(patches.shape) != (len(rows), *FEATURE_SHAPE) or not patches.is_contiguous():
        raise ValueError(f"cache tensor geometry mismatch for {category}/{view}")
    if payload["shape"] != list(patches.shape):
        raise ValueError(f"cache shape declaration mismatch for {category}/{view}")
    if not bool(torch.isfinite(patches).all()):
        raise ValueError(f"cache contains non-finite features for {category}/{view}")
    feature_hashes = payload.get("per_image_feature_sha256", [])
    if len(feature_hashes) != len(rows):
        raise ValueError(f"cache lacks per-image integrity records for {category}/{view}")
    raw_hashes = payload.get("per_image_source_sha256", [])
    if raw_hashes != [row["raw_png_sha256"] for row in rows]:
        raise ValueError(f"cache source identity mismatch for {category}/{view}")
    return {"images": len(rows), "bytes": patches.numel() * patches.element_size(),
            "finite": True, "shape": list(patches.shape)}


def extract_category(
    extractor,
    view: str,
    category: str,
    rows: list[dict],
    metadata: dict,
    manifest_digest: str,
    batch_size: int,
    threads: int,
    force: bool,
) -> dict:
    import torch

    from scripts.exps.hope_image_synchronous_memory import feature_equivalence

    cache_path, marker_path = category_paths(view, category)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file() and marker_path.is_file() and not force:
        marker = load_json(marker_path)
        if (marker.get("cache_sha256") == sha256_file(cache_path)
                and marker.get("view") == view and marker.get("class_name") == category
                and marker.get("source_manifest_digest") == manifest_digest
                and marker.get("checkpoint_sha256") == pinned_checkpoint_sha()):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            summary = validate_payload(payload, category, view, rows, metadata)
            log(f"reused completed {view} cache for {category} ({summary['images']} images)")
            return {"class_name": category, "view": view, "status": "REUSED",
                    "cache_path": str(cache_path), "cache_sha256": marker["cache_sha256"],
                    "cache_bytes": cache_path.stat().st_size, **summary,
                    "marker_path": str(marker_path)}
        log(f"existing {view} cache for {category} failed validation; re-extracting")
        marker_path.unlink(missing_ok=True)

    from exps.ol04.hope_outer_learning_pilot_v1 import mask_rgb
    from exps.ol03.hope_outer_learning_p1_v1 import CENTERS

    if view == "masked_rgb16":
        if len(CENTERS) != 16 or tuple(CENTERS) != tuple((r, c) for r in (3, 10, 17, 24) for c in (3, 10, 17, 24)):
            raise RuntimeError("mask oracle centers differ from the frozen contract")

    dataset = make_image_dataset(rows)
    dataset_seconds = 0.0
    forward_seconds = 0.0
    started = time.perf_counter()
    parts: list = []
    masked_input_hashes: list[str] = []
    mask_area_check = None
    populated = 0

    for start in range(0, len(rows), batch_size):
        batch_started = time.perf_counter()
        images = []
        for index in range(start, min(start + batch_size, len(rows))):
            sample = dataset[index]["images"].unsqueeze(0)
            if sample.shape != (1, 3, 224, 224):
                raise RuntimeError(f"preprocessing geometry differs for {rows[index]['relative_path']}")
            if view == "masked_rgb16":
                masked, mask = mask_rgb(sample)
                if mask_area_check is None:
                    mask_area_check = int(mask.sum())
                if int(mask.sum()) != 9216:
                    raise RuntimeError("mask area differs from 9216 pixels")
                for row, column in CENTERS:
                    window = mask[(row - 1) * 8:(row + 2) * 8, (column - 1) * 8:(column + 2) * 8]
                    if not bool(window.all()):
                        raise RuntimeError("center-to-pixel mapping differs")
                if not bool(torch.equal(masked[:, :, mask], torch.zeros_like(masked[:, :, mask]))):
                    raise RuntimeError("masked fill differs from normalized zero")
                if not bool(torch.equal(sample[:, :, ~mask], masked[:, :, ~mask])):
                    raise RuntimeError("unmasked pixels changed")
                images.append(masked[0])
                masked_input_hashes.append(tensor_sha256(masked[0]))
            else:
                images.append(sample[0])
        dataset_seconds += time.perf_counter() - batch_started

        batch = torch.stack(images)
        forward_started = time.perf_counter()
        features = extractor.extract_patch_features(batch).detach().cpu().to(torch.float32)
        forward_seconds += time.perf_counter() - forward_started
        if tuple(features.shape) != (batch.shape[0], *FEATURE_SHAPE):
            raise RuntimeError(f"extraction geometry differs for {category}/{view}: {tuple(features.shape)}")
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError(f"non-finite features for {category}/{view}")
        parts.append(features.contiguous())
        populated += batch.shape[0]
        if populated % (batch_size * 8) < batch_size or populated == len(rows):
            log(f"{view} {category}: {populated}/{len(rows)}")

    patches = torch.cat(parts, dim=0).contiguous()
    if tuple(patches.shape) != (len(rows), *FEATURE_SHAPE):
        raise RuntimeError(f"assembled cache geometry differs for {category}/{view}")

    payload = {
        "patches": patches,
        "relative_paths": [row["relative_path"] for row in rows],
        "class_name": category,
        "metadata": metadata,
        "selection_rule": SELECTION_RULE,
        "dtype": "float32",
        "shape": list(patches.shape),
        "view": view,
        "view_contract": VIEW_CONTRACT[view],
        "source_manifest_digest": manifest_digest,
        "per_image_source_sha256": [row["raw_png_sha256"] for row in rows],
        "per_image_feature_sha256": [tensor_sha256(patches[index]) for index in range(patches.shape[0])],
        "per_image_masked_input_sha256": masked_input_hashes,
        "extraction": {
            "batch_size": batch_size,
            "torch_num_threads": threads,
            "checkpoint_sha256": pinned_checkpoint_sha(),
            "device": str(extractor.device),
            "dataset_seconds": dataset_seconds,
            "forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
            "images_per_second": len(rows) / max(1e-9, time.perf_counter() - started),
            "code_hashes": code_hashes(),
        },
    }
    summary = validate_payload(payload, category, view, rows, metadata)
    summary["mask_pixel_area"] = mask_area_check if view == "masked_rgb16" else None
    payload["mask_pixel_area"] = summary["mask_pixel_area"]

    direct = check_direct_equivalence(extractor, dataset, view, rows, patches, feature_equivalence, batch_size)
    payload["direct_equivalence"] = direct

    temporary = cache_path.with_suffix(cache_path.suffix + ".part")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)
    cache_sha = sha256_file(cache_path)

    reopened = torch.load(cache_path, map_location="cpu", weights_only=False)
    reopened_summary = validate_payload(reopened, category, view, rows, metadata)
    if [tensor_sha256(reopened["patches"][i]) for i in range(min(3, reopened["patches"].shape[0]))] != \
            [tensor_sha256(payload["patches"][i]) for i in range(min(3, reopened["patches"].shape[0]))]:
        raise RuntimeError(f"reopened cache differs for {category}/{view}")

    marker = {
        "schema": "hope_cad_frozen_features_v1_done_marker",
        "class_name": category,
        "view": view,
        "images": len(rows),
        "shape": list(patches.shape),
        "dtype": str(patches.dtype).replace("torch.", ""),
        "cache_sha256": cache_sha,
        "cache_bytes": cache_path.stat().st_size,
        "checkpoint_sha256": pinned_checkpoint_sha(),
        "source_manifest_digest": manifest_digest,
        "batch_size": batch_size,
        "torch_num_threads": threads,
        "direct_equivalence": direct,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(marker_path, marker)
    extraction = payload["extraction"]
    record = {"class_name": category, "view": view, "status": "EXTRACTED",
              "cache_path": str(cache_path), "marker_path": str(marker_path),
              "cache_sha256": cache_sha, "cache_bytes": cache_path.stat().st_size,
              "marker_bytes": marker_path.stat().st_size, **reopened_summary,
              "dataset_seconds": extraction["dataset_seconds"],
              "forward_seconds": extraction["forward_seconds"],
              "total_seconds": extraction["total_seconds"],
              "images_per_second": extraction["images_per_second"],
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    del parts, patches, payload, reopened
    gc.collect()
    return record


def check_direct_equivalence(extractor, dataset, view, rows, patches, feature_equivalence, batch_size: int) -> dict:
    """Direct single-image extraction versus the assembled cache rows (bounded sample)."""
    import torch

    from exps.ol04.hope_outer_learning_pilot_v1 import mask_rgb

    indices = sorted({0, len(rows) // 2, len(rows) - 1})
    checked = []
    worst = {"relative_l2": 0.0, "max_abs": 0.0, "scaled_max_abs": 0.0}
    for index in indices:
        sample = dataset[index]["images"].unsqueeze(0)
        if view == "masked_rgb16":
            sample, _ = mask_rgb(sample)
        direct = extractor.extract_patch_features(sample).detach().cpu().to(torch.float32)[0]
        equivalence = feature_equivalence(direct, patches[index])
        if not equivalence["passed"]:
            raise RuntimeError(f"direct/cache equivalence failed for {rows[index]['relative_path']}: {equivalence}")
        worst["relative_l2"] = max(worst["relative_l2"], equivalence["relative_l2"])
        worst["max_abs"] = max(worst["max_abs"], equivalence["max_abs"])
        worst["scaled_max_abs"] = max(worst["scaled_max_abs"], equivalence["scaled_max_abs"])
        checked.append({"relative_path": rows[index]["relative_path"], **equivalence})
    repeat_batch = torch.stack([dataset[indices[0]]["images"]])
    if view == "masked_rgb16":
        repeat_batch, _ = mask_rgb(repeat_batch)
    first = extractor.extract_patch_features(repeat_batch).detach().cpu()
    second = extractor.extract_patch_features(repeat_batch).detach().cpu()
    return {"checked": checked, "worst": worst, "sample_count": len(indices),
            "repeat_inference_bitwise_equal": bool(torch.equal(first, second)),
            "tolerance_relative_l2": 256 * float(torch.finfo(torch.float32).eps),
            "tolerance_scaled_max_abs": 1024 * float(torch.finfo(torch.float32).eps)}


# --------------------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------------------

def required_free_bytes(view: str, manifest: dict, categories: list[str] | None) -> int:
    """Conservative remaining-space requirement: pending payloads plus the largest in-flight temporary."""
    per_image = FEATURE_SHAPE[0] * FEATURE_SHAPE[1] * 4
    pending = []
    for category in (categories or manifest["categories_in_scope"]):
        rows = dataset_rows_for_category(manifest, category)
        cache_path, marker_path = category_paths(view, category)
        complete = cache_path.is_file() and marker_path.is_file() and \
            load_json(marker_path).get("class_name") == category
        if not complete:
            pending.append(len(rows))
    if not pending:
        return 0
    return (sum(pending) + max(pending)) * per_image


def disk_free_bytes(path: Path) -> int:
    stats = os.statvfs(str(Path(path)))
    return stats.f_bavail * stats.f_frsize


def run_manifest(categories: list[str] | None = None) -> dict:
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_source_manifest(categories)
    atomic_json(FEATURES_ROOT / "source_manifest.json", manifest)
    log(f"source manifest: {manifest['total_source_images']} images, "
        f"{manifest['duplicate_content_check']['duplicate_group_count']} duplicate-content groups")
    return manifest


def run_extract(view: str, batch_size: int, threads: int, device: str, force: bool, categories: list[str] | None) -> dict:
    import torch

    torch.set_num_threads(threads)
    manifest = load_source_manifest()
    from scripts.hope_cad.probe_real_features import feature_metadata

    metadata = feature_metadata(CHECKPOINT, pinned_checkpoint_sha())
    if metadata["block_index"] != 8 or metadata["layer_number"] != 9 or metadata["feature_normalization"] != "none":
        raise RuntimeError("canonical feature metadata differs from the frozen contract")

    expected = required_free_bytes(view, manifest, categories)
    free = disk_free_bytes(FEATURES_ROOT)
    log(f"free space {free / 1024**3:.2f} GiB; conservative requirement for this run {expected / 1024**3:.2f} GiB")
    if free < expected:
        raise RuntimeError(f"insufficient disk space for the {view} view: {free} < {expected} bytes")

    contract_path = FEATURES_ROOT / "extractor_contract.json"
    extractor = build_extractor(device)
    contract = extractor_contract(extractor, batch_size, threads, device)
    if contract_path.is_file() and load_json(contract_path)["checkpoint_sha256"] != contract["checkpoint_sha256"]:
        raise RuntimeError("existing extractor contract belongs to a different checkpoint")
    atomic_json(contract_path, contract)
    log(f"extractor ready on {device}; checkpoint {contract['checkpoint_sha256'][:16]}...")

    order = categories or list(manifest["categories_in_scope"])
    records = []
    for category in order:
        rows = dataset_rows_for_category(manifest, category)
        record = extract_category(extractor, view, category, rows, metadata, manifest["manifest_digest"],
                                  batch_size, threads, force)
        records.append(record)
        log(f"{category}: {record['status']} {record['images']} images, "
            f"{record['cache_bytes'] / 1024**2:.1f} MiB, {record.get('images_per_second', float('nan')):.2f} img/s")
    extractor.close()
    atomic_json(FEATURES_ROOT / f"extraction_summary_{view}.json", {
        "schema": "hope_cad_frozen_features_v1_extraction_summary",
        "view": view,
        "checkpoint_sha256": pinned_checkpoint_sha(),
        "batch_size": batch_size,
        "torch_num_threads": threads,
        "device": device,
        "records": records,
        "total_images": sum(record["images"] for record in records),
        "total_bytes": sum(record["cache_bytes"] for record in records),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    return {"view": view, "records": records}


def r1_split_eligibility(manifest: dict, payloads: dict) -> dict:
    """Check that the frozen three-category identity split used by the project's P1/R1 tooling is
    satisfiable, covered and content-disjoint in both views.

    The role structure is the one already frozen in
    ``scripts.exps.hope_image_synchronous_memory.manifest(seed=0)`` (bottle 170 = 150 update + 20 probe,
    carpet 120 = 100 + 20, hazelnut 120 = 100 + 20).  No new split is invented here; only eligibility for
    the existing structure is measured.
    """
    from scripts.exps.hope_image_synchronous_memory import manifest as project_manifest

    rows, probes, availability = project_manifest(0)
    scoped = set(manifest["categories_in_scope"])
    role_rows = [dict(row, role_group="update") for row in rows if row["class_name"] in scoped]
    role_rows += [dict(row, role_group="probe") for row in probes if row["class_name"] in scoped]
    if not role_rows:
        return {"status": "NOT_APPLICABLE", "reason": "no in-scope category participates in the frozen split"}

    per_category: dict[str, dict] = {}
    used: dict[tuple[str, str], list[int]] = {}
    for row in role_rows:
        key = (row["class_name"], row["role_group"])
        used.setdefault(key, []).append(row["cache_index"])
    for category in manifest["categories_in_scope"]:
        rows_cat = dataset_rows_for_category(manifest, category)
        per_category[category] = {"available_train_good": len(rows_cat), "roles": {}}
        for role_group in ("update", "probe"):
            indices = sorted(used.get((category, role_group), []))
            if not indices:
                continue
            if max(indices) >= len(rows_cat):
                raise RuntimeError(f"frozen split role exceeds available images for {category}/{role_group}")
            paths = [rows_cat[index]["relative_path"] for index in indices]
            digests = [rows_cat[index]["raw_png_sha256"] for index in indices]
            per_category[category]["roles"][role_group] = {
                "images": len(indices), "index_min": indices[0], "index_max": indices[-1],
                "distinct_content": len(set(digests)) == len(digests),
                "clean_rows_present": all(payloads[("clean", category)]["patches"][index].shape == tuple(FEATURE_SHAPE)
                                          for index in indices),
                "masked_rows_present": all(payloads[("masked_rgb16", category)]["patches"][index].shape == tuple(FEATURE_SHAPE)
                                           for index in indices),
            }
            for index, path, digest in zip(indices, paths, digests):
                per_category[category]["roles"][role_group].setdefault("first_path", path)
                per_category[category]["roles"][role_group].setdefault("content_digests", set()).add(digest)

    overlap = {}
    for category in manifest["categories_in_scope"]:
        roles = per_category[category]["roles"]
        if "update" in roles and "probe" in roles:
            shared = roles["update"]["content_digests"] & roles["probe"]["content_digests"]
            overlap[category] = {"shared_content_digests": len(shared), "group_disjoint": not shared}
    prepared = {}
    for category in manifest["categories_in_scope"]:
        roles = per_category[category]["roles"]
        prepared[category] = {name: {k: (sorted(v) if isinstance(v, set) else v) for k, v in role.items()
                                     if k != "content_digests"}
                              for name, role in roles.items()}
    return {
        "status": "ELIGIBLE" if all(item["group_disjoint"] for item in overlap.values()) else "OVERLAP_DETECTED",
        "split_source": "scripts.exps.hope_image_synchronous_memory.manifest(seed=0) frozen role structure",
        "role_total_images": len(role_rows),
        "per_category": prepared,
        "content_disjointness": overlap,
        "available_counts": availability,
        "caveat": ("MVTec AD ships no acquisition/object group labels; only exact-content duplicates and "
                   "filename identity are checkable here."),
    }


def run_verify(direct_samples: int) -> dict:
    import torch

    from scripts.exps.hope_image_synchronous_memory import feature_equivalence
    from exps.ol04.hope_outer_learning_pilot_v1 import mask_rgb
    from scripts.hope_cad.probe_real_features import feature_metadata

    manifest = load_source_manifest()
    metadata = feature_metadata(CHECKPOINT, pinned_checkpoint_sha())
    integrity = {"schema": "hope_cad_frozen_features_v1_cache_integrity",
                 "checkpoint_sha256": pinned_checkpoint_sha(),
                 "source_manifest_digest": manifest["manifest_digest"],
                 "categories_in_scope": manifest["categories_in_scope"],
                 "views": {}, "comparisons": {}}
    payloads = {}
    extractor = build_extractor("cpu")
    for view in VIEWS:
        view_records = {}
        for category in manifest["categories_in_scope"]:
            rows = dataset_rows_for_category(manifest, category)
            cache_path, marker_path = category_paths(view, category)
            if not cache_path.is_file() or not marker_path.is_file():
                raise RuntimeError(f"missing cache or marker for {view}/{category}")
            marker = load_json(marker_path)
            cache_sha = sha256_file(cache_path)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            summary = validate_payload(payload, category, view, rows, metadata)
            recomputed = [tensor_sha256(payload["patches"][index]) for index in range(payload["patches"].shape[0])]
            if recomputed != payload["per_image_feature_sha256"]:
                raise RuntimeError(f"per-image feature hash mismatch for {view}/{category}")
            if marker["cache_sha256"] != cache_sha:
                raise RuntimeError(f"cache file hash mismatch for {view}/{category}")
            if view == "masked_rgb16":
                if len(payload["per_image_masked_input_sha256"]) != len(rows):
                    raise RuntimeError(f"masked contract records differ for {category}")
                if int(payload["view_contract"]["masked_pixel_area"]) != 9216:
                    raise RuntimeError(f"masked pixel area contract differs for {category}")
                stored_area = payload.get("mask_pixel_area")
                if stored_area is not None and int(stored_area) != 9216:
                    raise RuntimeError(f"stored masked pixel area differs for {category}: {stored_area}")
            view_records[category] = {"cache_path": str(cache_path), "cache_sha256": cache_sha,
                                      "cache_bytes": cache_path.stat().st_size, **summary}
            payloads[(view, category)] = payload
        integrity["views"][view] = view_records
        log(f"validated {view}: {len(view_records)} categories")

    for category in manifest["categories_in_scope"]:
        clean = payloads[("clean", category)]
        masked = payloads[("masked_rgb16", category)]
        if clean["relative_paths"] != masked["relative_paths"]:
            raise RuntimeError(f"clean/masked path sequence differs for {category}")
        if clean["per_image_source_sha256"] != masked["per_image_source_sha256"]:
            raise RuntimeError(f"clean/masked source identity differs for {category}")
        if clean["patches"].shape != masked["patches"].shape:
            raise RuntimeError(f"clean/masked geometry differs for {category}")

    direct_report = {}
    dataset_cache = {}
    for category in manifest["categories_in_scope"]:
        rows = dataset_rows_for_category(manifest, category)
        dataset = dataset_cache.setdefault(category, make_image_dataset(rows))
        indices = sorted({0, len(rows) // 2, len(rows) - 1})
        if direct_samples and len(rows) > 2 * direct_samples:
            indices = sorted(set(indices + [len(rows) * index // direct_samples for index in range(direct_samples)]))
        checked = []
        mask_invariants = []
        for view in VIEWS:
            payload = payloads[(view, category)]
            for index in indices:
                sample = dataset[index]["images"].unsqueeze(0)
                if view == "masked_rgb16":
                    sample, mask = mask_rgb(sample)
                    mask_invariants.append({
                        "relative_path": rows[index]["relative_path"],
                        "masked_pixel_area": int(mask.sum()),
                        "disjoint_regions": int(mask.sum()) == 16 * 24 * 24,
                        "fill_normalized_zero": bool(torch.equal(sample[:, :, mask], torch.zeros_like(sample[:, :, mask]))),
                        "unmasked_unchanged": bool(torch.equal(
                            dataset[index]["images"].unsqueeze(0)[:, :, ~mask], sample[:, :, ~mask])),
                        "centers_fully_masked": all(
                            bool(mask[(row - 1) * 8:(row + 2) * 8, (column - 1) * 8:(column + 2) * 8].all())
                            for row, column in [(r, c) for r in (3, 10, 17, 24) for c in (3, 10, 17, 24)]),
                    })
                direct = extractor.extract_patch_features(sample).detach().cpu().to(torch.float32)[0]
                equivalence = feature_equivalence(direct, payload["patches"][index])
                checked.append({"view": view, "relative_path": rows[index]["relative_path"],
                                "passed": equivalence["passed"], "max_abs": equivalence["max_abs"],
                                "relative_l2": equivalence["relative_l2"]})
                if not equivalence["passed"]:
                    raise RuntimeError(f"direct/cache mismatch for {view}/{category} index {index}")
        direct_report[category] = {"checked": checked, "mask_invariants": mask_invariants}
    extractor.close()
    integrity["direct_extraction_checks"] = direct_report
    integrity["r1_split_eligibility"] = r1_split_eligibility(manifest, payloads)
    integrity["comparisons"]["clean_masked_aligned"] = True
    integrity["comparisons"]["total_clean_images"] = sum(record["images"] for record in integrity["views"]["clean"].values())
    integrity["comparisons"]["total_masked_images"] = sum(
        record["images"] for record in integrity["views"]["masked_rgb16"].values())
    atomic_json(FEATURES_ROOT / "cache_integrity.json", integrity)
    log(f"direct-extraction checks: {sum(len(v['checked']) for v in direct_report.values())} pairs passed")
    return integrity


def run_contract(batch_size: int, threads: int, device: str) -> dict:
    """Regenerate the machine-readable extractor contract without extracting features."""
    import torch

    torch.set_num_threads(threads)
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(device)
    contract = extractor_contract(extractor, batch_size, threads, device)
    extractor.close()
    atomic_json(FEATURES_ROOT / "extractor_contract.json", contract)
    log(f"extractor contract refreshed (checkpoint {contract['checkpoint_sha256'][:16]}...)")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild frozen ViT-B/8 feature caches for authorized MVTec classes")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_parser = sub.add_parser("manifest", help="build the canonical source manifest")
    manifest_parser.add_argument("--categories", nargs="*", default=None)

    extract = sub.add_parser("extract", help="extract one feature view")
    extract.add_argument("--view", choices=VIEWS, required=True)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--categories", nargs="*", default=None)

    verify = sub.add_parser("verify", help="reopen and validate every cache")
    verify.add_argument("--direct-samples", type=int, default=3)

    contract = sub.add_parser("contract", help="regenerate the machine-readable extractor contract")
    contract.add_argument("--batch-size", type=int, default=8)
    contract.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    contract.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "manifest":
        run_manifest(args.categories)
    elif args.command == "extract":
        run_extract(args.view, args.batch_size, args.threads, args.device, args.force, args.categories)
    elif args.command == "verify":
        run_verify(args.direct_samples)
    elif args.command == "contract":
        run_contract(args.batch_size, args.threads, args.device)


if __name__ == "__main__":
    main()
