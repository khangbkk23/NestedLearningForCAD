"""Extract only permitted first-task normal-prefix features for W6 calibration."""
import argparse
from collections import OrderedDict
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from dataset.benchmark_manifest_v1 import canonical_digest
from dataset.benchmark_protocol_v1 import _ImageDataset
from dataset.benchmark_protocol_v2 import (
    calibration_roles,
    file_hash,
    validate_prefix_manifest,
)
from dataset.hope_mask_views import mask_normalized_rgb
from models.feature_extractors.cadic_vit_v1 import (
    CADICViTConfig,
    CADICViTFeatureExtractor,
)
from models.hope_cad_v2.state import tensor_digest
from models.hope_cad_v2.teacher_head import frozen_teacher
from scripts.hope_cad_v2.artifacts import provenance, write_json
from training.benchmark_config_v1 import load_configs
from training.benchmark_config_v2 import validate_configs


def prepare(protocol_path, method_path, run_dir, output, device, batch_size=None):
    protocol, method = load_configs(protocol_path, method_path)
    resolved = validate_configs(protocol, method)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest_train.json").read_text(encoding="utf-8"))
    lock = json.loads((run_dir / "baseline_lock.json").read_text(encoding="utf-8"))
    if lock["identity"]["manifest_digest"] != manifest["digest"]:
        raise ValueError("baseline lock and training manifest differ")
    validate_prefix_manifest(manifest, protocol, verify_bytes=True)
    role_manifest = calibration_roles(manifest)
    permitted = {
        row["relative_path"]
        for role in ("meta_support", "meta_query", "normal_validation")
        for row in role_manifest["roles"][role]
    }
    rows = [row for row in manifest["entries"] if row["relative_path"] in permitted]
    if {row["relative_path"] for row in rows} != permitted:
        raise ValueError("role manifest does not resolve to permitted train images")

    preprocessing = method["preprocessing"]
    dataset = _ImageDataset(
        protocol["dataset"]["root"],
        rows,
        preprocessing["image_size"],
        preprocessing["mean"],
        preprocessing["std"],
        training=True,
    )
    effective_batch_size = batch_size or method["runtime"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    extractor_config = dict(method["extractor"])
    extractor_config.update(
        image_size=preprocessing["image_size"],
        mean=tuple(preprocessing["mean"]),
        std=tuple(preprocessing["std"]),
    )
    extractor = CADICViTFeatureExtractor(
        CADICViTConfig(**{
            key: value
            for key, value in extractor_config.items()
            if key in CADICViTConfig.__dataclass_fields__
        }),
        device,
    )

    images = OrderedDict()
    started = time.perf_counter()
    try:
        for batch in loader:
            rgb = batch["images"].to(device)
            paths = batch["relative_path"]
            with torch.no_grad():
                teacher = frozen_teacher(extractor.extract_patch_features(rgb))
                views = []
                for view in range(4):
                    masked = mask_normalized_rgb(rgb, view)
                    features = extractor.extract_patch_features(masked)
                    views.append(features.detach().cpu())
            for index, relative_path in enumerate(paths):
                tensors = {
                    f"masked_{view}": views[view][index:index + 1].contiguous()
                    for view in range(4)
                }
                tensors["teacher"] = teacher[index:index + 1].detach().cpu().contiguous()
                images[relative_path] = {
                    "sha256": tensor_digest((tensors,)),
                    "tensors": tensors,
                }
    finally:
        extractor.close()

    expected_keys = {
        row["relative_path"]
        for role in ("meta_support", "meta_query", "normal_validation")
        for row in role_manifest["roles"][role]
    }
    if set(images) != expected_keys:
        raise RuntimeError("feature extraction escaped or omitted permitted role identities")

    source_files = (
        "conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml",
        "conf/benchmarks/methods/cadic_matched_prefix_v2.yaml",
        "dataset/benchmark_protocol_v2.py",
        "dataset/hope_mask_views.py",
        "models/feature_extractors/cadic_vit_v1.py",
        "models/hope_cad_v2/teacher_head.py",
        "scripts/hope_cad_v2/prepare_prefix_features.py",
        "docs/main_spec/architecture_decision_v3.md",
        "pixi.lock",
        "pixi.toml",
        "requirements.txt",
    )
    source_hashes = {name: file_hash(ROOT / name) for name in source_files}
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "timm": version("timm"),
        "safetensors": version("safetensors"),
        "cuda_device": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "pixi_lock_sha256": source_hashes["pixi.lock"],
    }
    feature_identity = {
        "design_id": "hope_cad_v3_masked_teacher_cms",
        "backbone_sha256": method["extractor"]["checkpoint_sha256"],
        "backbone_identity": method["extractor"]["checkpoint_identity"],
        "preprocessing_sha256": canonical_digest(method["preprocessing"]),
        "mask_sha256": file_hash(ROOT / "dataset/hope_mask_views.py"),
        "source_sha256": source_hashes,
        "environment": environment,
        "extraction_batch_size": effective_batch_size,
        "manifest_digest": manifest["digest"],
        "roles_digest": role_manifest["digest"],
        "baseline_lock_digest": lock["digest"],
    }
    bundle = {
        "schema": "hope_frozen_prefix_features_v3",
        "role_manifest": role_manifest,
        "feature_identity": feature_identity,
        "environment": environment,
        "images": dict(images),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(output)
    evidence = {
        "status": "features_prepared_not_selected",
        "reportable": False,
        "protocol_eligible": resolved["reportable"],
        "protocol_id": protocol["id"],
        "method_id": method["id"],
        "manifest_digest": manifest["digest"],
        "roles_digest": role_manifest["digest"],
        "baseline_lock_digest": lock["digest"],
        "feature_identity": feature_identity,
        "feature_count": len(images),
        "role_counts": {
            role: len(role_manifest["roles"][role])
            for role in ("meta_support", "meta_query", "normal_validation", "normal_probe")
        },
        "probe_features_extracted": False,
        "official_test_accessed": False,
        "bundle_sha256": file_hash(output),
        "bundle_bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "source": provenance(),
    }
    write_json(output.with_suffix(".json"), evidence)
    print(json.dumps({key: evidence[key] for key in (
        "status", "feature_count", "role_counts", "probe_features_extracted",
        "official_test_accessed", "bundle_sha256", "bundle_bytes", "elapsed_seconds",
    )}, indent=2))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="conf/benchmarks/protocols/mvtec_1x15_prefix_cal_v2.yaml")
    parser.add_argument("--method", default="conf/benchmarks/methods/cadic_matched_prefix_v2.yaml")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="prepared W1 run directory containing the immutable lock and manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    prepare(args.protocol, args.method, args.run_dir, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()