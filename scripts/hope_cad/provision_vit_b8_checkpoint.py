# scripts/hope_cad/provision_vit_b8_checkpoint.py
"""Provision and verify the CADIC-compatible ViT-B/8 AugReg ImageNet-21K checkpoint.

The historical local artifact
``checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth``
is not recoverable.  This script rebuilds an auditable, explicitly versioned
replacement from the pinned public source

    https://huggingface.co/timm/vit_base_patch8_224.augreg_in21k
    revision 3c68ba3fef589ef43b862ec19254ecb1146506c3

and never modifies pretrained tensor values.  Both upstream serializations are
downloaded and cross-checked bitwise before a flat state dict compatible with
``timm.create_model("vit_base_patch8_224", pretrained=False, num_classes=0)``
is written for the existing :mod:`models.feature_extractors.cadic_vit_v1`
adapter (which loads with ``strict=True``).

This module is additive: it does not touch any existing scientific core file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

SOURCE_REPO = "timm/vit_base_patch8_224.augreg_in21k"
SOURCE_REVISION = "3c68ba3fef589ef43b862ec19254ecb1146506c3"
SOURCE_URL_TEMPLATE = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"

# Published LFS sha256 values from the Hugging Face model API at the pinned
# revision; these are the upstream integrity anchors for this rebuild.
UPSTREAM_FILES = {
    "pytorch_model.bin": {
        "size": 410475437,
        "sha256": "6c22b721047efbfae19cb5d52e2cf1bd65bd35aa80b65948c8a3c29e46a7efe8",
    },
    "model.safetensors": {
        "size": 410434646,
        "sha256": "2c4839589adbbf701573dd155e1814f8adb145cd4d235bb86a93269369e111ec",
    },
}

CANONICAL_CHECKPOINT = REPO_ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"
HISTORICAL_CHECKPOINT_SHA256 = "ae3012808a9b406a19b799381bd26b253634ad26125937c26d02dbcbbc85dd92"

# Architecture/preprocessing metadata published with the pinned weights (num_classes=21843,
# tag "augreg_in21k").  Recorded for provenance; the feature contract of this project does NOT use the
# generic timm mean/std/bicubic defaults listed there.
UPSTREAM_CONFIG = {
    "filename": "config.json",
    "sha256": "fc1850d333dec2f479a82d4c01fcc10d2cd6540fc3e57a6734ef7cef874467bf",
}

HISTORICAL_SHA_INVESTIGATION = {
    "historical_sha256": HISTORICAL_CHECKPOINT_SHA256,
    "byte_identity_established": False,
    "reason": ("the historical file is absent, so byte-level serialization cannot be compared; "
               "torch.save output depends on the writer's torch version, pickle protocol, tensor storage "
               "layout and dict insertion order, and possibly on whether the payload was wrapped in a "
               "container dict or retained the 21843-class head"),
    "tensor_level_provenance_established": [
        "both upstream serializations (pytorch_model.bin, model.safetensors) match their published "
        "Hugging Face LFS sha256 values at the pinned revision",
        "the two upstream serializations are bitwise identical tensor-by-tensor",
        "the selected .pt payload loads with strict=True into timm vit_base_patch8_224 num_classes=0",
        "upstream config.json declares architecture vit_base_patch8_224, tag augreg_in21k, num_features 768",
    ],
    "consequence": ("the rebuilt checkpoint is a NEW, explicitly versioned frozen representation; it must not be "
                    "represented as the exact historical artifact, and historical hashes in production code must "
                    "not be edited to accept it"),
}


def fetch_upstream_config(destination: Path) -> dict:
    """Download and hash-check the upstream config.json at the pinned revision."""
    url = SOURCE_URL_TEMPLATE.format(repo=SOURCE_REPO, revision=SOURCE_REVISION, filename=UPSTREAM_CONFIG["filename"])
    raw = urllib.request.urlopen(url, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != UPSTREAM_CONFIG["sha256"]:
        raise RuntimeError("upstream config.json sha256 differs from the pinned value")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    return {"url": url, "sha256": digest, "bytes": len(raw), "path": str(destination)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pinned(filename: str, destination: Path) -> dict:
    """Stream one pinned upstream artifact, verifying size and sha256."""
    expected = UPSTREAM_FILES[filename]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        record = {
            "path": str(destination),
            "size": destination.stat().st_size,
            "sha256": sha256_file(destination),
            "reused_existing_download": True,
        }
    else:
        url = SOURCE_URL_TEMPLATE.format(repo=SOURCE_REPO, revision=SOURCE_REVISION, filename=filename)
        digest = hashlib.sha256()
        temporary = destination.with_suffix(destination.suffix + ".part")
        written = 0
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
                written += len(chunk)
        record = {"path": str(destination), "size": written, "sha256": digest.hexdigest(),
                  "url": url, "reused_existing_download": False}
        os.replace(temporary, destination)
    record["expected_size"] = expected["size"]
    record["expected_sha256"] = expected["sha256"]
    record["size_matches"] = record["size"] == expected["size"]
    record["sha256_matches"] = record["sha256"] == expected["sha256"]
    if not (record["size_matches"] and record["sha256_matches"]):
        raise RuntimeError(f"upstream artifact integrity mismatch for {filename}: {record}")
    return record


def load_upstream_tensors(pytorch_bin: Path, safetensors_path: Path) -> tuple[dict, dict]:
    from safetensors.torch import load_file as load_safetensors

    binary = torch.load(pytorch_bin, map_location="cpu", weights_only=True)
    if not isinstance(binary, dict):
        raise ValueError("pytorch_model.bin is not a state-dict mapping")
    safe = load_safetensors(str(safetensors_path), device="cpu")
    return binary, safe


def cross_check_serializations(binary: dict, safe: dict) -> dict:
    if set(binary) != set(safe):
        raise RuntimeError("upstream serializations disagree on key sets")
    mismatches = []
    nonfinite = []
    for key in sorted(binary):
        left, right = binary[key], safe[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatches.append({"key": key, "reason": "geometry"})
            continue
        if not torch.equal(left, right):
            mismatches.append({"key": key, "reason": "value", "max_abs": float((left.double() - right.double()).abs().max())})
        if not bool(torch.isfinite(left).all()):
            nonfinite.append(key)
    if mismatches:
        raise RuntimeError(f"upstream serializations differ: {mismatches[:5]}")
    if nonfinite:
        raise RuntimeError(f"non-finite upstream tensors: {nonfinite}")
    return {"keys": len(binary), "bitwise_identical": True, "nonfinite_tensors": nonfinite}


def to_extractor_state_dict(upstream: dict, model_keys: set) -> tuple[dict, dict]:
    """Select exactly the keys the CADIC adapter's num_classes=0 model requires."""
    state = {key: value.detach().clone() for key, value in upstream.items() if key in model_keys}
    dropped = sorted(key for key in upstream if key not in model_keys)
    missing = sorted(model_keys - set(state))
    if missing:
        raise RuntimeError(f"upstream artifact lacks required tensors: {missing}")
    return state, {"dropped_upstream_keys": dropped, "required_keys": sorted(model_keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision the pinned ViT-B/8 AugReg IN21K checkpoint")
    parser.add_argument("--download-dir", default="/tmp/opencode/vit_b8_upstream")
    parser.add_argument("--output", default=str(CANONICAL_CHECKPOINT))
    parser.add_argument("--provenance", default=str(REPO_ROOT / "results/hope_cad/frozen_features_v1/checkpoint_provenance.json"))
    parser.add_argument("--force", action="store_true", help="rewrite the canonical checkpoint even if present")
    args = parser.parse_args()

    import timm

    download_dir = Path(args.download_dir)
    upstream = {name: download_pinned(name, download_dir / name) for name in UPSTREAM_FILES}
    binary, safe = load_upstream_tensors(download_dir / "pytorch_model.bin", download_dir / "model.safetensors")
    cross = cross_check_serializations(binary, safe)

    model = timm.create_model("vit_base_patch8_224", pretrained=False, num_classes=0, img_size=224)
    state, selection = to_extractor_state_dict(binary, set(model.state_dict()))
    model.load_state_dict(state, strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and not args.force:
        existing = torch.load(output, map_location="cpu", weights_only=True)
        if not isinstance(existing, dict) or set(existing) != set(state):
            raise RuntimeError(f"existing canonical checkpoint {output} has a different key set; refusing to overwrite")
        differing = [key for key in state if not torch.equal(existing[key], state[key])]
        if differing:
            raise RuntimeError(f"existing canonical checkpoint {output} differs from the pinned upstream tensors: {differing[:5]}")
    else:
        temporary = output.with_suffix(output.suffix + ".part")
        torch.save(state, temporary)
        os.replace(temporary, output)

    provisional = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_url_template": SOURCE_URL_TEMPLATE,
        "upstream_files": upstream,
        "upstream_cross_check": cross,
        "upstream_config": fetch_upstream_config(
            Path(args.provenance).parent / "upstream_config.json"),
        "serialization": {
            "format": "torch.save of a flat state dict (pickle protocol default of this torch build)",
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "num_classes": 0,
            "dropped_upstream_keys": selection["dropped_upstream_keys"],
            "key_count": len(state),
            "tensor_dtype": "float32",
            "numerical_transform": "none (identity copy of upstream tensors)",
        },
        "canonical_path": str(output),
        "canonical_sha256": sha256_file(output),
        "canonical_bytes": output.stat().st_size,
        "historical_checkpoint_sha256": HISTORICAL_CHECKPOINT_SHA256,
        "historical_sha_reproduced": sha256_file(output) == HISTORICAL_CHECKPOINT_SHA256,
        "historical_sha_investigation": HISTORICAL_SHA_INVESTIGATION,
        "parameter_bytes": sum(v.numel() * v.element_size() for v in state.values()),
    }
    provenance_path = Path(args.provenance)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provisional, indent=2) + "\n")
    print(json.dumps(provisional, indent=2))


if __name__ == "__main__":
    main()
