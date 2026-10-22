"""Portable, canonical manifests built exclusively from train/good."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def manifest_digest(manifest: dict) -> str:
    return canonical_digest({k: v for k, v in manifest.items() if k != "digest"})

def validate_manifest(manifest: dict, protocol: dict, seed: int) -> None:
    if manifest.get("digest") != manifest_digest(manifest):
        raise ValueError("Training manifest digest mismatch")
    if (manifest["task_order"] != protocol["task_order"] or manifest["seed"] != seed
            or manifest["protocol_id"] != protocol["id"]):
        raise ValueError("Manifest does not match resolved protocol/seed")
    seen = set()
    for row in manifest["entries"]:
        task_id = row["task_id"]
        if not isinstance(task_id, int) or not 0 <= task_id < len(manifest["task_order"]):
            raise ValueError("Invalid task id")
        task = manifest["task_order"][task_id]
        path = PurePosixPath(row["relative_path"])
        if (row["task_name"] != task or row["split"] != "train" or path.is_absolute()
                or len(path.parts) != 4 or path.parts[:3] != (task, "train", "good")
                or ".." in path.parts or path.suffix.lower() != ".png"
                or path.as_posix() in seen):
            raise ValueError("Manifest must contain unique train/good PNG paths only")
        seen.add(path.as_posix())
    for task_id in range(len(manifest["task_order"])):
        if not any(e["task_id"] == task_id for e in manifest["entries"]):
            raise ValueError("Manifest contains an empty task")

def build_training_manifest(protocol: dict, seed: int, git_commit: str) -> dict:
    root = Path(protocol["dataset"]["root"]).resolve()
    entries = []
    for task_id, task in enumerate(protocol["task_order"]):
        folder = root / task / "train" / "good"
        files = sorted(folder.glob("*.png"), key=lambda p: p.name)
        if not files:
            raise FileNotFoundError(f"No train/good PNG samples: {folder}")
        for path in files:
            relative = path.relative_to(root).as_posix()
            if path.resolve() != root / relative:
                raise ValueError(f"Symlinked training sample rejected: {relative}")
            entries.append(dict(task_id=task_id, task_name=task, split="train", relative_path=relative))
    manifest = dict(schema_version=1, protocol_id=protocol["id"], dataset_name="MVTec AD",
                    seed=seed, task_order=list(protocol["task_order"]), git_commit=git_commit,
                    ordering_rule="task_order_then_lexicographic_relative_path", entries=entries)
    manifest["digest"] = manifest_digest(manifest)
    validate_manifest(manifest, protocol, seed)
    return manifest