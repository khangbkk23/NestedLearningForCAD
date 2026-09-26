"""Deterministic development manifests derived only from MVTec train/good.

The manifest is intentionally independent from the official test set.  It is
for configuration diagnostics only; the CADIC exact stream must exclude every
DEV path from its coreset and memory updates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _digest(paths: Iterable[str]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_train_good_manifest(
    root_dir: str | Path,
    categories: Iterable[str],
    dev_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return a stable train/dev manifest without opening any test paths.

    Selection is deterministic: category-local sorted paths are assigned by a
    hash of ``seed:category:path``.  The first ``ceil(fraction*n)`` ranked
    paths become DEV; all others are CADIC train paths.
    """
    if not 0.0 < float(dev_fraction) < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    root = Path(root_dir)
    task_records: List[Dict[str, Any]] = []
    all_dev: List[str] = []
    all_train: List[str] = []
    for category in categories:
        files = sorted((root / str(category) / "train" / "good").glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No train/good PNG files for {category}")
        ranked = sorted(
            files,
            key=lambda p: hashlib.sha256(
                f"{seed}:{category}:{p.as_posix()}".encode("utf-8")
            ).hexdigest(),
        )
        n_dev = max(1, int(round(len(files) * float(dev_fraction))))
        dev = [p.as_posix() for p in ranked[:n_dev]]
        train = [p.as_posix() for p in files if p.as_posix() not in set(dev)]
        all_dev.extend(dev)
        all_train.extend(train)
        task_records.append({
            "category": str(category),
            "disk_count": len(files),
            "dev_count": len(dev),
            "train_count": len(train),
            "dev": dev,
            "train": train,
        })
    return {
        "schema": "cadic_dev_manifest_v1",
        "root_dir": str(root),
        "seed": int(seed),
        "dev_fraction": float(dev_fraction),
        "tasks": task_records,
        "dev_count": len(all_dev),
        "train_count": len(all_train),
        "manifest_sha256": _digest(all_dev + all_train),
    }


def write_train_good_manifest(
    root_dir: str | Path,
    categories: Iterable[str],
    output_path: str | Path,
    dev_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    manifest = build_train_good_manifest(root_dir, categories, dev_fraction, seed)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
