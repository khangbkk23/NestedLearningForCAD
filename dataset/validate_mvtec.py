#!/usr/bin/env python3
"""Read-only validation of the local MVTec AD tree (data/mvtec), no re-download.

Mirrors what the current pipeline consumes:

  1. Tree layout required by dataset/download_mvtec.py
     (15 canonical categories x {train/good, test, ground_truth}), plus the
     test/mask pairing required by dataset/benchmark_protocol_v1.py at eval time
     (every defective test PNG has ground_truth/<defect>/<stem>_mask.png).
  2. No symlinks anywhere in the tree.
  3. train/good filename listings of all 15 categories reproduce the digest
     recorded by scripts/hope_cad/rebuild_frozen_features_v1.py
     (results/hope_cad/frozen_features_v1/source_manifest.json).
  4. Byte-level SHA-256 + size of the train/good images consumed by the current
     R1/P0/P1 runs match that same manifest.

Exit code: 0 = PASS, 1 = FAIL. No arguments, no network access.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
TARGET_DIR = DATA_DIR / "mvtec"
SOURCE_MANIFEST = REPO_ROOT / "results/hope_cad/frozen_features_v1/source_manifest.json"
PROTOCOL_PATH = REPO_ROOT / "conf/benchmarks/protocols/mvtec_1x15_v1.yaml"

try:  # run as `python dataset/validate_mvtec.py`
    from download_mvtec import CATEGORIES, validate_tree
    from benchmark_manifest_v1 import build_training_manifest
except ImportError:  # run as `python -m dataset.validate_mvtec`
    from dataset.download_mvtec import CATEGORIES, validate_tree
    from dataset.benchmark_manifest_v1 import build_training_manifest

PROBLEMS: list[str] = []
WARNINGS: list[str] = []


def problem(message: str) -> None:
    PROBLEMS.append(message)
    print(f"    [FAIL] {message}")


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"    [WARN] {message}")


def ok(message: str) -> None:
    print(f"    [OK]   {message}")


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def count_pngs(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.rglob("*.png") if p.is_file())


def check_tree() -> None:
    print(f"[1] Tree and test/mask pairing ({TARGET_DIR})")
    if not TARGET_DIR.is_dir():
        problem(f"missing dataset directory: {TARGET_DIR}")
        return
    if validate_tree(TARGET_DIR):
        ok("download_mvtec.validate_tree: all 15 categories have train/good, test, ground_truth")
    else:
        problem("download_mvtec.validate_tree failed for this tree")

    print(f"    {'category':<12} {'train_good':>10} {'test_png':>9} {'test_good':>9} {'gt_masks':>9}  pairing")
    totals = [0, 0, 0, 0]
    for category in CATEGORIES:
        root = TARGET_DIR / category
        train_dir = root / "train" / "good"
        test_dir = root / "test"
        gt_dir = root / "ground_truth"

        train_count = sum(1 for p in train_dir.glob("*.png") if p.is_file()) if train_dir.is_dir() else 0
        test_count = count_pngs(test_dir)
        good_count = sum(1 for p in (test_dir / "good").glob("*.png") if p.is_file()) if (test_dir / "good").is_dir() else 0
        gt_count = count_pngs(gt_dir)
        defective_count = test_count - good_count

        missing_masks = []
        if test_dir.is_dir() and gt_dir.is_dir():
            for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                if defect_dir.name == "good":
                    continue
                for image in sorted(defect_dir.glob("*.png")):
                    mask = gt_dir / defect_dir.name / f"{image.stem}_mask.png"
                    if not mask.is_file():
                        missing_masks.append(mask.relative_to(TARGET_DIR).as_posix())

        pairing = "OK" if not missing_masks else f"MISSING {len(missing_masks)}"
        print(f"    {category:<12} {train_count:>10} {test_count:>9} {good_count:>9} {gt_count:>9}  {pairing}")

        totals[0] += train_count
        totals[1] += test_count
        totals[2] += good_count
        totals[3] += gt_count

        if not train_dir.is_dir() or train_count == 0:
            problem(f"{category}: no train/good PNG images")
        if not test_dir.is_dir() or test_count == 0:
            problem(f"{category}: no test PNG images")
        if missing_masks:
            shown = ", ".join(missing_masks[:5]) + (" ..." if len(missing_masks) > 5 else "")
            problem(f"{category}: {len(missing_masks)} defective test image(s) without ground-truth mask: {shown}")
        elif gt_count != defective_count:
            warn(f"{category}: ground_truth masks {gt_count} != defective test images {defective_count}")
        if train_dir.is_dir():
            extras = [p.name for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() != ".png"]
            if extras:
                warn(f"{category}: non-PNG file(s) in train/good are ignored by manifest build: "
                     + ", ".join(extras[:5]) + (" ..." if len(extras) > 5 else ""))
        if test_dir.is_dir():
            strays = [p.name for p in test_dir.iterdir() if p.is_file()]
            if strays:
                warn(f"{category}: file(s) directly under test/ are ignored by the eval loader: "
                     + ", ".join(strays[:5]) + (" ..." if len(strays) > 5 else ""))

    print(f"    {'TOTAL':<12} {totals[0]:>10} {totals[1]:>9} {totals[2]:>9} {totals[3]:>9}")


def check_symlinks() -> None:
    print("[2] Symlink scan")
    links = []
    for dirpath, dirnames, filenames in os.walk(TARGET_DIR, followlinks=False):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                links.append(os.path.relpath(full, TARGET_DIR))
    if links:
        shown = ", ".join(links[:5]) + (" ..." if len(links) > 5 else "")
        problem(f"{len(links)} symlink(s) found; manifest build and training reject them: {shown}")
    else:
        ok("no symlinks")


def check_protocol_digest(source_manifest: dict | None) -> None:
    print("[3] train/good filename listings vs frozen-feature rebuild digest")
    if source_manifest is None:
        warn("reference manifest missing; digest check skipped")
        return
    try:
        import yaml
    except ImportError:
        warn("PyYAML unavailable; digest check skipped")
        return
    if not PROTOCOL_PATH.is_file():
        warn(f"protocol file missing: {PROTOCOL_PATH}")
        return
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text())
    protocol["dataset"]["root"] = str(TARGET_DIR)
    try:
        rebuilt = build_training_manifest(protocol, 0, "feature_rebuild_v1")
    except Exception as exc:  # missing/empty categories land here too
        problem(f"cannot rebuild training manifest from current tree: {exc}")
        return
    recorded = source_manifest.get("protocol_manifest_digest")
    if rebuilt["digest"] == recorded:
        ok(f"digest matches recorded value ({recorded[:16]}...): all 15 train/good listings unchanged")
    else:
        problem(f"train/good listings changed since the rebuild "
                f"(recorded {recorded[:16]}..., recomputed {rebuilt['digest'][:16]}...)")


def check_consumed_images(source_manifest: dict | None) -> None:
    print("[4] Consumed train/good image bytes (source_manifest.json)")
    if source_manifest is None:
        warn("reference manifest missing; byte check skipped")
        return

    payload = {k: v for k, v in source_manifest.items() if k != "manifest_digest"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recomputed == source_manifest.get("manifest_digest"):
        ok(f"manifest self-digest valid ({recomputed[:16]}...)")
    else:
        problem("source_manifest.json self-digest mismatch (file was modified after the rebuild)")

    entries = source_manifest.get("entries", [])
    if not entries:
        problem("source_manifest.json contains no entries")
        return

    missing, bad_size, bad_sha = [], [], []
    verified_bytes = 0
    for index, entry in enumerate(entries, 1):
        relative = entry["relative_path"]
        path = TARGET_DIR / relative
        if not path.is_file():
            missing.append(relative)
            continue
        size = path.stat().st_size
        if size != entry["raw_png_bytes"]:
            bad_size.append(f"{relative} ({size} != {entry['raw_png_bytes']} bytes)")
            continue
        if sha256_file(path) != entry["raw_png_sha256"]:
            bad_sha.append(relative)
            continue
        verified_bytes += size
        if index % 200 == 0:
            print(f"    ... hashed {index}/{len(entries)}")

    for label, items in (("missing", missing), ("size mismatch", bad_size), ("SHA-256 mismatch", bad_sha)):
        if items:
            shown = ", ".join(items[:5]) + (" ..." if len(items) > 5 else "")
            problem(f"{len(items)} file(s) {label}: {shown}")

    verified = len(entries) - len(missing) - len(bad_size) - len(bad_sha)
    if verified == len(entries):
        ok(f"{verified}/{len(entries)} files verified (sha256 + size, {verified_bytes / 2**20:.1f} MiB)")
    else:
        warn(f"only {verified}/{len(entries)} files verified ({verified_bytes / 2**20:.1f} MiB verified)")


def main() -> int:
    print("== MVTec dataset validation (read-only, no download) ==")
    print(f"root: {TARGET_DIR}")
    env_root = os.environ.get("MVTEC_ROOT")
    if env_root:
        resolved = Path(env_root).expanduser().resolve()
        if resolved != TARGET_DIR.resolve():
            warn(f"MVTEC_ROOT={env_root} points elsewhere; this run validated {TARGET_DIR} only")

    source_manifest = None
    if SOURCE_MANIFEST.is_file():
        source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    else:
        warn(f"reference manifest not found: {SOURCE_MANIFEST}")

    check_tree()
    if TARGET_DIR.is_dir():
        check_symlinks()
        check_protocol_digest(source_manifest)
        check_consumed_images(source_manifest)

    print()
    if PROBLEMS:
        print(f"[RESULT] FAIL - {len(PROBLEMS)} problem(s), {len(WARNINGS)} warning(s)")
        for item in PROBLEMS:
            print(f"  - {item}")
        return 1
    print(f"[RESULT] PASS - data/mvtec is consistent with the current run configuration "
          f"({len(WARNINGS)} warning(s))")
    for item in WARNINGS:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
