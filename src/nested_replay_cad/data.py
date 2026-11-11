"""Prospectively fixed full-stream metadata and read-only development cohort."""

import hashlib
import json
from pathlib import Path

SALT = "FND-FULLTEST-SEED0-02:20261107"
ORDER = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def selection(full):
    rows = full["records"]
    if (
        len(rows) != 3629
        or tuple(dict.fromkeys(r["task_audit_only"] for r in rows)) != ORDER
    ):
        raise ValueError("Expected registered full3629 task order")
    if [r["arrival"] for r in rows] != list(range(1, 3630)):
        raise ValueError("Full stream arrivals")
    if len({r["opaque_id"] for r in rows}) != 3629 or any(
        r["split"] != "train/good" for r in rows
    ):
        raise ValueError("Full stream identities/split")
    boundaries = []
    for category in ORDER:
        members = [r for r in rows if r["task_audit_only"] == category]
        if not members or any(
            not r["relative_path"].startswith(category + "/train/good/")
            for r in members
        ):
            raise ValueError("Task source binding")
        boundaries.append(max(r["arrival"] for r in members))
    return dict(
        schema="fnd_full15_selection_v1",
        records=rows,
        external_records=[],
        categories=list(ORDER),
        boundaries=boundaries,
        full_manifest_sha256=hashlib.sha256(canonical(full).encode()).hexdigest(),
    )


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def select_paths(paths):
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate test paths")
    chosen = []
    for category in ORDER:
        rows = sorted(p for p in paths if p.split("/")[0] == category)
        if (
            not rows
            or not any("/test/good/" in p for p in rows)
            or not any("/test/good/" not in p for p in rows)
        ):
            raise ValueError("Full-test category binary population " + category)
        chosen.extend(rows)
    return chosen


def prepare(dataset_root, full, output_dir):
    from io import BytesIO

    from nested_replay_cad.core.data.manifest import (
        DATASET_ID,
        _entries,
        _image_class,
        checked_path,
        checked_root,
        read_verified_bytes,
    )
    from nested_replay_cad.core.identity import json_digest

    dataset = checked_root(dataset_root)
    paths = []
    for category in ORDER:
        for defect in sorted(
            _entries(checked_path(dataset, category + "/test")), key=lambda p: p.name
        ):
            if not defect.is_dir():
                raise ValueError("Expected defect directory")
            for path in sorted(_entries(defect), key=lambda p: p.name):
                if not path.is_file() or path.suffix != ".png":
                    raise ValueError("Expected test PNG")
                paths.append(f"{category}/test/{defect.name}/{path.name}")
    if len(paths) != 1725 or len(set(paths)) != 1725:
        raise ValueError("Expected exactly1725 unique full-test PNG paths")
    Image = _image_class()
    records = []
    for path in select_paths(paths):
        raw = read_verified_bytes(dataset, path)
        if len(raw) > 16 * 1024**2:
            raise ValueError("Development image cap")
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.width * image.height > 4_000_000:
                raise ValueError("Development PNG geometry")
            shape = [image.height, image.width, len(image.getbands())]
        parts = path.split("/")
        digest = hashlib.sha256(raw).hexdigest()
        label = int(parts[2] != "good")
        row = dict(
            image_id=json_digest(
                dict(
                    dataset_id=DATASET_ID,
                    split="test-development",
                    path=path,
                    sha256=digest,
                )
            ),
            path=path,
            image_sha256=digest,
            bytes=len(raw),
            shape=shape,
            category=parts[0],
            image_label=label,
            mask_path=None,
            mask_sha256=None,
            mask_bytes=0,
        )
        if label:
            mask_path = "/".join(
                (parts[0], "ground_truth", parts[2], parts[3][:-4] + "_mask.png")
            )
            mask = read_verified_bytes(dataset, mask_path)
            if len(mask) > 16 * 1024**2:
                raise ValueError("Mask byte cap")
            with Image.open(BytesIO(mask)) as image:
                if (
                    image.format != "PNG"
                    or image.mode not in ("1", "L")
                    or [image.height, image.width] != shape[:2]
                ):
                    raise ValueError("Mask header")
            with Image.open(BytesIO(mask)) as image:
                if image.getbbox() is None:
                    raise ValueError("Original anomaly mask is empty")
                row["mask_survives_nearest256"] = (
                    image.resize((256, 256), Image.Resampling.NEAREST).getbbox()
                    is not None
                )
            row.update(
                mask_path=mask_path,
                mask_sha256=hashlib.sha256(mask).hexdigest(),
                mask_bytes=len(mask),
            )
        records.append(row)
    body = dict(
        schema="fnd_full15_readonly_development_v1",
        selection_salt=SALT,
        selection="all official test PNGs, sorted category/defect/file, seen categories at each boundary",
        scope="development only; evaluator has no learner callback",
        records=records,
    )
    if len(records) != 1725 or len({r["image_id"] for r in records}) != 1725:
        raise ValueError("Full-test manifest identities")
    expected_masks = {r["mask_path"] for r in records if r["image_label"]}
    actual_masks = set()
    for category in ORDER:
        for defect in _entries(checked_path(dataset, category + "/ground_truth")):
            if not defect.is_dir():
                raise ValueError("Ground-truth defect directory")
            for path in _entries(defect):
                if not path.is_file() or path.suffix != ".png":
                    raise ValueError("Ground-truth PNG")
                actual_masks.add(
                    category + "/ground_truth/" + defect.name + "/" + path.name
                )
    if actual_masks != expected_masks:
        raise ValueError("Missing/orphan/extra ground-truth mask")
    chosen = selection(full)
    counts = {c: sum(r["category"] == c for r in records) for c in ORDER}
    observations = sum(counts[c] * (15 - i) for i, c in enumerate(ORDER))
    body.update(
        category_counts=counts,
        observations_per_arm=observations,
        all_test_images=len(records),
        mask_resolution=[256, 256],
        image_label_counts={
            str(y): sum(r["image_label"] == y for r in records) for y in (0, 1)
        },
        mask_mapping_verified=True,
        anomalous_masks_lost_at256=sum(
            r.get("mask_survives_nearest256") is False for r in records
        ),
        training_split="train/good",
        evaluation_only=True,
    )
    estimated_raw = 3 * (observations * (256 * 256 * 5 + 1024) + 1400000000)
    if estimated_raw > 18 * 1024**3:
        raise ValueError("Prospective full-test raw output estimate exceeds18GiB")
    print(
        "FULLTEST_POPULATION",
        json.dumps(
            dict(
                images=len(records),
                counts=counts,
                observations_per_arm=observations,
                estimated_raw_bytes=estimated_raw,
            )
        ),
        flush=True,
    )
    (output_dir / "full15_selection.json").write_text(canonical(chosen))
    (output_dir / "full15_cohort.json").write_text(canonical(body))
    print(
        "FULL15_INPUTS",
        json.dumps(
            dict(
                arrivals=3629,
                categories=list(ORDER),
                boundaries=chosen["boundaries"],
                cohort_images=len(records),
                cohort_sha256=hashlib.sha256(canonical(body).encode()).hexdigest(),
            )
        ),
        flush=True,
    )
    return chosen
