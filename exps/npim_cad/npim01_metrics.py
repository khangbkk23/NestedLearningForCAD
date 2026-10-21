# exps/npim_cad/npim01_metrics.py
"""EXP-NPIM-01 native metrics and diagnostics (E2 only).

The module is imported by the evaluator but its mask-reading functions are only
called after the E1 prediction seal has been verified.  Point Pixel AP uses the
repository's exact tie-aware AP routine; image AUROC uses the exact image-level
routine.  Whole-image paired bootstrap is provided for image AUROC (all groups)
and native Pixel AP (selected primary/acquisition groups) with an explicitly
conservative paired bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from exps.task_p.p1_readout import (
    BoundedPixelBootstrap,
    bootstrap_weights,
    count_metrics,
    image_metrics,
    score_histogram,
)
from models.hope_cad.npim_cad import native_map

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_image(root: Path, record) -> torch.Tensor:
    """Decode one admitted DEV raw image exactly as the frozen contract."""

    path = root / record.relative_path
    if sha256_file(path) != record.sha256:
        raise ValueError(f"raw DEV image changed: {record.relative_path}")
    with Image.open(path) as source:
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(array)
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    return ((tensor - mean) / std)[None]


def load_native_mask(root: Path, record) -> np.ndarray:
    """Read one native annotation mask. E2 only, after the prediction seal."""

    h, w = record.hw
    if record.label == 0:
        return np.zeros((h, w), dtype=bool)
    if not record.mask_path or not record.mask_sha256:
        raise ValueError(f"anomalous DEV record lacks a pinned mask: {record.relative_path}")
    path = root / record.mask_path
    if sha256_file(path) != record.mask_sha256:
        raise ValueError(f"native mask content changed: {record.mask_path}")
    with Image.open(path) as source:
        mask = np.array(source)
    if mask.shape != (h, w):
        raise ValueError(f"native mask geometry mismatch: {mask.shape} != {(h, w)}")
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError("native mask has unresolved values")
    binary = mask > 0
    if not bool(binary.any()):
        raise ValueError("anomalous DEV image has an empty native mask")
    return binary


def native_scores(patch_scores: np.ndarray, hw: Sequence[int]) -> np.ndarray:
    return native_map(torch.from_numpy(np.asarray(patch_scores, dtype=np.float32)), hw).numpy()


def group_point_metrics(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    image_scores: Sequence[float],
) -> Dict[str, Any]:
    """Exact point metrics for one boundary × arm × category group."""

    if not (len(maps) == len(masks) == len(labels) == len(image_scores)):
        raise ValueError("group arrays differ in length")
    flat_scores = np.concatenate([np.asarray(m, dtype=np.float64).ravel() for m in maps])[None, :]
    flat_masks = np.concatenate([np.asarray(m, dtype=bool).ravel() for m in masks])[None, :]
    histogram = score_histogram(flat_scores, flat_masks)
    pixel = count_metrics(histogram["positive"][::-1], histogram["total"][::-1])
    image = image_metrics(np.asarray(labels), np.asarray(image_scores))
    normal = np.asarray(image_scores)[np.asarray(labels) == 0]
    anomaly = np.asarray(image_scores)[np.asarray(labels) == 1]
    return {
        "images": int(len(labels)),
        "normal_images": int((np.asarray(labels) == 0).sum()),
        "anomaly_images": int((np.asarray(labels) == 1).sum()),
        "positive_pixels": int(flat_masks.sum()),
        "total_pixels": int(flat_masks.size),
        "prevalence": float(flat_masks.mean()),
        "pixel_ap": pixel["ap"],
        "pixel_auroc": pixel["auroc"],
        "pixel_degenerate": pixel["degenerate"],
        "image_ap": image["ap"],
        "image_auroc": image["auroc"],
        "image_degenerate": image["degenerate"],
        "normal_image_score_mean": float(normal.mean()) if len(normal) else None,
        "normal_image_score_p95": float(np.quantile(normal, 0.95)) if len(normal) else None,
        "normal_image_score_max": float(normal.max()) if len(normal) else None,
        "anomaly_image_score_min": float(anomaly.min()) if len(anomaly) else None,
        "anomaly_image_score_max": float(anomaly.max()) if len(anomaly) else None,
        "normal_minus_anomaly_gap": (
            float(normal.mean() - anomaly.mean()) if len(normal) and len(anomaly) else None
        ),
    }


def macro_over_categories(groups: Mapping[tuple[int, str, str], Mapping[str, Any]], boundary: int, arm: str) -> Dict[str, Any]:
    selected = [
        value
        for (b, a, _category), value in groups.items()
        if b == boundary and a == arm and value.get("pixel_ap") is not None
    ]
    if not selected:
        return {"pixel_ap_mean": None, "image_auroc_mean": None, "categories": 0}
    return {
        "pixel_ap_mean": float(np.mean([row["pixel_ap"] for row in selected if row["pixel_ap"] is not None])),
        "pixel_auroc_mean": float(np.mean([row["pixel_auroc"] for row in selected if row["pixel_auroc"] is not None])),
        "image_auroc_mean": float(np.mean([row["image_auroc"] for row in selected if row["image_auroc"] is not None])),
        "image_ap_mean": float(np.mean([row["image_ap"] for row in selected if row["image_ap"] is not None])),
        "categories": len(selected),
    }


def paired_image_auroc_bootstrap(
    labels: Sequence[int],
    scores_by_arm: Mapping[str, Sequence[float]],
    *,
    reps: int,
    seed: int,
) -> Dict[str, Any]:
    """Exact paired whole-image AUROC bootstrap with shared multiplicities."""

    labels_array = np.asarray(labels, dtype=np.int64)
    generator = np.random.default_rng(seed)
    draws = np.stack([bootstrap_weights(labels_array, generator) for _ in range(reps)], axis=0)
    values: Dict[str, np.ndarray] = {}
    for arm, scores in scores_by_arm.items():
        row = np.empty(reps, dtype=np.float64)
        for index in range(reps):
            row[index] = image_metrics(labels_array, np.asarray(scores), draws[index])["auroc"]
        values[arm] = row
    output: Dict[str, Any] = {"reps": reps, "seed": seed, "auroc": {}}
    for arm, row in values.items():
        output["auroc"][arm] = {
            "mean": float(row.mean()),
            "interval": np.quantile(row, [0.025, 0.975]).tolist(),
        }
    for contrast in (("C3", "C2"), ("C3", "C1")):
        if contrast[0] in values and contrast[1] in values:
            difference = values[contrast[0]] - values[contrast[1]]
            output[f"{contrast[0]}-{contrast[1]}"] = {
                "mean": float(difference.mean()),
                "interval": np.quantile(difference, [0.025, 0.975]).tolist(),
                "probability_positive": float((difference > 0).mean()),
            }
    return output


def _binned_bootstrap(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    reps: int,
    bins: int,
    seed: int,
) -> Dict[str, Any]:
    histograms = [score_histogram(np.asarray(m, dtype=np.float64), np.asarray(k, dtype=bool)) for m, k in zip(maps, masks)]
    bootstrap = BoundedPixelBootstrap(histograms, bins=bins)
    generator = np.random.default_rng(seed)
    weights = np.stack([bootstrap_weights(np.asarray(labels), generator) for _ in range(reps)], axis=0)
    lower, upper = bootstrap.bounds(weights)
    return {
        "reps": int(reps),
        "bins": int(bins),
        "seed": int(seed),
        "lower": float(np.quantile(lower, 0.025)),
        "upper": float(np.quantile(upper, 0.975)),
        "within_bin_conservative": True,
    }


def paired_pixel_ap_bootstrap(
    per_arm: Mapping[str, Mapping[str, Any]],
    *,
    reps: int,
    bins: int,
    seed: int,
) -> Dict[str, Any]:
    """Conservative paired native-Pixel-AP bounds for one category/boundary.

    ``per_arm`` maps arm -> {"maps", "masks", "labels", "point_ap"}.  The same
    whole-image multiplicities are used for every arm.  The paired interval is
    the conservative ``lower(C3) - upper(C2)`` / ``upper(C3) - lower(C2)`` range.
    """

    arms = list(per_arm)
    results: Dict[str, Any] = {"reps": int(reps), "bins": int(bins), "seed": int(seed), "arms": {}}
    for arm in arms:
        payload = per_arm[arm]
        results["arms"][arm] = _binned_bootstrap(
            payload["maps"], payload["masks"], payload["labels"],
            reps=reps, bins=bins, seed=seed,
        )
        results["arms"][arm]["point_ap"] = float(payload["point_ap"])
    for contrast in (("C3", "C2"), ("C3", "C1")):
        if contrast[0] in results["arms"] and contrast[1] in results["arms"]:
            left, right = results["arms"][contrast[0]], results["arms"][contrast[1]]
            results[f"{contrast[0]}-{contrast[1]}"] = {
                "point_delta": left["point_ap"] - right["point_ap"],
                "conservative_interval": [left["lower"] - right["upper"], left["upper"] - right["lower"]],
            }
    return results


def retention_table(
    groups: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    arms: Sequence[str],
    acquisition: Mapping[str, int],
    boundaries: Sequence[int],
) -> Dict[str, Any]:
    """Acquisition-to-later AP/AUROC deltas; no 'forgetting' claim."""

    table: Dict[str, Any] = {}
    for arm in arms:
        arm_rows: Dict[str, Any] = {}
        for category, acquired_at in acquisition.items():
            points = [
                (boundary, groups[(boundary, arm, category)])
                for boundary in boundaries
                if boundary >= acquired_at and (boundary, arm, category) in groups
            ]
            if not points:
                continue
            first_boundary, first = points[0]
            arm_rows[category] = {
                "acquisition_boundary": int(first_boundary),
                "points": {
                    str(boundary): {
                        "pixel_ap": row["pixel_ap"],
                        "image_auroc": row["image_auroc"],
                        "normal_image_score_mean": row["normal_image_score_mean"],
                    }
                    for boundary, row in points
                },
                "deltas": {
                    str(boundary): {
                        "pixel_ap": row["pixel_ap"] - first["pixel_ap"],
                        "image_auroc": row["image_auroc"] - first["image_auroc"],
                    }
                    for boundary, row in points[1:]
                },
            }
        table[arm] = arm_rows
    return table


def contrast_table(
    groups: Mapping[tuple[int, str, str], Mapping[str, Any]],
    *,
    boundaries: Sequence[int],
    categories: Sequence[str],
) -> Dict[str, Any]:
    table: Dict[str, Any] = {}
    for boundary in boundaries:
        boundary_rows: Dict[str, Any] = {}
        for left, right in (("C3", "C2"), ("C3", "C1")):
            category_rows: Dict[str, Any] = {}
            for category in categories:
                key_left = (boundary, left, category)
                key_right = (boundary, right, category)
                if key_left not in groups or key_right not in groups:
                    continue
                category_rows[category] = {
                    "pixel_ap": groups[key_left]["pixel_ap"] - groups[key_right]["pixel_ap"],
                    "image_auroc": groups[key_left]["image_auroc"] - groups[key_right]["image_auroc"],
                }
            if category_rows:
                boundary_rows[f"{left}-{right}"] = {
                    "per_category": category_rows,
                    "macro_pixel_ap": float(np.mean([row["pixel_ap"] for row in category_rows.values()])),
                    "macro_image_auroc": float(np.mean([row["image_auroc"] for row in category_rows.values()])),
                    "worst_category_pixel_ap": float(min(row["pixel_ap"] for row in category_rows.values())),
                }
        table[str(boundary)] = boundary_rows
    return table


def decide(contrasts: Mapping[str, Any], groups: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> Dict[str, Any]:
    """Pre-registered progression rule at boundary 350; descriptive, not proof."""

    primary = contrasts.get("350", {}).get("C3-C2")
    if not primary:
        return {"verdict": "NPIM01_INCONCLUSIVE_DEV", "reason": "missing primary C3-C2 contrast at 350"}
    macro = float(primary["macro_pixel_ap"])
    worst = float(primary["worst_category_pixel_ap"])
    signal = macro >= 0.01 and worst >= -0.03
    if signal:
        verdict = "NPIM01_MULTITIMESCALE_SIGNAL_DEV"
        reason = "pre-registered macro AP(C3-C2) >= +0.01 and no category < -0.03 at 350"
    elif macro <= 0.0 and worst >= -0.03:
        verdict = "NPIM01_COMPRESSION_ONLY_DEV"
        reason = "no positive multi-timescale AP contrast and no catastrophic category loss"
    else:
        verdict = "NPIM01_INCONCLUSIVE_DEV"
        reason = "mixed or non-positive category outcomes; pre-registered rule not met"
    return {
        "verdict": verdict,
        "reason": reason,
        "macro_pixel_ap_C3_minus_C2_at_350": macro,
        "worst_category_pixel_ap_C3_minus_C2_at_350": worst,
        "rule": "macro >= +0.01 and worst >= -0.03",
        "category_deltas": primary["per_category"],
    }


def save_qualitative(
    out_dir: Path,
    groups: Mapping[tuple[int, str, str], Mapping[str, Any]],
    records: Sequence[Any],
    maps_by_key: Mapping[tuple[int, str, str], List[np.ndarray]],
    masks_by_index: Mapping[int, np.ndarray],
) -> List[Dict[str, Any]]:
    """Save a bounded set of failure/success maps at boundary 350."""

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Dict[str, Any]] = []
    for category in ("bottle", "carpet", "hazelnut"):
        for arm in ("C1", "C2", "C3"):
            key = (350, arm, category)
            if key not in maps_by_key or key not in groups:
                continue
            rows = [record for record in records if record.category == category]
            maps = maps_by_key[key]
            image_scores = [float(np.asarray(m).max()) for m in maps]
            normals = [i for i, record in enumerate(rows) if record.label == 0]
            anomalies = [i for i, record in enumerate(rows) if record.label == 1]
            picks = []
            if normals:
                picks.append(("hard_normal", max(normals, key=lambda i: image_scores[i])))
            if anomalies:
                picks.append(("hard_anomaly_failure", min(anomalies, key=lambda i: image_scores[i])))
                picks.append(("easy_anomaly", max(anomalies, key=lambda i: image_scores[i])))
            for tag, index in picks:
                record = rows[index]
                native = maps[index]
                path = out_dir / f"{category}_{arm}_{tag}_{record.index:02d}.npy"
                np.save(path, native.astype(np.float32))
                png = out_dir / f"{category}_{arm}_{tag}_{record.index:02d}.png"
                lo, hi = float(native.min()), float(native.max())
                scaled = np.zeros_like(native, dtype=np.uint8) if hi <= lo else (
                    (native - lo) / (hi - lo) * 255.0
                ).astype(np.uint8)
                Image.fromarray(scaled).save(png)
                mask_png = None
                if record.label == 1:
                    mask_png = out_dir / f"{category}_{arm}_{tag}_{record.index:02d}_mask.png"
                    Image.fromarray((masks_by_index[record.index].astype(np.uint8) * 255)).save(mask_png)
                saved.append(
                    {
                        "category": category,
                        "arm": arm,
                        "tag": tag,
                        "record_index": record.index,
                        "relative_path": record.relative_path,
                        "image_score": image_scores[index],
                        "map_npy": str(path),
                        "map_png": str(png),
                        "mask_png": str(mask_png) if mask_png else None,
                    }
                )
    return saved
