# exps/npim_cad/forensic/spatial.py
"""Independent spatial-correctness audit for EXP-NPIM-02.

Checks the flat patch index, 28x28 -> native bilinear geometry, orientation,
non-square handling, corners, interpolation settings and the existing
qualitative maps.  It never changes the production evaluator; if a real mapping
defect is found, it is reported as a blocker and reproduced by a minimal fixture.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from exps.task_p.p1_readout import count_metrics, score_histogram
from models.hope_cad.npim_cad import native_map

GRID = 28


def flat_index(row: int, column: int, grid: int = GRID) -> int:
    if not (0 <= row < grid and 0 <= column < grid):
        raise ValueError("patch coordinates out of range")
    return row * grid + column


def manual_bilinear(source: np.ndarray, height: int, width: int) -> np.ndarray:
    """Independent align_corners=False, antialias=False bilinear oracle.

    Source coordinates follow ``(dst + 0.5) * src_size / dst_size - 0.5`` with
    endpoint clamping, exactly matching PyTorch's linear upsample convention.
    """

    source = np.asarray(source, dtype=np.float64)
    if source.shape != (GRID, GRID):
        raise ValueError("source must be 28x28")
    output = np.zeros((height, width), dtype=np.float64)
    for row in range(height):
        source_y = max(0.0, min(GRID - 1.0, (row + 0.5) * GRID / height - 0.5))
        y0 = int(math.floor(source_y))
        y1 = min(y0 + 1, GRID - 1)
        wy = source_y - y0
        for column in range(width):
            source_x = max(0.0, min(GRID - 1.0, (column + 0.5) * GRID / width - 0.5))
            x0 = int(math.floor(source_x))
            x1 = min(x0 + 1, GRID - 1)
            wx = source_x - x0
            top = (1.0 - wx) * source[y0, x0] + wx * source[y0, x1]
            bottom = (1.0 - wx) * source[y1, x0] + wx * source[y1, x1]
            output[row, column] = (1.0 - wy) * top + wy * bottom
    return output


def production_bilinear(source: np.ndarray, height: int, width: int) -> np.ndarray:
    flat = np.asarray(source, dtype=np.float32).reshape(-1)
    return native_map(torch.from_numpy(flat), (height, width)).numpy()


def delta_fixture(row: int, column: int, grid: int = GRID) -> np.ndarray:
    source = np.zeros((grid, grid), dtype=np.float64)
    source[row, column] = 1.0
    return source


def _argmax_location(values: np.ndarray) -> tuple[int, int]:
    flat = int(np.argmax(values))
    return divmod(flat, values.shape[1])


def _expected_center(row: int, column: int, height: int, width: int, grid: int = GRID) -> tuple[float, float]:
    return ((row + 0.5) * height / grid, (column + 0.5) * width / grid)


def geometric_audit() -> Dict[str, Any]:
    """Delta, orientation, non-square, corner and interpolation checks."""

    cases = []
    for row, column, height, width in (
        (0, 0, 32, 32),
        (0, 27, 32, 32),
        (27, 0, 32, 32),
        (27, 27, 32, 32),
        (14, 7, 48, 64),
        (3, 24, 45, 71),
        (13, 13, 7, 11),
    ):
        source = delta_fixture(row, column)
        manual = manual_bilinear(source, height, width)
        production = production_bilinear(source, height, width)
        max_abs = float(np.abs(manual - production).max())
        peak = _argmax_location(production)
        expected = _expected_center(row, column, height, width)
        distance = math.dist(peak, expected)
        cases.append(
            {
                "patch_row": row,
                "patch_column": column,
                "native_hw": [height, width],
                "manual_vs_production_max_abs": max_abs,
                "orientation_ok": bool(
                    peak[0] < height / 2 if row < GRID / 2 else peak[0] >= height / 2
                )
                and bool(peak[1] < width / 2 if column < GRID / 2 else peak[1] >= width / 2),
                "peak_native_rc": [int(peak[0]), int(peak[1])],
                "expected_center": [float(expected[0]), float(expected[1])],
                "peak_distance_pixels": float(distance),
                "peak_distance_ok": bool(distance <= 1.5),
            }
        )
    flat = [flat_index(row, column) for row in range(GRID) for column in range(GRID)]
    return {
        "grid": GRID,
        "flat_index_ok": bool(flat == list(range(GRID * GRID))),
        "manual_vs_production_max_abs": max(case["manual_vs_production_max_abs"] for case in cases),
        "all_cases_ok": bool(
            all(
                case["manual_vs_production_max_abs"] <= 2e-6
                and case["orientation_ok"]
                and case["peak_distance_ok"]
                for case in cases
            )
        ),
        "cases": cases,
        "interpolation": {"mode": "bilinear", "align_corners": False, "antialias": False},
        "note": "production uses native_map; the manual oracle is independent of the evaluator",
    }


def _ap(scores: np.ndarray, mask: np.ndarray) -> float | None:
    histogram = score_histogram(np.asarray(scores, dtype=np.float64), np.asarray(mask, dtype=bool))
    metrics = count_metrics(histogram["positive"][::-1], histogram["total"][::-1])
    return metrics["ap"]


def _transform(values: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return values
    if name == "transpose":
        return values.T
    if name == "fliplr":
        return np.fliplr(values)
    if name == "flipud":
        return np.flipud(values)
    if name == "rot90":
        return np.rot90(values)
    raise ValueError(name)


def qualitative_audit(
    qualitative_root: Path,
    sealed_predictions: Mapping[str, Any],
    *,
    boundary: int = 350,
) -> Dict[str, Any]:
    """Verify saved maps against sealed scores and audit orientation without selection."""

    sealed_index: Dict[tuple[str, int, int], np.ndarray] = {}
    arms = list(sealed_predictions["arms"])
    for entry in range(len(sealed_predictions["patch_scores"])):
        arm = arms[int(sealed_predictions["arm_index"][entry])]
        record = int(sealed_predictions["record_index"][entry])
        b = int(sealed_predictions["boundary"][entry])
        sealed_index[(arm, b, record)] = sealed_predictions["patch_scores"][entry].numpy()

    cases: List[Dict[str, Any]] = []
    identity_best = 0
    identity_count = 0
    map_parity_max = 0.0
    for path in sorted(qualitative_root.glob("*_hard_anomaly_failure_*.npy")) + sorted(
        qualitative_root.glob("*_easy_anomaly_*.npy")
    ):
        stem = path.stem
        parts = stem.split("_")
        category, arm, tag = parts[0], parts[1], "_".join(parts[2:-1])
        record = int(parts[-1])
        mask_path = path.with_name(stem + "_mask.png")
        if not mask_path.is_file():
            continue
        mask = np.array(Image.open(mask_path)) > 0
        stored = np.load(path).astype(np.float32)
        if stored.shape != mask.shape:
            raise ValueError(f"qualitative map/mask shape mismatch: {path}")
        sealed = sealed_index[(arm, boundary, record)]
        expected = production_bilinear(sealed, mask.shape[0], mask.shape[1])
        parity = float(np.abs(expected - stored).max())
        map_parity_max = max(map_parity_max, parity)
        scores = {name: _ap(_transform(stored, name), mask) for name in
                  ("identity", "transpose", "fliplr", "flipud", "rot90")}
        best = max(scores, key=lambda key: (-1.0 if scores[key] is None else scores[key]))
        identity_count += 1
        identity_best += int(best == "identity")
        cases.append(
            {
                "file": str(path),
                "category": category,
                "arm": arm,
                "tag": tag,
                "record_index": record,
                "shape": list(mask.shape),
                "sealed_map_parity_max_abs": parity,
                "ap_by_transform": {key: (None if value is None else float(value)) for key, value in scores.items()},
                "best_transform": best,
                "identity_is_best": bool(best == "identity"),
            }
        )
    return {
        "cases": cases,
        "case_count": len(cases),
        "sealed_map_parity_max_abs": float(map_parity_max),
        "sealed_map_parity_ok": bool(map_parity_max <= 1e-6),
        "identity_best_count": int(identity_best),
        "identity_case_count": int(identity_count),
        "orientation_audit_note": "identity is compared with transpose/flip/rot90 only; no transform is selected or applied",
        "all_shapes_native": bool(all(case["shape"] == list(np.load(case["file"]).shape) for case in cases)),
    }
