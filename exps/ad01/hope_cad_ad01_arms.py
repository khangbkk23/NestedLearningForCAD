# exps/ad01/hope_cad_ad01_arms.py
"""AD-01 four-arm endpoint evaluation on the permitted development split.

This driver runs the predeclared two-factor design:

    allocation  in {global, spatial}
    scoring     in {global, local}

on the bottle / carpet / hazelnut development data that was already authorised
for the anomaly-signal study. It reads the frozen development manifest rather
than re-deriving a split, so no new test-driven development set is created and
the sealed confirmation manifest is never opened.

Design commitments fixed before any endpoint number was observed:

* Backbone, layer, preprocessing, feature normalisation: the frozen CADIC
  extractor, unchanged.
* Total exemplar budget: 2500 vectors for every arm, at every order seed.
* Spatial partition: one fixed 4x4 grid over the 28x28 patch lattice, used for
  both allocation and scoring locality, so the two factors share one notion of
  position. Partition sensitivity (2x2, 7x7) is reported as a diagnostic only.
* Order seeds: three paired within-task orders in which the category *sequence*
  is permuted but each category's own image order is fixed, matching the
  established paired-order-seed convention.
* No anomaly label enters admission, eviction, scoring or normalisation. Labels
  are read only to compute the reported metrics.
* Primary pixel metric is the benchmark's native-resolution AUPR; the 28x28
  grid AUPR is reported alongside as a declared secondary diagnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]

CATEGORIES = ("bottle", "carpet", "hazelnut")
ORDER_SEEDS = (0, 1, 2)
TOTAL_BUDGET = 2500
PRIMARY_GRID = 4
DIAGNOSTIC_GRIDS = (2, 7)
IMAGE_NEIGHBORS = 9

DEV_MANIFEST = ROOT / "results/hope_cad/anomaly_signal_gate/manifests/anomaly_dev_manifest.parquet"
CONFIRMATION_MANIFEST = (
    ROOT / "results/hope_cad/anomaly_signal_gate/manifests/anomaly_confirmation_manifest.parquet"
)
PROTOCOL_CONFIG = ROOT / "conf/benchmarks/protocols/mvtec_1x15_v1.yaml"
METHOD_CONFIG = ROOT / "conf/benchmarks/methods/cadic_compatible_v1.yaml"
CHECKPOINT = ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"


def order_permutation(seed: int, categories: Sequence[str] = CATEGORIES) -> list[str]:
    """Paired within-task order seed: permute the category sequence only.

    Image order inside a category is never permuted, so a seed changes the
    *stream order* of tasks and nothing else. All arms share the permutation,
    which makes arm comparisons paired.

    Permutations are drawn from a seeded shuffle of the full permutation list,
    so the three declared seeds are guaranteed to be distinct. A naive
    `default_rng(seed).permutation` can repeat an order (`seed=0` and `seed=2`
    both yielded the same order here), which would silently reduce the number of
    distinct streams below the declared three.
    """
    from itertools import permutations

    all_orders = sorted(permutations(categories))
    if not 0 <= seed < len(all_orders):
        raise ValueError(f"order seed {seed} outside [0,{len(all_orders)})")
    return list(all_orders[seed])


# ------------------------------------------------------------------ manifests


def load_dev_manifest(path: Path = DEV_MANIFEST) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"category", "relative_path", "image_id", "label", "mask_path", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"development manifest is missing columns {sorted(missing)}")
    if set(frame["split"]) != {"DEVELOPMENT"}:
        raise ValueError("development manifest contains non-DEVELOPMENT rows")
    confirmation = pd.read_parquet(CONFIRMATION_MANIFEST)
    overlap = set(frame["image_id"]) & set(confirmation["image_id"])
    if overlap:
        raise ValueError(f"development/confirmation overlap detected: {len(overlap)} ids")
    return frame


def dev_rows_for(frame: pd.DataFrame, category: str) -> list[dict[str, Any]]:
    subset = frame[frame["category"] == category]
    rows: list[dict[str, Any]] = []
    for record in subset.to_dict("records"):
        mask_path = record["mask_path"]
        rows.append(
            dict(
                relative_path=record["relative_path"],
                label=int(record["label"]),
                mask_path=None if mask_path is None or pd.isna(mask_path) else str(mask_path),
            )
        )
    if not rows:
        raise ValueError(f"no development rows for {category}")
    return rows


# --------------------------------------------------------------- evaluation


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def average_precision(masks: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(masks, scores))


@dataclass
class TaskObservation:
    """One evaluated (state, task) pair on the frozen development images."""

    state_after: str
    task: str
    i_auroc: float
    p_aupr_native: float
    p_aupr_grid28: float
    mean_normal_score: float
    mean_defect_score: float


def evaluate_task(
    memory,
    patches: torch.Tensor,
    masks: Sequence[np.ndarray],
    masks28: Sequence[np.ndarray],
    labels: np.ndarray,
    scoring: str,
) -> dict[str, Any]:
    """Score the development images and return image and pixel metrics.

    `masks` are native-resolution; `masks28` are their 28x28 nearest-downsampled
    counterparts used only by the secondary grid diagnostic.
    """
    images, pixel = memory.score(patches, scoring=scoring)
    image_scores = images.detach().cpu().numpy().astype(np.float64)
    pixel_scores = pixel.detach().cpu().numpy().astype(np.float64)

    native_scores, native_masks = [], []
    for index in range(pixel_scores.shape[0]):
        patch_map = torch.from_numpy(pixel_scores[index]).reshape(1, 1, 28, 28)
        resized = torch.nn.functional.interpolate(
            patch_map, size=masks[index].shape[-2:], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        native_scores.append(resized.reshape(-1))
        native_masks.append(masks[index].reshape(-1))

    return {
        "i_auroc": auroc(labels, image_scores),
        "p_aupr_native": average_precision(
            np.concatenate(native_masks), np.concatenate(native_scores)
        ),
        "p_aupr_grid28": average_precision(
            np.concatenate([m.reshape(-1) for m in masks28]),
            pixel_scores.reshape(-1),
        ),
        "image_scores": image_scores,
        "mean_normal_score": float(image_scores[labels == 0].mean()),
        "mean_defect_score": float(image_scores[labels == 1].mean()),
    }


def forgetting_from_matrix(matrix: np.ndarray, task_names: Sequence[str]) -> dict[str, Any]:
    """Task-boundary forgetting: mean over tasks of (max later - final).

    `matrix[k, j]` is the performance on task `j` after task `k` was learned.
    Only entries `k >= j` are defined; future cells are never evaluated.
    """
    n = matrix.shape[0]
    if n < 2:
        return {"fm": None, "reason": "fewer than two tasks", "per_task": None}
    per_task = []
    for j in range(n):
        observed = matrix[j:, j]
        finite = observed[~np.isnan(observed)]
        if finite.size < 2:
            per_task.append(float("nan"))
            continue
        per_task.append(float(finite.max() - finite[-1]))
    values = np.array(per_task, dtype=np.float64)
    finite = values[~np.isnan(values)]
    return {
        "fm": float(finite.mean()) if finite.size else None,
        "per_task": {task_names[j]: float(values[j]) for j in range(n)},
        "definition": "mean_j(max_{k>=j} a[k,j] - a[n-1,j])",
    }


def membership_grid(patches: torch.Tensor) -> np.ndarray:
    """Pixel score map under the raw 28x28 grid, for the bounded score grid."""
    return patches.reshape(28, 28)
