# training/benchmark_metrics_v1.py
"""Benchmark metrics with continual-learning-aware forgetting."""

from __future__ import annotations

import numpy as np


def _arrays(scores, labels, name):
    s = np.asarray(scores, dtype=float).reshape(-1)
    y = np.asarray(labels).reshape(-1)

    if s.shape != y.shape or s.size == 0:
        raise ValueError(f"{name} scores/labels shape mismatch")
    if np.unique(y).size < 2:
        return None

    return s, y.astype(int)


def image_auroc(scores, labels):
    a = _arrays(scores, labels, "image")
    if a is None:
        return float("nan")

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(*a[::-1]))


def image_ap(scores, labels):
    a = _arrays(scores, labels, "image")
    if a is None:
        return float("nan")

    from sklearn.metrics import average_precision_score
    return float(average_precision_score(*a[::-1]))


def pixel_aupr(maps, masks):
    s = np.asarray(maps, dtype=float)
    y = np.asarray(masks).astype(int)

    if s.ndim != 3 or y.shape != s.shape:
        raise ValueError("pixel shapes must be [B,H,W] and equal")

    a = _arrays(s.reshape(-1), y.reshape(-1), "pixel")
    if a is None:
        return float("nan")

    from sklearn.metrics import average_precision_score
    return float(average_precision_score(*a[::-1]))


def pixel_auroc(maps, masks):
    s = np.asarray(maps, dtype=float)
    y = np.asarray(masks).astype(int)

    if s.ndim != 3 or y.shape != s.shape:
        raise ValueError("pixel shapes must be [B,H,W] and equal")

    a = _arrays(s.reshape(-1), y.reshape(-1), "pixel")
    if a is None:
        return float("nan")

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(*a[::-1]))


def compute_metrics(image_scores, labels, maps, masks):
    return {
        "i_auroc": image_auroc(image_scores, labels),
        "i_ap": image_ap(image_scores, labels),
        "p_aupr": pixel_aupr(maps, masks),
        "p_auroc": pixel_auroc(maps, masks),
        "n_images": int(np.asarray(labels).size),
        "n_pixels": int(np.asarray(masks).size),
    }


def macro_task_metrics(per_task):
    out = {}
    for key in ("i_auroc", "i_ap", "p_aupr", "p_auroc"):
        vals = [float(v[key]) for v in per_task.values()]
        out[key] = float(np.mean(vals)) if vals else float("nan")

    out["task_count"] = len(per_task)
    out["macro_final_i_auroc"] = out["i_auroc"]
    out["macro_final_p_aupr"] = out["p_aupr"]
    return out


def _json_safe_matrix(matrix):
    return [
        [float(x) if np.isfinite(x) else None for x in row]
        for row in matrix
    ]


def forgetting_matrix(performance, formula="mean_prior_max_minus_final"):
    """Compute forgetting using only history after each task was learned.

    Matrix convention:
      row i = checkpoint after learning task i
      col j = performance on task j

    For every old task j < k-1:
      F_j = max_{l=j..k-2} a[l,j] - a[k-1,j]

    Future-task cells j > i are ignored.
    """
    if formula != "mean_prior_max_minus_final":
        raise ValueError("unsupported FM formula")

    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("FM matrix must be square")

    k = matrix.shape[0]
    if k == 0:
        raise ValueError("FM requires at least one task")

    values = []
    for task in range(k - 1):
        # Task j becomes valid only after checkpoint j.
        prior = matrix[task:k - 1, task]
        available = prior[np.isfinite(prior)]
        final = matrix[-1, task]

        if not available.size or not np.isfinite(final):
            raise ValueError(
                "FM needs finite learned-history and final scores for every old task"
            )

        values.append(float(np.max(available) - final))

    return {
        "matrix": _json_safe_matrix(matrix),
        "per_task_forgetting": values,
        "fm": float(np.mean(values)) if values else None,
        "denominator": k - 1,
        "formula": formula,
        "clamped_at_zero": False,
        "unlearned_cells": "not_evaluated",
        "single_task_convention": "undefined",
    }
