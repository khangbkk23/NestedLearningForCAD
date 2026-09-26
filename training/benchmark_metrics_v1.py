"""Single strict metric implementation for benchmark reporting."""
from __future__ import annotations
import numpy as np


def _arrays(scores, labels, name):
    s, y = np.asarray(scores, dtype=float).reshape(-1), np.asarray(labels).reshape(-1)
    if s.shape != y.shape or s.size == 0:
        raise ValueError(f"{name} scores/labels shape mismatch")
    if np.unique(y).size < 2:
        return None
    return s, y.astype(int)


def image_auroc(scores, labels):
    a = _arrays(scores, labels, "image")
    if a is None: return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(*a[::-1]))


def image_ap(scores, labels):
    a = _arrays(scores, labels, "image")
    if a is None: return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(*a[::-1]))


def pixel_aupr(maps, masks):
    s, y = np.asarray(maps, dtype=float), np.asarray(masks).astype(int)
    if s.ndim != 3 or y.shape != s.shape: raise ValueError("pixel shapes must be [B,H,W] and equal")
    a = _arrays(s.reshape(-1), y.reshape(-1), "pixel")
    if a is None: return float("nan")
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(*a[::-1]))


def pixel_auroc(maps, masks):
    s, y = np.asarray(maps, dtype=float), np.asarray(masks).astype(int)
    if s.ndim != 3 or y.shape != s.shape: raise ValueError("pixel shapes must be [B,H,W] and equal")
    a = _arrays(s.reshape(-1), y.reshape(-1), "pixel")
    if a is None: return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(*a[::-1]))


def compute_metrics(image_scores, labels, maps, masks):
    return {"i_auroc": image_auroc(image_scores, labels), "i_ap": image_ap(image_scores, labels),
            "p_aupr": pixel_aupr(maps, masks), "p_auroc": pixel_auroc(maps, masks),
            "n_images": int(np.asarray(labels).size), "n_pixels": int(np.asarray(masks).size)}


def macro_task_metrics(per_task):
    out = {}
    for key in ("i_auroc", "i_ap", "p_aupr", "p_auroc"):
        vals = [float(v[key]) for v in per_task.values() if np.isfinite(v[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    out["task_count"] = len(per_task)
    out["macro_final_i_auroc"] = out["i_auroc"]
    out["macro_final_p_aupr"] = out["p_aupr"]
    return out


def forgetting_matrix(performance, formula="mean_prior_max_minus_final"):
    if formula != "mean_prior_max_minus_final": raise ValueError("unsupported FM formula")
    matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]: raise ValueError("FM matrix must be square")
    values = []
    for task in range(matrix.shape[1]):
        prior = matrix[: matrix.shape[0], task]
        final = matrix[-1, task]
        values.append(float(np.max(prior[:-1]) - final) if task < matrix.shape[0] - 1 else 0.0)
    return {"matrix": matrix.tolist(), "per_task_forgetting": values,
            "fm": float(np.mean(values)) if values else 0.0, "formula": formula}
