"""CALM-00A: native-resolution metrics, paired image bootstrap and AP-level Shapley."""
from __future__ import annotations

import numpy as np
import torch

from exps.task_p.p1_readout import count_metrics, image_metrics, native_map

BINS = 16384  # bootstrap-only discretisation; point AP is exact


def maps_from_patches(patches: np.ndarray, hw_list) -> list[np.ndarray]:
    """patches [N,784] -> native FP32 maps via the frozen P1 readout."""
    return [native_map(torch.from_numpy(np.ascontiguousarray(row)), hw)
            for row, hw in zip(patches, hw_list)]


def pooled_ap(maps, masks) -> dict:
    s = np.concatenate([m.ravel() for m in maps])
    y = np.concatenate([m.ravel() for m in masks])
    values, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    positive = np.bincount(inv, weights=y.astype(np.int64), minlength=len(values))
    # count_metrics expects thresholds in descending score order.
    out = count_metrics(positive[::-1].astype(np.int64), counts[::-1].astype(np.int64))
    out["n_positive"] = int(y.sum())
    out["n_total"] = int(y.size)
    return out


def binned_hist(maps, masks, bins: int = BINS):
    lo = min(float(m.min()) for m in maps)
    hi = max(float(m.max()) for m in maps)
    if not hi > lo:
        raise RuntimeError("degenerate score range")
    pos, tot = [], []
    for m, y in zip(maps, masks):
        idx = np.clip(((m.ravel() - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
        tot.append(np.bincount(idx, minlength=bins).astype(np.float64))
        pos.append(np.bincount(idx, weights=y.ravel().astype(np.float64), minlength=bins))
    return np.stack(pos), np.stack(tot), (lo, hi)


def ap_from_hist(pos, tot) -> float:
    return count_metrics(pos[::-1].astype(np.int64), tot[::-1].astype(np.int64))["ap"]


def shapley(ap00: float, ap01: float, ap10: float, ap11: float) -> dict:
    """Exact two-player Shapley split plus the order-dependence (interaction) term.

    intrusion effect when applied first  : ap01 - ap00
    intrusion effect when applied second : ap11 - ap10
    eviction  effect when applied first  : ap10 - ap00
    eviction  effect when applied second : ap11 - ap01
    interaction = (intrusion|second) - (intrusion|first)
    The sum of the two Shapley values is exactly total by construction, so a residual
    computed as total - shapley_intr - shapley_evict is identically zero and is NOT
    an interaction term.
    """
    intr = 0.5 * ((ap01 - ap00) + (ap11 - ap10))
    evict = 0.5 * ((ap10 - ap00) + (ap11 - ap01))
    inter = (ap11 - ap10) - (ap01 - ap00)
    return {"intrusion": intr, "eviction": evict, "interaction": inter,
            "intrusion_first": ap01 - ap00, "intrusion_second": ap11 - ap10,
            "eviction_first": ap10 - ap00, "eviction_second": ap11 - ap01,
            "total": ap11 - ap00, "shapley_sum": intr + evict}


def bootstrap_weights(labels: np.ndarray, reps: int, seed: int) -> list[np.ndarray]:
    """Paired image weights, resampled within label strata. Pixels are never resampled."""
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(np.asarray(labels) == v) for v in (0, 1)]
    out = []
    for _ in range(reps):
        w = np.zeros(len(labels), dtype=np.float64)
        for idx in strata:
            draw = rng.integers(0, len(idx), size=len(idx))
            w[idx] = np.bincount(draw, minlength=len(idx))
        out.append(w)
    return out


def weighted_ap(hists, w) -> float:
    pos, tot = hists[0], hists[1]
    return ap_from_hist(pos.T @ w, tot.T @ w)


def image_auroc(labels, image_scores) -> dict:
    return image_metrics(np.asarray(labels), np.asarray(image_scores))
