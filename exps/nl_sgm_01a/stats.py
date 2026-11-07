"""Category-clustered paired inference (R4).

Unit of observation: the whole image (its 784 patch values are first reduced to one per-image statistic; patches
are never treated as replicates). Primary: category-cluster bootstrap of the macro (equal-weight over categories)
paired contrast. Sensitivity: nested bootstrap (categories, then images within each drawn category).
"""
from __future__ import annotations

import numpy as np

from exps.task_p.p0_normal_support import require

B, SEED = 2000, 20261106


def macro(per_image, cats):
    """per_image [n], cats [n] -> macro mean over categories (each category = mean of its images)."""
    u = np.unique(cats)
    return float(np.mean([per_image[cats == c].mean() for c in u]))


def clustered_ci(diff, cats, nested=False, b=B, seed=SEED, alpha=0.05):
    """Bootstrap (2.5%, 97.5%) and point estimate of the macro mean of per-image paired differences."""
    diff, cats = np.asarray(diff, float), np.asarray(cats)
    require(diff.shape == cats.shape and np.isfinite(diff).all(), "bad paired differences")
    u = np.unique(cats)
    groups = [diff[cats == c] for c in u]
    rng = np.random.default_rng(seed)
    out = np.empty(b)
    for i in range(b):
        pick = rng.integers(len(u), size=len(u))
        vals = []
        for j in pick:
            g = groups[j]
            vals.append(g[rng.integers(len(g), size=len(g))].mean() if nested else g.mean())
        out[i] = np.mean(vals)
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return {"point": macro(diff, cats), "lo": float(lo), "hi": float(hi), "b": b, "seed": seed,
            "scheme": "nested" if nested else "category-cluster"}


def relative_harm(arm, comp, tau):
    """Per-category relative harm (arm - comp) / max(|comp|, tau); tau guards near-zero denominators."""
    return (np.asarray(arm, float) - np.asarray(comp, float)) / np.maximum(np.abs(np.asarray(comp, float)), tau)
