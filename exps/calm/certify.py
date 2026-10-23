"""CALM-00A: category-level certificate proxy and the violation ratio (labels used in A2 only)."""
from __future__ import annotations

import numpy as np
import torch

from exps.task_p.p0_normal_support import distances


@torch.no_grad()
def category_radius(bank: torch.Tensor, features: list[torch.Tensor], q: float = 0.95, chunk: int = 64) -> float:
    """Stable certificate proxy: the q-quantile of a category's own normal NN distances.

    Computed only from normal patches of that category against the support at its own
    acquisition boundary. Phase A uses this category-level proxy because per-row radii from
    7,840 normals over 2,702 rows are too sparse to estimate a 0.95 quantile per row.
    """
    vals = []
    for feat in features:
        for start in range(0, 784, chunk):
            vals.append(distances(feat[start:start + chunk], bank).min(1).values)
    d = torch.cat(vals)
    return float(torch.quantile(d, q))


def violation(d1: np.ndarray, owner1: np.ndarray, query_cat: int, radii: dict[int, float]) -> dict:
    lut = np.array([radii[i] for i in range(max(radii) + 1)], dtype=np.float64)
    r = lut[owner1.astype(np.int64)]
    foreign = owner1 != query_cat
    violating = foreign & (d1 > r)
    return {"foreign": foreign, "violating": violating}


def violation_summary(viol: dict, d1: np.ndarray, d2: np.ndarray, labels: np.ndarray) -> dict:
    foreign, violating = viol["foreign"], viol["violating"]
    lab = np.asarray(labels).astype(bool)          # per image
    norm = ~lab
    out = {
        "foreign_rate_normal": float(foreign[norm].mean()),
        "foreign_rate_defect": float(foreign[lab].mean()),
        "violating_rate_normal": float(violating[norm].mean()),
        "violating_rate_defect": float(violating[lab].mean()),
        "defect_patches": int(violating[lab].size),
        "affected_defect_patches": int(violating[lab].sum()),
        "normal_patches": int(violating[norm].size),
        "affected_normal_patches": int(violating[norm].sum()),
    }
    nr, dr = out["violating_rate_normal"], out["violating_rate_defect"]
    out["vir"] = float(dr / nr) if nr > 0 else (float("inf") if dr > 0 else float("nan"))
    out["affected_defect_fraction"] = dr
    gain = (d2 - d1)[violating]
    out["mean_restoration"] = float(np.mean(gain)) if gain.size else 0.0
    out["max_restoration"] = float(np.max(gain)) if gain.size else 0.0
    return out
