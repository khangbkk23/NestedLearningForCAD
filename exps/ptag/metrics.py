# exps/ptag/metrics.py
"""Exact native AP in bounded GPU memory, host-cluster bootstrap weights, weighted quantile, Kendall tau-b."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau
from torch.nn.functional import interpolate

from exps.ptag.common import REPS, SEED
from exps.task_p.p0_normal_support import require


def native_maps(patch_scores, hw, device):
    """Frozen P1 readout: FP32 28x28 -> native bilinear, align_corners=False, antialias=False."""
    x = torch.as_tensor(patch_scores, dtype=torch.float32).reshape(-1, 1, 28, 28).to(device)
    return interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False, antialias=False)[:, 0]


@torch.no_grad()
def exact_ap(values, labels, buckets=8):
    """Non-interpolated AP with ties grouped (identical to `p2.metrics.exact_ranking`).

    Values are split into descending value ranges at sampled thresholds; ties never span a range, so each
    range is ranked exactly on its own and carries the cumulative counts of all higher ranges.
    """
    v, y = values.flatten(), labels.flatten()
    positives = float(y.sum())
    require(0 < positives < v.numel(), "degenerate synthetic pixel metric")
    sample = torch.sort(v[:: max(1, v.numel() // 65536)]).values
    cuts = torch.unique(sample[(torch.arange(1, buckets, device=v.device) * len(sample)) // buckets])
    edges = [float("inf")] + cuts.flip(0).tolist() + [-float("inf")]
    tp_before, total_before, ap = 0.0, 0.0, 0.0
    for hi, lo in zip(edges[:-1], edges[1:]):
        sel = (v < hi) & (v >= lo) if hi != float("inf") else (v >= lo)
        if not bool(sel.any()):
            continue
        vals, order = torch.sort(v[sel], descending=True, stable=True)
        pos = y[sel][order].to(torch.float64).cumsum(0)
        ends = torch.cat((torch.nonzero(vals[:-1] != vals[1:]).flatten(),
                          torch.tensor([vals.numel() - 1], device=v.device)))
        tp = pos[ends] + tp_before
        total = ends.to(torch.float64) + 1 + total_before
        inc = tp - torch.cat((tp.new_full((1,), tp_before), tp[:-1]))
        ap += float(((inc / positives) * tp / total).sum())
        tp_before, total_before = float(tp[-1]), float(total[-1])
    require(abs(tp_before - positives) < 0.5 and total_before == v.numel(), "AP bucket accounting")
    return ap


def host_weights(cat_index, hosts):
    rng = np.random.default_rng(SEED + 100 + cat_index)
    return np.stack([np.bincount(rng.integers(0, hosts, hosts), minlength=hosts) for _ in range(REPS)])


def weighted_quantile(values, weights, q):
    """Inverted-CDF quantile: smallest value whose cumulative weight reaches q of the total."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, q * cum[-1], side="left")])


def tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(kendalltau(x, y, variant="b").statistic)
