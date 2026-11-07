# exps/cadfinal/metrics.py
"""One metric implementation for every candidate (readouts differ, metrics do not).

exact_ap      exact non-interpolated pixel AP with tied scores grouped over native-resolution maps; the
              algorithm of ``exps.calm02.native.native_ap`` (uint64 bit-packing, in-place sort), generalised
              from 28x28 patch grids to any native map. Equal to sklearn (tests).
hist          per-image (positive, total) histograms on a fixed range, for the paired image bootstrap and
              for pooled AP across categories (exact pooled AP over ~2-3e9 native pixels does not fit in RAM;
              the binning error is reported against the exact per-category AP).
Readouts      memory arms: frozen P1 readout (28x28 -> native, bilinear, align_corners=False, antialias=False);
              Dinomaly: official map chain (R_D: 392 -> 256, Gaussian) -> native bilinear (exps.dinomaly.readout).
"""
from __future__ import annotations

import numpy as np
import torch

from exps.calm02.native import CHUNK, native_hw, native_mask           # noqa: F401  (re-exported)
from exps.task_p.p0_normal_support import require
from exps.task_p.p1_readout import native_map

NEG_TOL = 1e-5            # Dinomaly maps are 1 - cos >= 0 up to rounding; tiny negatives are clamped to 0


def to_native(kind, x, hw):
    """kind 'patch': x [784] patch distances; kind 'dinomaly': x [256,256] R_D map. Returns FP32 numpy >= 0."""
    if kind == "patch":
        m = native_map(torch.as_tensor(np.asarray(x)), hw)
    else:
        from exps.dinomaly.readout import native
        m = native(torch.as_tensor(np.asarray(x, dtype=np.float32)), hw).numpy()
    lo = float(m.min())
    require(np.isfinite(m).all() and lo >= -NEG_TOL, f"bad map (min {lo})")
    return np.maximum(m, np.float32(0.0)).astype(np.float32, copy=False)


def _keys(m, y):
    s = np.ascontiguousarray(m, dtype=np.float32) + np.float32(0.0)
    return (s.view(np.uint32).astype(np.uint64) << np.uint64(1)) | y.astype(np.uint64)


def exact_ap(maps, masks):
    """maps / masks: sequences of equally shaped [H,W] arrays (FP32 >= 0 / bool)."""
    total = int(sum(y.size for y in masks))
    keys = np.empty(total, dtype=np.uint64)
    pos, P = 0, 0
    for m, y in zip(maps, masks):
        require(m.shape == y.shape, "map/mask shape mismatch")
        k = _keys(m, y).ravel()
        keys[pos:pos + k.size] = k; pos += k.size; P += int(y.sum())
    require(pos == total and 0 < P < total, "degenerate pixel set")
    keys.sort()
    desc = keys[::-1]
    ap, tp_carry, tp_last = 0.0, 0, 0
    for start in range(0, total, CHUNK):
        c = np.ascontiguousarray(desc[start:start + CHUNK])
        score, lab = c >> np.uint64(1), (c & np.uint64(1)).astype(np.int64)
        tp = tp_carry + np.cumsum(lab)
        end = np.empty(len(c), dtype=bool)
        end[:-1] = score[:-1] != score[1:]
        end[-1] = start + len(c) == total or score[-1] != (desc[start + len(c)] >> np.uint64(1))
        idx = np.flatnonzero(end)
        tpb = tp[idx]
        prev = np.concatenate(([tp_last], tpb[:-1]))
        ap += float(np.sum((tpb - prev) * (tpb / (start + idx + 1.0))))
        if len(tpb):
            tp_last = int(tpb[-1])
        tp_carry = int(tp[-1])
    return {"ap": ap / P, "n_positive": P, "n_total": total}


def hist(m, y, lo, hi, bins):
    b = np.clip(((m.ravel().astype(np.float64) - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
    return (np.bincount(b, weights=y.ravel().astype(np.float64), minlength=bins),
            np.bincount(b, minlength=bins).astype(np.float64))


def ap_from_hist(pos, tot):
    """AP of binned scores, ties grouped within a bin (bins in ascending score order)."""
    p, t = pos[::-1], tot[::-1]
    keep = t > 0
    p, t = p[keep], t[keep]
    tp, n = np.cumsum(p), np.cumsum(t)
    P = tp[-1]
    require(P > 0, "no positives")
    return float(np.sum(p * (tp / n)) / P)


def auroc(labels, scores, weights=None):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores, sample_weight=weights))


def bootstrap_weights(labels, reps, seed):
    """Paired image weights resampled within label strata (exps.calm.stats rule)."""
    from exps.calm.stats import bootstrap_weights as bw
    return np.stack(bw(np.asarray(labels), reps, seed))
