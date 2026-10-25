# exps/calm02/native.py
"""Exact native-resolution pixel AP for hundreds of millions of pixels (primary CALM-02 endpoint).

Maps are the frozen P1 readout (``exps/task_p/p1_readout.native_map``: FP32 [28,28] -> native H,W,
bilinear, align_corners=False, antialias=False). AP is non-interpolated with tied scores grouped,
equal to ``sklearn.metrics.average_precision_score``. Scores are non-negative FP32, so their bit
patterns order like their values; each pixel is packed as ``bits << 1 | label`` into one uint64,
sorted in place, and AP is accumulated in chunks over the descending order.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from exps.task_p.p0_normal_support import require
from exps.task_p.p1_readout import native_map

CHUNK = 1 << 24
BINS = 16384


def native_hw(path):
    with Image.open(path) as im:                 # header only
        return im.height, im.width


def native_mask(path, hw):
    if path is None:
        return np.zeros(hw, dtype=bool)
    with Image.open(path) as im:
        m = np.array(im.convert("L")) > 0
    require(m.shape == tuple(hw), f"mask/image size mismatch: {path}")
    return m


def _keys(score_map, mask):
    s = np.ascontiguousarray(score_map, dtype=np.float32) + np.float32(0.0)   # -0.0 -> +0.0
    require(bool(np.isfinite(s).all()) and bool((s >= 0).all()), "scores must be finite and >= 0")
    return (s.view(np.uint32).astype(np.uint64) << np.uint64(1)) | mask.astype(np.uint64)


def native_ap(patch_scores, hws, masks):
    """patch_scores: [N,784] (torch or numpy); hws: [(H,W)]; masks: list of bool [H,W]."""
    total = int(sum(h * w for h, w in hws))
    keys = np.empty(total, dtype=np.uint64)
    pos, P = 0, 0
    for d, hw, y in zip(patch_scores, hws, masks):
        m = native_map(torch.as_tensor(np.asarray(d)), hw)
        k = _keys(m, y).ravel()
        keys[pos:pos + k.size] = k; pos += k.size; P += int(y.sum())
    require(pos == total and P > 0 and P < total, "degenerate pixel set")
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
    return {"ap": ap / P, "prevalence": P / total, "n_positive": P, "n_total": total}


def native_hists(patch_scores, hws, masks, bins=BINS):
    """Per-image binned (positive, total) native-pixel histograms for the paired bootstrap only.

    The range comes from the patch scores: a bilinear map lies within [min, max] of its patches.
    """
    lo = float(min(np.min(d) for d in patch_scores)); hi = float(max(np.max(d) for d in patch_scores))
    require(hi > lo, "degenerate score range")
    pos = np.zeros((len(hws), bins)); tot = np.zeros((len(hws), bins))
    for i, (d, hw, y) in enumerate(zip(patch_scores, hws, masks)):
        m = native_map(torch.as_tensor(np.asarray(d)), hw).ravel()
        b = np.clip(((m - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
        tot[i] = np.bincount(b, minlength=bins); pos[i] = np.bincount(b, weights=y.ravel().astype(np.float64), minlength=bins)
    return pos, tot
