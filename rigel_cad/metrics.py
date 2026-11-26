"""RIGEL-CAD evaluation metrics (frozen definitions of the paper tables).

* Pixel AP: native-resolution, binned (8,192 bins over [0, max energy of the category's grid + 1e-12]); each bin is one
  tie group (exps/nanm_r0/audit.py::binned_ap / image_hists).
* Image AUROC: sklearn roc_auc_score of the top-1 % image score.
* Paired bootstrap: 2,000 image-resampling replicates, seeded by [20261120, position of the category in the order].
* Stream summaries: final, forgetting (max over stages after acquisition minus final), acquisition, prequential.
* MVTec AD 2: AU-PRO (FPR limits 0.05, 0.30) by an exact histogram over the 65,536 float16 bit patterns of the
  submitted maps, equivalent to the official MVTec evaluation code (compute_pro + trapezoid); SegF1 aggregated over
  all pixels of a scenario.
"""
from __future__ import annotations

from bisect import bisect
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from rigel_cad import readout
from rigel_cad.config import AP_BINS, BOOT_REPLICATES, BOOT_SEED


# ---------------------------------------------------------------------- pixel AP / image AUROC
def native_hw(path):
    with Image.open(path) as im:
        return im.height, im.width


def native_mask(path, hw) -> np.ndarray:
    if path is None:
        return np.zeros(hw, dtype=bool)
    with Image.open(path) as im:
        m = np.array(im.convert('L')) > 0
    if m.shape != tuple(hw):
        raise ValueError(f'mask/image size mismatch: {path}')
    return m


def binned_ap(pos, tot) -> float:
    """AP from per-bin positive / total counts (bins ascending in score; each bin = one tie group)."""
    pos, tot = pos[::-1], tot[::-1]
    P = pos.sum()
    if P <= 0:
        return float('nan')
    tp, n = np.cumsum(pos), np.cumsum(tot)
    keep = tot > 0
    return float(np.sum(pos[keep] / P * (tp[keep] / n[keep])))


def image_hists(maps, hws, masks, lo, hi, bins=AP_BINS):
    """Per-image native-resolution (pos, tot) histograms of 392 maps over a fixed [lo, hi] range."""
    pos = np.zeros((len(maps), bins), np.float32)
    tot = np.zeros((len(maps), bins), np.float32)
    for i, (m, hw, y) in enumerate(zip(maps, hws, masks)):
        v = readout.to_native_torch(m, hw).ravel()
        b = np.clip(((v - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
        tot[i] = np.bincount(b, minlength=bins)
        pos[i] = np.bincount(b, weights=y.ravel().astype(np.float64), minlength=bins)
    return pos, tot


def boot_weights(n: int, position: int, B: int = BOOT_REPLICATES) -> np.ndarray:
    """Paired image-resampling weights of one category; identical for every method scored on the same order."""
    rng = np.random.default_rng([BOOT_SEED, position])
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float32)


class Category:
    """Ground truth of one test category: native sizes, masks, labels (rows in the scoring order)."""

    def __init__(self, root, samples):
        from pathlib import Path
        root = Path(root)
        self.rows = [s.relative_path for s in samples]
        self.hws = [native_hw(root / s.relative_path) for s in samples]
        self.masks = [native_mask(root / s.mask_path if s.mask_path else None, hw) for s, hw in zip(samples, self.hws)]
        self.y = np.array([s.label for s in samples])


def evaluate_category(cat: Category, grid: torch.Tensor, stages: Sequence[int], boot_w=None):
    """grid: [len(stages), n, 28, 28] fp32 energies of one category, one stage per row (stage = task index of the
    read-out state). The histogram range is shared by all stages of the grid. Returns
    ({stage: {'ap', 'i_auroc'}}, bootstrap replicates of the last stage or None)."""
    from sklearn.metrics import roc_auc_score
    y = cat.y
    hi = float(grid.max()) + 1e-12
    out, boot = {}, None
    for i, l in enumerate(stages):
        maps = readout.anomaly_maps(grid[i])
        img = readout.image_scores(maps)
        h = image_hists(maps, cat.hws, cat.masks, 0., hi)
        out[int(l)] = {'ap': binned_ap(h[0].sum(0), h[1].sum(0)), 'i_auroc': float(roc_auc_score(y, img))}
        if i == len(stages) - 1 and boot_w is not None:
            P, Tt = boot_w @ h[0], boot_w @ h[1]
            ap = np.array([binned_ap(P[r], Tt[r]) for r in range(len(P))])
            au = np.array([roc_auc_score(y, img, sample_weight=w) if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0
                           else np.nan for w in boot_w])
            boot = {'ap': ap.astype(np.float32), 'auroc': au.astype(np.float32)}
    return out, boot


def stream_summary(order: Sequence[str], per_task: Dict[str, Dict[int, dict]]) -> dict:
    """per_task[task][stage] = {'ap', 'i_auroc'} for stages >= the task's position. Returns the paper summaries."""
    T = len(order)
    ap = {t: {int(l): v['ap'] for l, v in per_task[t].items()} for t in order}
    au = {t: {int(l): v['i_auroc'] for l, v in per_task[t].items()} for t in order}
    return {'pixel_ap': float(np.mean([ap[t][T - 1] for t in order])),
            'image_auroc': float(np.mean([au[t][T - 1] for t in order])),
            'forgetting_ap': float(np.mean([max(ap[t][l] for l in range(j, T - 1)) - ap[t][T - 1]
                                            for j, t in enumerate(order[:-1])])) if T > 1 else 0.0,
            'acquisition_ap': float(np.mean([ap[t][j] for j, t in enumerate(order)])),
            'prequential_ap': float(np.mean([np.mean([ap[order[j]][l] for j in range(l + 1)]) for l in range(T)])),
            'per_task_final': {t: {'ap': ap[t][T - 1], 'i_auroc': au[t][T - 1]} for t in order}}


def ci95(boot: List[np.ndarray]) -> List[float]:
    """Percentile CI of the mean over categories of paired bootstrap replicates."""
    m = np.mean(boot, axis=0)
    return [float(x) for x in np.nanpercentile(m, [2.5, 97.5])]


# ---------------------------------------------------------------------- MVTec AD 2: AU-PRO and SegF1
def _regions(mask: np.ndarray):
    from scipy.ndimage import label
    lab, n = label(mask, np.ones((3, 3), dtype=int))
    flat = lab.ravel()
    idx = np.flatnonzero(flat)
    if n == 0:
        return idx.astype(np.int64), np.zeros(0), 0
    sizes = np.bincount(flat, minlength=n + 1).astype(np.float64)
    return idx.astype(np.int64), 1.0 / sizes[flat[idx]], n


class ProAccumulator:
    """Exact PRO curve over float16 maps: every distinct float16 value is a threshold (official tie handling)."""

    def __init__(self):
        self.tot = np.zeros(65536, np.int64)
        self.gt = np.zeros(65536, np.int64)
        self.pro = np.zeros(65536)
        self.nreg = 0

    def add(self, map16: np.ndarray, mask: np.ndarray):
        bits = map16.astype(np.float16).view(np.uint16).ravel()
        if (bits & 0x8000).any():
            raise ValueError('AU-PRO histogram requires non-negative maps')
        self.tot += np.bincount(bits, minlength=65536)
        idx, w, n = _regions(mask)
        if len(idx):
            b = bits[idx]
            self.gt += np.bincount(b, minlength=65536)
            self.pro += np.bincount(b, weights=w, minlength=65536)
        self.nreg += n

    def curve(self):
        ok = self.tot - self.gt
        keep = np.flatnonzero(self.tot > 0)[::-1]
        fpr = np.cumsum(ok[keep]).astype(np.float64) / ok.sum()
        pro = np.cumsum(self.pro[keep]) / self.nreg
        fpr = np.clip(fpr.astype(np.float32), None, 1.)
        pro = np.clip(pro, None, 1.)
        return np.concatenate(([0.], fpr, [1.])), np.concatenate(([0.], pro, [1.]))

    def au_pro(self, limit: float) -> float:
        f, p = self.curve()
        return float(trapezoid(f, p, x_max=limit) / limit)


def trapezoid(x, y, x_max=None) -> float:
    """Area under (x, y) up to x_max with linear interpolation at x_max; same semantics as
    generic_util.trapezoid of the MVTec AD evaluation code (Copyright 2022 MVTec Software GmbH, BSD-style licence)."""
    x, y = np.asarray(x), np.asarray(y)
    fin = np.logical_and(np.isfinite(x), np.isfinite(y))
    x, y = x[fin], y[fin]
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            ins = bisect(x, x_max)
            assert 0 < ins < len(x)
            y_i = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_i + y[ins - 1]) * (x_max - x[ins - 1])
        m = x <= x_max
        x, y = x[m], y[m]
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction)


class SegF1:
    """Pixel F1 aggregated over all images of a scenario for binary maps (map > theta)."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, binary: np.ndarray, mask: np.ndarray):
        g = mask.astype(bool)
        self.tp += int((binary & g).sum())
        self.fp += int((binary & ~g).sum())
        self.fn += int((~binary & g).sum())

    def value(self):
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else None
