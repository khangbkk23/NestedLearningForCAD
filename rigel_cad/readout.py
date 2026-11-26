"""A1 read-out: patch-energy grid -> anomaly map -> image score (frozen; exps/nar_lite_01 + exps/nanm_r2)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from rigel_cad.config import IMAGE_TOP, INPUT_SIZE, SMOOTH_KSIZE, SMOOTH_SIGMA


def _gauss():
    ax = torch.arange(SMOOTH_KSIZE, dtype=torch.float32) - SMOOTH_KSIZE // 2
    g = torch.exp(-ax ** 2 / (2 * SMOOTH_SIGMA ** 2))
    return g / g.sum()


def anomaly_maps(grid: torch.Tensor) -> torch.Tensor:
    """[n, 28, 28] energies -> [n, 392, 392] maps: bilinear upsampling to the input size, separable Gaussian
    (sigma 4, 33 taps, zero padding), then a batch-level shift so the minimum is >= 0
    (exps/nanm_r2/run.py::maps_of)."""
    g = _gauss()
    up = F.interpolate(grid[:, None].float(), size=(INPUT_SIZE, INPUT_SIZE), mode='bilinear', align_corners=False)
    up = F.conv2d(up, g.view(1, 1, SMOOTH_KSIZE, 1), padding=(SMOOTH_KSIZE // 2, 0))
    m = F.conv2d(up, g.view(1, 1, 1, SMOOTH_KSIZE), padding=(0, SMOOTH_KSIZE // 2))[:, 0]
    return m - min(0., float(m.min()))


def image_scores(maps: torch.Tensor) -> np.ndarray:
    """Mean of the top 1 % values of each map (exps/nanm_r0/audit.py::image_scores)."""
    k = max(1, int(IMAGE_TOP * maps.shape[-1] * maps.shape[-2]))
    return maps.flatten(1).topk(k, dim=1).values.mean(1).numpy()


def to_native_torch(m392: torch.Tensor, hw) -> np.ndarray:
    """392 map -> native (H, W), torch bilinear, align_corners False, no antialias (used by the pixel-AP metric)."""
    return F.interpolate(m392.float().reshape(1, 1, *m392.shape[-2:]), size=tuple(hw), mode='bilinear',
                         align_corners=False, antialias=False)[0, 0].numpy()


def to_native_cv2(m392: torch.Tensor, hw) -> np.ndarray:
    """392 map -> native (H, W), OpenCV INTER_LINEAR float32 (used for the MVTec AD 2 submission maps)."""
    import cv2
    return cv2.resize(m392.numpy().astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
