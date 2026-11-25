"""Fixed Anomaly Readout (no parameters): 28x28 energies -> 392x392 map -> image score; native-resolution map.

Oracle: exps/nanm_r2/run.py::smooth_nc / ::maps_of (CPU), exps/nanm_r0/audit.py::image_scores,
exps/nar_lite_01/run.py::map_native.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def maps392(grid, size, sigma, ksize):
    """bilinear 28->392 (align_corners False), separable Gaussian (zero padding), then the oracle's min-shift."""
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    up = F.interpolate(grid[:, None].float(), size=(size, size), mode='bilinear', align_corners=False)
    up = F.conv2d(up, g.view(1, 1, ksize, 1), padding=(ksize // 2, 0))
    m = F.conv2d(up, g.view(1, 1, 1, ksize), padding=(0, ksize // 2))[:, 0]
    return m - min(0., float(m.min()))


def image_score(maps, top_fraction):
    k = max(1, int(top_fraction * maps.shape[-1] * maps.shape[-2]))
    return maps.flatten(1).topk(k, dim=1).values.mean(1)


def to_native(map392, hw):
    return F.interpolate(map392.float().reshape(1, 1, *map392.shape[-2:]), size=tuple(hw), mode='bilinear',
                         align_corners=False, antialias=False)[0, 0]


class FixedAnomalyReadout:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, grid):
        m = maps392(grid, self.cfg.image_size, self.cfg.sigma, self.cfg.ksize)
        return m, image_score(m, self.cfg.top_fraction)


def to_224(maps392):
    """Resized-224 evaluation map: bilinear 392 -> 224 (align_corners False, no antialias), as bench_g3 maps224."""
    return F.interpolate(maps392[:, None].float(), size=(224, 224), mode='bilinear', align_corners=False,
                         antialias=False)[:, 0]
