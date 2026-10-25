# exps/dinomaly/readout.py
"""Readouts. R_D: official Dinomaly map chain, then native upsampling (deviation D2). R_P1: canonical P1."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exps.dinomaly import encoder as E
from exps.dinomaly import official as O

_GAUSS = {}


def gauss(device):
    if device not in _GAUSS:
        _GAUSS[device] = O.get_gaussian_kernel(*O.RECIPE["gaussian"]).to(device)
    return _GAUSS[device]


def grid(en, de):
    """[B,28,28] mean over the two groups of 1 - cos (FP32)."""
    return torch.stack([1 - F.cosine_similarity(e, d) for e, d in zip(en, de)], dim=1).mean(dim=1)


@torch.no_grad()
def dinomaly(en, de):
    """Official evaluation_batch chain: 392 (align_corners=True) -> 256 -> Gaussian 5x5/sigma 4; top-1% mean."""
    a, _ = O.cal_anomaly_maps(en, de, E.SIZE)
    a = F.interpolate(a, size=O.RECIPE["eval_resize_mask"], mode="bilinear", align_corners=False)
    s = gauss(a.device)(a)
    flat = s.flatten(1)
    image = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1] * O.RECIPE["max_ratio"])].mean(dim=1)
    return s[:, 0], image


def native(map2d, hw):
    """Bilinear to native H,W (align_corners=False, antialias=False); masks are never resized."""
    return F.interpolate(map2d[None, None].float(), size=tuple(hw), mode="bilinear", align_corners=False,
                         antialias=False)[0, 0]
