# Adapted from Dinomaly commit 1f252be03a918789b19848f0ca37166e7c28dada; Apache-2.0; see root LICENSE.
"""Dinomaly anomaly maps and top-1% image scores."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nested_replay_cad.model import decoder as O
from nested_replay_cad.model import encoder as E

_GAUSS = {}


def gauss(device):
    if device not in _GAUSS:
        _GAUSS[device] = O.get_gaussian_kernel(*O.RECIPE["gaussian"]).to(device)
    return _GAUSS[device]


@torch.no_grad()
def dinomaly(en, de):
    """Official evaluation_batch chain: 392 (align_corners=True) -> 256 -> Gaussian 5x5/sigma 4; top-1% mean."""
    a, _ = O.cal_anomaly_maps(en, de, E.SIZE)
    a = F.interpolate(
        a, size=O.RECIPE["eval_resize_mask"], mode="bilinear", align_corners=False
    )
    s = gauss(a.device)(a)
    flat = s.flatten(1)
    image = torch.sort(flat, dim=1, descending=True)[0][
        :, : int(flat.shape[1] * O.RECIPE["max_ratio"])
    ].mean(dim=1)
    return s[:, 0], image
