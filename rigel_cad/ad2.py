"""MVTec AD 2 protocol for RIGEL-CAD: one continual stream over the scenarios (alphabetical), fitting on train/ only.

* Threshold (fixed before any test result): theta_s = mean + 3 std of all validation/ pixel scores of scenario s,
  computed with the model state at the scenario's acquisition stage (native-resolution float32 maps).
* Submission: final-stage maps at native resolution, float16 zlib TIFF under anomaly_images/, binary PNG
  (float32 map > theta_s) under anomaly_images_thresholded/. Use the official MVTec AD 2 code utils to check and
  compress the folder before uploading.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from rigel_cad import readout


def validation_threshold(grid: torch.Tensor, hws, k: float = 3.0):
    """grid: [n, 28, 28] validation energies at the acquisition stage. Returns (theta, stats)."""
    maps = readout.anomaly_maps(grid)
    s1 = s2 = 0.0
    n = 0
    for m, hw in zip(maps, hws):
        v = readout.to_native_cv2(m, hw).astype(np.float64)
        s1 += v.sum()
        s2 += (v * v).sum()
        n += v.size
    mu = s1 / n
    sd = float(np.sqrt(max(0.0, s2 / n - mu * mu)))
    return float(mu + k * sd), {'mean': mu, 'std': sd, 'n_pixels': n, 'n_images': len(hws), 'k': k}


def write_maps(out_dir, scenario: str, split: str, rows, hws, grid: torch.Tensor, theta: float) -> int:
    """Write the float16 TIFF and binary PNG maps of one scenario/split. Returns the number of images written."""
    import cv2
    import tifffile
    out_dir = Path(out_dir)
    maps = readout.anomaly_maps(grid)
    n = 0
    for r, hw, m in zip(rows, hws, maps):
        nat = readout.to_native_cv2(m, hw)
        name = Path(r).stem
        pa = out_dir / 'anomaly_images' / scenario / split / f'{name}.tiff'
        pt = out_dir / 'anomaly_images_thresholded' / scenario / split / f'{name}.png'
        pa.parent.mkdir(parents=True, exist_ok=True)
        pt.parent.mkdir(parents=True, exist_ok=True)
        tmp = pa.with_suffix('.tmp16')
        tifffile.imwrite(tmp, nat.astype(np.float16), compression='zlib')
        os.replace(tmp, pa)
        cv2.imwrite(str(pt), ((nat > theta) * 255).astype(np.uint8))
        n += 1
    return n
