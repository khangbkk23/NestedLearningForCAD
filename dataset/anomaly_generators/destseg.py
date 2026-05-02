"""
DeSTSegAnomalyGenerator

Reference: DeSTSeg — Segmentation-Based Deep Anomaly Detection with Self-Supervised
           Training (Zhang et al., CVPR 2023)

Pipeline
--------
  1. Perlin noise mask (same as PerlinAnomalyGenerator)
  2. Raw DTD texture (no augmentation)
  3. Blend:  I*(1-mask) + (1-β)*DTD*mask + β*I*mask
"""

import glob
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase
from .perlin import rand_perlin_2d 


class DeSTSegAnomalyGenerator(AnomalyGeneratorBase):
    """
    DeSTSeg-style: Perlin mask + raw (unaugmented) DTD texture.

    Config keys (all optional):
        dtd_dir             : path to DTD images directory
        perlin_scale        : max log2 of Perlin frequency  (default 6)
        min_perlin_scale    : min log2                       (default 0)
        perlin_threshold    : binarisation threshold         (default 0.5)
        destseg_beta_range  : (min, max) blend factor        (default [0.1, 0.9])
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)
        beta_lo, beta_hi      = cfg.get("destseg_beta_range", [0.1, 0.9])
        self.beta_lo, self.beta_hi = beta_lo, beta_hi

        dtd_dir = cfg.get("dtd_dir", "")
        if not dtd_dir:
            raise FileNotFoundError("[DeSTSeg] dtd_dir is required for DTD textures.")
        self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
        if not self.dtd_file_list:
            raise FileNotFoundError(
                f"[DeSTSeg] No DTD images found under: {dtd_dir}"
            )

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))

        # random rotation
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        return (noise > self.threshold).astype(np.float32)

    def _dtd_source_raw(self, h: int, w: int) -> np.ndarray:
        """Load DTD texture WITHOUT any colour augmentation (key DeSTSeg difference)."""
        if not self.dtd_file_list:
            raise FileNotFoundError("[DeSTSeg] DTD textures are missing.")
        path = np.random.choice(self.dtd_file_list)
        tex = cv2.imread(path)
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        return cv2.resize(tex, (w, h)).astype(np.float32)

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        mask = self._perlin_mask(h, w)

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        dtd = self._dtd_source_raw(h, w)
        beta = np.random.uniform(self.beta_lo, self.beta_hi)

        m = mask[:, :, None]
        # Same DRAEM blend formula
        result = img_np * (1 - m) + (1 - beta) * dtd * m + beta * img_np * m

        return result.astype(np.float32), mask, True
