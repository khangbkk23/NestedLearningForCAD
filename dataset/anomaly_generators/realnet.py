"""
Reference: RealNet — A Feature Selection Network with Realistic Synthetic Anomaly
           for Anomaly Detection (Zhang et al., CVPR 2024)

Pipeline
--------
  1. Perlin noise mask × foreground mask  → final mask   (both-aware)
  2. Probabilistic source choice:
       • 'dtd'  (p = dtd_weight)   → DTD texture + 3 colour augments
       • 'sdas' (p = 1-dtd_weight) → SDAS images (class-specific generated
                                      anomalies, or replay-buffer images)
  3. Blend:
       factor * (mask * src) + (1-factor) * (mask * img) + (1-mask) * img
"""

import glob
import math
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase
from .perlin import rand_perlin_2d, _rand_aug      # reuse helpers

_TEXTURE_CATEGORIES = {
    "carpet", "leather", "tile", "wood", "cable", "transistor", "grid",
}


class RealnetAnomalyGenerator(AnomalyGeneratorBase):
    """
    Realnet-style: dual-source (DTD + SDAS/replay) with foreground-aware mask.

    Config keys (all optional):
        dtd_dir                 : path to DTD images
        sdas_dir                : path to SDAS / replay-buffer images
        realnet_dtd_weight      : probability of choosing DTD source (default 0.5)
        realnet_dtd_factor_range: (min, max) blend factor for DTD   (default [0.2, 0.8])
        realnet_sdas_factor_range: (min, max) blend factor for SDAS (default [0.1, 0.6])
        perlin_scale            : max log2 Perlin frequency          (default 6)
        min_perlin_scale        : min log2 Perlin frequency          (default 0)
        perlin_threshold        : binarisation threshold             (default 0.5)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg.get("name", "mvtec").lower()

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)

        self.dtd_weight  = cfg.get("realnet_dtd_weight", 0.5)
        dtd_lo, dtd_hi   = cfg.get("realnet_dtd_factor_range", [0.2, 0.8])
        sd_lo,  sd_hi    = cfg.get("realnet_sdas_factor_range", [0.1, 0.6])
        self.dtd_lo, self.dtd_hi = dtd_lo, dtd_hi
        self.sd_lo,  self.sd_hi  = sd_lo,  sd_hi

        # DTD source
        self.dtd_file_list = []
        dtd_dir = cfg.get("dtd_dir", "")
        if dtd_dir:
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))

        # SDAS / replay-buffer source
        self.sdas_file_list = []
        sdas_dir = cfg.get("sdas_dir", "")
        if sdas_dir and os.path.isdir(sdas_dir):
            self.sdas_file_list = glob.glob(os.path.join(sdas_dir, "**", "*.*"),
                                            recursive=True)
            self.sdas_file_list = [
                p for p in self.sdas_file_list
                if p.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

        import logging
        log = logging.getLogger(__name__)
        if not self.dtd_file_list:
            log.warning("[Realnet] No DTD images found.")
        if not self.sdas_file_list:
            log.info("[Realnet] No SDAS/replay images found. Will use DTD only.")
            self.dtd_weight = 1.0   # force DTD

    # ── foreground mask (same logic as Superpixel) ───────────────────────────

    def _foreground_mask(self, img_np: np.ndarray, category: str) -> np.ndarray:
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cat  = category.lower()

        if cat in _TEXTURE_CATEGORIES or self.dataset_name != "mvtec":
            return np.ones_like(gray, dtype=np.float32)

        if cat in {"pill", "hazelnut", "metal_nut", "toothbrush"}:
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif cat in {"bottle", "capsule", "screw", "zipper"}:
            _, bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            fg = cv2.bitwise_not(bg)
        else:
            _, fg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        return (fg > 0).astype(np.float32)

    # ── Perlin mask ───────────────────────────────────────────────────────────

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        return (noise > self.threshold).astype(np.float32)

    # ── anomaly sources ───────────────────────────────────────────────────────

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        if not self.dtd_file_list:
            return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8).astype(np.float32)
        tex = cv2.imread(np.random.choice(self.dtd_file_list))
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        tex = cv2.resize(tex, (w, h))
        tex = _rand_aug(tex)   # 3 colour augments (DRAEM-style)
        return tex.astype(np.float32)

    def _sdas_source(self, h: int, w: int) -> np.ndarray:
        """
        Class-specific anomaly images (SDAS) or replay-buffer images.
        No augmentation — the image already looks anomalous.
        """
        img = cv2.imread(np.random.choice(self.sdas_file_list))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        return img.astype(np.float32)

    # ── public API ───────────────────────────────────────────────────────────

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]

        fg_mask    = self._foreground_mask(img_np, category)
        prl_mask   = self._perlin_mask(h, w)
        mask       = (prl_mask * fg_mask).astype(np.float32)   # intersection

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        # ── source selection ─────────────────────────────────────────────
        use_dtd = (not self.sdas_file_list) or (np.random.rand() < self.dtd_weight)

        if use_dtd:
            src    = self._dtd_source(h, w)
            factor = np.random.uniform(self.dtd_lo, self.dtd_hi)
        else:
            src    = self._sdas_source(h, w)
            factor = np.random.uniform(self.sd_lo, self.sd_hi)

        # ── Realnet blend ────────────────────────────────────────────────
        # factor*(mask*src) + (1-factor)*(mask*img) + (1-mask)*img
        m      = mask[:, :, None]
        result = factor * (m * src) + (1 - factor) * (m * img_np) + (1 - m) * img_np

        return result.astype(np.float32), mask, True
