"""
Reference: DRAEM — Discriminatively trained Reconstruction Embedding for
           Surface Anomaly Detection (Zavrtanik et al., ICCV 2021)

Pipeline
--------
  1. Generate 2-D Perlin noise (multi-scale, random resolution)
  2. Rotate the noise map randomly (−90° → +90°)
  3. Threshold at 0.5 → binary mask
  4. Pick DTD texture → augment with 3 random colour transforms
  5. Blend:  I*(1-mask) + (1-β)*DTD*mask + β*I*mask
"""

import glob
import math
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase

# ────────────────────────────────────────────────────────────────────────────
# Perlin noise (pure numpy, no extra deps)
# ────────────────────────────────────────────────────────────────────────────

def _lerp(a, b, w):
    return (b - a) * w + a


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(shape, res):
    """Return a 2-D Perlin noise array of `shape` with frequency `res`."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def tile(s1, s2):
        return np.repeat(np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], 0), d[1], 1)

    def dot(g, shift):
        coords = np.stack(
            [grid[:shape[0], :shape[1], 0] + shift[0],
             grid[:shape[0], :shape[1], 1] + shift[1]], axis=-1
        )
        return (coords * g[:shape[0], :shape[1]]).sum(-1)

    n00 = dot(tile([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile([0, -1], [1, None]), [0, -1])
    n11 = dot(tile([1, None], [1, None]), [-1, -1])
    t = _fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp(_lerp(n00, n10, t[..., 0]),
                                 _lerp(n01, n11, t[..., 0]), t[..., 1])


# ────────────────────────────────────────────────────────────────────────────
# Colour augmenters (same helpers as superpixel, duplicated for independence)
# ────────────────────────────────────────────────────────────────────────────

def _rand_aug(image: np.ndarray) -> np.ndarray:
    """Apply 3 random colour transforms (DRAEM augmenter set)."""
    aug_pool = [
        lambda x: np.clip(np.power(x / 255.0, np.random.uniform(0.5, 2.0)) * 255, 0, 255).astype(np.uint8),           # gamma
        lambda x: np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-30, 30), 0, 255).astype(np.uint8),     # brightness
        lambda x: cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x, (0, 0), 1.0), -0.5, 0),                                # sharpness
        lambda x: _hue_sat(x),
        lambda x: np.where(x < np.random.randint(32, 129), x, 255 - x).astype(np.uint8),                              # solarize
        lambda x: ((x >> (8 - np.random.randint(3, 7))) << (8 - np.random.randint(3, 7))).astype(np.uint8),           # posterize
        lambda x: (255 - x).astype(np.uint8),                                                                          # invert
        lambda x: _autocontrast(x),
        lambda x: _equalize(x),
        lambda x: cv2.warpAffine(x,
            cv2.getRotationMatrix2D((x.shape[1]/2, x.shape[0]/2), float(np.random.uniform(-45,45)), 1.0),
            (x.shape[1], x.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
    ]
    for fn in np.random.choice(aug_pool, 3, replace=False):  # type: ignore[arg-type]
        image = fn(image)
    return image


def _hue_sat(x):
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + np.random.randint(-50, 51)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _autocontrast(x):
    out = x.astype(np.float32)
    for c in range(out.shape[2]):
        lo, hi = out[..., c].min(), out[..., c].max()
        if hi > lo:
            out[..., c] = (out[..., c] - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _equalize(x):
    ycrcb = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ────────────────────────────────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────────────────────────────────

class PerlinAnomalyGenerator(AnomalyGeneratorBase):
    """
    DRAEM-style Perlin noise mask + DTD texture blending.

    Config keys (all optional):
        dtd_dir             : path to DTD images directory
        perlin_scale        : max log2 of Perlin frequency  (default 6)
        min_perlin_scale    : min log2 of Perlin frequency  (default 0)
        perlin_threshold    : binarisation threshold        (default 0.5)
        anomaly_blend_range : (min, max) beta range         (default [0.1, 0.8])
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

        self.perlin_scale     = cfg.get("perlin_scale", 6)
        self.min_perlin_scale = cfg.get("min_perlin_scale", 0)
        self.threshold        = cfg.get("perlin_threshold", 0.5)
        beta_lo, beta_hi      = cfg.get("anomaly_blend_range", [0.1, 0.8])
        self.beta_lo, self.beta_hi = beta_lo, beta_hi

        # DTD
        dtd_dir = cfg.get("dtd_dir", "")
        if not dtd_dir:
            raise FileNotFoundError("[Perlin/DRAEM] dtd_dir is required for DTD textures.")
        self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
        if not self.dtd_file_list:
            raise FileNotFoundError(
                f"[Perlin/DRAEM] No DTD images found under: {dtd_dir}"
            )

    def _perlin_mask(self, h: int, w: int) -> np.ndarray:
        """Binary mask from 2-D Perlin noise, random scale + rotation."""
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((h, w), (sx, sy))

        # random rotation via OpenCV
        angle = float(np.random.uniform(-90, 90))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        return (noise > self.threshold).astype(np.float32)

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        if not self.dtd_file_list:
            raise FileNotFoundError("[Perlin/DRAEM] DTD textures are missing.")
        path = np.random.choice(self.dtd_file_list)
        tex = cv2.imread(path)
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        tex = cv2.resize(tex, (w, h))
        tex = _rand_aug(tex)           # 3 colour augmentations
        return tex.astype(np.float32)

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        mask = self._perlin_mask(h, w)

        if mask.sum() == 0:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        dtd = self._dtd_source(h, w)
        beta = np.random.uniform(self.beta_lo, self.beta_hi)

        m = mask[:, :, None]
        # DRAEM blend: I*(1-mask) + (1-β)*DTD*mask + β*I*mask
        result = img_np * (1 - m) + (1 - beta) * dtd * m + beta * img_np * m

        return result.astype(np.float32), mask, True
