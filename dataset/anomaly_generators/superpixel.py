"""
Nested Learning's own method

  1. Foreground mask      — Otsu thresholding, category-aware
  2. Semantic mask        — SLIC superpixel selection + area filter (0.5%–15%)
  3. Anomaly source       — DTD texture (augmented) OR self-shift + luminance jitter
  4. Alpha blend          — factor * (mask * src) + (1-factor) * (mask * img)

"""

import glob
import os

import cv2
import numpy as np

from .base import AnomalyGeneratorBase


_AUGMENTER_NAMES = [
    "gamma_contrast", "brightness", "sharpness", "hue_saturation",
    "solarize", "posterize", "invert", "autocontrast", "equalize", "rotate",
]

_TEXTURE_CATEGORIES = {
    "carpet", "leather", "tile", "wood", "cable", "transistor", "grid",
}


def _apply_aug(image: np.ndarray, aug_name: str) -> np.ndarray:
    if aug_name == "gamma_contrast":
        gamma = np.random.uniform(0.5, 2.0)
        return np.clip(np.power(image / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)

    if aug_name == "brightness":
        return np.clip(image * np.random.uniform(0.8, 1.2) + np.random.uniform(-30, 30),
                       0, 255).astype(np.uint8)

    if aug_name == "sharpness":
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    if aug_name == "hue_saturation":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + np.random.randint(-50, 51)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + np.random.randint(-50, 51), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if aug_name == "solarize":
        threshold = np.random.randint(32, 129)
        return np.where(image < threshold, image, 255 - image).astype(np.uint8)

    if aug_name == "posterize":
        shift = 8 - np.random.randint(3, 7)
        return ((image >> shift) << shift).astype(np.uint8)

    if aug_name == "invert":
        return (255 - image).astype(np.uint8)

    if aug_name == "autocontrast":
        out = image.astype(np.float32).copy()
        for c in range(out.shape[2]):
            lo, hi = out[..., c].min(), out[..., c].max()
            if hi > lo:
                out[..., c] = (out[..., c] - lo) * (255.0 / (hi - lo))
        return np.clip(out, 0, 255).astype(np.uint8)

    if aug_name == "equalize":
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if aug_name == "rotate":
        h, w = image.shape[:2]
        angle = float(np.random.uniform(-45, 45))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return image


# Genetar

class SuperpixelAnomalyGenerator(AnomalyGeneratorBase):
    """Nested Learning original method."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg.get("name", "mvtec").lower()

        # DTD texture source
        self.use_dtd = cfg.get("use_dtd", False)
        self.dtd_file_list = []
        if self.use_dtd:
            dtd_dir = cfg.get("dtd_dir", "")
            if not dtd_dir:
                raise FileNotFoundError("[Superpixel] dtd_dir is required when use_dtd=True.")
            self.dtd_file_list = glob.glob(os.path.join(dtd_dir, "*/*.*"))
            if not self.dtd_file_list:
                raise FileNotFoundError(
                    f"[Superpixel] No DTD images found under: {dtd_dir}"
                )

        # SLIC
        self.min_fg_coverage = cfg.get("superpixel_min_fg_coverage", 0.7)
        self.max_sp_fraction = cfg.get("superpixel_max_fraction", 0.15)
        self.area_min = cfg.get("anomaly_area_min", 0.005)    # fraction of image
        self.area_max = cfg.get("anomaly_area_max", 0.15)
        self.mask_retries = cfg.get("superpixel_retries", 3)

    def _foreground_mask(self, img_np: np.ndarray, category: str) -> np.ndarray:
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        cat = category.lower()

        if cat in _TEXTURE_CATEGORIES or self.dataset_name != "mvtec":
            return np.ones_like(gray, dtype=np.float32)

        if cat in {"pill", "hazelnut", "metal_nut", "toothbrush"}:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        elif cat in {"bottle", "capsule", "screw", "zipper"}:
            _, bg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = cv2.bitwise_not(bg)
        else:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.float32)

    def _semantic_mask(self, img_np: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
        h, w = img_np.shape[:2]
        reg_size = np.random.randint(15, 40)
        slic = cv2.ximgproc.createSuperpixelSLIC(
            img_np.astype(np.uint8),
            algorithm=cv2.ximgproc.SLIC,
            region_size=reg_size, ruler=15.0,
        )
        slic.iterate(10)
        labels = slic.getLabels()
        num_sp = slic.getNumberOfSuperpixels()

        mask = np.zeros((h, w), dtype=np.float32)
        if num_sp <= 1:
            return mask

        valid_sp = [
            sp for sp in range(num_sp)
            if np.mean(fg_mask[labels == sp]) > self.min_fg_coverage
        ]
        if not valid_sp:
            return mask

        max_select = max(3, int(len(valid_sp) * self.max_sp_fraction))
        num_select = np.random.randint(2, max_select + 1)
        for sp in np.random.choice(valid_sp, num_select, replace=False):
            mask[labels == sp] = 1.0

        k = np.random.choice([7, 11, 15])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return (mask > 0.4).astype(np.float32)

    def _dtd_source(self, h: int, w: int) -> np.ndarray:
        img = cv2.imread(np.random.choice(self.dtd_file_list))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h))
        for aug in np.random.choice(_AUGMENTER_NAMES, 3, replace=False):
            img = _apply_aug(img, aug)
        return img.astype(np.float32)

    def _self_shift_source(self, img_np: np.ndarray) -> tuple:
        h, w = img_np.shape[:2]
        sx = np.random.randint(-w // 15, w // 15)
        sy = np.random.randint(-h // 15, h // 15)
        src = np.roll(img_np, shift=(sy, sx), axis=(0, 1))
        lum = np.random.normal(0, 15, (h, w, 1))
        src = np.clip(src * np.random.uniform(0.85, 1.15) + lum, 0, 255)
        factor = np.random.uniform(0.4, 0.8)
        return src, factor

    # API

    def generate(self, img_np: np.ndarray, category: str):
        h, w = img_np.shape[:2]
        img_area = h * w

        fg_mask = self._foreground_mask(img_np, category)

        mask_noise = None
        for _ in range(self.mask_retries):
            candidate = self._semantic_mask(img_np, fg_mask)
            area = np.sum(candidate)
            if self.area_min * img_area <= area <= self.area_max * img_area:
                mask_noise = candidate
                break

        if mask_noise is None:
            return img_np.copy(), np.zeros((h, w), dtype=np.float32), False

        if self.use_dtd and np.random.rand() > 0.5:
            src = self._dtd_source(h, w)
            factor = np.random.uniform(0.3, 0.7)
        else:
            src, factor = self._self_shift_source(img_np)

        blurred = cv2.GaussianBlur(mask_noise, (7, 7), 0)
        m = blurred[:, :, None]
        blended = factor * (m * src) + (1 - factor) * (m * img_np)
        result = (1 - m) * img_np + blended

        return result.astype(np.float32), mask_noise, True
