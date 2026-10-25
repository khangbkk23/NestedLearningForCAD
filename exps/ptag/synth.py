# exps/ptag/synth.py
"""PTAG-01 synthetic anomaly families on normal training hosts (no real defect data).

Q1 feature-space Gaussian blobs, Q2 DRAEM Perlin+DTD (repo generator), Q3 CutPaste.
Every sample is seeded by (family, category, host, variant, attempt); masks are native-resolution.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image

from dataset.anomaly_generators.perlin import PerlinAnomalyGenerator, rand_perlin_2d
from exps.ptag.common import DATA, MAX_ATTEMPTS, ROOT, sample_seed
from exps.task_p.p0_normal_support import distances, require

MEAN, STD = (.485, .456, .406), (.229, .224, .225)
DTD = ROOT / "data/dtd/images"


def load_rgb(rel):
    with Image.open(DATA / rel) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def preprocess(arrays):
    """Canonical `_ImageDataset` path from an in-memory uint8 RGB array."""
    mean = torch.tensor(MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(STD, dtype=torch.float32)[:, None, None]
    out = []
    for a in arrays:
        im = Image.fromarray(a).resize((224, 224), Image.Resampling.BILINEAR)
        out.append((torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1)) / 255 - mean) / std)
    return torch.stack(out)


def extractor(device):
    from exps.calm02.extract import extractor as calm02_extractor
    return calm02_extractor(device)


@torch.no_grad()
def extract(model, arrays, device, batch=8):
    feats = []
    for i in range(0, len(arrays), batch):
        f = model.extract_patch_features(preprocess(arrays[i:i + batch]).to(device)).float().cpu()
        require(f.shape[1:] == (784, 768) and bool(torch.isfinite(f).all()), "bad features")
        feats.append(f)
    return torch.cat(feats)


def patch_mask(blob, hw):
    """Native pixel (y, x) is positive iff its patch (floor(28y/H), floor(28x/W)) is in the blob."""
    h, w = hw
    rows = (np.arange(h) * 28) // h
    cols = (np.arange(w) * 28) // w
    grid = np.zeros((28, 28), dtype=bool)
    for p in blob:
        grid[p // 28, p % 28] = True
    return grid[rows][:, cols]


def blob(rng):
    k = int(rng.integers(2, 7))
    members = [int(rng.integers(784))]
    while len(members) < k:
        m = members[int(rng.integers(len(members)))]
        r, c = divmod(m, 28)
        free = [nr * 28 + nc for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                if 0 <= nr < 28 and 0 <= nc < 28 and nr * 28 + nc not in members]
        if free:
            members.append(free[int(rng.integers(len(free)))])
    return sorted(members)


def normal_scale(clean):
    """m_c: median FP64 1-NN distance of each host patch to the patches of the other hosts."""
    n = len(clean)
    med = []
    for i in range(n):
        others = torch.cat([clean[j] for j in range(n) if j != i])
        med.append(distances(clean[i], others).min(1).values)
    return float(torch.cat(med).median())


def q1(clean_features, hw, sigma, cat_index, host, variant):
    rng = np.random.default_rng(sample_seed(1, cat_index, host, variant, 0))
    members = blob(rng)
    noise = torch.from_numpy(rng.standard_normal((len(members), 768))) * sigma
    x = clean_features.clone()
    x[members] = (x[members].double() + noise).float()
    return x, patch_mask(members, hw), {"blob": members}


class NativePerlin(PerlinAnomalyGenerator):
    """Repo generator; noise on an S x S canvas (S = multiple of 64 >= side) then cropped (task §2.5)."""

    def __init__(self):
        super().__init__({"dtd_dir": str(DTD)})
        self.dtd_file_list = sorted(self.dtd_file_list)
        require(len(self.dtd_file_list) > 5000, "DTD texture set missing")

    def _perlin_mask(self, h, w):
        s = 64 * math.ceil(max(h, w) / 64)
        sx = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        sy = 2 ** np.random.randint(self.min_perlin_scale, self.perlin_scale + 1)
        noise = rand_perlin_2d((s, s), (sx, sy))
        angle = float(np.random.uniform(-90, 90))
        m = cv2.getRotationMatrix2D((s / 2, s / 2), angle, 1.0)
        noise = cv2.warpAffine(noise, m, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return (noise[:h, :w] > self.threshold).astype(np.float32)


def q2(gen, image, cat, cat_index, host, variant):
    for attempt in range(MAX_ATTEMPTS):
        np.random.seed(sample_seed(2, cat_index, host, variant, attempt))
        out, mask, ok = gen.generate(image.astype(np.float32), cat)
        if ok and mask.any():
            return np.clip(np.rint(out), 0, 255).astype(np.uint8), mask > 0, {"attempts": attempt + 1}
    raise RuntimeError(f"Q2 failed after {MAX_ATTEMPTS} attempts")


def q3(image, cat_index, host, variant):
    h, w = image.shape[:2]
    rng = np.random.default_rng(sample_seed(3, cat_index, host, variant, 0))
    area = rng.uniform(0.02, 0.15) * h * w
    ratio = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
    ph = int(min(h, max(1, round(math.sqrt(area * ratio)))))
    pw = int(min(w, max(1, round(math.sqrt(area / ratio)))))
    sy, sx = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
    ty, tx = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
    patch = image[sy:sy + ph, sx:sx + pw].astype(np.float64)
    brightness, contrast = rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1)
    mean = patch.mean()
    patch = ((patch - mean) * contrast + mean) * brightness
    out = image.copy()
    out[ty:ty + ph, tx:tx + pw] = np.clip(np.rint(patch), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[ty:ty + ph, tx:tx + pw] = True
    return out, mask, {"source": [sy, sx], "target": [ty, tx], "size": [ph, pw]}
