# exps/npim_cad/uncertainty/moments.py
"""Image-grouped normal-residual moments for EXP-NPIM-03.

A calibration image contributes 784 correlated patch residuals.  The moments
are accumulated with Chan's parallel Welford combination, one image group at a
time, and the *image* count is reported as the effective sample size.  Whole-image
bootstrap resamples image groups, never patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

D = 768
PATCHES = 784


@dataclass
class PerChannelMoments:
    d: int = D
    mu: np.ndarray = field(default_factory=lambda: np.zeros(D, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(D, dtype=np.float64))
    n_patches: int = 0
    n_images: int = 0
    group_means: List[np.ndarray] = field(default_factory=list)
    group_m2: List[np.ndarray] = field(default_factory=list)
    group_n: List[int] = field(default_factory=list)

    def add_image_group(self, residual: np.ndarray) -> None:
        """Add one image's 784 residuals with a stable parallel update."""

        values = np.asarray(residual, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.d or values.shape[0] < 2:
            raise ValueError(f"expected [n_patches,{self.d}], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("non-finite calibration residuals")
        n2 = int(values.shape[0])
        mean2 = values.mean(axis=0)
        centered = values - mean2
        m2_group = (centered * centered).sum(axis=0)
        if self.n_images == 0:
            self.mu = mean2.copy()
            self.m2 = m2_group.copy()
            self.n_patches = n2
        else:
            n1 = self.n_patches
            n = n1 + n2
            delta = mean2 - self.mu
            self.mu = self.mu + delta * (n2 / n)
            self.m2 = self.m2 + m2_group + delta * delta * (n1 * n2 / n)
            self.n_patches = n
        self.n_images += 1
        self.group_means.append(mean2.copy())
        self.group_m2.append(m2_group.copy())
        self.group_n.append(n2)

    def variance(self, ddof: int = 1) -> np.ndarray:
        if self.n_patches <= ddof:
            raise ValueError("insufficient calibration patches for variance")
        return self.m2 / float(self.n_patches - ddof)

    def state(self) -> Dict[str, Any]:
        return {
            "d": int(self.d),
            "mu": self.mu.astype(np.float32),
            "m2": self.m2.astype(np.float32),
            "n_patches": int(self.n_patches),
            "n_images": int(self.n_images),
            "group_means": np.stack(self.group_means, axis=0).astype(np.float32) if self.group_means else np.zeros((0, self.d), dtype=np.float32),
            "group_m2": np.stack(self.group_m2, axis=0).astype(np.float32) if self.group_m2 else np.zeros((0, self.d), dtype=np.float32),
            "group_n": np.asarray(self.group_n, dtype=np.int64),
        }


def variance_from_groups(
    group_means: Sequence[np.ndarray],
    group_m2: Sequence[np.ndarray],
    group_n: Sequence[int],
    indices: Sequence[int],
) -> np.ndarray:
    """Pool selected image groups with Chan's parallel update (bootstrap helper)."""

    if len(indices) == 0:
        raise ValueError("at least one group is required")
    mu = None
    m2_total = None
    n_total = 0
    for index in indices:
        mean2 = np.asarray(group_means[index], dtype=np.float64)
        m2_group = np.asarray(group_m2[index], dtype=np.float64)
        n2 = int(group_n[index])
        if mu is None:
            mu = mean2.copy()
            m2_total = m2_group.copy()
            n_total = n2
            continue
        delta = mean2 - mu
        n_new = n_total + n2
        mu = mu + delta * (n2 / n_new)
        m2_total = m2_total + m2_group + delta * delta * (n_total * n2 / n_new)
        n_total = n_new
    if n_total <= 1:
        raise ValueError("insufficient patches for variance")
    return m2_total / float(n_total - 1)


def regularization(var_raw: np.ndarray, rho: float, tau: float, eps_abs: float) -> Dict[str, np.ndarray]:
    """Frozen shrinkage/flooring: (1-rho) vhat + rho mean(vhat), floored."""

    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1)")
    if tau <= 0.0 or eps_abs <= 0.0:
        raise ValueError("tau and eps_abs must be positive")
    var_raw = np.asarray(var_raw, dtype=np.float64)
    if not np.isfinite(var_raw).all() or (var_raw < 0).any():
        raise ValueError("invalid raw variances")
    mean_variance = float(var_raw.mean())
    shrunk = (1.0 - rho) * var_raw + rho * mean_variance
    safe = np.maximum(np.maximum(shrunk, tau * mean_variance), eps_abs)
    return {
        "var_raw": var_raw.astype(np.float32),
        "var_shrunk": shrunk.astype(np.float32),
        "var_safe": safe.astype(np.float32),
        "mean_variance": np.float32(mean_variance),
    }


def anisotropy(var: np.ndarray) -> Dict[str, Any]:
    """Descriptive variance geometry; no anomaly information."""

    var = np.asarray(var, dtype=np.float64)
    if not np.isfinite(var).all() or (var <= 0).any():
        raise ValueError("invalid variance for anisotropy")
    p5, p95 = np.quantile(var, [0.05, 0.95])
    total = float(var.sum())
    descending = np.sort(var)[::-1]
    return {
        "p95_over_p5": float(p95 / p5),
        "max_over_min": float(var.max() / var.min()),
        "coefficient_of_variation": float(var.std() / var.mean()),
        "top1_variance_fraction": float(descending[0] / total),
        "top10_variance_fraction": float(descending[:10].sum() / total),
        "mean_variance": float(var.mean()),
    }


def image_bootstrap_anisotropy(
    group_means: Sequence[np.ndarray],
    group_m2: Sequence[np.ndarray],
    group_n: Sequence[int],
    *,
    reps: int = 200,
    seed: int = 20261021,
) -> Dict[str, Any]:
    """Whole-image bootstrap of variance/anisotropy; patches are never resampled."""

    n_images = len(group_n)
    if n_images < 2:
        raise ValueError("at least two calibration images are required")
    generator = np.random.default_rng(seed)
    ratios = []
    cv_log = []
    var_samples = np.zeros((reps, group_means[0].size), dtype=np.float64)
    for rep in range(reps):
        indices = generator.integers(0, n_images, size=n_images)
        var = variance_from_groups(group_means, group_m2, group_n, indices)
        var_samples[rep] = var
        ratios.append(anisotropy(var)["p95_over_p5"])
        # Descriptive channel heterogeneity (not estimation uncertainty).
        cv_log.append(float(np.std(np.log(var)) / abs(np.mean(np.log(var)))))
    ratios = np.asarray(ratios)
    cv_log = np.asarray(cv_log)
    # True image-bootstrap estimation instability per channel.
    channel_cv = var_samples.std(axis=0) / var_samples.mean(axis=0)
    return {
        "reps": int(reps),
        "seed": int(seed),
        "images": int(n_images),
        "p95_over_p5_mean": float(ratios.mean()),
        "p95_over_p5_interval": np.quantile(ratios, [0.025, 0.975]).tolist(),
        "p95_over_p5_min": float(ratios.min()),
        "cv_log_variance_mean": float(cv_log.mean()),
        "cv_log_variance_interval": np.quantile(cv_log, [0.025, 0.975]).tolist(),
        "bootstrap_channel_cv_mean": float(channel_cv.mean()),
        "bootstrap_channel_cv_median": float(np.median(channel_cv)),
        "bootstrap_channel_cv_p95": float(np.quantile(channel_cv, 0.95)),
        "patches_counted_as_independent": False,
        "note": "cv_log_variance is across-channel heterogeneity; bootstrap_channel_cv is whole-image estimation instability",
    }
