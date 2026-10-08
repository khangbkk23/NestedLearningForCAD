# exps/hope_anomaly_signal.py
"""Read-only residual detection and normal statistical controls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F

from exps.hope_image_synchronous_memory import ImageSynchronousMemory, fingerprint


CATEGORIES = ("bottle", "carpet", "hazelnut")
METHOD_NAMES = {"FROZEN": "FROZEN", "P0": "P0_BASE", "P1": "P1_SYNC", "P2": "P2_PROJ"}
CHECKPOINTS = (0, 100, 200, 300)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_manifests(root: Path, limit: int = 10) -> tuple[list[dict], list[dict]]:
    """Fix disjoint development/confirmation images before detector evaluation."""
    def ordered(folder):
        return sorted(folder.glob("*.png"), key=lambda p: (p.name, sha256(p)))

    def round_robin(groups):
        return [paths[index] for index in range(max(map(len, groups), default=0))
                for paths in groups if index < len(paths)]

    def record(path, split):
        relative = path.relative_to(root).as_posix()
        category, _, defect, _ = relative.split("/")
        label = int(defect != "good")
        mask = f"{category}/ground_truth/{defect}/{path.stem}_mask.png" if label else None
        if mask is not None and not (root / mask).is_file():
            raise FileNotFoundError(root / mask)
        with Image.open(path) as image:
            width, height = image.size
        if label:
            with Image.open(root / mask) as source:
                if source.size != (width, height):
                    raise ValueError("image and native mask geometry differ")
        return {"category": category, "relative_path": relative, "image_id": relative,
                "label": label, "defect_type": defect, "mask_path": mask,
                "height": height, "width": width, "split": split,
                "image_sha256": sha256(path), "mask_sha256": sha256(root / mask) if mask else None}

    dev, confirmation = [], []
    for category in CATEGORIES:
        folder = root / category / "test"
        normal = ordered(folder / "good")
        anomaly = round_robin([ordered(p) for p in sorted(folder.iterdir()) if p.is_dir() and p.name != "good"])
        for group in (normal, anomaly):
            dev.extend(record(path, "DEVELOPMENT") for path in group[:limit])
            confirmation.extend(record(path, "CONFIRMATION_UNEVALUATED") for path in group[limit:2 * limit])
    if {row["image_id"] for row in dev} & {row["image_id"] for row in confirmation}:
        raise ValueError("development and confirmation overlap")
    return dev, confirmation


def native_mask(root: Path, row: dict) -> np.ndarray:
    if not row["label"]:
        return np.zeros((row["height"], row["width"]), dtype=bool)
    with Image.open(root / row["mask_path"]) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != (row["height"], row["width"]):
        raise ValueError("native mask shape differs")
    return mask


@torch.no_grad()
def residual_scores(model: ImageSynchronousMemory, features: torch.Tensor) -> dict[str, Any]:
    """Evaluate the current K/V residual and Q readout from one immutable state."""
    before = model.state_fingerprint()
    snapshot = model.snapshot_state()
    quantities = model.generate_update_quantities(features, snapshot)
    prediction = F.linear(quantities.keys, snapshot.weights["memory"])
    residual = prediction - quantities.values
    scores = residual.square().sum(-1)
    q_output = model.read_from_snapshot(quantities, snapshot)
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite associative residual")
    if model.state_fingerprint() != before:
        raise ValueError("residual evaluation mutated memory")
    return {"patch_scores": scores.detach().cpu(), "image_score": float(scores.max()),
            "kv_residual_rms": float(residual.double().square().mean().sqrt()),
            "key_prediction_rms": float(prediction.double().square().mean().sqrt()),
            "value_rms": float(quantities.values.double().square().mean().sqrt()),
            "q_output_rms": float(q_output.double().square().mean().sqrt()),
            "snapshot_hash": snapshot.identity}


def pixel_map(patch_scores: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    if patch_scores.numel() != 784 or not torch.isfinite(patch_scores).all():
        raise ValueError("784 finite spatial patch scores required")
    return F.interpolate(patch_scores.detach().cpu().reshape(1, 1, 28, 28).float(), size=shape,
                         mode="bilinear", align_corners=False)[0, 0].numpy()


class PooledCovariance:
    """Fixed-query pooled normal statistics with a declared shrinkage."""

    def __init__(self, dim: int, device="cpu"):
        self.count = torch.zeros((), dtype=torch.int64, device=device)
        self.completed_events = torch.zeros((), dtype=torch.int64, device=device)
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.scatter = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        self.factor = torch.zeros_like(self.scatter)

    @torch.no_grad()
    def update(self, queries: torch.Tensor):
        x = queries.detach().double()
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not torch.isfinite(x).all() or not len(x):
            raise ValueError("finite fixed-basis rows required")
        n, previous = len(x), int(self.count)
        batch_mean = x.mean(0)
        centered = x - batch_mean
        delta = batch_mean - self.mean
        self.scatter.add_(centered.T @ centered + torch.outer(delta, delta) * (previous * n / (previous + n)))
        self.mean.add_(delta * (n / (previous + n)))
        self.count.add_(n)
        self.completed_events.add_(1)
        self.factor.zero_()

    @torch.no_grad()
    def factorize(self):
        if int(self.count) < 2:
            raise ValueError("covariance detector is not fitted")
        covariance = self.scatter / (int(self.count) - 1)
        identity = torch.eye(len(self.mean), dtype=self.mean.dtype, device=self.mean.device)
        regularized = 0.9 * covariance + (0.1 * covariance.trace() / len(self.mean) + 1e-6) * identity
        self.factor.copy_(torch.linalg.cholesky(regularized))

    @torch.no_grad()
    def score(self, queries: torch.Tensor) -> torch.Tensor:
        if not torch.any(self.factor.diag()):
            raise ValueError("covariance factorization required before read-only scoring")
        centered = queries.detach().double() - self.mean
        whitened = torch.linalg.solve_triangular(self.factor, centered.T, upper=False)
        return whitened.square().sum(0).detach().cpu()

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu().clone()
                for name in ("count", "completed_events", "mean", "scatter", "factor")}

    def load_state_dict(self, state):
        if set(state) != {"count", "completed_events", "mean", "scatter", "factor"}:
            raise ValueError("covariance state schema differs")
        for name, value in state.items():
            reference = getattr(self, name)
            if value.shape != reference.shape or value.dtype != reference.dtype or not torch.isfinite(value).all():
                raise ValueError("covariance state geometry differs")
        for name, value in state.items():
            getattr(self, name).copy_(value.to(self.mean.device))

    def storage(self):
        state = self.state_dict()
        return {"persistent_bytes": sum(t.numel() * t.element_size() for t in state.values()),
                "factorization_cache_bytes": self.factor.numel() * self.factor.element_size(),
                "count": int(self.count), "image_events": int(self.completed_events)}


def bootstrap_image_counts(labels: np.ndarray, *, repetitions=400, seed=0) -> np.ndarray:
    """Paired stratified resampling units are images, never patches/pixels."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    counts = np.zeros((repetitions, len(labels)), dtype=np.int64)
    for label in (0, 1):
        ids = np.flatnonzero(labels == label)
        if not len(ids):
            raise ValueError("both image labels required")
        sampled = rng.choice(ids, (repetitions, len(ids)), replace=True)
        for row in range(repetitions):
            counts[row] += np.bincount(sampled[row], minlength=len(labels))
    return counts


def histogram_ap(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    positive, negative = np.atleast_2d(positive), np.atleast_2d(negative)
    tp, fp = np.cumsum(positive[:, ::-1], axis=1), np.cumsum(negative[:, ::-1], axis=1)
    precision = tp / np.maximum(tp + fp, 1)
    return (precision * positive[:, ::-1]).sum(1) / np.maximum(positive.sum(1), 1)


@dataclass
class MetricResult:
    metrics: dict[str, float]
    bootstrap_image_auroc: np.ndarray
    bootstrap_pixel_aupr: np.ndarray


def evaluate_metrics(image_scores, labels, maps, masks, counts) -> MetricResult:
    """Exact point estimates plus image-block percentile bootstrap intervals."""
    scores = np.asarray(image_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    if not (len(scores) == len(labels) == len(maps) == len(masks)) or counts.shape[1] != len(scores):
        raise ValueError("image identities and bootstrap units differ")
    if any(np.shape(values) != mask.shape for values, mask in zip(maps, masks)):
        raise ValueError("pixel scores and native masks differ")
    pixel_scores = np.concatenate([np.asarray(x, dtype=np.float32).ravel() for x in maps])
    pixel_labels = np.concatenate([np.asarray(x, dtype=np.int8).ravel() for x in masks])
    if not np.isfinite(scores).all() or not np.isfinite(pixel_scores).all():
        raise ValueError("non-finite anomaly scores")
    point = {"image_AUROC": float(roc_auc_score(labels, scores)),
             "image_AP": float(average_precision_score(labels, scores)),
             "pixel_AUPR": float(average_precision_score(pixel_labels, pixel_scores)),
             "pixel_AUROC": float(roc_auc_score(pixel_labels, pixel_scores)),
             "pixel_prevalence": float(pixel_labels.mean())}
    image_boot = np.array([roc_auc_score(labels, scores, sample_weight=row) for row in counts])
    # Only interval computation groups score thresholds.  Every image's full
    # native-resolution pixel mass stays together under its sampled weight.
    # The exact point estimate is independent of this computational reduction.
    bins = 16384
    while True:
        edges = np.linspace(float(pixel_scores.min()), float(pixel_scores.max()) + 1e-9, bins + 1)
        positive, negative = [], []
        for values, target in zip(maps, masks):
            values, target = np.asarray(values).ravel(), np.asarray(target).ravel()
            positive.append(np.histogram(values[target], edges)[0])
            negative.append(np.histogram(values[~target], edges)[0])
        positive, negative = np.asarray(positive), np.asarray(negative)
        approximate = float(histogram_ap(positive.sum(0), negative.sum(0))[0])
        error = abs(approximate - point["pixel_AUPR"])
        if error <= 2e-4 or bins == 262144:
            break
        bins *= 2
    if error > 2e-4:
        raise ValueError("pixel bootstrap threshold grouping exceeds declared AP tolerance")
    pixel_boot = []
    for start in range(0, len(counts), 16):
        weights = counts[start:start + 16]
        pixel_boot.extend(histogram_ap(weights @ positive, weights @ negative))
    pixel_boot = np.asarray(pixel_boot)
    for name, array in (("image_AUROC", image_boot), ("pixel_AUPR", pixel_boot)):
        point[f"{name}_ci_low"], point[f"{name}_ci_high"] = map(float, np.quantile(array, [0.025, 0.975]))
    point.update({"pixel_bootstrap_histogram_bins": bins, "pixel_bootstrap_point_error": error,
                  "bootstrap_repetitions": len(counts)})
    return MetricResult(point, image_boot, pixel_boot)


def score_distribution(image_scores, labels, maps, masks) -> dict[str, float]:
    scores, labels = np.asarray(image_scores), np.asarray(labels)
    normal, anomaly = scores[labels == 0], scores[labels == 1]
    defect = np.concatenate([m[gt] for m, gt in zip(maps, masks) if gt.any()])
    background = np.concatenate([m[~gt] for m, gt in zip(maps, masks)])
    normal_pixels = np.concatenate([m.ravel() for m, label in zip(maps, labels) if label == 0])
    anomaly_background = np.concatenate([m[~gt] for m, gt, label in zip(maps, masks, labels) if label == 1])
    return {"normal_score_mean": float(normal.mean()), "normal_score_median": float(np.median(normal)),
            "normal_score_q90": float(np.quantile(normal, .9)), "normal_score_q95": float(np.quantile(normal, .95)),
            "anomaly_score_mean": float(anomaly.mean()), "anomaly_score_median": float(np.median(anomaly)),
            "anomaly_score_q10": float(np.quantile(anomaly, .1)), "anomaly_score_q50": float(np.median(anomaly)),
            "separation_margin": float(np.median(anomaly) - np.median(normal)),
            "normal_pixel_mean": float(normal_pixels.mean()), "defect_score_mean": float(defect.mean()),
            "background_score_mean": float(background.mean()),
            "anomaly_background_score_mean": float(anomaly_background.mean()),
            "defect_background_ratio": float(defect.mean() / max(anomaly_background.mean(), np.finfo(float).tiny))}
