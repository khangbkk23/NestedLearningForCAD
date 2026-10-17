"""P1's single task-free readout and exact, image-grouped native pixel metrics.

Pixel AUPR means non-interpolated average precision (AP), not trapezoidal PR area.
Histograms below group *identical* FP32 scores; they do not quantize or resize pixels.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import digamma
from torch.nn.functional import interpolate

from exps.task_p.p0_normal_support import Support, fingerprint, require

ARMS = ("GRID16_GREEDY", "FULL784_GREEDY", "FULL784_RESERVOIR")
GRID16 = tuple(r * 28 + c for r in (3, 10, 17, 24) for c in (3, 10, 17, 24))


def features_ok(features):
    require(features.shape == (784, 768) and features.dtype == torch.float32,
            "expected raw FP32 [784,768] features")
    require(bool(torch.isfinite(features).all()), "nonfinite features")


def new_models(capacity=None):
    return {arm: Support("RESERVOIR" if arm.endswith("RESERVOIR") else "STREAM_GREEDY",
                         capacity=capacity) for arm in ARMS}


def restore(state):
    model = Support(state["method"], capacity=state["capacity"], dim=state["bank"].shape[1])
    for key, value in state.items():
        if key != "rng_state":
            setattr(model, key, value.clone() if isinstance(value, torch.Tensor) else value)
    model.rng.set_state(state["rng_state"])
    return model


def native_map(distances, hw):
    require(distances.shape == (784,) and bool(torch.isfinite(distances).all()), "bad patch scores")
    require(len(hw) == 2 and min(hw) > 0, "invalid native shape")
    # FP64 NN distances are rounded once, before the common FP32 bilinear readout.
    return interpolate(distances.float().reshape(1, 1, 28, 28), size=tuple(hw),
                       mode="bilinear", align_corners=False, antialias=False)[0, 0].numpy()


def readout(model, features, hw):
    """Receives no product identity, label, mask or per-category calibration."""
    features_ok(features)
    before = fingerprint(model.state())
    # Bound the FP64 query-bank workspace without changing the frozen distance utility.
    parts = [model.score(chunk)[0] for chunk in features.split(64)]
    d = torch.cat(parts)
    require(fingerprint(model.state()) == before, "evaluation mutated memory")
    return {"patch_distances": d.numpy(), "map": native_map(d, hw),
            "image_score": float(d.max()), "state_sha256": before}


def score_histogram(scores, mask):
    scores, mask = np.asarray(scores), np.asarray(mask)
    require(scores.shape == mask.shape and scores.ndim == 2, "native score/mask alignment mismatch")
    require(mask.dtype == np.bool_ and np.isfinite(scores).all(), "invalid mask or scores")
    values, inv, counts = np.unique(scores.ravel(), return_inverse=True, return_counts=True)
    positive = np.bincount(inv, weights=mask.ravel().astype(np.int64), minlength=len(values))
    return {"score": values, "positive": positive.astype(np.int64), "total": counts}


def count_metrics(positive, total):
    """Inputs are counts at distinct thresholds, descending score; exact ties handled together."""
    p, n = float(positive.sum()), float((total - positive).sum())
    if p == 0 or n == 0:
        return {"ap": None, "auroc": None, "prevalence": p / (p + n) if p + n else None,
                "degenerate": "missing positive or negative pixels/images"}
    tp, fp = np.cumsum(positive), np.cumsum(total - positive)
    ap = float(np.sum((positive / p) * np.divide(tp, tp + fp,
                   out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)))
    auc = float(np.trapezoid(np.r_[0., tp / p], np.r_[0., fp / n]))
    return {"ap": ap, "auroc": auc, "prevalence": p / (p + n), "degenerate": None}


class PixelRanking:
    """Sort once, then reweight complete images for paired cluster bootstrap.

    No pixels are sampled independently. RAM is proportional to the number of
    distinct native scores within each image; the runner preflights that cost.
    """
    def __init__(self, histograms):
        require(bool(histograms), "empty metric fixture")
        self.images = len(histograms)
        scores = np.concatenate([h["score"] for h in histograms])
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        self.starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        self.positive = np.concatenate([h["positive"] for h in histograms])[order]
        self.total = np.concatenate([h["total"] for h in histograms])[order]
        self.group = np.repeat(np.arange(self.images), [len(h["score"]) for h in histograms])[order]

    def metrics(self, image_weights=None):
        weights = np.ones(self.images) if image_weights is None else np.asarray(image_weights)
        require(weights.shape == (self.images,) and (weights >= 0).all(), "invalid image weights")
        w = weights[self.group]
        pos = np.add.reduceat(self.positive * w, self.starts)
        total = np.add.reduceat(self.total * w, self.starts)
        return count_metrics(pos, total)


def image_metrics(labels, scores, weights=None):
    y, s = np.asarray(labels), np.asarray(scores)
    require(y.ndim == s.ndim == 1 and len(y) == len(s) and np.isin(y, [0, 1]).all()
            and np.isfinite(s).all(), "invalid image labels or scores")
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    order = np.argsort(-s, kind="stable")
    starts = np.r_[0, np.flatnonzero(s[order][1:] != s[order][:-1]) + 1]
    return count_metrics(np.add.reduceat((y * w)[order], starts), np.add.reduceat(w[order], starts))


def bootstrap_weights(labels, generator):
    """Stratify by image class within category; all arms and boundaries share draws."""
    y = np.asarray(labels)
    weights = np.zeros(len(y), dtype=np.int64)
    for label in (0, 1):
        group = np.flatnonzero(y == label)
        require(len(group) > 0, "degenerate image category")
        weights += np.bincount(generator.choice(group, len(group), replace=True), minlength=len(y))
    return weights


def interval(samples):
    a = np.asarray(samples, dtype=float)
    require(np.isfinite(a).all() and len(a) > 0, "invalid uncertainty samples")
    return np.quantile(a, [.025, .975]).tolist()


class BoundedPixelBootstrap:
    """Whole-image AP intervals with rigorous within-bin ranking bounds.

    Exact point AP uses PixelRanking. Bootstrap uses fixed 65,536 score bins,
    but encloses every possible ordering *inside* each bin, including original
    exact scores and ties. Thus no unmeasured quantization-error tolerance is
    needed. Bins affect interval width only, never the reported point metric.
    """
    def __init__(self, histograms, bins=65536):
        lo = min(float(h["score"].min()) for h in histograms)
        hi = max(float(h["score"].max()) for h in histograms)
        self.positive = np.zeros((len(histograms), bins), dtype=np.float64)
        self.total = np.zeros_like(self.positive)
        for i, h in enumerate(histograms):
            ix = np.zeros(len(h["score"]), dtype=int) if hi == lo else np.minimum(
                ((h["score"].astype(float) - lo) / (hi - lo) * (bins - 1)).astype(int), bins - 1)
            ix = bins - 1 - ix
            self.positive[i] = np.bincount(ix, weights=h["positive"], minlength=bins)
            self.total[i] = np.bincount(ix, weights=h["total"], minlength=bins)

    def bounds(self, weights):
        weights = np.atleast_2d(weights)
        p, t = weights @ self.positive, weights @ self.total
        require((p.sum(1) > 0).all(), "missing anomaly pixels in bootstrap")
        prior_p, prior_t = np.cumsum(p, axis=1) - p, np.cumsum(t, axis=1) - t
        negatives = prior_t - prior_p
        def contribution(offset):
            # Sum_{j=1..p} (prior_p+j)/(prior_t+offset+j), using harmonic numbers.
            harmonic_delta = digamma(prior_t + offset + p + 1) - digamma(prior_t + offset + 1)
            return np.clip(p - (negatives + offset) * harmonic_delta, 0, p).sum(1) / p.sum(1)
        # Outward numerical padding for FP64 harmonic-number cancellation.
        return (np.maximum(0., contribution(t - p) - 1e-7),
                np.minimum(1., contribution(np.zeros_like(p)) + 1e-7))


def classify(final, acquisition, worst_loss, final_ci):
    """Project feasibility criteria; not a published benchmark pass threshold.

    final/acquisition map arm to category metrics; final_ci contains paired macro
    image AUROC and AP-minus-prevalence lower limits. Never pick another scorer.
    """
    def credible(categories):
        return sum(m["image_auroc"] >= .70 and m["pixel_ap"] >= max(3 * m["prevalence"],
                   m["prevalence"] + .05) for m in categories.values()) >= 2
    retained = any(credible(final[a]) and final_ci[a]["image_auroc"][0] > .5
                   and final_ci[a]["ap_minus_prevalence"][0] > 0 for a in ARMS)
    if any(loss > .10 for loss in worst_loss.values()):
        return "P1_CONTINUAL_RETENTION_FAILURE"
    if retained:
        return "P1_SIGNAL_VALID"
    if all(all(m["image_auroc"] <= .55 and m["pixel_ap"] <= m["prevalence"] + .02
               for m in acquisition[a].values()) for a in ARMS):
        return "P1_REPRESENTATION_OR_SCORER_FAILURE"
    return "P1_INCONCLUSIVE"
