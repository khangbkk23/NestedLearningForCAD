"""Exact tied-score ranking and conservative whole-image AP bootstrap on CUDA."""
from __future__ import annotations

import numpy as np
import torch


def exact_ranking(scores, targets):
    values, order = torch.sort(scores.flatten(), descending=True, stable=True)
    positive = targets.flatten()[order].to(torch.float64).cumsum(0)
    ends = torch.cat((torch.nonzero(values[:-1] != values[1:]).flatten(),
                      torch.tensor([values.numel()-1], device=values.device)))
    true_positive = positive[ends]
    total = ends.to(torch.float64) + 1
    increments = true_positive - torch.cat((true_positive.new_zeros(1), true_positive[:-1]))
    positives = true_positive[-1]
    negatives = total[-1] - positives
    if positives <= 0 or negatives <= 0:
        raise RuntimeError("degenerate native pixel metric")
    average_precision = ((increments / positives) * true_positive / total).sum()
    false_positive = total - true_positive
    auc = torch.trapezoid(torch.cat((true_positive.new_zeros(1), true_positive / positives)),
                          torch.cat((false_positive.new_zeros(1), false_positive / negatives)))
    return {"pixel_ap": float(average_precision), "pixel_auroc": float(auc),
            "prevalence": float(positives / total[-1]), "native_pixels": values.numel(),
            "positive_pixels": int(positives)}


def bootstrap_bounds(image_scores, masks, weights, bins=65536, chunk=32):
    device = image_scores[0].device
    lower_score = min(float(score.min()) for score in image_scores)
    upper_score = max(float(score.max()) for score in image_scores)
    positives, totals = [], []
    for score, mask in zip(image_scores, masks):
        indices = torch.zeros(score.numel(), dtype=torch.int64, device=device) if lower_score == upper_score else (
            (score.flatten().double()-lower_score) / (upper_score-lower_score) * (bins-1)).long().clamp(max=bins-1)
        indices = bins - 1 - indices
        positives.append(torch.bincount(indices[mask.flatten()], minlength=bins).double())
        totals.append(torch.bincount(indices, minlength=bins).double())
    positive_histograms, total_histograms = torch.stack(positives), torch.stack(totals)
    lower, upper = [], []
    for batch in torch.as_tensor(weights, dtype=torch.float64, device=device).split(chunk):
        positive = batch @ positive_histograms
        total = batch @ total_histograms
        prior_positive, prior_total = positive.cumsum(1)-positive, total.cumsum(1)-total
        negative_first = prior_total + total - positive
        harmonic = torch.special.digamma(negative_first + positive + 1) - torch.special.digamma(negative_first + 1)
        low = (positive - (negative_first-prior_positive)*harmonic).clamp(min=0)
        low = torch.minimum(low, positive).sum(1) / positive.sum(1)
        denominator = prior_total+positive
        high = torch.where(denominator > 0, positive*(prior_positive+positive)/denominator.clamp(min=1), 0)
        high = high.sum(1) / positive.sum(1)
        lower.append((low-1e-7).clamp(min=0).cpu().numpy())
        upper.append((high+1e-7).clamp(max=1).cpu().numpy())
    return np.concatenate(lower), np.concatenate(upper)


def image_auroc(labels, scores, weights):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    comparison = (scores[positive, None] > scores[negative]).astype(float)
    comparison += 0.5 * (scores[positive, None] == scores[negative])
    weights = np.atleast_2d(weights)
    return ((weights[:, positive] @ comparison) * weights[:, negative]).sum(1) / (
        weights[:, positive].sum(1) * weights[:, negative].sum(1))
