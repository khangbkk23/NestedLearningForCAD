# exps/npim_cad/test_metric_parity.py
"""G8 oracles: native bilinear geometry, tie-aware AP, image max, image-level bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.task_p.p1_readout import (
    PixelRanking,
    bootstrap_weights,
    count_metrics,
    image_metrics,
    native_map as p1_native_map,
    score_histogram,
)
from models.hope_cad.npim_cad import image_score, native_map


def brute_force_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    positives = labels.sum()
    true_positive = false_positive = 0
    total = 0.0
    index = 0
    while index < len(sorted_scores):
        end = index
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        group_positive = int(sorted_labels[index:end].sum())
        group_total = end - index
        true_positive += group_positive
        false_positive += group_total - group_positive
        if group_positive:
            total += (group_positive / positives) * (true_positive / (true_positive + false_positive))
        index = end
    return total


def test_native_map_matches_p1_bitwise():
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(784, generator=generator)
    for hw in ((512, 700), (1024, 1024), (900, 900), (256, 300)):
        ours = native_map(scores, hw).numpy()
        reference = p1_native_map(scores, hw)
        assert np.array_equal(ours, reference), hw
        assert ours.shape == tuple(hw)


def test_image_score_is_max_patch_score():
    generator = torch.Generator().manual_seed(1)
    scores = torch.rand(784, generator=generator)
    assert image_score(scores) == float(scores.max().item())


def test_tie_aware_ap_matches_independent_bruteforce():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 6, size=(24, 32)).astype(np.float64)  # many ties
    mask = rng.random((24, 32)) < 0.12
    mask[0, 0] = True
    histogram = score_histogram(scores, mask)
    # score_histogram returns ascending distinct scores; count_metrics expects
    # descending thresholds (as PixelRanking supplies them).
    metrics = count_metrics(histogram["positive"][::-1], histogram["total"][::-1])
    expected = brute_force_ap(scores, mask)
    assert abs(metrics["ap"] - expected) < 1e-12
    assert metrics["degenerate"] is None


def test_ap_is_not_trapezoidal_pr_auc():
    sklearn = pytest.importorskip("sklearn.metrics")
    scores = np.array([[3.0, 2.0, 1.0]])
    mask = np.array([[True, False, True]])
    histogram = score_histogram(scores, mask)
    metrics = count_metrics(histogram["positive"][::-1], histogram["total"][::-1])
    independent_ap = sklearn.average_precision_score(mask.ravel(), scores.ravel())
    precision, recall, _ = sklearn.precision_recall_curve(mask.ravel(), scores.ravel())
    trapezoid = sklearn.auc(recall, precision)
    assert abs(metrics["ap"] - independent_ap) < 1e-12
    assert abs(metrics["ap"] - trapezoid) > 1e-6


def test_image_metrics_handle_ties_exactly():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.5, 0.5, 0.1, 0.1])
    metrics = image_metrics(labels, scores)
    expected = brute_force_ap(scores, labels)
    assert abs(metrics["ap"] - expected) < 1e-12


def test_pixel_ranking_matches_pooled_counts():
    rng = np.random.default_rng(2)
    histograms = []
    all_scores, all_masks = [], []
    for _ in range(6):
        scores = rng.integers(0, 8, size=(20, 20)).astype(np.float64)
        mask = rng.random((20, 20)) < 0.1
        histograms.append(score_histogram(scores, mask))
        all_scores.append(scores.ravel())
        all_masks.append(mask.ravel())
    ranking = PixelRanking(histograms)
    metrics = ranking.metrics()
    # Comparing against the independent brute-force AP over the pooled native
    # pixels is the strongest check; `count_metrics` needs one aligned histogram.
    expected = brute_force_ap(np.concatenate(all_scores), np.concatenate(all_masks))
    assert abs(metrics["ap"] - expected) < 1e-12


def test_bootstrap_weights_are_whole_image_units():
    labels = np.array([0, 0, 1, 1, 1])
    generator = np.random.default_rng(3)
    weights = bootstrap_weights(labels, generator)
    assert weights.shape == labels.shape
    assert np.issubdtype(weights.dtype, np.integer)
    assert weights.sum() == len(labels)
    assert weights[labels == 0].sum() == 2
    assert weights[labels == 1].sum() == 3
    assert (weights >= 0).all()


def test_degenerate_metrics_are_explicit():
    histogram = score_histogram(np.zeros((3, 3)), np.zeros((3, 3), dtype=bool))
    metrics = count_metrics(histogram["positive"], histogram["total"])
    assert metrics["ap"] is None and metrics["degenerate"]
