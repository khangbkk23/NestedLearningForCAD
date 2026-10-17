"""Independent regression for exact AP ties; synthetic values only."""
import numpy as np
from sklearn.metrics import average_precision_score

from exps.task_p.p1_bootstrap_bounds import CorrectedBoundedPixelBootstrap
from exps.task_p.p1_readout import BoundedPixelBootstrap, PixelRanking, score_histogram


def test_frozen_upper_bound_counterexample_and_correction():
    # Ten higher-scoring negatives followed by two tied positives. AP=2/12.
    scores = np.array([[2.] * 10 + [1.] * 2], dtype=np.float32)
    mask = np.array([[False] * 10 + [True] * 2])
    histogram = score_histogram(scores, mask)
    point = average_precision_score(mask.ravel(), scores.ravel())
    old_lower, old_upper = BoundedPixelBootstrap([histogram], bins=3).bounds([[1]])
    assert old_upper[0] < point  # retained evidence of the original implementation bug
    lower, upper = CorrectedBoundedPixelBootstrap([histogram], bins=3).bounds([[1]])
    assert lower[0] <= point <= upper[0]


def test_corrected_bounds_cover_weighted_exact_ties_and_coarse_score_buckets():
    gen = np.random.default_rng(1028)
    for bins in (2, 3, 65536):
        for _ in range(20):
            scores = [gen.integers(0, 5, size=(11, 13)).astype(np.float32) for _ in range(4)]
            masks = [gen.random((11, 13)) < .2 for _ in range(4)]
            histograms = [score_histogram(s, m) for s, m in zip(scores, masks)]
            weights = gen.integers(0, 4, size=(1, 4))
            if not weights.any():
                weights[0, 0] = 1
            point = PixelRanking(histograms).metrics(weights[0])["ap"]
            lower, upper = CorrectedBoundedPixelBootstrap(histograms, bins=bins).bounds(weights)
            assert lower[0] <= point <= upper[0]
