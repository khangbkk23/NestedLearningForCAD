"""Analysis-only correction: AP upper bounds must include tied positive scores.

The frozen protocol requires bounds over *all* within-bin rankings, including
ties. The originally frozen harmonic upper expression covered strictly ordered
positives only. Tying all positives before negatives gives the correct upper
bound. Lower harmonic bound, point metrics, scoring, draws and gates are unchanged.
"""
import numpy as np
from scipy.special import digamma

from exps.task_p.p0_normal_support import require
from exps.task_p.p1_readout import BoundedPixelBootstrap


class CorrectedBoundedPixelBootstrap(BoundedPixelBootstrap):
    def bounds(self, weights):
        weights = np.atleast_2d(weights)
        p, t = weights @ self.positive, weights @ self.total
        require((p.sum(1) > 0).all(), "missing anomaly pixels in bootstrap")
        prior_p, prior_t = np.cumsum(p, axis=1) - p, np.cumsum(t, axis=1) - t
        negative_first = prior_t + t - p
        harmonic_delta = digamma(negative_first + p + 1) - digamma(negative_first + 1)
        low = np.clip(p - (negative_first - prior_p) * harmonic_delta, 0, p).sum(1) / p.sum(1)
        high_contribution = np.divide(p * (prior_p + p), prior_t + p,
                                      out=np.zeros_like(p), where=(prior_t + p) > 0)
        high = high_contribution.sum(1) / p.sum(1)
        return np.maximum(0., low - 1e-7), np.minimum(1., high + 1e-7)


if __name__ == "__main__":
    from exps.task_p import analyze_p1
    from exps.task_p.p1_protocol import configure
    configure()
    analyze_p1.BoundedPixelBootstrap = CorrectedBoundedPixelBootstrap
    result = analyze_p1.analyze()
    print(result["scientific_decision"], flush=True)
