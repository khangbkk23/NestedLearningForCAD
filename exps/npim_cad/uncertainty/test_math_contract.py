import math

import numpy as np
import pytest

from exps.npim_cad.uncertainty import math_contract as mc


def test_gaussian_nll_identity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    covariance = a @ a.T + np.eye(4)
    r = rng.normal(size=4)
    mu = rng.normal(size=4)
    delta = r - mu
    expected = delta @ np.linalg.solve(covariance, delta) + np.linalg.slogdet(covariance)[1] + 4 * math.log(2 * math.pi)
    assert mc.gaussian_nll(r, mu, covariance) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_scale_lrt_is_mahalanobis():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(3, 3))
    covariance = a @ a.T + np.eye(3)
    r = rng.normal(size=3)
    q = mc.mahalanobis(r, np.zeros(3), covariance)
    # -2 log LR = d log c + (1 - 1/c) q, c=4.
    expected = 3 * math.log(4.0) + (1.0 - 0.25) * q
    assert mc.scale_lr_statistic(r, np.zeros(3), covariance, 4.0) == pytest.approx(expected)


def test_2d_moments():
    result = mc.second_moment_2d()
    assert result["l2_normal_mean"] == 101.0
    assert result["l2_normal_variance"] == 20002.0
    assert result["l2_anomaly_mean"] == 117.0
    assert result["q_normal_mean"] == 2.0
    assert result["q_normal_variance"] == 4.0
    assert result["q_anomaly_mean"] == 18.0
    assert result["noncentrality"] == 16.0


def test_isotropic_is_monotone():
    result = mc.isotropic_equivalence(n=5000, seed=3)
    assert result["ranking_identical"] is True
    assert result["auc_l2"] == pytest.approx(result["auc_whitened"], abs=1e-12)


def test_epsilon_counterexample():
    result = mc.epsilon_counterexample(epsilon=0.1, n=150_000, seed=4)
    assert result["analytic_lrt_auc"] == pytest.approx(0.704832764, abs=1e-8)
    assert result["l2_close_to_lrt"] is True
    assert result["whitened_below_l2"] is True
    assert result["whitened_margin"] > 0.01
    reversal = mc.epsilon_point_reversal(0.1)
    assert reversal["l2_correct"] is True
    assert reversal["diag_correct"] is False
    assert reversal["rank_reversal"] is True


def test_high_variance_mean_shift_counterexample():
    result = mc.high_variance_mean_shift_counterexample(delta=20.0, n=150_000, seed=5)
    assert result["l2_better"] is True
    assert result["margin"] > 0.01
    reversal = mc.high_variance_point_reversal()
    assert reversal["l2_correct"] is True
    assert reversal["diag_correct"] is False
    assert reversal["rank_reversal"] is True


def test_correlated_diagonal_is_not_chi_square():
    result = mc.correlated_diagonal_counterexample(rho=0.9, n=100_000, seed=6)
    assert result["mean_diag"] == pytest.approx(2.0, abs=0.05)
    assert result["mean_full"] == pytest.approx(2.0, abs=0.05)
    assert result["variance_full"] == pytest.approx(4.0, abs=0.10)
    assert result["diag_not_chi2"] is True
    assert result["variance_diag"] > 5.0


def test_multimodal_counterexample():
    result = mc.multimodal_counterexample(n=120_000, seed=7)
    assert result["auc_true_mixture_lrt"] > result["auc_l2"] + 0.10
    assert result["auc_true_mixture_lrt"] > result["auc_single_gaussian_moment"] + 0.10


def test_coincident_distributions():
    result = mc.coincident_counterexample(n=80_000, seed=8)
    assert result["auc_l2"] == pytest.approx(0.5, abs=0.01)
