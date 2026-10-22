# exps/npim_cad/uncertainty/math_contract.py
"""EXP-NPIM-03 mathematical contract and counterexample oracles.

No anomaly data and no model state are involved.  The functions here establish:

* the Gaussian NLL/Mahalanobis identity and its determinant caveat;
* Neyman-Pearson optimality only for the known covariance-scale family;
* the noncentral-chi-square mean-shift geometry;
* explicit counterexamples where diagonal whitening ties or loses to L2;
* the adversarial epsilon model with an analytic LRT AUC and a simulated
  whitened/L2 comparison;
* correlated, multimodal and coincident null cases.

These are mathematical diagnostics, not NPIM anomaly results.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy.stats import norm, rankdata


def gaussian_logpdf(r: np.ndarray, mu: np.ndarray, covariance: np.ndarray) -> float:
    """Log N(r; mu, Sigma) for one vector, with a Cholesky solve."""

    r = np.asarray(r, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    d = r.size
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    delta = r - mu
    chol = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(chol, delta)
    return float(-0.5 * (d * math.log(2.0 * math.pi) + logdet + solved @ solved))


def gaussian_nll(r: np.ndarray, mu: np.ndarray, covariance: np.ndarray) -> float:
    return float(-2.0 * gaussian_logpdf(r, mu, covariance))


def mahalanobis(r: np.ndarray, mu: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(r, dtype=np.float64).reshape(-1) - np.asarray(mu, dtype=np.float64).reshape(-1)
    return float(delta @ np.linalg.solve(np.asarray(covariance, dtype=np.float64), delta))


def scale_lr_statistic(r: np.ndarray, mu: np.ndarray, covariance: np.ndarray, scale: float) -> float:
    """-2 log(p_A/p_N) for H_N N(mu,Sigma) vs H_A N(mu,cSigma), c>1, up to constants."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    d = np.asarray(r).size
    return float(d * math.log(scale) + (1.0 - 1.0 / scale) * mahalanobis(r, mu, covariance))


def lrt_auc_epsilon_model() -> float:
    """Analytic AUC of the exact LRT in the epsilon model.

    H_N: r1~N(0,1), H_A: r1~N(0,4).  P(r1_A^2 > r1_N^2)
    = 1 - (2/pi) atan(1/2) = 0.704832764...
    """

    return float(1.0 - (2.0 / math.pi) * math.atan(0.5))


def auc_from_scores(normal_scores: np.ndarray, anomaly_scores: np.ndarray) -> float:
    """Exact tie-aware AUC = P(anomaly > normal) from ranks."""

    normal_scores = np.asarray(normal_scores, dtype=np.float64).ravel()
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64).ravel()
    if normal_scores.size == 0 or anomaly_scores.size == 0:
        raise ValueError("both score sets are required")
    ranks = rankdata(np.concatenate([normal_scores, anomaly_scores]))
    n_normal = normal_scores.size
    n_anomaly = anomaly_scores.size
    u = ranks[n_normal:].sum() - n_anomaly * (n_anomaly + 1) / 2.0
    return float(u / (n_normal * n_anomaly))


def _simulate_auc(sampler, n: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    normal, anomaly = sampler(rng, n)
    return {name: auc_from_scores(normal[name], anomaly[name]) for name in normal}


def epsilon_counterexample(epsilon: float = 0.1, n: int = 200_000, seed: int = 0) -> Dict[str, Any]:
    """H_N: r1~N(0,1),r2~N(0,eps^2); H_A: r1~N(0,4),r2~N(0,eps^2).

    LRT ranks by r1^2; L2 adds a class-independent nuisance r2^2; whitening adds
    an amplified class-independent nuisance r2^2/eps^2.  The LRT AUC is analytic;
    the other AUCs are simulated.
    """

    def sampler(rng, count):
        z1n = rng.normal(0.0, 1.0, count)
        z2n = rng.normal(0.0, epsilon, count)
        z1a = rng.normal(0.0, 2.0, count)
        z2a = rng.normal(0.0, epsilon, count)
        normal = {
            "lrt": z1n ** 2,
            "l2": z1n ** 2 + z2n ** 2,
            "diag_whitened": z1n ** 2 + (z2n / epsilon) ** 2,
        }
        anomaly = {
            "lrt": z1a ** 2,
            "l2": z1a ** 2 + z2a ** 2,
            "diag_whitened": z1a ** 2 + (z2a / epsilon) ** 2,
        }
        return normal, anomaly

    simulated = _simulate_auc(sampler, n, seed)
    return {
        "epsilon": float(epsilon),
        "n_per_class": int(n),
        "seed": int(seed),
        "analytic_lrt_auc": lrt_auc_epsilon_model(),
        "simulated": simulated,
        "l2_close_to_lrt": bool(abs(simulated["l2"] - lrt_auc_epsilon_model()) <= 0.01),
        "whitened_below_l2": bool(simulated["diag_whitened"] < simulated["l2"] - 0.01),
        "whitened_margin": float(simulated["l2"] - simulated["diag_whitened"]),
    }


def epsilon_point_reversal(epsilon: float = 0.1) -> Dict[str, Any]:
    """One normal/anomaly pair where L2 ranks correctly and diagonal whitening reverses."""

    normal = np.array([0.1, 0.3])
    anomaly = np.array([1.5, 0.05])
    l2_normal = float(np.mean(normal ** 2))
    l2_anomaly = float(np.mean(anomaly ** 2))
    q_normal = float(np.mean(normal ** 2 / np.array([1.0, epsilon ** 2])))
    q_anomaly = float(np.mean(anomaly ** 2 / np.array([1.0, epsilon ** 2])))
    return {
        "epsilon": float(epsilon),
        "normal_point": normal.tolist(),
        "anomaly_point": anomaly.tolist(),
        "l2_normal": l2_normal,
        "l2_anomaly": l2_anomaly,
        "l2_correct": bool(l2_anomaly > l2_normal),
        "diag_normal": q_normal,
        "diag_anomaly": q_anomaly,
        "diag_correct": bool(q_anomaly > q_normal),
        "rank_reversal": bool(l2_anomaly > l2_normal and q_anomaly < q_normal),
    }


def high_variance_mean_shift_counterexample(
    delta: float = 20.0, n: int = 200_000, seed: int = 1
) -> Dict[str, Any]:
    """H_N: r1~N(0,100),r2~N(0,1); H_A: r1~N(delta,100),r2~N(0,1).

    The anomaly is a mean shift on the high-variance axis.  Diagonal whitening
    divides that axis by 100 and can lose to L2; this is a distribution-level
    numerical example, not an optimality theorem.
    """

    def sampler(rng, count):
        r1n = rng.normal(0.0, 10.0, count)
        r2n = rng.normal(0.0, 1.0, count)
        r1a = rng.normal(delta, 10.0, count)
        r2a = rng.normal(0.0, 1.0, count)
        return (
            {"l2": r1n ** 2 + r2n ** 2, "diag_whitened": r1n ** 2 / 100.0 + r2n ** 2},
            {"l2": r1a ** 2 + r2a ** 2, "diag_whitened": r1a ** 2 / 100.0 + r2a ** 2},
        )

    simulated = _simulate_auc(sampler, n, seed)
    return {
        "delta": float(delta),
        "n_per_class": int(n),
        "seed": int(seed),
        "simulated": simulated,
        "l2_better": bool(simulated["l2"] > simulated["diag_whitened"] + 0.01),
        "margin": float(simulated["l2"] - simulated["diag_whitened"]),
    }


def high_variance_point_reversal() -> Dict[str, Any]:
    normal = np.array([10.0, 10.0])
    anomaly = np.array([35.0, 1.0])
    l2_normal = float(np.mean(normal ** 2))
    l2_anomaly = float(np.mean(anomaly ** 2))
    q_normal = float(np.mean(normal ** 2 / np.array([100.0, 1.0])))
    q_anomaly = float(np.mean(anomaly ** 2 / np.array([100.0, 1.0])))
    return {
        "normal_point": normal.tolist(),
        "anomaly_point": anomaly.tolist(),
        "l2_normal": l2_normal,
        "l2_anomaly": l2_anomaly,
        "l2_correct": bool(l2_anomaly > l2_normal),
        "diag_normal": q_normal,
        "diag_anomaly": q_anomaly,
        "diag_correct": bool(q_anomaly > q_normal),
        "rank_reversal": bool(l2_anomaly > l2_normal and q_anomaly < q_normal),
    }


def isotropic_equivalence(sigma: float = 2.0, n: int = 20_000, seed: int = 2) -> Dict[str, Any]:
    """Under Sigma=sigma^2 I, whitening is a monotone rescaling of L2."""

    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, sigma, n)
    anomaly = rng.normal(0.0, 3.0 * sigma, n)
    l2_normal = normal ** 2
    l2_anomaly = anomaly ** 2
    q_normal = l2_normal / sigma ** 2
    q_anomaly = l2_anomaly / sigma ** 2
    order_same = bool(np.array_equal(np.argsort(l2_normal), np.argsort(q_normal)))
    return {
        "sigma": float(sigma),
        "auc_l2": auc_from_scores(l2_normal, l2_anomaly),
        "auc_whitened": auc_from_scores(q_normal, q_anomaly),
        "ranking_identical": order_same,
    }


def second_moment_2d() -> Dict[str, Any]:
    """Analytic moments for Sigma=diag(100,1), delta=(0,4)."""

    sigma = np.diag([100.0, 1.0])
    delta = np.array([0.0, 4.0])
    # L2: r1^2+r2^2. Normal E=101, Var=2(100^2+1)=20002.
    l2_normal_mean = 101.0
    l2_normal_var = 2.0 * (100.0 ** 2 + 1.0)
    l2_anomaly_mean = 100.0 + 1.0 + 16.0
    # Q=r1^2/100+r2^2. Normal E=2, Var=2(1+1)=4.
    q_normal_mean = 2.0
    q_normal_var = 4.0
    noncentrality = float(delta @ np.linalg.solve(sigma, delta))
    q_anomaly_mean = 2.0 + noncentrality
    return {
        "sigma": sigma.tolist(),
        "delta": delta.tolist(),
        "l2_normal_mean": l2_normal_mean,
        "l2_normal_variance": l2_normal_var,
        "l2_anomaly_mean": l2_anomaly_mean,
        "q_normal_mean": q_normal_mean,
        "q_normal_variance": q_normal_var,
        "q_anomaly_mean": q_anomaly_mean,
        "noncentrality": noncentrality,
    }


def correlated_diagonal_counterexample(rho: float = 0.9, n: int = 100_000, seed: int = 3) -> Dict[str, Any]:
    """Diagonal standardization ignores off-diagonal covariance."""

    rng = np.random.default_rng(seed)
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    sample = rng.multivariate_normal([0.0, 0.0], covariance, n)
    q_diag = sample[:, 0] ** 2 + sample[:, 1] ** 2
    q_full = np.einsum("ij,jk,ik->i", sample, np.linalg.inv(covariance), sample)
    return {
        "rho": float(rho),
        "variance_diag": float(np.var(q_diag)),
        "variance_full": float(np.var(q_full)),
        "diag_not_chi2": bool(float(np.var(q_diag)) > 5.0),
        "mean_diag": float(np.mean(q_diag)),
        "mean_full": float(np.mean(q_full)),
    }


def multimodal_counterexample(n: int = 200_000, seed: int = 4) -> Dict[str, Any]:
    """Normal is a two-mode mixture; a single Gaussian moment misranks the origin."""

    rng = np.random.default_rng(seed)
    component = rng.integers(0, 2, n)
    normal = np.where(component == 0, rng.normal(-3.0, 0.25, n), rng.normal(3.0, 0.25, n))
    anomaly = rng.normal(0.0, 0.25, n)
    p_normal = 0.5 * norm.pdf(normal, -3.0, 0.25) + 0.5 * norm.pdf(normal, 3.0, 0.25)
    p_anomaly = norm.pdf(normal, 0.0, 0.25)
    true_lrt_normal = np.log(p_anomaly) - np.log(np.maximum(p_normal, 1e-300))
    p_normal_a = 0.5 * norm.pdf(anomaly, -3.0, 0.25) + 0.5 * norm.pdf(anomaly, 3.0, 0.25)
    p_anomaly_a = norm.pdf(anomaly, 0.0, 0.25)
    true_lrt_anomaly = np.log(p_anomaly_a) - np.log(np.maximum(p_normal_a, 1e-300))
    # The single Gaussian matched to the mixture has mean 0 and variance 9.25;
    # its L2/Mahalanobis ranking is monotone in r^2.
    return {
        "auc_true_mixture_lrt": auc_from_scores(true_lrt_normal, true_lrt_anomaly),
        "auc_l2": auc_from_scores(normal ** 2, anomaly ** 2),
        "auc_single_gaussian_moment": auc_from_scores(normal ** 2 / 9.25, anomaly ** 2 / 9.25),
    }


def coincident_counterexample(n: int = 100_000, seed: int = 5) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, n)
    anomaly = rng.normal(0.0, 1.0, n)
    return {
        "auc_l2": auc_from_scores(normal ** 2, anomaly ** 2),
        "auc_same_distribution": 0.5,
    }
