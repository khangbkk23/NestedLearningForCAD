"""G1: exact bounded-support distance oracle and Hausdorff/margin mathematics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from exps.csg00 import contract as C
from exps.csg00.scoring import mode_distance


def _oracle_distance(z: np.ndarray, mu: np.ndarray, U: np.ndarray, R: np.ndarray,
                     rho: float, n_starts: int = 64) -> float:
    """Independent numerical oracle over the full support parameterisation.

    Variables are the tangent coordinates ``a`` (box-bounded) and an unconstrained
    vector ``g`` whose orthogonal projection, clipped to the ball of radius rho,
    is the orthogonal displacement ``v``.  The objective is the Euclidean
    residual; it never uses the closed-form clipped-tube formula under test.
    """
    delta = z - mu
    q = U.shape[1]

    def objective(params: np.ndarray) -> float:
        a = params[:q]
        g = params[q:]
        v = g - U @ (U.T @ g)
        nv = float(np.linalg.norm(v))
        if nv > rho and nv > 0.0:
            v = v * (rho / nv)
        return float(np.linalg.norm(delta - U @ a - v))

    rng = np.random.default_rng(0)
    scale = np.maximum(R, 1e-3)
    starts = []
    for _ in range(n_starts):
        a0 = rng.uniform(-1.0, 1.0, size=q) * (scale * 1.5)
        g0 = delta + rng.normal(size=delta.size) * max(rho, 1e-3) * 2.0
        starts.append(np.concatenate([a0, g0]))
    bounds = [(-float(R[j]), float(R[j])) for j in range(q)] + [(None, None)] * delta.size
    best = np.inf
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                          options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 800})
        best = min(best, float(result.fun))
    return best


def _random_mode(rng, d: int, q: int):
    A = rng.normal(size=(d, q))
    U, _ = np.linalg.qr(A)
    mu = rng.normal(size=d)
    R = rng.uniform(0.1, 2.0, size=q)
    rho = float(rng.uniform(0.1, 2.0))
    return mu, U, R, rho


@pytest.mark.parametrize("d,q", [(2, 1), (2, 2), (3, 2), (6, 3), (16, 4)])
def test_tube_distance_matches_independent_optimizer(d, q):
    rng = np.random.default_rng(1000 + d * 10 + q)
    for _ in range(12):
        mu, U, R, rho = _random_mode(rng, d, q)
        z = mu + rng.normal(size=d) * 1.5
        analytic = float(mode_distance(
            torch.tensor(z, dtype=torch.float64).unsqueeze(0),
            torch.tensor(mu, dtype=torch.float64),
            torch.tensor(U, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64), rho).item())
        oracle = _oracle_distance(z, mu, U, R, rho)
        assert analytic == pytest.approx(oracle, abs=2e-5, rel=2e-5)


def test_q0_degenerate_radii_and_point_mode():
    rng = np.random.default_rng(7)
    mu = rng.normal(size=8)
    z = mu + rng.normal(size=8)
    rho = 1.5
    expected = max(float(np.linalg.norm(z - mu)) - rho, 0.0)
    got = float(mode_distance(torch.tensor(z).unsqueeze(0), torch.tensor(mu),
                              torch.zeros((8, 0)), torch.zeros((0,)), rho).item())
    assert got == pytest.approx(expected, abs=1e-12)

    mu2 = rng.normal(size=5)
    z2 = mu2 + rng.normal(size=5)
    got2 = float(mode_distance(torch.tensor(z2).unsqueeze(0), torch.tensor(mu2),
                               torch.eye(5)[:, :2], torch.zeros(2), 0.0).item())
    assert got2 == pytest.approx(float(np.linalg.norm(z2 - mu2)), abs=1e-12)


def test_near_boundary_numerics_are_finite_and_consistent():
    rng = np.random.default_rng(11)
    d, q = 12, 3
    mu, U, R, rho = _random_mode(rng, d, q)
    e = rng.normal(size=d)
    e = e - U @ (U.T @ e)
    e = rho * e / np.linalg.norm(e)
    z = mu + U @ R + e
    value = mode_distance(torch.tensor(z).unsqueeze(0), torch.tensor(mu),
                          torch.tensor(U), torch.tensor(R), rho)
    assert torch.isfinite(value).all()
    assert float(value.item()) == pytest.approx(0.0, abs=1e-6)

    z2 = mu + U @ (R + 1e-3) + e * (1.0 + 1e-3)
    value2 = float(mode_distance(torch.tensor(z2).unsqueeze(0), torch.tensor(mu),
                                 torch.tensor(U), torch.tensor(R), rho).item())
    assert value2 == pytest.approx(np.sqrt(q * (1e-3) ** 2 + (rho * 1e-3) ** 2), rel=1e-5)


def test_no_ready_mode_is_a_blocked_condition():
    from exps.csg00 import scoring

    state = {
        "ready": np.zeros((C.BINS, C.K_MODES), dtype=np.uint8),
        "mu": np.zeros((C.BINS, C.K_MODES, C.DIM), dtype=np.float32),
        "U": np.zeros((C.BINS, C.K_MODES, C.DIM, C.Q_TANGENT), dtype=np.float32),
        "R": np.zeros((C.BINS, C.K_MODES, C.Q_TANGENT), dtype=np.float32),
        "rho": np.zeros((C.BINS, C.K_MODES), dtype=np.float32),
        "count": np.zeros((C.BINS, C.K_MODES), dtype=np.int64),
        "n_total": np.zeros((C.BINS,), dtype=np.int64),
    }
    queries = [torch.zeros((1, C.DIM), dtype=torch.float64) for _ in range(C.BINS)]
    with pytest.raises(RuntimeError, match="no ready mode"):
        scoring.score_modes(queries, state)


def test_two_point_gap_counterexample_and_single_mode_overreach():
    normals = np.array([[-1.0, 0.0], [1.0, 0.0]])
    z = np.array([0.0, 0.0])
    assert float(np.min(np.linalg.norm(normals - z, axis=1))) == pytest.approx(1.0)

    segment = float(mode_distance(torch.tensor(z).unsqueeze(0),
                                  torch.tensor([0.0, 0.0]),
                                  torch.tensor([[1.0], [0.0]]),
                                  torch.tensor([1.0]), 0.0).item())
    assert segment == pytest.approx(0.0, abs=1e-12)

    ball = float(mode_distance(torch.tensor(z).unsqueeze(0),
                               torch.tensor([0.0, 0.0]),
                               torch.zeros((2, 0)), torch.zeros((0,)), 1.0).item())
    assert ball == pytest.approx(0.0, abs=1e-12)

    d0 = float(mode_distance(torch.tensor(z).unsqueeze(0), torch.tensor(normals[0]),
                             torch.zeros((2, 0)), torch.zeros((0,)), 0.0).item())
    d1 = float(mode_distance(torch.tensor(z).unsqueeze(0), torch.tensor(normals[1]),
                             torch.zeros((2, 0)), torch.zeros((0,)), 0.0).item())
    assert min(d0, d1) == pytest.approx(1.0)
    # Convex-hull membership alone is not the issue; the point is that a single
    # bounded mode can span a normal-free gap while the empirical NN cannot.
    assert segment < min(d0, d1)


def test_symmetric_hausdorff_distance_and_two_epsilon_ranking_margin():
    rng = np.random.default_rng(23)
    S = rng.normal(size=(40, 3))
    Shat = S + rng.normal(size=S.shape) * 1e-3

    def one_sided(A, B):
        return float(np.max(np.min(np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2), axis=1)))

    eps = max(one_sided(S, Shat), one_sided(Shat, S))
    assert eps < 1e-2

    for _ in range(50):
        z = rng.normal(size=(1, 3))
        dS = float(np.min(np.linalg.norm(S - z, axis=1)))
        dH = float(np.min(np.linalg.norm(Shat - z, axis=1)))
        assert abs(dS - dH) <= eps + 1e-12

    # Construct an anomaly/normal pair with a > 2 eps margin and verify the
    # ordering is preserved for every random witness z.
    for _ in range(20):
        z_anom = rng.normal(size=(1, 3)) * 3.0
        z_norm = S[rng.integers(0, S.shape[0])]
        dS_a = float(np.min(np.linalg.norm(S - z_anom, axis=1)))
        dS_n = float(np.min(np.linalg.norm(S - z_norm, axis=1)))
        if dS_a - dS_n <= 2 * eps:
            continue
        dH_a = float(np.min(np.linalg.norm(Shat - z_anom, axis=1)))
        dH_n = float(np.min(np.linalg.norm(Shat - z_norm, axis=1)))
        assert dH_a > dH_n


def test_coverage_does_not_bound_overreach():
    normals = np.array([[-1.0, 0.0], [1.0, 0.0]])
    mu = np.array([0.0, 0.0])
    ball_radius = 1.0
    # Every normal point is covered (support distance 0) ...
    for x in normals:
        value = float(mode_distance(torch.tensor(x).unsqueeze(0), torch.tensor(mu),
                                    torch.zeros((2, 0)), torch.zeros((0,)),
                                    ball_radius).item())
        assert value == pytest.approx(0.0, abs=1e-12)
    # ... including the gap point (0,0), which is 1.0 away from every normal point.
    gap = float(mode_distance(torch.tensor([0.0, 0.0]).unsqueeze(0), torch.tensor(mu),
                              torch.zeros((2, 0)), torch.zeros((0,)),
                              ball_radius).item())
    assert gap == pytest.approx(0.0, abs=1e-12)
    assert float(np.min(np.linalg.norm(normals - np.array([0.0, 0.0]), axis=1))) == 1.0
