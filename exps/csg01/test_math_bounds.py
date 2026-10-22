"""G1: independent numerical checks of Propositions 1-6 and the UINT8 bound."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.csg01 import contract as C1
from exps.csg01 import quantize, scoring


def _dist(z, X):
    X = np.asarray(X, dtype=np.float64)
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    return float(np.min(np.linalg.norm(X - z[None, :], axis=1)))


def _epsilon(D, C):
    D = np.asarray(D, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    return float(np.max(np.min(np.linalg.norm(D[:, None, :] - C[None, :, :], axis=2), axis=1)))




def _local_quantize(X: np.ndarray) -> np.ndarray:
    """Independent implementation of the registered affine UINT8 decode for any d."""
    X = np.asarray(X, dtype=np.float32)
    xmin = X.min(axis=1).astype(np.float32)
    xmax = X.max(axis=1).astype(np.float32)
    step = ((xmax.astype(np.float64) - xmin.astype(np.float64)) / 255.0).astype(np.float32)
    q = np.zeros_like(X, dtype=np.uint8)
    nz = step > 0
    if np.any(nz):
        z = (X[nz].astype(np.float64) - xmin[nz].astype(np.float64)[:, None]) / \
            step[nz].astype(np.float64)[:, None]
        q[nz] = np.clip(np.rint(z), 0, 255).astype(np.uint8)
    return (xmin.astype(np.float64)[:, None] + step.astype(np.float64)[:, None] *
            q.astype(np.float64)).astype(np.float32)


def _delta(C, Q):
    C = np.asarray(C, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    return float(np.max(np.linalg.norm(C - Q, axis=1)))


def test_propositions_1_to_4_numerically():
    rng = np.random.default_rng(11)
    for dim in (3, 16, 64):
        D = rng.normal(size=(200, dim))
        idx = rng.choice(200, size=25, replace=False)
        C = D[idx]
        eps = _epsilon(D, C)
        # quantize C with the registered affine scheme
        Q = _local_quantize(C).astype(np.float64)
        delta = _delta(C.astype(np.float32), Q)
        for _ in range(50):
            z = rng.normal(size=dim)
            dD, dC, dQ = _dist(z, D), _dist(z, C), _dist(z, Q)
            assert dC >= dD - 1e-9                                  # Prop 1
            assert 0.0 <= dC - dD <= eps + 1e-9                      # Prop 2
            assert abs(dQ - dC) <= delta + 1e-6                      # Prop 3
            assert dD - delta <= dQ <= dD + eps + delta + 1e-6       # Prop 4


def test_ranking_margin_sufficient_condition():
    rng = np.random.default_rng(12)
    D = np.concatenate([rng.normal(loc=-5, size=(40, 8)),
                        rng.normal(loc=5, size=(40, 8))])
    C = D[rng.choice(80, size=10, replace=False)]
    Q = _local_quantize(C).astype(np.float64)
    eps, delta = _epsilon(D, C), _delta(C.astype(np.float32), Q)
    # a normal near its support and an anomaly far away
    n = C[0] + rng.normal(size=8) * 0.01
    a = np.full(8, 30.0)
    dDn, dDa = _dist(n, D), _dist(a, D)
    assert dDa - dDn > eps + 2 * delta
    dQn, dQa = _dist(n, Q), _dist(a, Q)
    assert dQa > dQn
    # the condition is sufficient, not necessary: a wide margin is not needed
    # for this particular pair, so we only assert the implication above.


def test_image_max_bound_from_patch_bounds():
    rng = np.random.default_rng(13)
    D = rng.normal(size=(50, 12))
    C = D[rng.choice(50, size=8, replace=False)]
    eps = _epsilon(D, C)
    S_D = np.array([_dist(rng.normal(size=12), D) for _ in range(20)])
    # construct image scores directly from the patch bounds: S_C in [S_D, S_D+eps]
    # for a subset arm; the image max then lies in [max S_D, max S_D + eps].
    S_C = S_D + rng.uniform(0, eps, size=S_D.size)
    assert S_D.max() <= S_C.max() + 1e-9
    assert S_C.max() <= S_D.max() + eps + 1e-9


def test_uint8_l2_bound():
    rng = np.random.default_rng(14)
    for scale in (1e-3, 1.0, 10.0, 100.0):
        X = torch.tensor(rng.normal(size=(100, C1.DIM)) * scale, dtype=torch.float32)
        enc = quantize.encode_support([X])
        bound = np.sqrt(C1.DIM) * enc["step"].astype(np.float64) / 2.0
        assert np.all(enc["error"] <= bound + C1.QUANT_TOL)


def test_spatial_index_and_bin_eligibility():
    bins = C1.C0.patch_bin_indices()
    assert bins.shape == (C1.PATCHES,)
    assert int(np.bincount(bins, minlength=C1.BINS).min()) == C1.TOKENS_PER_BIN
    assert int(np.bincount(bins, minlength=C1.BINS).max()) == C1.TOKENS_PER_BIN
    # p=0 and p=3 are bin 0; p=4 is bin 1; p=112 is bin 7
    assert int(bins[0]) == 0 and int(bins[3]) == 0
    assert int(bins[4]) == 1 and int(bins[112]) == 7
    # scoring uses the query's own bin only: a cross-bin decoy cannot lower Arm B
    rng = np.random.default_rng(15)
    full = [torch.tensor(rng.normal(size=(5, C1.DIM)), dtype=torch.float64) for _ in range(C1.BINS)]
    q = [torch.zeros((1, C1.DIM), dtype=torch.float64) for _ in range(C1.BINS)]
    q[0] = torch.tensor(rng.normal(size=(1, C1.DIM)), dtype=torch.float64)
    full[1] = q[0].clone()  # perfect cross-bin match
    scores = scoring.score_bins(q, full, device="cpu")
    same_bin = torch.cdist(q[0], full[0]).min().item()
    assert float(scores[0].item()) == pytest.approx(float(same_bin))


def test_score_orientation_and_zero_distance():
    rng = np.random.default_rng(16)
    X = torch.tensor(rng.normal(size=(1, C1.DIM)), dtype=torch.float64)
    q = torch.zeros((2, C1.DIM), dtype=torch.float64)
    q[0] = X[0]
    q[1] = X[0] + 5.0
    d = scoring.S0.nn_distance(q, X, device="cpu")
    assert float(d[0]) == pytest.approx(0.0)
    assert float(d[1]) == pytest.approx(5.0 * np.sqrt(C1.DIM))
    assert float(d[1]) > float(d[0])


def test_cpu_float64_nn_matches_bruteforce():
    rng = np.random.default_rng(71)
    X = torch.tensor(rng.normal(size=(200, C1.DIM)) * 3.0, dtype=torch.float32)
    Q = torch.tensor(rng.normal(size=(97, C1.DIM)) * 3.0, dtype=torch.float32)
    got = scoring.S0.nn_distance(Q, X, device="cpu").numpy()
    ref = np.linalg.norm(Q.numpy()[:, None, :].astype(np.float64) -
                         X.numpy()[None, :, :].astype(np.float64), axis=2).min(axis=1)
    assert np.max(np.abs(got - ref)) < 1e-8
    # tie handling: duplicate support points must give the same distance
    X2 = torch.cat([X, X[:1]], dim=0)
    got2 = scoring.S0.nn_distance(Q, X2, device="cpu").numpy()
    assert np.max(np.abs(got2 - got)) < 1e-12
