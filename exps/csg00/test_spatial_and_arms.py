"""G1: spatial eligibility, reservoir statistics, fitting and scoring determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.csg00 import contract as C
from exps.csg00 import fit_modes, reservoir, scoring


def test_patch_index_bins_partition():
    bins = C.patch_bin_indices()
    assert bins.shape == (C.PATCHES,)
    assert bins.min() == 0 and bins.max() == C.BINS - 1
    counts = np.bincount(bins, minlength=C.BINS)
    assert counts.tolist() == [C.TOKENS_PER_BIN] * C.BINS
    # p = 28*row + col
    assert bins[0] == 0            # row 0, col 0
    assert bins[3] == 0            # row 0, col 3
    assert bins[4] == 1            # row 0, col 4
    assert bins[28] == 0           # row 1, col 0
    assert bins[4 * 28] == 7       # row 4, col 0
    assert bins[27 * 28 + 27] == 48
    positions = C.bin_positions()
    assert all(len(p) == C.TOKENS_PER_BIN for p in positions)
    assert sorted(np.concatenate(positions).tolist()) == list(range(C.PATCHES))


def test_binned_arms_ignore_cross_bin_decoy_but_global_diagnostic_does_not():
    rng = np.random.default_rng(4)
    descriptors = [rng.normal(size=(3, C.DIM)) for _ in range(C.BINS)]
    query = rng.normal(size=(1, C.DIM))
    descriptors[0] = np.full((3, C.DIM), 10.0)
    descriptors[1] = query.copy()  # perfect cross-bin match
    queries = [torch.zeros((1, C.DIM), dtype=torch.float64) for _ in range(C.BINS)]
    queries[0] = torch.tensor(query, dtype=torch.float64)
    xbins = [torch.tensor(x, dtype=torch.float64) for x in descriptors]
    binned = scoring.score_binned(queries, xbins, device="cpu")
    global_scores = scoring.score_global(queries, xbins, device="cpu")
    assert float(binned[0].item()) == pytest.approx(np.linalg.norm(query - 10.0))
    assert float(global_scores[0].item()) == pytest.approx(0.0, abs=1e-12)
    assert float(global_scores[0].item()) < float(binned[0].item())


def _stream_bins(events: int, rng) -> list:
    data = rng.normal(size=(events * C.TOKENS_PER_BIN, C.DIM)).astype(np.float32)
    return [torch.tensor(data.copy()) for _ in range(C.BINS)]


def test_reservoir_uniform_inclusion_and_determinism():
    rng = np.random.default_rng(12)
    events, capacity, trials = 10, 10, 120
    n = events * C.TOKENS_PER_BIN
    data = rng.normal(size=(n, C.DIM))
    data32 = data.astype(np.float32)
    bins = [torch.tensor(data.copy()) for _ in range(C.BINS)]
    inclusion = np.zeros(n)
    first = None
    for trial in range(trials):
        state = reservoir.build_reservoir(bins, t=events, capacity=capacity, seed=1000 + trial)
        slot_items = state["res"][0]
        for item in range(n):
            if np.any(np.all(slot_items == data32[item], axis=1)):
                inclusion[item] += 1
        if trial == 0:
            first = state
        assert state["count"][0] == capacity
        assert state["seen"][0] == n
    frequencies = inclusion / trials
    assert abs(frequencies.mean() - capacity / n) < 0.02
    assert float(np.mean(np.abs(frequencies - capacity / n))) < 0.03

    again = reservoir.build_reservoir(bins, t=events, capacity=capacity, seed=1000)
    assert np.array_equal(again["res"], first["res"])
    other = reservoir.build_reservoir(bins, t=events, capacity=capacity, seed=999999)
    assert not np.array_equal(other["res"], first["res"])


def test_reservoir_rejects_unregistered_prefix_length_and_is_deterministic():
    rng = np.random.default_rng(13)
    bins = _stream_bins(5, rng)
    with pytest.raises(ValueError, match="expected"):
        reservoir.build_reservoir(bins, t=6, capacity=10, seed=7)
    first = reservoir.build_reservoir(bins, t=5, capacity=10, seed=7)
    second = reservoir.build_reservoir(bins, t=5, capacity=10, seed=7)
    assert np.array_equal(first["res"], second["res"])
    assert np.array_equal(first["seen"], second["seen"])


def test_two_mode_fit_is_deterministic_and_orthonormal():
    rng = np.random.default_rng(21)
    centers = rng.normal(size=(2, C.DIM)) * 10.0
    labels = rng.integers(0, 2, size=400)
    data = centers[labels] + rng.normal(size=(400, C.DIM)) * 0.3
    D = torch.tensor(data, dtype=torch.float64)
    first = fit_modes.fit_two_mode(D, 0)
    second = fit_modes.fit_two_mode(D, 0)
    for key in ("ready", "count", "mu", "U", "R", "rho", "lam"):
        assert np.array_equal(first["arrays"][key], second["arrays"][key]), key
    assert int(first["arrays"]["ready"].sum()) >= 1
    assert int(first["arrays"]["count"].sum()) <= D.shape[0]
    est_centers = first["arrays"]["mu"][first["arrays"]["ready"] == 1]
    for center in est_centers:
        assert min(np.linalg.norm(center - centers[0]), np.linalg.norm(center - centers[1])) < 1.0
    for k in range(C.K_MODES):
        if first["arrays"]["ready"][k] == 0:
            continue
        U = torch.tensor(first["arrays"]["U"][k], dtype=torch.float64)
        err = float((U.T @ U - torch.eye(C.Q_TANGENT, dtype=torch.float64)).abs().max())
        assert err < 1e-5
        assert np.isfinite(first["arrays"]["R"][k]).all()
        assert np.isfinite(first["arrays"]["rho"][k])


def test_score_modes_deterministic_and_reload_independent():
    rng = np.random.default_rng(31)
    centers = rng.normal(size=(2, C.DIM)) * 8.0
    labels = rng.integers(0, 2, size=300)
    data = centers[labels] + rng.normal(size=(300, C.DIM))
    result = fit_modes.fit_two_mode(torch.tensor(data, dtype=torch.float64), 0)
    state = {key: np.stack([result["arrays"][key]] * C.BINS) for key in result["arrays"]}
    queries = [torch.tensor(rng.normal(size=(7, C.DIM)), dtype=torch.float64)
               for _ in range(C.BINS)]
    queries[0] = torch.tensor(data[:7], dtype=torch.float64)
    first = scoring.score_modes(queries, state, device="cpu")
    _ = rng.normal(size=10000)  # read-only scoring must not depend on RNG state
    second = scoring.score_modes(queries, state, device="cpu")
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    # in-support training points must be finite and non-negative
    assert float(first[0].min()) >= 0.0


def test_high_score_means_more_anomalous():
    mu = torch.zeros(C.DIM, dtype=torch.float64)
    U = torch.zeros((C.DIM, C.Q_TANGENT), dtype=torch.float64)
    U[: C.Q_TANGENT, :] = torch.eye(C.Q_TANGENT, dtype=torch.float64)
    R = torch.ones(C.Q_TANGENT, dtype=torch.float64)
    rho = 1.0
    near = torch.zeros((1, C.DIM), dtype=torch.float64)
    far = torch.zeros((1, C.DIM), dtype=torch.float64)
    far[0, 0] = 5.0
    d_near = float(scoring.mode_distance(near, mu, U, R, rho).item())
    d_far = float(scoring.mode_distance(far, mu, U, R, rho).item())
    assert d_near == pytest.approx(0.0)
    assert d_far > d_near
    assert d_far == pytest.approx(4.0)


def test_eligible_prefix_matches_protocol_order():
    assert C.ELIGIBLE_PREFIX == {100: 19, 200: 39, 300: 59, 350: 59}
    assert [c for c, _ in C.PROBE_BLOCKS] == ["bottle", "carpet", "hazelnut"]
    assert sum(n for _, n in C.PROBE_BLOCKS) == 59
