import numpy as np
import pytest

from exps.npim_cad.uncertainty.moments import (
    PerChannelMoments,
    anisotropy,
    image_bootstrap_anisotropy,
    regularization,
    variance_from_groups,
)


def test_welford_matches_direct_pooled_moments():
    rng = np.random.default_rng(0)
    d = 8
    images = rng.normal(0.2, 1.7, size=(5, 11, d))
    accumulator = PerChannelMoments(d=d)
    for image in images:
        accumulator.add_image_group(image)
    pooled = images.reshape(-1, d)
    assert accumulator.n_images == 5
    assert accumulator.n_patches == 55
    assert np.allclose(accumulator.mu, pooled.mean(axis=0), atol=1e-12, rtol=0)
    assert np.allclose(accumulator.variance(), pooled.var(axis=0, ddof=1), atol=1e-12, rtol=0)


def test_group_pooling_matches_accumulator():
    rng = np.random.default_rng(1)
    d = 4
    images = rng.normal(0, 2, size=(6, 9, d))
    accumulator = PerChannelMoments(d=d)
    for image in images:
        accumulator.add_image_group(image)
    pooled = variance_from_groups(
        accumulator.group_means, accumulator.group_m2, accumulator.group_n, list(range(6))
    )
    assert np.allclose(pooled, accumulator.variance(), atol=1e-12, rtol=0)


def test_regularization_constant_and_floor():
    var = np.full(16, 4.0)
    result = regularization(var, rho=0.1, tau=0.01, eps_abs=1e-8)
    assert np.allclose(result["var_shrunk"], 4.0)
    assert np.allclose(result["var_safe"], 4.0)
    heterogeneous = np.array([1.0, 100.0, 1.0, 100.0])
    result = regularization(heterogeneous, rho=0.1, tau=0.01, eps_abs=1e-8)
    assert np.all(result["var_safe"] >= 1e-8)
    assert np.all(result["var_safe"] >= 0.01 * heterogeneous.mean())


def test_anisotropy_null_and_scaled():
    null = anisotropy(np.ones(32))
    assert null["p95_over_p5"] == pytest.approx(1.0)
    assert null["coefficient_of_variation"] == pytest.approx(0.0)
    scaled = anisotropy(np.array([1.0, 2.0, 4.0, 8.0]))
    assert scaled["max_over_min"] == pytest.approx(8.0)
    assert scaled["p95_over_p5"] > 1.0


def test_whole_image_bootstrap_is_deterministic_and_patch_safe():
    rng = np.random.default_rng(2)
    d = 6
    images = rng.normal(0, 1, size=(8, 12, d))
    accumulator = PerChannelMoments(d=d)
    for image in images:
        accumulator.add_image_group(image)
    first = image_bootstrap_anisotropy(
        accumulator.group_means, accumulator.group_m2, accumulator.group_n, reps=20, seed=99
    )
    second = image_bootstrap_anisotropy(
        accumulator.group_means, accumulator.group_m2, accumulator.group_n, reps=20, seed=99
    )
    assert first == second
    assert first["images"] == 8
    assert first["patches_counted_as_independent"] is False
    assert first["bootstrap_channel_cv_mean"] >= 0.0
    assert first["cv_log_variance_mean"] >= 0.0
    low, high = first["p95_over_p5_interval"]
    assert low <= high
