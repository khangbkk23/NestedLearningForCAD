# exps/npim_cad/test_proximal_math.py
"""G3/G5 oracles: closed-form proximal regression and sufficient statistics."""

from __future__ import annotations

import pytest
import torch

from models.hope_cad.npim_proximal import (
    ProximalConfig,
    ProximalLevel,
    drift_bound,
    proximal_lstsq_oracle,
    proximal_solve_oracle,
    strong_convexity_floor,
)


def random_pairs(d: int = 32, r: int = 8, n: int = 64, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    K = torch.randn(n, r, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(n, d, generator=generator)
    return K, Y


def make_level(d: int = 32, r: int = 8, lam: float = 0.1) -> ProximalLevel:
    return ProximalLevel(ProximalConfig(d=d, r=r, period=1, lam=lam), device="cpu")


def test_zero_keys_leave_W_unchanged():
    level = make_level()
    generator = torch.Generator().manual_seed(1)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    before = level.W.clone()
    Skk = torch.zeros(level.config.r, level.config.r)
    Syk = torch.zeros(level.config.d, level.config.r)
    candidate = level.candidate_on_boundary(Skk, Syk, 100)
    assert torch.allclose(candidate, before, atol=1e-6)


def test_targets_equal_W0_k_leave_W_unchanged():
    level = make_level()
    K, _ = random_pairs(level.config.d, level.config.r, n=100, seed=2)
    generator = torch.Generator().manual_seed(3)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    before = level.W.clone()
    Y = K @ before.T
    Skk, Syk, n = level.stage_image(K, Y)
    candidate = level.candidate_on_boundary(Skk, Syk, n)
    assert torch.allclose(candidate, before, atol=2e-6, rtol=1e-5)


def test_large_lambda_freezes_update():
    level = make_level(lam=1.0e6)
    K, Y = random_pairs(level.config.d, level.config.r, n=100, seed=4)
    generator = torch.Generator().manual_seed(5)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    before = level.W.clone()
    Skk, Syk, n = level.stage_image(K, Y)
    candidate = level.candidate_on_boundary(Skk, Syk, n)
    assert (candidate - before).abs().max().item() < 1e-3


def test_one_update_drift_bound_holds():
    level = make_level()
    K, Y = random_pairs(level.config.d, level.config.r, n=200, seed=6)
    generator = torch.Generator().manual_seed(7)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    Skk, Syk, n = level.stage_image(K, Y)
    candidate = level.candidate_on_boundary(Skk, Syk, n)
    bound = drift_bound(Skk, Syk, n, level.W, level.config.lam)
    actual = (candidate - level.W).norm().item()
    assert actual <= bound * (1.0 + 1e-4) + 1e-6, (actual, bound)


def test_strong_convexity_floor_at_least_lambda():
    level = make_level(lam=0.37)
    K, _ = random_pairs(level.config.d, level.config.r, n=300, seed=8)
    Skk, _, n = level.stage_image(K, torch.zeros(K.shape[0], level.config.d))
    assert strong_convexity_floor(Skk, n, level.config.lam) >= level.config.lam - 1e-9


@pytest.mark.parametrize("d,r", [(32, 8), (768, 64)])
def test_cholesky_matches_two_fp64_oracles(d: int, r: int):
    level = make_level(d=d, r=r, lam=0.1)
    K, Y = random_pairs(d, r, n=128, seed=9)
    generator = torch.Generator().manual_seed(10)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    Skk, Syk, n = level.stage_image(K, Y)
    candidate = level.candidate_on_boundary(Skk, Syk, n)
    solve = proximal_solve_oracle(Skk, Syk, n, level.W, level.config.lam)
    lstsq = proximal_lstsq_oracle(Skk, Syk, n, level.W, level.config.lam)
    assert torch.isfinite(solve).all() and torch.isfinite(lstsq).all()
    assert (candidate.double() - solve).abs().max().item() <= 1e-4
    assert (candidate.double() - lstsq).abs().max().item() <= 1e-4
    assert (solve - lstsq).abs().max().item() <= 1e-8


def test_non_spd_system_is_rejected():
    level = make_level(lam=0.1)
    Skk = -1.0e3 * torch.eye(level.config.r)
    Syk = torch.zeros(level.config.d, level.config.r)
    with pytest.raises(RuntimeError):
        level.candidate_on_boundary(Skk, Syk, 10)


def test_overflowing_statistics_are_rejected():
    level = make_level()
    K = torch.full((4, level.config.r), 1.0e30)
    Y = torch.zeros(4, level.config.d)
    Skk, Syk, n = level.stage_image(K, Y)
    assert not torch.isfinite(Skk).all()
    with pytest.raises(ValueError):
        level.candidate_on_boundary(Skk, Syk, n)


def test_sufficient_statistics_match_concatenated_objective():
    level = make_level()
    chunks = [random_pairs(level.config.d, level.config.r, n=37, seed=11 + i) for i in range(5)]
    Skk = torch.zeros(level.config.r, level.config.r)
    Syk = torch.zeros(level.config.d, level.config.r)
    pairs = 0
    all_k, all_y = [], []
    for K, Y in chunks:
        dA, dB, n = level.stage_image(K, Y)
        Skk += dA
        Syk += dB
        pairs += n
        all_k.append(K)
        all_y.append(Y)
    generator = torch.Generator().manual_seed(12)
    level.W.copy_(torch.randn(level.W.shape, generator=generator))
    accumulated = level.candidate_on_boundary(Skk, Syk, pairs)
    one_shot = level.candidate_on_boundary(
        torch.cat(all_k).T @ torch.cat(all_k),
        torch.cat(all_y).T @ torch.cat(all_k),
        pairs,
    )
    assert torch.allclose(accumulated, one_shot, atol=1e-6, rtol=1e-6)


def test_pending_statistics_do_not_replay_raw_pairs():
    level = make_level()
    K, Y = random_pairs(level.config.d, level.config.r, n=50, seed=13)
    Skk, Syk, n = level.stage_image(K, Y)
    level.stage_pending(Skk, Syk, n)
    assert level.pair_count == n
    assert torch.equal(level.Skk, Skk)
    assert torch.equal(level.Syk, Syk)
    # Only statistics and counters exist; no raw K/V are stored on the level.
    for name in ("K", "Y", "V", "patches", "replay"):
        assert not hasattr(level, name)


def test_unlearnable_normal_structure_keeps_high_residual():
    """Negative finding: unpredictable normal targets are not compressed."""

    d, r = 32, 8
    K, _ = random_pairs(d, r, n=400, seed=14)
    Y = torch.randn(400, d, generator=torch.Generator().manual_seed(15))  # independent of K
    train_k, train_y = K[:300], Y[:300]
    held_k, held_y = K[300:], Y[300:]
    level = make_level(d=d, r=r, lam=0.1)
    Skk, Syk, n = level.stage_image(train_k, train_y)
    level.W.copy_(level.candidate_on_boundary(Skk, Syk, n))
    before = held_y.square().sum(dim=1).mean().item()
    after = (held_y - held_k @ level.W.T).square().sum(dim=1).mean().item()
    assert after > 0.70 * before, (before, after)
