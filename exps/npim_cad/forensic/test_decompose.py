# exps/npim_cad/forensic/test_decompose.py
"""G1 math tests for the NPIM mixture/residual decomposition."""

from __future__ import annotations

import pytest
import torch

from exps.npim_cad.forensic.decompose import (
    PAIR_INDEX,
    decompose_mixture,
    effective_W,
    mix_from_statistics,
    ready_weights,
)
from exps.npim_cad.npim01_contract import PILOT, arm_config
from models.hope_cad.npim_cad import NPIMDetector
from models.hope_cad.npim_state import load_state_file, rng_fingerprint


def load_arm(arm: str, boundary: int) -> NPIMDetector:
    detector = NPIMDetector(arm_config(arm), device="cpu")
    detector.load_state_dict(load_state_file(PILOT / "states" / f"{arm}_{boundary}.pt"))
    return detector


def fixture(d: int = 768, r: int = 8, n: int = 64, levels: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    K = torch.randn(n, r, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(n, d, generator=generator)
    Ws = [torch.randn(d, r, generator=generator) for _ in range(levels)]
    weights = torch.softmax(torch.randn(levels, generator=generator), dim=0)
    return K, Y, Ws, weights


def test_mixture_identity_and_effective_weight():
    K, Y, Ws, weights = fixture()
    decomposition = decompose_mixture(Y, K, Ws, weights)
    assert decomposition.identity_max_abs <= 1e-5
    assert decomposition.effective_weight_error <= 1e-5
    assert torch.allclose(decomposition.fused_scores, decomposition.self_terms.sum(0) + decomposition.cross_terms.sum(0), atol=1e-5)
    # r = sum omega_l r_l
    residuals = []
    for W in Ws:
        residuals.append(Y - K @ W.T)
    fused = sum(weights[index] * residuals[index] for index in range(len(Ws)))
    assert torch.allclose(fused, Y - sum(weights[index] * (K @ Ws[index].T) for index in range(len(Ws))), atol=1e-5)


def test_effective_W_matches_sum_of_level_predictions():
    K, Y, Ws, weights = fixture(seed=1)
    W_eff = effective_W(Ws, weights)
    fused = sum(weights[index] * (K @ Ws[index].T) for index in range(len(Ws)))
    assert torch.allclose(K @ W_eff.T, fused, atol=1e-5)


def test_zero_cross_terms_reduce_to_weighted_self_terms():
    level_scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dots = torch.zeros((1, 2))
    weights = torch.tensor([0.5, 0.5])
    mixed = mix_from_statistics(level_scores, dots, weights, d=768)
    expected = 0.25 * level_scores[0] + 0.25 * level_scores[1]
    assert torch.allclose(mixed, expected, atol=1e-6)


def test_negative_cross_terms_cancel_signal():
    # r_2 = -r_1 => <r_1,r_2> = -d; with equal weights the fused residual is zero.
    level_scores = torch.ones((2, 4))
    dots = -768.0 * torch.ones((1, 4))
    weights = torch.tensor([0.5, 0.5])
    mixed = mix_from_statistics(level_scores, dots, weights, d=768)
    assert torch.allclose(mixed, torch.zeros_like(mixed), atol=1e-6)


def test_mix_from_statistics_matches_decomposition():
    K, Y, Ws, weights = fixture(seed=2)
    decomposition = decompose_mixture(Y, K, Ws, weights)
    mixed = mix_from_statistics(decomposition.level_scores, decomposition.dot_products, weights, d=768)
    assert torch.allclose(mixed, decomposition.fused_scores, atol=1e-5)
    assert decomposition.dot_products.shape == (len(PAIR_INDEX), K.shape[0])


def test_actual_ready_weights_and_cold_start():
    c3_100 = load_arm("C3", 100)
    indices, weights = ready_weights(c3_100)
    assert indices == [0, 1]
    assert torch.allclose(weights, torch.tensor([0.5, 0.5]))
    c3_350 = load_arm("C3", 350)
    indices, weights = ready_weights(c3_350)
    assert indices == [0, 1, 2]
    assert torch.allclose(weights, torch.tensor([1.0 / 3.0] * 3))
    c1 = load_arm("C1", 350)
    indices, weights = ready_weights(c1)
    assert indices == [0]
    assert torch.allclose(weights, torch.tensor([1.0]))


def test_decomposition_is_read_only_and_matches_score_keys():
    detector = load_arm("C3", 350)
    generator = torch.Generator().manual_seed(3)
    K = torch.randn(784, 64, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(784, 768, generator=generator)
    indices, weights = ready_weights(detector)
    Ws = [detector.levels[index].W for index in indices]
    before_hash = detector.state_hash()
    before_rng = rng_fingerprint()
    decomposition = decompose_mixture(Y, K, Ws, weights)
    reference = detector.score_keys(K, Y)["patch_scores"]
    assert torch.allclose(decomposition.fused_scores, reference, atol=1e-4, rtol=1e-5)
    assert torch.allclose(decomposition.fused_scores.max(), torch.tensor(float(detector.score_keys(K, Y)["image_score"])), atol=1e-6)
    assert detector.state_hash() == before_hash
    assert rng_fingerprint() == before_rng
