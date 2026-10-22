# exps/npim_cad/forensic/test_reconstruct.py
"""G1 reconstruction/parity tests (synthetic; no DEV masks)."""

from __future__ import annotations

import torch

from exps.npim_cad.forensic.decompose import decompose_mixture, mix_from_statistics, ready_weights
from exps.npim_cad.forensic.reconstruct import MAX_ABS_TOLERANCE, REL_L2_TOLERANCE
from exps.npim_cad.npim01_contract import PILOT, arm_config
from models.hope_cad.npim_cad import NPIMDetector
from models.hope_cad.npim_state import load_state_file, rng_fingerprint


def load_arm(arm: str, boundary: int) -> NPIMDetector:
    detector = NPIMDetector(arm_config(arm), device="cpu")
    detector.load_state_dict(load_state_file(PILOT / "states" / f"{arm}_{boundary}.pt"))
    return detector


def test_pre_registered_parity_tolerances():
    assert MAX_ABS_TOLERANCE == 1.0e-4
    assert REL_L2_TOLERANCE == 1.0e-5


def test_reconstruction_matches_frozen_score_and_is_read_only():
    detector = load_arm("C3", 350)
    generator = torch.Generator().manual_seed(0)
    K = torch.randn(784, 64, generator=generator)
    K = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Y = torch.randn(784, 768, generator=generator)
    indices, weights = ready_weights(detector)
    Ws = [detector.levels[index].W for index in indices]
    before_hash = detector.state_hash()
    before_rng = rng_fingerprint()
    decomposition = decompose_mixture(Y, K, Ws, weights)
    reference = detector.score_keys(K, Y)["patch_scores"]
    max_abs = float((decomposition.fused_scores - reference).abs().max())
    relative = float((decomposition.fused_scores - reference).norm() / reference.norm())
    assert max_abs <= MAX_ABS_TOLERANCE
    assert relative <= REL_L2_TOLERANCE
    assert detector.state_hash() == before_hash
    assert rng_fingerprint() == before_rng


def test_fast_mid_counterfactual_uses_dot_terms():
    level_scores = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    dots = -768.0 * torch.ones((1, 3))
    fast_mid = mix_from_statistics(level_scores, dots, torch.tensor([0.5, 0.5]), d=768)
    assert torch.allclose(fast_mid, torch.zeros_like(fast_mid), atol=1e-6)


def test_c3_cold_start_has_no_slow_counterfactual():
    from exps.npim_cad.forensic.analysis import diagnostic_kinds

    c3_100 = load_arm("C3", 100)
    ready = torch.tensor([level.ready for level in c3_100.levels])
    kinds = [name for name, _index in diagnostic_kinds("C3", ready)]
    assert "slow" not in kinds
    assert "fast_mid" in kinds
    c3_350 = load_arm("C3", 350)
    ready = torch.tensor([level.ready for level in c3_350.levels])
    kinds = [name for name, _index in diagnostic_kinds("C3", ready)]
    assert "slow" in kinds
    assert "fast_mid" in kinds
    assert diagnostic_kinds("C1", ready) == [("single", None)]
