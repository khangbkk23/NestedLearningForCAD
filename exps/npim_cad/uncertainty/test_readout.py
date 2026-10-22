import numpy as np
import pytest
import torch

from exps.npim_cad.npim01_contract import load_state_detectors
from exps.npim_cad.uncertainty.readout import (
    a0_parity,
    image_max,
    score_keys,
    score_patch_residual,
)


@pytest.fixture(scope="module")
def detectors():
    return load_state_detectors("cpu")


def test_a0_parity_and_read_only(detectors):
    detector = detectors[("C1", 100)]
    generator = torch.Generator().manual_seed(0)
    K = torch.randn(784, detector.config.r, generator=generator)
    Y = torch.randn(784, 768, generator=generator)
    before = detector.state_hash()
    parity = a0_parity(detector, K, Y)
    assert parity["pass"] is True
    assert parity["max_abs"] <= 1e-6
    assert detector.state_hash() == before


def test_center_and_diagonal_identities(detectors):
    detector = detectors[("C2", 350)]
    generator = torch.Generator().manual_seed(1)
    K = torch.randn(784, detector.config.r, generator=generator)
    Y = torch.randn(784, 768, generator=generator)
    scored = score_keys(detector, K, Y)
    residual = scored["residual"]
    a0 = score_patch_residual(residual, kind="A0")
    zeros = torch.zeros(768)
    ones = torch.ones(768)
    a1 = score_patch_residual(residual, kind="A1", mu=zeros, var_safe=ones)
    a2_one = score_patch_residual(residual, kind="A2", mu=zeros, var_safe=ones)
    a2_two = score_patch_residual(residual, kind="A2", mu=zeros, var_safe=2.0 * ones)
    assert torch.allclose(a0, a1, atol=1e-7, rtol=0)
    assert torch.allclose(a0, a2_one, atol=1e-7, rtol=0)
    assert torch.allclose(a2_two, a0 / 2.0, atol=1e-7, rtol=0)
    assert torch.allclose(scored["patch_scores"], a0, atol=1e-7, rtol=0)


def test_nonpositive_variance_fails_closed(detectors):
    detector = detectors[("C1", 100)]
    generator = torch.Generator().manual_seed(2)
    residual = torch.randn(16, 768, generator=generator)
    with pytest.raises(ValueError):
        score_patch_residual(residual, kind="A2", mu=torch.zeros(768), var_safe=torch.zeros(768))
    with pytest.raises(ValueError):
        score_patch_residual(residual, kind="A2", mu=torch.zeros(768), var_safe=-torch.ones(768))


def test_image_max():
    assert image_max(torch.tensor([1.0, 4.0, 2.0])) == 4.0
    with pytest.raises(ValueError):
        image_max(torch.tensor([[1.0, 2.0]]))
