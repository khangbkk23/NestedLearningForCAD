# exps/test_hope_outer_learning_h50_v1.py
"""Focused synthetic checks for the contextual-signal construction."""

from __future__ import annotations

import torch

from exps.ol04e.hope_outer_learning_h50_v1 import CENTERS, ridge_fit, predict


def test_ridge_prediction_shape_and_train_fit():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(6, 16, 4, generator=generator)
    y = torch.randn(6, 16, 4, generator=generator)
    fit = ridge_fit(x, y)
    prediction = predict(fit, x, torch.zeros(16, 4))
    assert prediction.shape == y.shape
    assert fit["B"].shape == (4, 4)
    assert fit["lambda"] > 0


def test_shuffled_fit_is_deterministic():
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(8, 16, 5, generator=generator)
    y = torch.randn(8, 16, 5, generator=generator)
    a = ridge_fit(x, y, seed=4404)
    b = ridge_fit(x, y, seed=4404)
    torch.testing.assert_close(a["B"], b["B"], rtol=0, atol=0)
    assert a["lambda"] == b["lambda"]


def test_center_contract_is_fixed():
    assert len(CENTERS) == 16
    assert CENTERS[0] == (3, 3)
    assert CENTERS[-1] == (24, 24)

