# exps/nlcad/test_forensic_06a.py
"""CPU checks for the NLCAD-06A forensic operators (no data, no checkpoints)."""
from __future__ import annotations

import json

import torch

from exps.nlcad import forensic_06a as F


def _sd(seed):
    g = torch.Generator().manual_seed(seed)
    return {f"{l}.weight": torch.randn(768, 768, generator=g) for l in F.LAYERS} | {"other": torch.randn(3, generator=g)}


def test_edited_endpoints_and_linearity():
    prev, full = _sd(0), _sd(1)
    eye, zero = torch.eye(768, dtype=torch.float64), torch.zeros(768, 768, dtype=torch.float64)
    keep = F.edited(full, prev, {l: zero for l in F.LAYERS})
    back = F.edited(full, prev, {l: eye for l in F.LAYERS})
    for l in F.LAYERS:
        k = l + ".weight"
        assert torch.allclose(keep[k], full[k], atol=1e-5) and torch.allclose(back[k], prev[k], atol=1e-5)
    U = F.rand_basis(50, torch.Generator().manual_seed(2))
    out = F.edited(full, prev, {F.LAYERS[0]: U @ U.T})
    d = (out[F.LAYERS[0] + ".weight"] - prev[F.LAYERS[0] + ".weight"]).double()
    assert float((d @ U).abs().max()) < 1e-4                    # protected update has no component on U
    assert out["other"] is full["other"]                           # untouched parameters are the stage-end ones


def test_basis_orthonormal_and_energy():
    g = torch.Generator().manual_seed(3)
    a = torch.randn(4000, 768, generator=g, dtype=torch.float64) * torch.linspace(3, 0.01, 768, dtype=torch.float64)
    cv = a.T @ a / len(a)
    for tau in F.TAUS:
        U, w = F.basis(cv, tau)
        assert torch.allclose(U.T @ U, torch.eye(U.shape[1], dtype=U.dtype), atol=1e-8)
        assert F.frac_in(cv, U) >= tau - 1e-9
        assert F.frac_in(cv, U[:, :-1]) < tau                      # minimal rank
    assert F.basis(F.eig(cv), 0.97)[0].shape == F.basis(cv, 0.97)[0].shape


def test_projection_before_elementwise_normaliser_leaks():
    """Projecting a gradient before an elementwise (Adam-like) normaliser does not keep the step out of U."""
    g = torch.Generator().manual_seed(4)
    G = torch.randn(768, 768, generator=g, dtype=torch.float64)
    U = F.rand_basis(100, g)
    gp = G - G @ U @ U.T
    assert float((gp @ U).abs().max()) < 1e-9
    step = gp / gp.abs()
    assert float((step @ U).pow(2).sum() / step.pow(2).sum()) > 0.01      # vs < 1e-18 when projecting after


def test_write_partial_overwrites(tmp_path):
    p = tmp_path / "x.partial.json"
    F.write_partial(p, {"a": 1})
    F.write_partial(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
