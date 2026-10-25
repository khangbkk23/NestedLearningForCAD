# exps/rh01b/test_rh01b.py
"""Pre-run checks for RH-01b: the D4' patched forward and the stage-seed formulas."""
from __future__ import annotations

import pytest
import torch

from exps.dinomaly import official as O
from exps.rh01 import numerics as NUM
from exps.rh01b import run as RB


def _att(seed=0):
    torch.manual_seed(seed)
    return O.LinearAttention2(64, num_heads=4, qkv_bias=True)


def test_fp32core_bitwise_identical_in_fp32():
    att, x = _att(), torch.randn(2, 21, 64)
    NUM.use(False)
    a = att(x)[0]
    NUM.use(True)
    b = att(x)[0]
    NUM.use(False)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32core_finite_where_bf16_original_vanishes():
    att = _att().cuda()
    with torch.no_grad():                                   # head 0 queries of token 3 strongly negative -> phi(q)=0 in BF16
        att.qkv.weight.zero_()
        att.qkv.bias.zero_()
        att.qkv.bias[:16] = -10.0
    x = torch.zeros(1, 21, 64, device="cuda")
    outs = {}
    for fp32core in (False, True):
        NUM.use(fp32core)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outs[fp32core] = att(x)[0].float()
    NUM.use(False)
    assert not bool(torch.isfinite(outs[False]).all()) and bool(torch.isfinite(outs[True]).all())


def test_stage_seeds():
    for seed in RB.SEEDS:
        assert RB.stage_seeds("R15", seed, 1) == (seed * 100 + 1, seed * 100 + 1)      # DRF-01 arm A stage 1
        assert RB.stage_seeds("R15", seed, 5) == (seed * 100 + 5, seed * 100 + 55)    # RH-01 R15 formula
        assert RB.stage_seeds("J15", seed, 1) == (seed * 100 + 1, seed * 100 + 71)
