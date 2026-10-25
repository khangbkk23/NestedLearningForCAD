# exps/nlcad/test_nlcad_r1.py
"""CPU tests for the proposed slow-level revision (controller_r1.py)."""
from __future__ import annotations

import copy

import torch

from exps.nlcad.controller import BLOCKS, BlockController


def test_mean_controller_follows_mean_not_majority_sign():
    from exps.nlcad.controller_r1 import MeanController
    seq = [-1.0, -1.0, -1.0, -1.0, 10.0]                         # majority negative, mean positive (heavy tail)
    mc, ac = MeanController(step=0.1, window=5, weight_decay=0.0), BlockController(lr=0.05, weight_decay=0.0)
    for v in seq:
        s = {b: 0.0 for b in BLOCKS}
        s["decoder.2"] = v
        mc.slow_step(s)
        ac.slow_step(s)
    assert mc.updates == 1 and mc.a()["decoder.2"] < 1.0           # follows the mean: smaller decoder.2 updates
    assert ac.a()["decoder.2"] > 1.0                               # Adam followed the majority sign


def test_mean_controller_global_scale_identity_bounds_and_roundtrip():
    from exps.nlcad.controller_r1 import MeanController
    mc = MeanController(step=0.5, window=2, global_scale=True)
    assert mc.a() == {b: 1.0 for b in BLOCKS}
    for _ in range(40):
        mc.slow_step({b: 1.0 for b in BLOCKS})                     # every block's larger update hurts -> gamma down
    a = mc.a()
    assert all(v < 1.0 for v in a.values()) and min(a.values()) >= 0.5 / 2 ** 0.5 - 1e-9 and mc.gamma() >= 0.5 - 1e-12
    snap = copy.deepcopy(mc.snapshot())
    mc2 = MeanController(step=0.5, window=2, global_scale=True)
    mc2.restore(snap)
    s = {b: float(i) for i, b in enumerate(BLOCKS)}
    for _ in range(2):
        mc.slow_step(s)
        mc2.slow_step(s)
    assert torch.equal(mc.phi, mc2.phi) and torch.equal(mc.psi, mc2.psi)
