# exps/nlcad/test_nlcad_02.py
"""NLCAD-02 CPU tests: arm configurations differ from Stage C N only where the task says, the F1/RND profile rules,
fixed-profile update parity in the fast loop, and the open-loop evidence behind lambda = 1e-5."""
from __future__ import annotations

import numpy as np
import torch

from exps.dinomaly import common as D
from exps.nlcad import analyze_02 as A2
from exps.nlcad import run_02 as R2
from exps.nlcad.controller import BLOCKS, BlockController
from exps.nlcad.objectives import NormalLookahead
from exps.nlcad.test_nlcad import _run, feats

STAGE_C = D.read(R2.OUT / "runs/N_per_task_seed2711_from6/CONFIG.json")


def test_n1_differs_from_stage_c_n_only_in_lambda():
    n1 = R2.config("N1")
    assert set(n1) == set(STAGE_C)
    assert {k for k in n1 if n1[k] != STAGE_C[k]} == {"wd_phi"} and n1["wd_phi"] == 1e-5 and STAGE_C["wd_phi"] == 1e-3


def test_fixed_arms_differ_from_n1_only_in_arm_learning_and_profile():
    prof = dict(zip(BLOCKS, np.linspace(-0.3, 0.3, len(BLOCKS)).tolist()))
    n1, f = R2.config("N1"), R2.config("F1", prof)
    assert {k for k in n1 if n1[k] != f[k]} == {"arm", "learn", "profile"}
    assert f["arm"] == "F" and f["learn"] is False and f["outer"] == "normal" and f["profile"] == prof


def test_f1_is_the_mean_recorded_phi():
    ev = [{"phi": [float(i + j) for j in range(len(BLOCKS))]} for i in range(4)]
    assert np.allclose(R2.f1_phi(ev), [1.5 + j for j in range(len(BLOCKS))])


def test_rnd_matches_centred_log_multiplier_norm_and_is_deterministic():
    phi = np.array([-0.6, -0.6, 0.0, 0.3, 0.4, -0.2, 0.1, 0.3, 0.5])
    target = float(np.linalg.norm(R2.centred_log_a(phi)))
    z, s, rp = R2.rnd_phi(target)
    z2, s2, rp2 = R2.rnd_phi(target)
    assert np.array_equal(rp, rp2) and abs(z.mean()) < 1e-12 and s > 0
    assert abs(np.linalg.norm(R2.centred_log_a(rp)) - target) < 1e-9
    a = R2.multipliers(rp)
    assert abs(a.mean() - 1) < 1e-12 and a.max() / a.min() <= 2 + 1e-12
    _, _, other = R2.rnd_phi(target, seed=R2.RND_SEED + 1)
    assert not np.allclose(rp, other)


def test_fixed_profile_is_applied_unchanged_at_every_step():
    """learn=False with a non-zero profile: phi never moves and every slow event records the profile multipliers."""
    phi = torch.tensor([-0.5, 0.2, 0.0, 0.1, -0.1, 0.3, 0.0, 0.2, -0.4], dtype=torch.float64)
    ctrl = BlockController()
    with torch.no_grad():
        ctrl.phi.copy_(phi)
    want = ctrl.a()
    _, (_, skips, ev) = _run(ctrl=ctrl, outer=NormalLookahead(feats(32, 1), feats(8, 2), seed=9, device="cpu",
                                                               autocast=False), learn=False)
    assert torch.equal(ctrl.phi.detach(), phi) and skips == [] and len(ev) == 5
    assert all(e["a_used"] == want for e in ev)


def test_learning_parity_rule_matches_the_loop():
    """For a learning controller, each event's applied multipliers are the previous event's post-step multipliers."""
    ctrl = BlockController(lr=0.05, weight_decay=1e-5)
    _, (_, _, ev) = _run(ctrl=ctrl, outer=NormalLookahead(feats(32, 1), feats(8, 2), seed=9, device="cpu",
                                                          autocast=False))
    res = A2.parity("N1", {7: ev})
    assert res["events"] == 5 and res["mismatches"] == 0 and res["nonfinite_events"] == 0


def test_open_loop_evidence_for_lambda_1e5_reproduces():
    ol = D.read(R2.OUT / "OPENLOOP_N_per_task_seed2711_from6.json")
    pred = next(r for r in ol["replay_counterfactual"] if r["lam"] == 1e-5)
    ctrl = BlockController(lr=0.05, weight_decay=1e-5)
    for e in A2.OL.events("N_per_task_seed2711_from6", 6):
        ctrl.slow_step(e["dJ_da"])
    assert np.allclose(ctrl.phi.detach().numpy(), pred["phi_end"], rtol=0, atol=1e-12)
    assert pred["saturated_blocks_end"] == 0 and pred["max_abs_a_minus_1"] > 0.1
