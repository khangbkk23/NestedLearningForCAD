"""CPU invariants for the NLCAD-05A record-only harness."""
from __future__ import annotations

import math

import numpy as np
import torch

from exps.nlcad.controller import BLOCKS
from exps.nlcad.retention_credit import CreditRecorder, cosine, phi_hypergrad, sign_agreement
from exps.nlcad.test_nlcad import feats, tiny
from exps.nlcad.test_nlcad_04 import REPLAY_KEYS, tiny_bank


def test_full_normalized_phi_jacobian_identity_and_nonidentity():
    for phi in (torch.zeros(9, dtype=torch.float64), torch.linspace(-.8, .8, 9, dtype=torch.float64)):
        s = {b: float(v) for b, v in zip(BLOCKS, torch.linspace(-.7, .9, 9, dtype=torch.float64))}
        got = np.asarray(phi_hypergrad(phi, s))
        def f(z):
            h = np.log(2.) / 2 * np.tanh(z)
            a = np.exp(h) / np.exp(h).mean()
            return float(np.dot(a, np.asarray(list(s.values()))))
        fd = []
        for j in range(9):
            e = torch.zeros(9, dtype=torch.float64); e[j] = 1e-5
            fd.append((f(phi + e) - f(phi - e)) / 2e-5)
        assert np.allclose(got, fd, rtol=2e-5, atol=2e-7)


def test_credit_recorder_gradient_and_host_eligibility():
    bank, allowed = tiny_bank()
    rec = CreditRecorder(bank, {}, allowed, "leather", REPLAY_KEYS, 7, 2711, "cpu")
    ctx = {"iter": 31, "ci": [0, 1], "ri": [0]}
    for src in ("current", "replay"):
        items = rec.sample("Q3", src, ctx, "check_" + src)
        assert len(items) == 8
        assert all(it["host"] in allowed and it["host"] not in rec.inner_hosts(ctx) for it in items)
        assert all((it["host"][0] == "leather") == (src == "current") for it in items)
    dec = tiny()
    out, grads = rec.event(dec, feats(32, 4), feats(8, 5), ctx, q2=True)
    assert out["Q3_current"]["used"] > 0 and out["Q3_replay"]["used"] > 0 and out["Q2_replay"]["used"] > 0
    assert all(np.isfinite(v) for g in grads.values() if g is not None for b in BLOCKS for v in [sum(float(x.abs().sum()) for x in g[b])])
    assert all(p.grad is None for p in dec.parameters())


def test_paired_queries_and_rng_are_deterministic():
    bank, allowed = tiny_bank()
    ctx = {"iter": 63, "ci": [2, 3], "ri": [1]}
    r1 = CreditRecorder(bank, {}, allowed, "leather", REPLAY_KEYS, 8, 2711, "cpu")
    r2 = CreditRecorder(bank, {}, allowed, "leather", REPLAY_KEYS, 8, 2711, "cpu")
    assert [[x["host"] for x in r1.sample("Q2", "replay", ctx, "paired")] ==
            [x["host"] for x in r2.sample("Q2", "replay", ctx, "paired")]]
    assert r1.normal_query(feats(32, 4), feats(8, 5), ctx, "normal") == r2.normal_query(feats(32, 4), feats(8, 5), ctx, "normal")


def test_credit_summary_invariants():
    a = {b: float(i - 4) for i, b in enumerate(BLOCKS)}
    b = {k: -v for k, v in a.items()}
    assert -1 <= cosine(a, b) <= 1 and sign_agreement(a, b) == 1 / 9


def _counterfactual_setup():
    import math
    from exps.dinomaly import official as O
    from exps.nlcad.fast import block_optimizer
    bank, allowed = tiny_bank()
    rec = CreditRecorder(bank, {}, allowed, "leather", REPLAY_KEYS, 7, 2711, "cpu")
    dec = tiny()
    assert any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in dec.modules())   # the bottleneck dropout
    dec.train()
    opt = block_optimizer(dec); sch = O.make_scheduler(opt, 400)
    for _ in range(150):                        # past the warm-up (lr = 0 at step 0)
        sch.step()
    x = feats(16, 9)
    ctx = {"iter": 31, "ci": [0, 1], "ri": [0], "q3_current_credit": {b: 0.1 * i for i, b in enumerate(BLOCKS)}}
    s = {b: (-1) ** i * 0.01 * (i + 1) for i, b in enumerate(BLOCKS)}
    return dec, opt, sch, rec, x, ctx, s


def test_counterfactual_leaves_rng_and_next_step_identical_to_identity_branch():
    """R3 regression: the branches and their metrics must not consume the trajectory's RNG, so the real next step
    equals the identity branch bitwise (R2 failed this on GPU: its normal-query forward ran with dropout active)."""
    from exps.nlcad.retention_credit import counterfactual, fast_one_step, state_hash
    dec, opt, sch, rec, x, ctx, s = _counterfactual_setup()
    torch.manual_seed(123)
    before = torch.get_rng_state()
    cf = counterfactual(dec, opt, sch, rec, feats(32, 4), feats(8, 5), ctx, x, 0.5, s, torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())
    assert cf["restoration_exact"]
    fast_one_step(dec, opt, sch, x, 0.5, {b: 1.0 for b in BLOCKS}, torch.device("cpu"))
    assert state_hash(dec) == cf["variants"]["identity"]["state_hash"]
    v = cf["variants"]
    assert v["identity"]["effective_update_norm"] > 0 and v["identity"]["a"] == {b: 1.0 for b in BLOCKS}
    for name in ("q3_replay", "matched_norm_random", "q3_current_plus_replay"):
        assert abs(math.sqrt(sum(p ** 2 for p in v[name]["phi"])) - 0.15) < 1e-6          # phi is built from a float32 tensor
