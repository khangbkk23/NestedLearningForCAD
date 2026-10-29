# exps/nlcad/test_anchor_06c.py
"""CPU checks for NLCAD-06C-ANCHOR (tiny decoder and synthetic tensors, plus one check on the real stage-7 sets)."""
from __future__ import annotations

import numpy as np
import torch

from exps.dinomaly import run as DR
from exps.nlcad import anchor_06c as A
from exps.nlcad import closed_loop_06b as M
from exps.nlcad.test_nlcad import feats, state, tiny
from exps.nlcad.test_nlcad_06b import _run, layer_bases


def test_loop_without_anchor_is_bitwise_the_06b_loop():
    U = layer_bases(4, 50)
    for kind, alpha in (("id", 0.0), ("rep", 0.5)):
        ref = _run(lambda dec, cur, rep, gen, it: M.train_stage_protected(
            dec, cur, it, gen, rep, op=M.Protector(dec, kind, alpha, U, U, "cpu"), device="cpu", autocast=False))
        out = _run(lambda dec, cur, rep, gen, it: A.train_stage_06c(
            dec, cur, it, gen, rep, op=M.Protector(dec, kind, alpha, U, U, "cpu"), device="cpu", autocast=False))
        assert all(torch.equal(out[0][k], ref[0][k]) for k in ref[0]), kind
        assert torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2]), kind


def test_batch_and_replay_identities_follow_the_06b_stream():
    current, replay = feats(20, 2), feats(10, 3)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    c1, r1 = DR.Batches(20, 8, g1), DR.Batches(10, 8, g1)
    c2, r2 = DR.Batches(20, 8, g2), DR.Batches(10, 8, g2)
    for _ in range(5):
        x = current[c1.next()]                                   # the 06B sequence
        x = torch.cat([x, replay[r1.next()]])
        x = x[torch.randperm(len(x), generator=g1)]
        y, rid = A.next_batch(current, replay, c2, r2, g2)
        assert torch.equal(x, y)
        m = rid >= 0
        assert int(m.sum()) == 8 and torch.equal(y[m], replay[rid[m]])
    assert torch.equal(g1.get_state(), g2.get_state())


def test_zero_weight_anchor_leaves_training_unchanged():
    tg = A.anchor_targets(tiny(seed=1), feats(10, 3), "cpu")
    ref = _run(lambda dec, cur, rep, gen, it: A.train_stage_06c(dec, cur, it, gen, rep, device="cpu", autocast=False))
    out = _run(lambda dec, cur, rep, gen, it: A.train_stage_06c(dec, cur, it, gen, rep, device="cpu", autocast=False,
                                                              anchor={"lam": 0.0, "targets": tg}))
    assert all(torch.equal(out[0][k], ref[0][k]) for k in ref[0])
    assert torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])


def test_anchor_changes_training_and_logs_finite_values():
    seen = {}

    def loop(lam):
        def f(dec, cur, rep, gen, it):
            tg = A.anchor_targets(dec, rep, "cpu")
            seen[lam] = A.train_stage_06c(dec, cur, it, gen, rep, device="cpu", autocast=False,
                                          anchor={"lam": lam, "targets": tg})
        return f
    a = _run(loop(0.0))
    b = _run(loop(1.0))
    assert any(not torch.equal(a[0][k], b[0][k]) for k in a[0])
    losses, al, gn, sk = seen[1.0]
    assert not sk and np.all(np.isfinite(al)) and min(al) >= 0 and max(al) > 0
    assert all(x == 0 for x in seen[0.0][1])
    assert len(gn) == len(losses) and np.all(np.isfinite(gn))


def test_anchor_gradient_is_not_hard_mined():
    """grad(micro loss with anchor) - grad(without) == grad(anchor term alone): the hard-mining hook does not act on
    the anchor branch (it would scale its easy points by hm_factor)."""
    dec = tiny(seed=3)
    rep = feats(6, 4)
    tg = A.anchor_targets(tiny(seed=9), rep, "cpu")             # targets from another state: nonzero anchor
    xb, rb = rep[:4], torch.tensor([0, 1, 2, 3])
    dec.eval()                                                  # deterministic forward for the comparison

    def grads(fn):
        dec.zero_grad(set_to_none=True)
        fn().backward()
        return {k: v.grad.clone() for k, v in dec.named_parameters() if v.grad is not None}
    anc = {"lam": 1.0, "targets": tg}
    g_with = grads(lambda: A.micro_loss(dec, xb, rb, 0.9, 8, 4, "cpu", False, anc)[0])
    g_without = grads(lambda: A.micro_loss(dec, xb, rb, 0.9, 8, 4, "cpu", False, None)[0])

    def anchor_only():
        _, de = dec(xb[:, 0], xb[:, 1])
        return A.anchor_distance(de, [t[rb] for t in tg]).sum() / 4
    g_a = grads(anchor_only)
    for k in g_with:
        assert torch.allclose(g_with[k] - g_without.get(k, 0), g_a[k], atol=1e-6, rtol=1e-4), k


def test_anchor_targets_consume_no_rng_and_restore_mode():
    dec = tiny(seed=2)
    dec.train()
    s0 = torch.get_rng_state()
    t1 = A.anchor_targets(dec, feats(5, 1), "cpu", micro=2)
    assert torch.equal(torch.get_rng_state(), s0) and dec.training
    t2 = A.anchor_targets(dec, feats(5, 1), "cpu", micro=5)                # FP32: not bitwise batch-size invariant
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(t1, t2)) and t1[0].shape[0] == 5


def test_stage_sets_with_seed_2711_equal_the_06b_sets():
    from exps.nlcad import forensic_06a as F
    from exps.rh01 import run as R1
    from exps.rh01b import common as C
    from exps.rh01b import run as RB
    n = R1.n_train(RB.verify_lock())
    sub = F.subsets(n)
    held = {c: DR.load_split("heldout", c) for c in C.EVAL}
    a, b = A.stage_sets(2711, 7, n, sub, held), M.stage_sets(7, n, sub, held)
    assert a["replay_keys"] == b["replay_keys"] and a["groups"] == b["groups"]
    assert torch.equal(a["replay"], b["replay"]) and torch.equal(a["old_x"], b["old_x"])
    keep = A.o6_keep(sub, held)
    assert len(keep) == 78 and sum(keep) == 77 and b["groups"][keep.index(False)] == "cable"
    assert A.stage_sets(2712, 7, n, sub, held)["replay_keys"] != a["replay_keys"]


def test_shock_ratio():
    assert A.shock_ratio([0.06, 0.045, 0.043, 0.131, 0.168, 0.094]) == (0.168 / 0.043, 200)
    assert A.shock_ratio([0.08, 0.05, 0.048, 0.046, 0.044])[0] < 1


def _sm(g, acq_ok=True, prohibitive=False):
    seeds = ["1", "2", "3"]
    out = {"seeds": seeds, "g": {}, "acq": {}}
    for arm, vals in g.items():
        out["g"][arm] = {k: {s: {"point": v} for s, v in zip(("7", "8", "9"), vals[i])} for i, k in enumerate(seeds)}
        out["acq"][arm] = {"retained_all": acq_ok, "prohibitive_any": prohibitive}
    return out


def test_decision_branches():
    rep_rev = [(-0.007, 0.0, -0.001)] * 3                       # REP-05 reverts: P ~ 0.14
    keep = [(-0.006, -0.005, -0.004)] * 3                       # anchor persists: P ~ 0.67
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": keep}))[0] == "EQUILIBRIUM_SHIFT_SUPPORTED"
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": rep_rev}))[0] == "EQUILIBRIUM_SHIFT_REJECTED"
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": [(-0.001, 0, 0)] * 3}))[0] == "NO_STAGE7_EFFECT"
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": [(-0.006, -0.003, -0.002)] * 3}))[0] == "INCONCLUSIVE"
    mixed = [(-0.006, -0.005, -0.004)] * 2 + [(-0.006, -0.005, 0.001)]
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": mixed}))[0] == "INCONCLUSIVE"
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": keep}, prohibitive=True))[0] == "EQUILIBRIUM_SHIFT_REJECTED"
    assert A.decide(_sm({"REP-05": rep_rev, "ANCHOR": keep}))[1] == "REP-05 reversion replicated"
