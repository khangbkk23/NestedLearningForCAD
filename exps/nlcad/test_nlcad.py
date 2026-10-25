# exps/nlcad/test_nlcad.py
"""Stage A CPU tests for NL-CAD (synthetic tensors only: no image, no mask, no checkpoint)."""
from __future__ import annotations

import copy
import math

import torch

from exps.dinomaly import official as O
from exps.nlcad.controller import BLOCKS, C_BOUND, BlockController, block_of
from exps.nlcad.fast import nl_train_stage, slow_event
from exps.nlcad.objectives import NormalLookahead, ranking_loss


def tiny(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    dec = O.Decoder(32, 4).to(dtype)
    O.init_trainable(dec)
    return dec


def feats(n, seed, dtype=torch.float32):
    return torch.randn(n, 2, 21, 32, generator=torch.Generator().manual_seed(seed)).to(dtype)


def state(dec):
    return {k: v.detach().clone() for k, v in dec.state_dict().items()}


def test_blocks_cover_every_decoder_parameter():
    assert {block_of(n) for n, _ in O.Decoder(768, 12).named_parameters()} == set(BLOCKS)


def test_identity_bounds_and_sign():
    c = BlockController()
    assert c.a() == {b: 1.0 for b in BLOCKS}
    with torch.no_grad():
        c.phi.copy_(torch.randn(len(BLOCKS), dtype=torch.float64) * 10)
    m = c.multipliers()
    m = m.detach()
    assert abs(float(m.mean()) - 1) < 1e-12 and float(m.max() / m.min()) <= math.exp(2 * C_BOUND) + 1e-12
    c = BlockController(lr=0.1)
    s = {b: 0.0 for b in BLOCKS}
    s["decoder.3"] = 1.0                                         # a larger decoder.3 update would raise the outer loss
    c.slow_step(s)
    a = c.a()
    assert a["decoder.3"] < 1.0 and all(a[b] > 1.0 for b in BLOCKS if b != "decoder.3")


def test_grad_phi_matches_finite_difference():
    c = BlockController(weight_decay=0.0)
    with torch.no_grad():
        c.phi.copy_(torch.linspace(-1, 1, len(BLOCKS), dtype=torch.float64))
    s = {b: float(v) for b, v in zip(BLOCKS, torch.randn(len(BLOCKS), dtype=torch.float64))}
    g = c.grad_phi(s)
    f = lambda: float(sum(s[b] * v for b, v in zip(BLOCKS, c.multipliers().detach())))
    for j in range(len(BLOCKS)):
        with torch.no_grad():
            c.phi[j] += 1e-6
        up = f()
        with torch.no_grad():
            c.phi[j] -= 2e-6
        down = f()
        with torch.no_grad():
            c.phi[j] += 1e-6
        assert abs((up - down) / 2e-6 - float(g[j])) < 1e-6


def test_stableadamw_update_is_linear_in_lr():
    """Post-step rescaling of the applied update by a equals running the optimizer at lr * a (weight decay included)."""
    torch.manual_seed(4)                                         # FP64: tests the arithmetic, not FP32 rounding (~1e-6)
    w0, grads = torch.randn(50, 7, dtype=torch.float64), [torch.randn(50, 7, dtype=torch.float64) for _ in range(5)]
    a = 0.7
    pa = torch.nn.Parameter(w0.clone())
    pb = torch.nn.Parameter(w0.clone())
    oa = O.StableAdamW([pa], lr=2e-3 * a, weight_decay=1e-4, amsgrad=True, eps=1e-10)
    ob = O.StableAdamW([pb], lr=2e-3, weight_decay=1e-4, amsgrad=True, eps=1e-10)
    for g in grads:                                              # same gradient sequence for both (isolates the step)
        pa.grad, pb.grad = g.clone(), g.clone()
        oa.step()
        old = pb.detach().clone()
        ob.step()
        with torch.no_grad():
            pb.copy_(old + a * (pb - old))
    assert torch.allclose(pa, pb, rtol=0, atol=1e-12)


def _run(ctrl=None, outer=None, learn=True, K=8, iters=40):
    dec = tiny()
    cur, rep = feats(32, 1), feats(8, 2)
    torch.manual_seed(7)
    out = nl_train_stage(dec, cur, iters, torch.Generator().manual_seed(3), rep, ctrl=ctrl, outer=outer, K=K,
                         learn=learn, device="cpu", autocast=False)
    return dec, out


def test_identity_controller_equals_replay_baseline_and_fixed_machinery_is_inert():
    d0, (l0, s0, e0) = _run()
    d1, (l1, s1, e1) = _run(ctrl=BlockController())
    cur, rep = feats(32, 1), feats(8, 2)
    d2, (l2, s2, e2) = _run(ctrl=BlockController(), outer=NormalLookahead(cur, rep, seed=9, device="cpu",
                                                                         autocast=False), learn=False)
    for d in (d1, d2):
        assert all(torch.equal(a, b) for a, b in zip(state(d0).values(), state(d).values()))
    assert l0 == l1 == l2 and s0 == s1 == s2 == [] and e0 == e1 == [] and len(e2) == 5
    assert all(ev["finite"] and ev["hypergrad"] is not None for ev in e2)


def test_learning_controller_changes_phi_and_trajectory():
    cur, rep = feats(32, 1), feats(8, 2)
    ctrl = BlockController(lr=0.05)
    d0, _ = _run()
    d3, (_, _, ev) = _run(ctrl=ctrl, outer=NormalLookahead(cur, rep, seed=9, device="cpu", autocast=False))
    assert ctrl.events == 5 and float(ctrl.phi.abs().max()) > 0
    assert any(not torch.equal(a, b) for a, b in zip(state(d0).values(), state(d3).values()))


def test_query_disjoint_from_inner_and_within_allowed_pools():
    cur, rep = feats(32, 1), feats(8, 2)
    q = NormalLookahead(cur, rep, seed=1, device="cpu", autocast=False)
    for t in range(20):
        ci, ri = list(range(t % 4, 32, 4))[:8], [t % 8, (t + 3) % 8]
        qc, qr = q.query({"ci": ci, "ri": ri})
        assert len(qc) == 8 and not set(qc) & set(ci) and all(0 <= j < 32 for j in qc)
        assert len(qr) == 6 and not set(qr) & set(ri) and all(0 <= j < 8 for j in qr)
    q1 = NormalLookahead(cur, None, seed=1, device="cpu", autocast=False)
    qc, qr = q1.query({"ci": list(range(16)), "ri": None})
    assert len(qc) == 16 and qr == [] and min(qc) >= 16


def test_lookahead_hypergradient_matches_finite_difference():
    """One fast step from theta_old with multipliers a(phi); J(phi) = L_Q(theta_old + a(phi) * Delta1), fixed query."""
    dec = tiny(dtype=torch.float64)
    for m in dec.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    cur, rep = feats(32, 1, torch.float64), feats(8, 2, torch.float64)
    outer = NormalLookahead(cur, rep, seed=5, device="cpu", autocast=False)
    fixed = outer.query({"ci": list(range(8)), "ri": [0, 1, 2, 3]})
    outer.query = lambda ctx: fixed
    ctrl = BlockController(weight_decay=0.0)
    with torch.no_grad():
        ctrl.phi.copy_(torch.linspace(-0.8, 0.8, len(BLOCKS), dtype=torch.float64))
    a = ctrl.a()
    named = list(dec.named_parameters())
    old = [p.detach().clone() for _, p in named]
    delta1 = [torch.randn_like(p) * 1e-3 for _, p in named]      # a fixed base update direction (held fixed, as in the method)
    with torch.no_grad():
        for (n, p), o, d in zip(named, old, delta1):
            p.copy_(o + a[block_of(n)] * d)
    ev = slow_event(dec, named, old, a, outer, {"iter": 0, "ci": [], "ri": []}, ctrl, learn=False, device="cpu")
    g = torch.tensor(ev["hypergrad"])

    def J(phi):
        with torch.no_grad():
            ctrl.phi.copy_(phi)
            av = dict(zip(BLOCKS, ctrl.multipliers().tolist()))
            for (n, p), o, d in zip(named, old, delta1):
                p.copy_(o + av[block_of(n)] * d)
            x = torch.cat([cur[fixed[0]], rep[fixed[1]]])
            tot = 0.
            for xb in x.split(8):
                en, de = dec(xb[:, 0], xb[:, 1])
                tot += float(O.global_cosine(en, de)) * len(xb) / len(x)
            return tot

    base = ctrl.phi.detach().clone()
    for j in range(len(BLOCKS)):
        e = torch.zeros_like(base)
        e[j] = 1e-4
        fd = (J(base + e) - J(base - e)) / 2e-4
        assert abs(fd - float(g[j])) <= 1e-6 + 1e-3 * abs(fd), (j, fd, float(g[j]))


def test_controller_snapshot_roundtrip():
    c = BlockController(lr=0.1)
    s = {b: float(i) for i, b in enumerate(BLOCKS)}
    c.slow_step(s)
    snap = copy.deepcopy(c.snapshot())
    c2 = BlockController(lr=0.1)
    c2.restore(snap)
    c.slow_step(s)
    c2.slow_step(s)
    assert torch.equal(c.phi, c2.phi) and c.events == c2.events == 2 and c.persistent_bytes() == 9 * 24 + 8


def test_ranking_loss_direction():
    s = torch.zeros(1, 28, 28, requires_grad=True)
    cov = torch.zeros(1, 28, 28)
    cov[0, :4, :4] = 1.0
    loss = ranking_loss(s, cov, tau=0.05, gen=torch.Generator().manual_seed(0))
    loss.backward()
    assert float(s.grad[0, :4, :4].sum()) < 0 and float(s.grad[0, 10:, 10:].sum()) >= 0


def test_no_restart_schedule_matches_recipe_formula():
    from exps.nlcad.run import global_schedule
    for total in (667, 6003):
        p = torch.nn.Parameter(torch.zeros(1))
        s = O.WarmCosineScheduler(O.StableAdamW([p], lr=2e-3), 2e-3, 2e-4, total, warmup_iters=100)
        assert abs(global_schedule(total) - s.schedule).max() < 1e-15


def test_lr_at_override_sets_learning_rate():
    seen = []

    def lr_at(i):
        seen.append(i)
        return 1e-3
    dec = tiny()
    torch.manual_seed(7)
    nl_train_stage(dec, feats(32, 1), 12, torch.Generator().manual_seed(3), feats(8, 2), device="cpu",
                   autocast=False, lr_at=lr_at)
    assert seen == list(range(12))


def test_patch_coverage_and_allowed_hosts():
    import numpy as np
    from exps.nlcad import bank as B
    from exps.rh01b import common as RC
    m = np.zeros((900, 900), bool)
    m[:450, :450] = True                                          # top-left quarter of the image
    cov = B.patch_coverage(m)
    assert cov.shape == (28, 28) and float(cov[:14, :14].min()) == 1.0 and float(cov[14:, 14:].max()) == 0.0
    n = {c: 100 for c in RC.STREAM}
    hosts = B.host_sets(2711, n)
    for s in (1, 5, 15):
        keys = B.allowed(hosts, 2711, n, s)
        cats = {k[0] for k in keys}
        assert RC.STREAM[s - 1] in cats and cats <= set(RC.STREAM[:s])  # never a future category
        for d in RC.STREAM[:s - 1]:                               # only buffer members of completed categories
            idx = RC.buffer_index(2711, n, list(RC.STREAM[:s - 1]))[d]
            assert {k[1] for k in keys if k[0] == d} == set(idx)


# --- Stage B: GPU equivalence with the locked RH-01 loop (tiny decoder, synthetic features, D4' numerics) ----------

import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_nl_loop_equals_locked_guarded_stage():
    from exps.rh01 import numerics as NUM
    from exps.rh01 import run as R1
    torch.cuda.set_per_process_memory_fraction(2.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    NUM.use(True)
    cur, rep = feats(32, 1), feats(8, 2)
    outs = []
    for mode in ("locked", "nl_none", "nl_identity_fixed"):
        torch.manual_seed(0)
        dec = O.Decoder(32, 4, checkpoint=True).cuda()
        O.init_trainable(dec)
        torch.manual_seed(7)
        gen = torch.Generator().manual_seed(3)
        if mode == "locked":
            losses, skips = R1.train_stage_guarded(dec, cur, 40, gen, rep)
        elif mode == "nl_none":
            losses, skips, _ = nl_train_stage(dec, cur, 40, gen, rep)
        else:
            losses, skips, ev = nl_train_stage(dec, cur, 40, gen, rep, ctrl=BlockController(),
                                               outer=NormalLookahead(cur, rep, seed=9), K=8, learn=False)
            assert len(ev) == 5 and all(e["finite"] for e in ev)
        outs.append((state(dec), losses, skips))
    NUM.use(False)
    for st, l, s in outs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(outs[0][0].values(), st.values())) and l == outs[0][1]
