# exps/nlcad/test_nlcad_05b.py
"""NLCAD-05B CPU tests for the retention slow objective J_ret (tiny decoder and synthetic tensors; no image, no real
mask, no checkpoint)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from exps.nlcad.controller import BLOCKS, BlockController, block_of
from exps.nlcad.fast import slow_event
from exps.nlcad.objectives_ret import RetentionRanking, category_loss, retention_loss
from exps.nlcad.test_nlcad import state, tiny
from exps.nlcad.test_nlcad_04 import CUR, OLD, REPLAY_KEYS, _stage, tiny_bank


def objective(bank, allowed, seed=0, **kw):
    return RetentionRanking(bank, "Q3", "Q2", allowed, CUR, REPLAY_KEYS, seed=seed, micro=4, device="cpu",
                            autocast=False, n_neg=8, **kw)


def fp64(monkeypatch):
    import exps.nlcad.objectives_ret as OR
    monkeypatch.setattr(OR, "_forward", lambda dec, xb, device, autocast: dec(xb[:, 0], xb[:, 1]))
    dec = tiny(dtype=torch.float64)
    for m in dec.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    return dec


def test_hosts_are_old_legal_outside_inner_batch_one_view_each():
    bank, allowed = tiny_bank()
    allowed.discard(("carpet", 3))
    obj = objective(bank, allowed, seed=3, max_hosts=4)
    for t in range(30):
        ri = [t % 12, (t + 5) % 12]
        inner = {REPLAY_KEYS[j] for j in ri}
        hs = obj.hosts({"ci": [0, 1], "ri": ri})
        assert set(hs) <= set(OLD)
        for c, keys in hs.items():
            assert 1 <= len(keys) <= 4 and len(set(keys)) == len(keys)
            assert all(k[0] == c and k in allowed and k not in inner for k in keys)
        sel = obj.select(hs, "Q3")
        for c, items in sel.items():
            assert [tuple(it["host"]) for it in items] == hs[c]          # exactly one view per host identity
            assert all(bool((it["coverage"] >= 0.5).any()) for it in items)


def test_category_loss_quota_cross_pairs_and_balance():
    torch.manual_seed(0)
    cov = torch.zeros(4, 4)
    cov[0, :2] = 1.0
    maps = [torch.randn(4, 4) for _ in range(3)]
    loss, pairs, cross = category_loss(maps, [cov] * 3, tau=0.05, n_neg=8)
    m = math.ceil(8 / 3)                                               # per-host quota of hard negatives
    neg = torch.cat([s[cov == 0].topk(m).values for s in maps])
    ref = torch.stack([F.softplus((neg[None] - s[cov >= 0.5][:, None]) / 0.05).mean() for s in maps]).mean()
    assert torch.allclose(loss, ref) and pairs == 3 * 2 * 3 * m and cross == 3 * 2 * 2 * m
    one, p1, c1 = category_loss(maps[:1], [cov], 0.05, 8)
    assert c1 == 0 and p1 == 2 * 8                                      # single host: within-image pairs only
    # category balance: a category with three hosts weighs the same as a category with one
    J, per, _, _ = retention_loss({"a": maps, "b": maps[:1]}, {"a": [cov] * 3, "b": [cov]}, 0.05, 8)
    assert torch.allclose(J, (per["a"] + per["b"]) / 2)


def test_two_pass_gradient_equals_single_graph_gradient(monkeypatch):
    dec = fp64(monkeypatch)
    bank, allowed = tiny_bank(torch.float64)
    obj = objective(bank, allowed, seed=5, check_tol=1e-12)
    sel = obj.select(obj.hosts({"ri": [0, 7]}), "Q3")
    J, info = obj.loss(dec, sel, grad=True)
    g2 = [p.grad.clone() for p in dec.parameters()]
    assert info["map_reproduction_max_abs"] == 0.0 and info["categories"] == 2
    assert info["cross_host_pair_fraction"] > 0.5
    dec.zero_grad(set_to_none=True)
    maps = {c: [obj._maps(dec, [it], True)[0] for it in sel[c]] for c in sel}
    J1, _, _, _ = retention_loss(maps, {c: [it["coverage"] for it in sel[c]] for c in sel}, obj.tau, obj.n_neg)
    J1.backward()
    assert abs(float(J1.detach()) - J) < 1e-12
    for a, p in zip(g2, dec.parameters()):
        assert torch.allclose(a, p.grad, rtol=1e-10, atol=1e-14)


def test_retention_lookahead_hypergradient_matches_finite_difference(monkeypatch):
    dec = fp64(monkeypatch)
    bank, allowed = tiny_bank(torch.float64)
    obj = objective(bank, allowed, seed=2, check_tol=1e-12)
    fixed = obj.select(obj.hosts({"ri": [1, 2]}), "Q3")
    obj.loss_fixed = lambda d, grad: obj.loss(d, fixed, grad)
    outer = lambda d, ctx: {"loss": obj.loss_fixed(d, True)[0], "used": 1}
    ctrl = BlockController(weight_decay=0.0)
    with torch.no_grad():
        ctrl.phi.copy_(torch.linspace(-0.6, 0.9, len(BLOCKS), dtype=torch.float64))
    a = ctrl.a()
    named = list(dec.named_parameters())
    old = [p.detach().clone() for _, p in named]
    delta1 = [torch.randn_like(p) * 1e-2 for _, p in named]
    with torch.no_grad():
        for (n, p), o, d in zip(named, old, delta1):
            p.copy_(o + a[block_of(n)] * d)
    ev = slow_event(dec, named, old, a, outer, {"iter": 0, "ci": [], "ri": []}, ctrl, learn=False, device="cpu")
    g = torch.tensor(ev["hypergrad"])
    assert ev["finite"] and float(g.abs().max()) > 0

    def J(phi):
        with torch.no_grad():
            ctrl.phi.copy_(phi)
            av = dict(zip(BLOCKS, ctrl.multipliers().tolist()))
            for (n, p), o, d in zip(named, old, delta1):
                p.copy_(o + av[block_of(n)] * d)
        return obj.loss_fixed(dec, False)[0]

    base = ctrl.phi.detach().clone()
    for j in range(len(BLOCKS)):
        e = torch.zeros_like(base)
        e[j] = 1e-4
        fd = (J(base + e) - J(base - e)) / 2e-4
        assert abs(fd - float(g[j])) <= 1e-8 + 1e-4 * abs(fd), (j, fd, float(g[j]))


def test_identity_controller_with_retention_objective_leaves_fast_trajectory_unchanged():
    bank, allowed = tiny_bank()
    d0, (l0, s0, _) = _stage()
    d1, (l1, s1, ev) = _stage(ctrl=BlockController(), outer=objective(bank, allowed), learn=False)
    assert all(torch.equal(a, b) for a, b in zip(state(d0).values(), state(d1).values()))
    assert l0 == l1 and s0 == s1 == [] and len(ev) == 5
    assert all(e["finite"] and e["hypergrad"] is not None and e["train"]["categories"] >= 1 for e in ev)
    assert all(e["train"]["map_reproduction_max_abs"] == 0.0 for e in ev)
    assert all(p.grad is None for p in d1.parameters())


def test_learning_controller_moves_phi_with_retention_objective():
    bank, allowed = tiny_bank()
    ctrl = BlockController(lr=0.05, weight_decay=1e-5)
    _stage(ctrl=ctrl, outer=objective(bank, allowed, seed=1))
    assert ctrl.events == 5 and float(ctrl.phi.abs().max()) > 0
