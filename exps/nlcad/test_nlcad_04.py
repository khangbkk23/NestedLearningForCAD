# exps/nlcad/test_nlcad_04.py
"""NLCAD-04 Stage A CPU tests for the Variant S slow objective (tiny decoder and synthetic tensors, plus the real
stream/bank host rules; no image, no real mask, no checkpoint)."""
from __future__ import annotations

import io

import torch

from exps.dinomaly import official as O
from exps.nlcad.controller import BLOCKS, BlockController, block_of
from exps.nlcad.fast import nl_train_stage, slow_event
from exps.nlcad.objectives import ranking_loss
from exps.nlcad.objectives_s import SyntheticRankingS, ranking_terms
from exps.nlcad.test_nlcad import feats, state, tiny

CUR = "leather"
OLD = ("bottle", "carpet")


def tiny_bank(dtype=torch.float32):
    """Q2/Q3 items on a 4x4 patch grid: 6 current hosts and 6 buffer hosts per old category, 2 variants each."""
    g = torch.Generator().manual_seed(11)
    hosts = [(CUR, i) for i in range(6)] + [(c, i) for c in OLD for i in range(6)]
    bank = {}
    for name in ("Q2", "Q3"):
        items = []
        for h in hosts:
            for v in range(2):
                cov = torch.zeros(4, 4)
                r, c = torch.randint(0, 3, (2,), generator=g).tolist()
                cov[r:r + 2, c:c + 2] = 1.0
                cov[3, 3] = 0.3                                    # a partial patch: neither positive nor negative
                items.append({"host": h, "variant": v, "coverage": cov,
                              "features": torch.randn(2, 21, 32, generator=g).to(dtype)})
        bank[name] = items
    return bank, set(hosts)


def objective(bank, allowed, replay_keys, seed=0, **kw):
    return SyntheticRankingS(bank, "Q3", "Q2", allowed, CUR, replay_keys, seed=seed, n_current=4, n_replay=4,
                             micro=4, device="cpu", autocast=False, **kw)


REPLAY_KEYS = [(c, i) for c in OLD for i in range(6)]          # row j of the fast step's replay tensor


def test_hosts_disjoint_from_inner_batch_stratified_and_allowed():
    bank, allowed = tiny_bank()
    allowed.discard(("carpet", 3))                                 # e.g. a member dropped from the buffer
    obj = objective(bank, allowed, REPLAY_KEYS, seed=3)
    for t in range(30):
        ctx = {"ci": [t % 6, (t + 1) % 6], "ri": [t % 6]}
        inner = {(CUR, t % 6), (CUR, (t + 1) % 6), REPLAY_KEYS[t % 6]}
        groups = obj.sample("Q3", ctx)
        assert len(groups["current"]) == 4 and len(groups["replay"]) == 4
        for src, items in groups.items():
            for it in items:
                assert it["host"] in allowed and it["host"] not in inner
                assert (it["host"][0] == CUR) == (src == "current")
    assert obj.inner_hosts({"ci": None, "ri": None}) == set()


def test_ranking_terms_and_loss_skip_rules():
    s = torch.zeros(4, 4)
    cov = torch.zeros(4, 4)
    assert ranking_terms(s, cov, 0.05) is None                     # no positive patch -> skipped, never silent zero
    cov[0, 0], cov[0, 1] = 1.0, 0.4
    t = ranking_terms(s, cov, 0.05)
    assert t["n_pos"] == 1 and t["n_neg"] == 14                    # the partial patch is excluded from both sides
    assert abs(float(t["loss"]) - float(torch.nn.functional.softplus(torch.tensor(0.)))) < 1e-7
    assert ranking_loss(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4)) is None


def test_s_lookahead_hypergradient_matches_finite_difference(monkeypatch):
    """J(phi) = L_S(theta_old + a(phi) * Delta1) on a fixed synthetic sample (all negatives). The objective's forward
    casts decoder outputs to FP32 by design (as in training); here it is kept in FP64 so the check is exact."""
    import exps.nlcad.objectives_s as OS
    monkeypatch.setattr(OS, "_forward", lambda dec, xb, device, autocast: dec(xb[:, 0], xb[:, 1]))
    dec = tiny(dtype=torch.float64)
    for m in dec.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    bank, allowed = tiny_bank(torch.float64)
    obj = objective(bank, allowed, REPLAY_KEYS, seed=5)
    fixed = obj.sample("Q3", {"ci": [0], "ri": [0]})
    obj.sample = lambda name, ctx: fixed
    ctrl = BlockController(weight_decay=0.0)
    with torch.no_grad():
        ctrl.phi.copy_(torch.linspace(-0.8, 0.8, len(BLOCKS), dtype=torch.float64))
    a = ctrl.a()
    named = list(dec.named_parameters())
    old = [p.detach().clone() for _, p in named]
    delta1 = [torch.randn_like(p) * 1e-2 for _, p in named]
    with torch.no_grad():
        for (n, p), o, d in zip(named, old, delta1):
            p.copy_(o + a[block_of(n)] * d)
    ev = slow_event(dec, named, old, a, obj, {"iter": 0, "ci": [], "ri": []}, ctrl, learn=False, device="cpu")
    g = torch.tensor(ev["hypergrad"])
    assert ev["finite"] and ev["used"] == 8 and float(g.abs().max()) > 0

    def J(phi):
        with torch.no_grad():
            ctrl.phi.copy_(phi)
            av = dict(zip(BLOCKS, ctrl.multipliers().tolist()))
            for (n, p), o, d in zip(named, old, delta1):
                p.copy_(o + av[block_of(n)] * d)
            return obj._loss(dec, fixed, grad=False)[0]

    base = ctrl.phi.detach().clone()
    for j in range(len(BLOCKS)):
        e = torch.zeros_like(base)
        e[j] = 1e-4
        fd = (J(base + e) - J(base - e)) / 2e-4
        assert abs(fd - float(g[j])) <= 1e-7 + 1e-4 * abs(fd), (j, fd, float(g[j]))


def _stage(ctrl=None, outer=None, learn=True, iters=40, dec=None, seed=3):
    dec = dec if dec is not None else tiny()
    torch.manual_seed(7)
    out = nl_train_stage(dec, feats(32, 1), iters, torch.Generator().manual_seed(seed), feats(12, 2), ctrl=ctrl,
                         outer=outer, K=8, learn=learn, device="cpu", autocast=False)
    return dec, out


def test_identity_controller_with_s_objective_leaves_fast_trajectory_bitwise_unchanged():
    bank, allowed = tiny_bank()
    d0, (l0, s0, _) = _stage()
    d1, (l1, s1, ev) = _stage(ctrl=BlockController(), outer=objective(bank, allowed, REPLAY_KEYS), learn=False)
    assert all(torch.equal(a, b) for a, b in zip(state(d0).values(), state(d1).values()))
    assert l0 == l1 and s0 == s1 == [] and len(ev) == 5
    assert all(e["finite"] and e["used"] == 8 and e["hypergrad"] is not None for e in ev)
    assert all(len(e["hosts"]["replay"]) == 4 for e in ev)        # 12 buffer rows, 8 inner: 4 hosts x 2 variants left
    assert all(p.grad is None for p in d1.parameters())            # the slow event leaves no gradient behind


def test_learning_s_controller_moves_phi_and_resumes_exactly_across_a_stage_boundary():
    bank, allowed = tiny_bank()
    ctrl_a = BlockController(lr=0.05, weight_decay=1e-5)
    da, _ = _stage(ctrl=ctrl_a, outer=objective(bank, allowed, REPLAY_KEYS, seed=1))
    assert ctrl_a.events == 5 and float(ctrl_a.phi.abs().max()) > 0
    buf = io.BytesIO()                                             # stage-end snapshot, as run_04 writes it
    torch.save({"state": state(da), "controller": ctrl_a.snapshot()}, buf)
    _stage(ctrl=ctrl_a, outer=objective(bank, allowed, REPLAY_KEYS, seed=2), dec=da, seed=4)
    buf.seek(0)
    snap = torch.load(buf, weights_only=False)
    db = tiny()
    db.load_state_dict(snap["state"])
    ctrl_b = BlockController(lr=0.05, weight_decay=1e-5)
    ctrl_b.restore(snap["controller"])
    _stage(ctrl=ctrl_b, outer=objective(bank, allowed, REPLAY_KEYS, seed=2), dec=db, seed=4)
    assert torch.equal(ctrl_a.phi, ctrl_b.phi) and ctrl_a.events == ctrl_b.events == 10
    assert all(torch.equal(a, b) for a, b in zip(state(da).values(), state(db).values()))


def test_real_stream_hosts_allowed_bank_coverage_and_no_future_category():
    """Real seed-2711 host plan: every allowed host exists in the bank manifest, no future category, buffer hosts are
    exactly the stage's Replay-24 members (the replay tensor the fast step uses), current hosts are in range."""
    from exps.dinomaly import common as D
    from exps.nlcad import bank as BK
    from exps.rh01 import run as R1
    from exps.rh01b import common as C
    from exps.rh01b import run as RB
    n = R1.n_train(RB.verify_lock())
    man = D.read(D.ROOT / "results/hope_cad/nlcad/BANK_seed2711.json")
    hosts = {c: {k: list(v) for k, v in h.items()} for c, h in man["hosts"].items()}
    for c in C.STREAM:
        assert all(0 <= i < n[c] for i in hosts[c]["current"] + hosts[c]["buffer"])
        assert man["files"][f"Q2_{c}"]["n"] == man["files"][f"Q3_{c}"]["n"] == \
            2 * len(set(hosts[c]["current"]) | set(hosts[c]["buffer"]))
    for s in range(1, len(C.STREAM) + 1):
        keys = BK.allowed(hosts, 2711, n, s)
        cur, done = C.STREAM[s - 1], list(C.STREAM[:s - 1])
        assert {k[0] for k in keys} <= set(C.STREAM[:s])
        assert {k for k in keys if k[0] == cur} == {(cur, i) for i in hosts[cur]["current"]}
        idx = C.buffer_index(2711, n, done) if done else {}
        assert {k for k in keys if k[0] != cur} == {(d, i) for d in done for i in idx[d]}
        if done:
            assert sum(len(v) for v in idx.values()) == 24
        for k in keys:
            assert k[1] in set(hosts[k[0]]["current"]) | set(hosts[k[0]]["buffer"])
