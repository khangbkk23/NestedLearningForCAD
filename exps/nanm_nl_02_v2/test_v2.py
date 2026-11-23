"""Focused synthetic tests for NANM-NL-02 Stage 0 v2 (optimizer layer, ledger, v1 parity). CPU, no real data."""
import math

import pytest
import torch

from exps.nanm_nl_02 import core as C
from exps.nanm_nl_02 import run as R
from exps.nanm_nl_02_v2 import core_v2 as O
from exps.nanm_nl_02_v2 import run as V

D, F64 = C.D, torch.float64


def _rand(seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return [scale * torch.randn(D, C.RANK, generator=g, dtype=F64) for _ in range(4)]


def test_sgd_kind_bitwise_equals_v1_sgd_step():
    P, Q, gP, gQ = _rand(0)
    Q = 0.01 * Q
    for lr in (1e-4, 3e-3, 1e-2, 5.0):
        a = C.sgd_step(P, Q, gP, gQ, lr)
        b = O.FastOpt('sgd', lr).step(P, Q, gP, gQ)
        assert a[2] == b[2] and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_adam_matches_torch_adam():
    P, Q, _, _ = _rand(1, 0.03)
    tgt = _rand(2, 0.03)
    Pt, Qt = P.clone().requires_grad_(True), Q.clone().requires_grad_(True)
    ref = torch.optim.Adam([Pt, Qt], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    opt = O.FastOpt('adam', 1e-3); Pm, Qm = P.clone(), Q.clone()
    for _ in range(25):
        L = ((Pt - tgt[0]) ** 2).sum() + ((Qt - tgt[1]) ** 3).abs().sum()
        ref.zero_grad(); L.backward(); ref.step()
        Pg, Qg = Pm.clone().requires_grad_(True), Qm.clone().requires_grad_(True)
        L2 = ((Pg - tgt[0]) ** 2).sum() + ((Qg - tgt[1]) ** 3).abs().sum()
        gP, gQ = torch.autograd.grad(L2, (Pg, Qg))
        Pm, Qm, h = opt.step(Pm, Qm, gP, gQ)
        assert h == 0
    assert float((Pm - Pt.detach()).abs().max()) < 1e-12 and float((Qm - Qt.detach()).abs().max()) < 1e-12
    assert opt.t == 25 and opt.persistent_bytes() == 2 * C.PHI_B + 8 == 196616


def test_sign_step_and_zero_gradient():
    P, Q, gP, gQ = _rand(3)
    Q = 0.01 * Q
    P2, Q2, h = O.FastOpt('sign', 1e-4).step(P, Q, torch.zeros_like(gP), gQ)
    assert h == 0 and torch.equal(P2, P)                                         # sign(0) = 0: P frozen at Q = 0 init (as SGD)
    assert torch.allclose(Q2 - Q, -1e-4 * torch.sign(gQ), atol=0, rtol=1e-12)
    assert O.FastOpt('sign', 1.0).persistent_bytes() == 0


@pytest.mark.parametrize('kind', O.KINDS)
def test_det_backtracking_keeps_det_positive(kind):
    P, Q = C.init_phi(5)
    opt = O.FastOpt(kind, 50.0)
    g = torch.Generator().manual_seed(9)
    for _ in range(20):
        gP = torch.randn(D, C.RANK, generator=g, dtype=F64); gQ = torch.randn(D, C.RANK, generator=g, dtype=F64)
        P, Q, h = opt.step(P, Q, gP, gQ)
        s, la = C.logdet_A(P, Q)
        assert float(s) > 0 and math.isfinite(float(la))
    assert opt.stats['halvings'] + opt.stats['rejected'] > 0                    # the safety actually engaged


def test_nonfinite_gradient_rejected_before_moment_update():
    P, Q, gP, gQ = _rand(4, 0.03)
    opt = O.FastOpt('adam', 1e-3)
    P1, Q1, _ = opt.step(P, Q, gP, gQ)
    m = [x.clone() for x in opt.m]; t = opt.t
    bad = gP.clone(); bad[0, 0] = float('nan')
    P2, Q2, h = opt.step(P1, Q1, bad, gQ)
    assert h == -1 and torch.equal(P2, P1) and torch.equal(Q2, Q1) and opt.t == t and all(torch.equal(a, b) for a, b in zip(m, opt.m))
    assert opt.stats['nonfinite_grad'] == 1 and opt.stats['rejected'] == 1


def test_clone_is_deep_and_resets_stats():
    P, Q, gP, gQ = _rand(6, 0.03)
    opt = O.FastOpt('adam', 1e-3); opt.step(P, Q, gP, gQ)
    c = opt.clone()
    assert c.t == opt.t and all(torch.equal(a, b) for a, b in zip(c.m + c.v, opt.m + opt.v)) and c.stats['steps'] == 0
    c.step(P, Q, gP, gQ)
    assert opt.t == 1 and c.t == 2 and not torch.equal(c.m[1], opt.m[1])


@pytest.mark.parametrize('kind', ('adam', 'sign'))
def test_c4_update_projection_is_orthogonal(kind):
    P, Q, gP, gQ = _rand(7, 0.03)
    B, _ = torch.linalg.qr(torch.randn(D, 16, generator=torch.Generator().manual_seed(1), dtype=F64))
    proj = lambda G: G - B @ (B.T @ G)
    opt = O.FastOpt(kind, 1e-3)
    P2, Q2, h = opt.step(P, Q, proj(gP), proj(gQ), proj)
    assert h >= 0
    assert float((B.T @ (P2 - P)).abs().max()) < 1e-14 and float((B.T @ (Q2 - Q)).abs().max()) < 1e-14


def test_ledger_v2():
    for kind in O.KINDS:
        for n in range(1, 5):
            for arm in ('C1', 'C2', 'C3', 'C4', 'C5'):
                assert O.reserve_alloc_v2(arm, n, kind) == R.reserve_alloc(arm, n)
                fb = 32 * n if arm in ('C2', 'C3', 'C4') else 0
                l1 = C.ledger(arm, n, [126] * n, fb, R.reserve_alloc(arm, n))
                l2 = O.ledger_v2(arm, kind, n, [126] * n, fb, O.reserve_alloc_v2(arm, n, kind))
                assert l2['total_charged'] == l1['total_charged'] + O.OPT_STATE_B[kind]
                assert l2['total_charged'] <= O.budget_v2(n, kind, 126)
    assert O.extra_v2('C3', 'adam') - O.extra_v2('C3', 'sgd') == 196616 and O.extra_v2('C0', 'adam') == 0


def test_v2_outcome_mapping():
    ok, no = {'pass': True, 'outcome': 'PASS'}, {'pass': False, 'outcome': 'X'}
    assert V.v2_outcome(False, ok, ok)[0] == 'NL02_V2_PROTOCOL_BLOCKED'
    assert V.v2_outcome(True, no, ok)[0] == 'NL02_V2_ADAPTATION_NOT_SUPPORTED'
    assert V.v2_outcome(True, ok, no)[0] == 'NL02_V2_CONVENTIONAL_SUFFICIENT'
    assert V.v2_outcome(True, ok, ok)[0] == 'NL02_V2_DESIGN_READY'


# ---------------------------------------------------------------- synthetic stream: v1 parity and zero-step identity
def _synthetic_stream(seed=2711, tasks=2):
    data = []
    for k in range(tasks):
        g = torch.Generator().manual_seed(100 + k)
        scale = torch.linspace(3.0, 0.2, D, dtype=F64)
        def tok(n_img, per, base_id):
            base = torch.randn(n_img, D, generator=g, dtype=F64) * scale
            x = base.repeat_interleave(per, 0) + 0.3 * torch.randn(n_img * per, D, generator=g, dtype=F64) * scale
            return torch.nn.functional.normalize(x, dim=1).half(), (base_id + torch.arange(n_img)).repeat_interleave(per)
        td = {}
        td['FIT'], td['FIT_img'] = tok(16, 784, k * 1000)
        td['FEEDBACK'], td['FEEDBACK_img'] = tok(4, 8, k * 1000 + 16)
        td['RESERVE'], td['RESERVE_img'] = tok(4, 8, k * 1000 + 20)
        td['EVAL'], td['EVAL_img'] = tok(8, 16, k * 1000 + 24)
        data.append(td)
    return R.Stream(data, seed, 'a')


@pytest.fixture(scope='module')
def stream():
    return _synthetic_stream()


def test_zero_step_identity_all_kinds(stream):
    for kind in O.KINDS:
        P, Q = V.train_stage(stream, 0, 'C1', C.init_phi(2711), R.Aux(), {}, O.FastOpt(kind, 1e-3), steps=0)
        assert torch.equal(C.A_of(P, Q), torch.eye(D, dtype=F64))


def test_O0_trajectories_bitwise_equal_v1(stream, monkeypatch):
    monkeypatch.setattr(R, 'STEPS', 6); monkeypatch.setattr(V, 'STEPS', 6)
    hp1 = {'eta': 3e-3, 'ewc_c': 1.0, 'ema': 0.9}; hp2 = {'opt': 'sgd', 'lr': 3e-3, 'ewc_c': 1.0, 'ema': 0.9}
    for arm in ('C1', 'C2', 'C3', 'C4', 'C5'):
        e1, p1 = R.run_traj(stream, arm, hp1); e2, p2, _, _ = V.run_traj(stream, arm, hp2)
        assert e1 == e2, arm
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(p1, p2)), arm


def test_adam_trajectory_deterministic_and_state_persists(stream, monkeypatch):
    monkeypatch.setattr(V, 'STEPS', 5)
    hp = {'opt': 'adam', 'lr': 1e-3}
    e1, p1, s1, _ = V.run_traj(stream, 'C1', hp); e2, p2, s2, _ = V.run_traj(stream, 'C1', hp)
    assert e1 == e2 and all(torch.equal(a[0], b[0]) for a, b in zip(p1, p2))
    assert [s.t for s in s1] == [5, 10]                                           # moments carried across the boundary
    assert all(math.isfinite(x) for r in e1 for x in r)
