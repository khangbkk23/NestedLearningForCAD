"""NAR-PILOT-00 mechanics tests for core.py (synthetic, CPU; no dataset, feature or label file is read).

Passing these establishes mechanical correctness only.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from exps.dinomaly import official as O
from exps.nar_pilot_00 import core as K

NAMES = ("l0", "l1", "l2")


def net(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential()
    m.add_module("l0", nn.Linear(12, 16))
    m.add_module("a0", nn.Tanh())
    m.add_module("l1", nn.Linear(16, 16))
    m.add_module("a1", nn.Tanh())
    m.add_module("l2", nn.Linear(16, 6))
    return m.double()


def make_opt(cfg):
    return lambda groups: O.StableAdamW(groups, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay,
                                        amsgrad=True, eps=cfg.eps)


def task(seed, n=96):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 12, generator=g, dtype=torch.float64)
    w = torch.randn(12, 6, generator=g, dtype=torch.float64)
    return x, torch.tanh(x @ w)


def run(cfg, lam, learn, seed=0, perm=None, data_seed=1, init=None, hook=None):
    m = net(0)
    lo = K.attach(m, NAMES, cfg.rank)
    for w in lo.values():
        w.double()
    K.set_state(lo, init if init is not None else K.init_state(lo, 7))
    x, y = task(data_seed)
    xf, yf, xc, yc = x[:64], y[:64], x[64:], y[64:]
    g = torch.Generator().manual_seed(seed)

    def fit(i):
        idx = torch.randperm(64, generator=g)[:16]
        return ((m(xf[idx]) - yf[idx]) ** 2).mean()

    def cred():
        return ((m(xc) - yc) ** 2).mean()

    lam2, log = K.acquire(lo, fit, cred, cfg, lam, learn, make_opt(cfg), perm=perm, on_step=hook)
    return m, lo, lam2, log, float(cred())


CFG = K.Cfg(rank=2, steps=60, lr=1e-2, warmup=5, m=5, clip=10.0)


def test_lora_zero_increment_and_frozen_base():
    m0 = net(0)
    m = copy.deepcopy(m0)
    lo = K.attach(m, NAMES, 2)
    for w in lo.values():
        w.double()
    K.set_state(lo, K.init_state(lo, 7))
    x = torch.randn(5, 12, dtype=torch.float64)
    assert torch.equal(m(x), m0(x))                                    # B = 0: the increment is exactly zero
    train = [n for n, q in m.named_parameters() if q.requires_grad]
    assert train and all(n.endswith((".A", ".B")) for n in train)


def test_fixed_rule_equivalence_when_slow_learning_disabled():
    """eta_lam = 0 (learn=False) and lam = 0: bitwise the plain StableAdamW run with the same schedule."""
    m1, lo1, lam1, _, _ = run(CFG, torch.zeros(3), learn=False)
    # independent re-implementation of the fixed rule
    m = net(0)
    lo = K.attach(m, NAMES, 2)
    for w in lo.values():
        w.double()
    K.set_state(lo, K.init_state(lo, 7))
    x, y = task(1)
    g = torch.Generator().manual_seed(0)
    opt = make_opt(CFG)([{"params": [w.A, w.B]} for w in lo.values()])
    ps = [q for w in lo.values() for q in (w.A, w.B)]
    for i in range(CFG.steps):
        for grp in opt.param_groups:
            grp["lr"] = K.lr_at(CFG, i)
        opt.zero_grad()
        idx = torch.randperm(64, generator=g)[:16]
        ((m(x[:64][idx]) - y[:64][idx]) ** 2).mean().backward()
        torch.nn.utils.clip_grad_norm_(ps, CFG.clip)
        opt.step()
    assert torch.equal(lam1, torch.zeros(3, dtype=torch.float64))
    for n in NAMES:
        assert torch.equal(lo1[n].A, lo[n].A) and torch.equal(lo1[n].B, lo[n].B)


def test_credit_sign_matches_finite_difference():
    """Level-1 credit: the sign of cos(grad L_cred, window displacement) equals the sign of the finite-difference
    derivative of the held-out loss w.r.t. the group log step size over one window (full-batch, plain SGD so the
    first-order formula is exact up to curvature)."""
    cfg = K.Cfg(rank=2, steps=5, lr=1e-2, warmup=1, final_ratio=1.0, m=5, clip=1e9, weight_decay=0.0)
    sgd = lambda groups: torch.optim.SGD(groups, lr=cfg.lr)

    def once(lam):
        m = net(0)
        lo = K.attach(m, NAMES, 2)
        for w in lo.values():
            w.double()
        st = K.init_state(lo, 7)
        st = {n: (a, torch.full_like(b, 0.05)) for n, (a, b) in st.items()}      # nonzero B: all groups move
        K.set_state(lo, st)
        x, y = task(1)
        fit = lambda i: ((m(x[:64]) - y[:64]) ** 2).mean()
        cred = lambda: ((m(x[64:]) - y[64:]) ** 2).mean()
        lam2, log = K.acquire(lo, fit, cred, cfg, lam, True, sgd)
        return float(cred()), lam2

    base, lam2 = once(torch.zeros(3))
    eps = 1e-3
    for g in range(3):
        d = torch.zeros(3)
        d[g] = eps
        fd = (once(d)[0] - once(-d)[0]) / (2 * eps)                  # dL_cred / d lam_g
        step = float(lam2[g])                                         # = -eta_lam * h_g, h_g ~ sign(dL/dlam_g)
        assert fd != 0 and math.copysign(1, -step) == math.copysign(1, fd), (g, fd, step)


def test_slow_level_learns_on_misscaled_groups_and_permuted_credit_differs():
    """Positive control: starting from a far too small step size, the slow level raises lam and reaches a lower
    held-out loss than the fixed rule; the dynamics are nonzero and nondegenerate; permuted credit gives a
    different lam."""
    cfg = K.Cfg(rank=2, steps=60, lr=2e-4, warmup=5, m=5, clip=10.0, eta_lam=0.12)
    *_, lam_f, _, loss_f = run(cfg, torch.zeros(3), learn=False)
    *_, lam_n, log, loss_n = run(cfg, torch.zeros(3), learn=True)
    *_, lam_x, _, _ = run(cfg, torch.zeros(3), learn=True, perm=[1, 2, 0])
    assert float(lam_n.mean()) > 0.5 and loss_n < loss_f
    assert float(lam_n.abs().max()) <= cfg.lam_max + 1e-12
    assert not torch.equal(lam_n, lam_x)
    assert all(math.isfinite(v) for v in log["fit"]) and len(log["h_mean"]) == cfg.steps // cfg.m


def test_lam_bounded_and_stable_under_extreme_rates():
    cfg = K.Cfg(rank=2, steps=40, lr=5e-2, warmup=1, m=2, clip=10.0, eta_lam=5.0)
    *_, lam, log, loss = run(cfg, torch.zeros(3), learn=True)
    assert float(lam.abs().max()) <= cfg.lam_max + 1e-12 and math.isfinite(loss)


def test_save_reload_equivalence(tmp_path):
    """Stopping after a task, saving (fast state, lam, phi) and reloading continues bitwise."""
    m, lo, lam, _, _ = run(CFG, torch.zeros(3), learn=True)
    a = K.get_state(lo)
    phi = K.reptile(K.init_state(lo, 7), a, CFG.beta)
    torch.save({"a": a, "lam": lam, "phi": phi}, tmp_path / "s.pt")
    z = torch.load(tmp_path / "s.pt", weights_only=False)
    r1 = run(CFG, lam, True, seed=3, data_seed=2, init=phi)
    r2 = run(CFG, z["lam"], True, seed=3, data_seed=2, init=z["phi"])
    assert torch.equal(r1[2], r2[2])
    for n in NAMES:
        assert torch.equal(r1[1][n].A, r2[1][n].A) and torch.equal(r1[1][n].B, r2[1][n].B)


def test_reptile_recurrence_and_bounded_bytes():
    m = net(0)
    lo = K.attach(m, NAMES, 2)
    phi = K.init_state(lo, 7)
    a = {n: (x + 1.0, y + 2.0) for n, (x, y) in phi.items()}
    p0 = K.reptile(phi, a, 0.0)
    p1 = K.reptile(phi, a, 1.0)
    ph = K.reptile(phi, a, 0.5)
    for n in NAMES:
        assert torch.equal(p0[n][0], phi[n][0]) and torch.equal(p1[n][1], a[n][1])
        assert torch.allclose(ph[n][0], phi[n][0] + 0.5)
    # bytes: fast state per task and slow state are constants of (rank, shapes), independent of the task count
    b = K.state_bytes(a)
    assert b == 4 * 2 * ((12 + 16) + (16 + 16) + (16 + 6))
    assert K.slow_bytes(torch.zeros(3), ph) == 12 + b
    for _ in range(5):
        ph = K.reptile(ph, a, 0.5)
    assert K.slow_bytes(torch.zeros(3), ph) == 12 + b


def test_structural_retention_is_bitwise():
    """An old task's function (base + its frozen fast state) is unchanged by any later acquisition."""
    m, lo, lam, _, _ = run(CFG, torch.zeros(3), learn=True)
    a1 = K.get_state(lo)
    x = torch.randn(9, 12, dtype=torch.float64)
    y1 = m(x).clone()
    phi = K.reptile(K.init_state(lo, 7), a1, 0.5)
    K.set_state(lo, phi)
    x2, t2 = task(5)
    K.acquire(lo, lambda i: ((m(x2) - t2) ** 2).mean(), lambda: ((m(x2) - t2) ** 2).mean(), CFG, lam, True,
              make_opt(CFG))
    assert not torch.equal(m(x), y1)
    K.set_state(lo, a1)
    assert torch.equal(m(x), y1)


def test_seed_reproducibility():
    r1 = run(CFG, torch.zeros(3), True, seed=11)
    r2 = run(CFG, torch.zeros(3), True, seed=11)
    r3 = run(CFG, torch.zeros(3), True, seed=12)
    assert torch.equal(r1[2], r2[2]) and torch.equal(r1[1]["l1"].B, r2[1]["l1"].B)
    assert not torch.equal(r1[1]["l1"].B, r3[1]["l1"].B)


def test_credit_observation_matches_fixed_fast_trajectory():
    cfg = K.Cfg(rank=2, steps=10, warmup=2, m=2, clip=10.)
    def once(observe):
        m = net(0)
        lo = K.attach(m, NAMES, 2)
        for w in lo.values():
            w.double()
        K.set_state(lo, K.init_state(lo, 7))
        x, y = task(1)
        lam, log = K.acquire(lo, lambda i: ((m(x[:64])-y[:64])**2).mean(),
                             lambda: ((m(x[64:])-y[64:])**2).mean(), cfg,
                             torch.zeros(3), False, make_opt(cfg), observe_credit=observe)
        return K.get_state(lo), lam, log
    a, la, _ = once(False)
    b, lb, log = once(True)
    assert torch.equal(la, lb) and len(log['credit']) == 5
    assert all(torch.equal(a[n][i], b[n][i]) for n in NAMES for i in (0, 1))


def test_old_expert_score_isolation_and_reload(tmp_path):
    m, lo, lam, _, _ = run(CFG, torch.zeros(3), True)
    m.eval()
    old = K.get_state(lo)
    shared = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    x = torch.randn(4, 12, dtype=torch.float64)
    with torch.no_grad():
        old_score = m(x).square()
    phi = K.reptile(K.init_state(lo, 7), old, CFG.beta)
    m2, lo2, _, _, _ = run(CFG, lam, True, data_seed=8, init=phi)
    new = K.get_state(lo2)
    assert all(torch.equal(old[n][i], K.get_state(lo)[n][i]) for n in NAMES for i in (0,1))
    assert any(not torch.equal(old[n][1], new[n][1]) for n in NAMES)
    for n,p in m2.named_parameters():
        if n in shared:
            assert torch.equal(p, shared[n])
    torch.save({'model': m.state_dict(), 'old': old, 'phi': phi, 'lam': lam}, tmp_path/'expert.pt')
    restored = torch.load(tmp_path/'expert.pt', weights_only=True)
    m3 = net(0)
    lo3 = K.attach(m3, NAMES, 2)
    for w in lo3.values():
        w.double()
    m3.load_state_dict(restored['model'])
    m3.eval()
    K.set_state(lo3, new)
    K.set_state(lo3, restored['old'])
    with torch.no_grad():
        assert torch.equal(m3(x).square(), old_score)


def test_prefix_initialization_and_rate_credit_change_next_task():
    _, lo, lam, _, _ = run(CFG, torch.zeros(3), True, data_seed=1)
    fixed = K.init_state(lo, 7)
    endpoint = K.get_state(lo)
    phi = K.reptile(fixed, endpoint, CFG.beta)
    fixed_next = run(CFG, torch.zeros(3), False, seed=21, data_seed=2, init=fixed)
    warm_next = run(CFG, torch.zeros(3), False, seed=21, data_seed=2, init=endpoint)
    init_next = run(CFG, torch.zeros(3), False, seed=21, data_seed=2, init=phi)
    rate_next = run(CFG, lam, False, seed=21, data_seed=2, init=phi)
    assert not torch.equal(fixed_next[1]['l1'].B, init_next[1]['l1'].B)
    assert not torch.equal(warm_next[1]['l1'].B, init_next[1]['l1'].B)
    assert not torch.equal(init_next[1]['l1'].B, rate_next[1]['l1'].B)
    assert all(torch.equal(endpoint[n][i], K.get_state(lo)[n][i]) for n in NAMES for i in (0,1))


def test_membership_roles_are_image_disjoint_normal_current_only():
    import pytest
    rows = [{'relative_path': f'bottle/train/good/{i:03}.png'} for i in range(20)]
    p = K.normal_partition(rows)
    assert len(p['FIT']) == 12 and len(p['CREDIT']) == len(p['EVAL']) == 4
    assert set(p['FIT']).isdisjoint(p['CREDIT']) and set(p['EVAL']).isdisjoint(p['CREDIT'])
    assert sorted(sum(p.values(), [])) == list(range(20))
    for bad in ([{'relative_path':'bottle/test/good/000.png'}]*20,
                rows[:19] + [{'relative_path':'carpet/train/good/999.png'}],
                rows[:19] + [{'relative_path':'bottle/../train/good/x.png'}]):
        with pytest.raises(RuntimeError):
            K.normal_partition(bad)
    assert K.normal_partition(rows) == p


def test_actual_ledger_grows_with_tasks_and_counts_fp64_rates():
    base = net(0)
    m = net(0)
    lo = K.attach(m, NAMES, 2)
    for w in lo.values():
        w.double()
    state = K.get_state(lo)
    lam = torch.zeros(3, dtype=torch.float64)
    assert K.slow_bytes(lam, state) == 24 + K.tensor_state_bytes(state)
    l1 = K.ledger(base, [state], lam, state)
    l2 = K.ledger(base, [state, state], lam, state)
    assert l2['total'] - l1['total'] == K.tensor_state_bytes(state)
    assert l1['shared_base'] == sum(x.numel()*x.element_size() for x in base.state_dict().values())


def test_task_agnostic_aggregation_changes_detector_despite_fixed_old_expert():
    old = torch.ones(2, 3, 3)
    new = torch.zeros_like(old)
    old_copy = old.clone()
    a, ai = K.task_agnostic_grid([old])
    b, bi = K.task_agnostic_grid([old, new])
    assert torch.equal(old, old_copy) and torch.equal(ai, torch.ones(2))
    assert torch.equal(bi, torch.zeros(2)) and not torch.equal(a, b)
    reverse, ri = K.task_agnostic_grid([new, old])
    assert torch.equal(b, reverse) and torch.equal(bi, ri)


def test_one_step_stableadam_credit_sign_without_multi_step_claim():
    cfg = K.Cfg(rank=2,steps=1,warmup=1,m=1,clip=10.,weight_decay=0.)
    m = net(0)
    lo = K.attach(m,NAMES,2)
    initial = {n:(a,torch.full_like(b,.05)) for n,(a,b) in K.init_state(lo,7).items()}
    def once(lam):
        return run(cfg,lam,True,init=initial)
    nominal = once(torch.zeros(3))
    for group in range(3):
        perturb = torch.zeros(3,dtype=torch.float64)
        perturb[group] = 1e-4
        fd = (once(perturb)[4]-once(-perturb)[4])/(2e-4)
        h = -float(nominal[2][group])/cfg.eta_lam
        assert abs(fd)>1e-12 and math.copysign(1,fd)==math.copysign(1,h)


def test_pilot_runner_refuses_unauthorized_execution():
    import pytest
    from exps.nar_pilot_00 import run as R
    with pytest.raises(RuntimeError,match='owner'):
        R.prepare(False)
    with pytest.raises(RuntimeError,match='owner'):
        R.execute(False)
    with pytest.raises(RuntimeError,match='owner'):
        R.gpu_preflight(False)


def test_pilot_cache_guard_has_no_label_or_future_dependency():
    import pytest
    from exps.nar_pilot_00 import run as R
    c = {'rows':[{'relative_path':f'bottle/train/good/{j:03}.png'} for j in range(10)],
         'features':torch.zeros(10,784,768)}
    assert len(R.assert_cache(c,'bottle')['FIT'])==6
    with pytest.raises(RuntimeError):
        R.assert_cache(c,'carpet')
    c['rows'][0]['relative_path']='bottle/test/good/000.png'
    with pytest.raises(RuntimeError):
        R.assert_cache(c,'bottle')
