"""Focused synthetic mechanical tests for NANM-NL-02 Stage 0 (CPU, no real data)."""
import math

import torch

from exps.nanm_nl_02 import core as C
from nanm import g3

D = C.D


def _tokens(n_img, per_img, seed, scale=None):
    """Anisotropic synthetic unit tokens with image ids (fp16, G3 convention)."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.linspace(3.0, 0.2, D, dtype=torch.float64) if scale is None else scale
    base = torch.randn(n_img, D, generator=g, dtype=torch.float64) * scale
    x = base.repeat_interleave(per_img, 0) + 0.3 * torch.randn(n_img * per_img, D, generator=g, dtype=torch.float64) * scale
    v = torch.nn.functional.normalize(x, dim=1).half()
    return v, torch.arange(n_img).repeat_interleave(per_img)


def _setup(seed=0):
    v, img = _tokens(12, 40, seed)
    st, key_img = C.g3_state_fit_only(v, img, seed=2711, k=48)
    W, _ = C.stage_W([st])
    Vk = st.keys.double(); Vq = v.double()
    return Vq, img, Vk, key_img, W, st


# ---------------------------------------------------------------- Correction 1
def test_zero_zero_init_is_degenerate():
    Vq, img, Vk, kimg, W, _ = _setup()
    P = torch.zeros(D, C.RANK, dtype=torch.float64); Q = torch.zeros_like(P)
    _, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
    assert float(gP.abs().max()) == 0.0 and float(gQ.abs().max()) == 0.0     # the reported saddle


def test_init_identity_exact():
    P, Q = C.init_phi(7)
    assert torch.equal(C.A_of(P, Q), torch.eye(D, dtype=torch.float64))
    s, la = C.logdet_A(P, Q)
    assert float(s) == 1.0 and float(la) == 0.0
    assert float(P.abs().max()) > 0                                             # non-degenerate factor


def test_c0_equals_zero_step_c1_scores():
    Vq, img, Vk, kimg, W, st = _setup()
    P, Q = C.init_phi(7)
    W32 = W.float()
    Weff = (W @ C.A_of(P, Q).T).float()
    q16 = Vq[:200].half()
    s0 = g3.score(q16, st.keys.float(), W32)                                    # canonical frozen-G3 scorer
    s1 = g3.score(q16, st.keys.float(), Weff)                                   # C1 after zero steps
    assert torch.equal(s0, s1)


def test_first_step_gradient_nonzero_and_A_moves():
    Vq, img, Vk, kimg, W, _ = _setup()
    P, Q = C.init_phi(7)
    L0, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
    assert float(gP.abs().max()) == 0.0                                         # dL/dP = G Q = 0 at Q = 0 (expected)
    assert float(gQ.norm()) > 1e-8                                              # dL/dQ = G^T P != 0
    P1, Q1, h = C.sgd_step(P, Q, gP, gQ, lr=1e-3)
    assert h >= 0 and not torch.equal(C.A_of(P1, Q1), torch.eye(D, dtype=torch.float64))
    _, gP1, gQ1, _ = C.grads(P1, Q1, Vq, Vk, img, kimg, W)
    assert float(gP1.norm()) > 1e-10                                            # P becomes trainable after step 1
    L1 = C.fast_loss(P1, Q1, Vq, Vk, img, kimg, W)[0]
    assert float(L1) < L0                                                       # descent on the training batch


def test_finite_difference_gradient():
    Vq, img, Vk, kimg, W, _ = _setup()
    P, Q = C.init_phi(7)
    _, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
    P, Q, _ = C.sgd_step(P, Q, gP, gQ, lr=1e-3)                                # move off the special point
    _, gP, gQ, j = C.grads(P, Q, Vq, Vk, img, kimg, W)
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        dP = torch.randn(P.shape, generator=g, dtype=torch.float64); dQ = torch.randn(Q.shape, generator=g, dtype=torch.float64)
        eps = 1e-6
        f = lambda s: float(C.fast_loss(P + s * dP, Q + s * dQ, Vq, Vk, img, kimg, W, j=j)[0])
        fd = (f(eps) - f(-eps)) / (2 * eps)
        an = float((gP * dP).sum() + (gQ * dQ).sum())
        assert abs(fd - an) <= 1e-6 * max(1.0, abs(an)), (fd, an)
        # full objective (assignment recomputed) agrees when the assignment is locally constant
        jp = C.assign(Vq @ W, Vk @ W, img, kimg, P + eps * dP, Q + eps * dQ)
        jm = C.assign(Vq @ W, Vk @ W, img, kimg, P - eps * dP, Q - eps * dQ)
        if torch.equal(jp, j) and torch.equal(jm, j):
            ff = lambda s: float(C.fast_loss(P + s * dP, Q + s * dQ, Vq, Vk, img, kimg, W)[0])
            assert abs((ff(eps) - ff(-eps)) / (2 * eps) - an) <= 1e-6 * max(1.0, abs(an))


def test_nll_and_det_finite_over_steps():
    Vq, img, Vk, kimg, W, _ = _setup()
    P, Q = C.init_phi(7)
    for _ in range(40):
        _, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
        P, Q, h = C.sgd_step(P, Q, gP, gQ, lr=5e-3)
        s, la = C.logdet_A(P, Q)
        assert float(s) > 0 and math.isfinite(float(la))
    Lm, ld = C.maps_for(W, P, Q)
    nll = C.nll_mixture(Vq[:100], Vk, Lm, ld)
    assert bool(torch.isfinite(nll).all())


# ---------------------------------------------------------------- Correction 2
def test_g3_state_uses_fit_only():
    fit, fimg = _tokens(10, 30, 1)
    st_a, ki_a = C.g3_state_fit_only(fit, fimg, seed=2711, k=40)
    # FEEDBACK / EVAL tokens are structurally not inputs of g3_state_fit_only; re-running is bit-identical.
    # (The real-data noise-replacement check is G.0 item M3 in the registration.)
    st_b, ki_b = C.g3_state_fit_only(fit, fimg, seed=2711, k=40)
    assert torch.equal(st_a.kc, st_b.kc) and torch.equal(st_a.S, st_b.S) and st_a.n == st_b.n
    assert torch.equal(ki_a, ki_b) and set(ki_a.tolist()) <= set(fimg.tolist())
    # residual rows = FIT non-members only
    assert st_a.n == len(fit) - 40
    try:
        C.require_disjoint({'FIT': {'a', 'b'}, 'EVAL': {'b'}})
        raise AssertionError('overlap not detected')
    except ValueError:
        pass


# ---------------------------------------------------------------- Correction 3
def test_reference_measure_invariance():
    """Under an invertible linear reparametrisation v' = v B (models and data transformed together),
    every model's NLL shifts by the same log|det B|, so model differences are invariant."""
    Vq, img, Vk, kimg, W, st = _setup()
    P, Q = C.init_phi(7)
    _, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
    P, Q, _ = C.sgd_step(P, Q, gP, gQ, lr=1e-3)
    s2 = C.iso_sigma2(st.S, st.n)
    g = torch.Generator().manual_seed(5)
    B = torch.eye(D, dtype=torch.float64) + 0.05 * torch.randn(D, D, generator=g, dtype=torch.float64)
    Binv = torch.linalg.inv(B); ldB = float(torch.linalg.slogdet(B)[1])
    q, k = Vq[:64], Vk
    for Lm, ld in (C.maps_for(W, sigma2=s2), C.maps_for(W), C.maps_for(W, P, Q)):
        a = C.nll_mixture(q, k, Lm, ld)
        b = C.nll_mixture(q @ B, k @ B, Binv @ Lm, ld - ldB)                    # same density in new coordinates
        assert torch.allclose(b, a + ldB, rtol=0, atol=1e-6)


def test_effect_scale_floor():
    U, raw, bound = C.effect_scale([0.5, 0.2, 0.9])
    assert U == 0.5 and not bound
    U, raw, bound = C.effect_scale([-0.3, -0.1, 0.0])
    assert U == C.U_FLOOR and bound


# ---------------------------------------------------------------- Correction 4
def test_decomposition_telescopes():
    out = C.decompose(nll_before=1.0, nll_geo=1.2, nll_naive=1.7, nll_ctrl={'C2': 1.4}, nll_oracle=0.9)
    assert abs(out['geometry'] + out['damage'] - 0.7) < 1e-12
    assert abs(out['recovery']['C2'] - 0.3) < 1e-12 and abs(out['oracle_advantage'] - 0.3) < 1e-12


# ---------------------------------------------------------------- Correction 6
def test_projection_acts_on_both_factors():
    g = torch.Generator().manual_seed(2)
    S, _ = torch.linalg.qr(torch.randn(D, C.RANK, generator=g, dtype=torch.float64))
    beta = torch.randn(C.RANK, generator=g, dtype=torch.float64)
    H = torch.eye(D, dtype=torch.float64) - S @ torch.diag(torch.sigmoid(beta)) @ S.T
    for _ in range(2):                                                          # once for grad P, once for grad Q
        G = torch.randn(D, C.RANK, generator=g, dtype=torch.float64)
        assert torch.allclose(C.apply_H(S, beta, G), H @ G, atol=1e-12)


def test_ledger_arithmetic():
    led = C.ledger('C3', n_tasks=4, keys_per_task=[126] * 4, n_fb_probes=128)
    assert led['keys'] == 4 * 126 * 1544 and led['g3_S'] == 2_359_304 and led['arm_state'] == 3 * 98_304
    assert led['probes'] == 128 * 1540
    assert led['total_charged'] == 778_176 + 2_359_304 + 294_912 + 197_120 + 128 + 16
    assert C.PHI_B == 98_304 and C.PSI_B == 49_220


# ---------------------------------------------------------------- owner conditional GO (2026-11-22)
def test_fast_loss_w_matches_reference():
    Vq, img, Vk, kimg, W, _ = _setup()
    P, Q = C.init_phi(7)
    _, gP, gQ, _ = C.grads(P, Q, Vq, Vk, img, kimg, W)
    P, Q, _ = C.sgd_step(P, Q, gP, gQ, lr=1e-3)
    ref = lambda P, Q: C.fast_loss(P, Q, Vq, Vk, img, kimg, W)[0]
    W2 = W @ W
    eff = lambda P, Q: C.fast_loss_w(P, Q, Vq, Vq @ W, img, torch.full((len(Vq),), 1.0 / len(Vq), dtype=torch.float64),
                                     Vk, Vk @ W, kimg, W2, torch.trace(W2))[0]
    Pa, Qa = P.clone().requires_grad_(True), Q.clone().requires_grad_(True)
    Pb, Qb = P.clone().requires_grad_(True), Q.clone().requires_grad_(True)
    La, Lb = ref(Pa, Qa), eff(Pb, Qb)
    assert abs(float(La.detach()) - float(Lb.detach())) <= 1e-9 * max(1.0, abs(float(La.detach())))
    ga = torch.autograd.grad(La, (Pa, Qa)); gb = torch.autograd.grad(Lb, (Pb, Qb))
    assert all(torch.allclose(x, y, rtol=1e-9, atol=1e-12) for x, y in zip(ga, gb))
    Lm, ld = C.Lmap_phi(W, P, Q)
    assert torch.allclose(Lm, W @ C.A_of(P, Q).T, atol=1e-12)


def _g0a_cells(c0, c0q, c1, r1=None):
    return [{'ACQ': {'C0': [c0] * 4, 'C0q': [c0q] * 4, 'C1': [c1] * 4, 'R1': [r1 if r1 is not None else c1] * 4}} for _ in range(9)]


def test_gate_g0a_ignores_r1():
    from exps.nanm_nl_02 import run as R
    U = 1.0
    # privileged R1 far better, deployable C1 equal to frozen G3 -> must fail
    g = R.gate_g0a(_g0a_cells(10.0, 10.0, 10.0, r1=0.0), U)
    assert not g['pass'] and g['outcome'] == 'NL02_NO_ADAPTATION_VALUE'
    # C1 beats C0 but not C0q -> quadratic sufficient
    g = R.gate_g0a(_g0a_cells(10.0, 9.0, 9.5), U)
    assert not g['pass'] and g['outcome'] == 'NL02_QUADRATIC_SUFFICIENT'
    # C1 beats both closed-form controls by >= 0.10 U in all cells -> pass
    g = R.gate_g0a(_g0a_cells(10.0, 9.9, 9.7), U)
    assert g['pass'] and abs(g['mean9_delta_acq'] - 0.2) < 1e-9
    # 6/9 positive cells is not enough even with a large mean
    cells = _g0a_cells(10.0, 10.0, 9.0)
    for c in cells[:3]:
        c['ACQ']['C1'] = [10.5] * 4
    assert not R.gate_g0a(cells, U)['pass']


def test_gate_g0b_logic():
    from exps.nanm_nl_02 import run as R
    X = ('C2', 'C3', 'C4', 'C5')
    mk = lambda dmg, rec, gap: [{'damage': [dmg] * 6, 'rec': {x: [rec[x]] * 6 for x in X}, 'acqcs_gap': {x: gap[x] for x in X}} for _ in range(9)]
    U = 1.0
    g = R.gate_g0b(mk(0.05, dict.fromkeys(X, 0.0), dict.fromkeys(X, 0.0)), U)
    assert g['outcome'] == 'NL02_NO_INTERFERENCE'
    g = R.gate_g0b(mk(0.5, {'C2': 0.48, 'C3': 0.1, 'C4': 0.1, 'C5': 0.1}, dict.fromkeys(X, 0.0)), U)
    assert g['outcome'] == 'NL02_CONVENTIONAL_SUFFICIENT' and g['X_star'] == 'C2'
    # the best recoverer violates the acquisition guard -> excluded
    g = R.gate_g0b(mk(0.5, {'C2': 0.48, 'C3': 0.1, 'C4': 0.1, 'C5': 0.1}, {'C2': 0.5, 'C3': 0.0, 'C4': 0.0, 'C5': 0.0}), U)
    assert g['X_star'] == 'C3' and g['pass']
    assert R.stage0_verdict(True, {'pass': True}, {'pass': True}) == 'NL02_PROTOCOL_BLOCKED'
