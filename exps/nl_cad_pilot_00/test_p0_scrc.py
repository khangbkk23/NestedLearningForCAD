"""NL-CAD-PILOT-00 P0 numerical tests (synthetic matrices only; no dataset file is opened here).

Run: python -m pytest exps/nl_cad_pilot_00/test_p0_scrc.py -q
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exps.nl_sgm_01a import exact as X
from exps.nl_cad_pilot_00 import es as E
from exps.nl_cad_pilot_00 import ledger as L
from exps.nl_cad_pilot_00 import nnx
from exps.nl_cad_pilot_00 import scrc as S

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(6)


def rnd_state(d, r, g, scale=1.0, k=None):
    k = r if k is None else k
    U = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))[0]
    lam = torch.sort(torch.rand(r, generator=g, dtype=torch.float64) * scale, descending=True).values
    lam[k:] = 0
    U[:, k:] = 0
    return S.FastState(U.float(), lam.float(), 1)


def rnd_R(m, d, g, scale=1.0, rank=None):
    if rank is None:
        return (torch.randn(m, d, generator=g, dtype=torch.float64) * scale).float()
    A = torch.randn(m, rank, generator=g, dtype=torch.float64) @ torch.randn(rank, d, generator=g, dtype=torch.float64)
    return (A * scale / math.sqrt(rank)).float()


def top_r(A, r):
    ev, V = torch.linalg.eigh(A)
    ev, V = ev.flip(0), V.flip(1)
    return (V[:, :r] * ev[:r].clamp_min(0)[None]) @ V[:, :r].T, ev


def F_obj(K, C, Kp, rho):
    return 0.5 * (K - C).square().sum() + 0.5 * rho * (K - Kp).square().sum()


# ------------------------------------------------------------------ T1: proximal step = exact constrained minimiser

@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0, 7.0, 32.0, 64.0])
def test_T1_prox_equals_eckart_young_truncation(rho):
    g = torch.Generator().manual_seed(1)
    d, r, m = 24, 4, 50
    st = rnd_state(d, r, g, scale=3.0)
    R = rnd_R(m, d, g)
    new, _ = S.prox_update(st, R, rho, r)
    Kt = k_t = S.dense_K(new)
    Ktil = S.k_tilde_dense(st, R, rho)
    ref, ev = top_r(Ktil, r)
    assert float((Kt - ref).norm() / ref.norm()) <= 1e-6
    # objective identity F(K) = (1+rho)/2 ||K - K_tilde||^2 + const
    C = R.double().T @ R.double() / m
    Kp = S.dense_K(st)
    A1, A2 = torch.randn(d, d, generator=g, dtype=torch.float64), torch.randn(d, d, generator=g, dtype=torch.float64)
    lhs = F_obj(A1, C, Kp, rho) - F_obj(A2, C, Kp, rho)
    rhs = 0.5 * (1 + rho) * ((A1 - Ktil).square().sum() - (A2 - Ktil).square().sum())
    assert abs(float(lhs - rhs)) <= 1e-9 * abs(float(lhs)) + 1e-9
    # brute force: random rank<=r PSD candidates and local perturbations never beat K_t
    f0 = float(F_obj(k_t, C, Kp, rho))
    for i in range(2000):
        kk = int(torch.randint(1, r + 1, (1,), generator=g))
        V = torch.randn(d, kk, generator=g, dtype=torch.float64) * float(torch.rand(1, generator=g)) * 2
        assert float(F_obj(V @ V.T, C, Kp, rho)) >= f0 - 1e-9
    U, lam = S.factors64(new)
    Vt = U * lam.sqrt()[None]
    for i in range(500):
        P = Vt + 1e-3 * torch.randn(d, r, generator=g, dtype=torch.float64)
        assert float(F_obj(P @ P.T, C, Kp, rho)) >= f0 - 1e-6 * abs(f0)
    # factored gradient descent from random starts reaches no lower objective
    for s in range(3):
        V = torch.randn(d, r, generator=g, dtype=torch.float64).requires_grad_(True)
        opt = torch.optim.Adam([V], lr=0.02)
        for _ in range(3000):
            opt.zero_grad()
            loss = F_obj(V @ V.T, C, Kp, rho)
            loss.backward()
            opt.step()
        assert float(loss) >= f0 - 1e-6 * abs(f0)


def test_T1_full_size_svd_matches_dense_eigh():
    g = torch.Generator().manual_seed(2)
    d, r, m = 768, 32, 4096
    st = rnd_state(d, r, g, scale=5.0)
    R = rnd_R(m, d, g, rank=120)
    new, dg = S.prox_update(st, R, 3.0, r, return_dense_check=True)
    Ktil = S.k_tilde_dense(st, R, 3.0)
    ref, ev = top_r(Ktil, r)
    assert float((S.dense_K(new) - ref).norm() / ref.norm()) <= 1e-6
    assert float((dg["_ev"][:r + 1] - ev[:r + 1]).abs().max() / ev[0]) <= 1e-12


def test_T1_gram_path_equals_svd_path():
    g = torch.Generator().manual_seed(21)
    d, r, m = 768, 32, 4096
    st = rnd_state(d, r, g, scale=5.0)
    R = rnd_R(m, d, g, rank=200)
    Cd = R.double().T @ R.double() / m
    for rho in (0.0, 2.5, 64.0):
        a, _ = S.prox_update(st, R, rho, r)
        b, _ = S.prox_update_gram(st, Cd, rho, r)
        assert float((S.dense_K(a) - S.dense_K(b)).norm() / S.dense_K(a).norm()) <= 1e-6


# ------------------------------------------------------------------ T2: P1 convex weights, boundedness, truncation bound

def test_T2_P1_convex_memory_1000_streams():
    g = torch.Generator().manual_seed(3)
    d, r = 12, 3
    worst = 0.0
    for s in range(1000):
        T = int(torch.randint(2, 9, (1,), generator=g))
        Cs, rhos = [], [0.0]
        for t in range(T):
            m = int(torch.randint(2, 20, (1,), generator=g))
            sc = float(10 ** (torch.rand(1, generator=g) * 4 - 2))
            Cs.append(rnd_R(m, d, g, scale=sc))
            if t:
                rhos.append(float(torch.rand(1, generator=g) * 64))
        st = S.FastState.empty(d, r)
        Kex = torch.zeros(d, d, dtype=torch.float64)
        bound = 0.0
        Cd = [R.double().T @ R.double() / len(R) for R in Cs]
        for t in range(T):
            rho = rhos[t]
            Ktil = S.k_tilde_dense(st, Cs[t], rho)
            lam_r1 = float(torch.linalg.eigvalsh(Ktil).flip(0)[r])
            st, _ = S.prox_update(st, Cs[t], rho, r)
            Kex = (Cd[t] + rho * Kex) / (1 + rho)
            bound = rho / (1 + rho) * bound + max(lam_r1, 0.0)
        # convex weights
        w = [(1 / (1 + rhos[tau])) * np.prod([rhos[j] / (1 + rhos[j]) for j in range(tau + 1, T)]) for tau in range(T)]
        assert abs(sum(w) - 1) <= 1e-12
        assert float((sum(wi * c for wi, c in zip(w, Cd)) - Kex).norm()) <= 1e-9 * float(Kex.norm()) + 1e-15
        cmax = max(float(torch.linalg.matrix_norm(c, 2)) for c in Cd)
        K = S.dense_K(st)
        assert float(torch.linalg.matrix_norm(Kex, 2)) <= cmax * (1 + 1e-12)
        assert float(torch.linalg.matrix_norm(K, 2)) <= cmax * (1 + 1e-6)
        dev = float(torch.linalg.matrix_norm(K - Kex, 2))
        assert dev <= bound + 1e-5 * cmax
        worst = max(worst, (dev - bound) / cmax)
    assert worst <= 1e-5


# ------------------------------------------------------------------ T3: P2 fixed point and contraction

@pytest.mark.parametrize("rho", [0.0, 1.0, 32.0, 64.0])
def test_T3_P2_stationary_fixed_point(rho):
    g = torch.Generator().manual_seed(4)
    d, r, m = 40, 5, 200
    R = rnd_R(m, d, g, rank=12)
    C = R.double().T @ R.double() / m
    target, ev = top_r(C, r)
    assert float(ev[r - 1] - ev[r]) > 0
    st = S.FastState.empty(d, r)
    st, _ = S.prox_update(st, R, 0.0, r)                       # rho_1 = 0
    for t in range(20):
        st, _ = S.prox_update(st, R, rho, r)
        assert float((S.dense_K(st) - target).norm() / target.norm()) <= 2e-6
    # contraction of the untruncated error from an arbitrary start: exactly rho/(1+rho)
    st0 = rnd_state(d, r, g, scale=float(ev[0]) * 2)
    Kp = S.dense_K(st0)
    Ktil = S.k_tilde_dense(st0, R, rho)
    ratio = float((Ktil - C).norm() / (Kp - C).norm())
    assert abs(ratio - rho / (1 + rho)) <= 1e-12
    # From an arbitrary start the truncated recursion converges to *a* fixed point, not necessarily Pi_r C_hat:
    # every fixed point is K* = sum_{i in S} lam_i v_i v_i^T over an eigen-subset S of C_hat with
    # min_{S} lam >= max_{not S} lam / (1 + rho)  (P0 finding: non-uniqueness / subspace lock-in).
    st = st0
    for t in range(3000 if rho else 1):
        st, _ = S.prox_update(st, R, rho, r)
    K = S.dense_K(st)
    evC, V = torch.linalg.eigh(C)
    evC, V = evC.flip(0), V.flip(1)
    tol = 5e-6 * (1 + rho)                         # FP32 persistence noise is amplified ~(1 + rho) at steady state
    assert float((K @ C - C @ K).norm() / (K.norm() * C.norm())) <= tol          # commutes with C_hat
    proj = torch.einsum("di,de,ei->i", V, K, V)                                   # K's value on each eigvec
    S_idx = torch.argsort(proj, descending=True)[:r]
    assert float((proj[S_idx] - evC[S_idx]).abs().max() / evC[0]) <= tol         # kept values = C_hat eigenvalues
    out = torch.ones(len(evC), dtype=torch.bool); out[S_idx] = False
    assert float(evC[S_idx].min()) >= float(evC[out].max()) / (1 + rho) * (1 - 1e-6)
    if rho == 0:
        assert float((K - target).norm() / target.norm()) <= 2e-6


@pytest.mark.parametrize("rho", [1.0, 4.0, 32.0, 64.0])
def test_T3_spurious_fixed_points_exist_but_are_unstable(rho):
    """Non-uniqueness of P2: the eigen-subset S = {2..r+1} (top eigenvector dropped) is an exact fixed point whenever
    lam_{r+1} >= lam_1/(1+rho). Linearisation: a rotation eps of the weakest kept direction towards the dropped one
    grows by g = rho lam_S / ((1+rho) lam_S - lam_0) > 1 per boundary (unstable, but g -> 1 as rho grows)."""
    d, r = 20, 4
    lam = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0] + [0.0] * 15, dtype=torch.float64)
    V = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(0), dtype=torch.float64))[0]
    R = ((V * lam.sqrt()[None]).T * math.sqrt(d)).float()      # C_hat = V diag(lam) V^T
    Sset = [1, 2, 3, 4]
    Kfix = (V[:, Sset] * lam[Sset][None]) @ V[:, Sset].T
    st = S.FastState(V[:, Sset].float(), lam[Sset].float(), 1)
    st1, _ = S.prox_update(st, R, rho, r)
    assert float((S.dense_K(st1) - Kfix).norm() / Kfix.norm()) <= 1e-6        # exact fixed point
    eps = 1e-4
    U = V[:, Sset].clone()
    U[:, 3] = (V[:, 4] + eps * V[:, 0]) / math.sqrt(1 + eps * eps)
    st = S.FastState(U.float(), lam[Sset].float(), 1)
    angles = []
    for t in range(6):
        st, _ = S.prox_update(st, R, rho, r)
        Uk, _ = S.factors64(st)
        angles.append(float((Uk.T @ V[:, 0]).norm()))
    g_meas = (angles[-1] / angles[0]) ** (1 / 5)
    g_pred = rho * 6.0 / ((1 + rho) * 6.0 - 10.0)
    assert g_meas > 1.0 and abs(g_meas - g_pred) <= 0.02 * g_pred


def test_T3_zero_input_only_gives_zero_state_and_no_decay():
    g = torch.Generator().manual_seed(5)
    d, r = 30, 4
    st, _ = S.prox_update(S.FastState.empty(d, r), torch.zeros(10, d), 0.0, r)
    assert float(st.lam.abs().max()) == 0.0
    R = rnd_R(60, d, g)
    st, _ = S.prox_update(S.FastState.empty(d, r), R, 0.0, r)
    tr0 = float(st.lam.sum())
    for t in range(200):                                       # NL-CORE-01R failure mode: decay to 0
        st, _ = S.prox_update(st, R, 64.0, r)
    assert abs(float(st.lam.sum()) - tr0) <= 1e-5 * tr0


# ------------------------------------------------------------------ T4: P3 sandwich and ranking distortion

def test_T4_P3_sandwich_1e5():
    g = torch.Generator().manual_seed(6)
    d, r = 768, 32
    n_cfg, n_q, n_b = 100, 1000, 64
    worst_lo, worst_hi, viol_rank = 0.0, 0.0, 0
    for k in range(n_cfg):
        st = rnd_state(d, r, g, scale=float(10 ** (torch.rand(1, generator=g) * 4 - 2)),
                       k=int(torch.randint(1, r + 1, (1,), generator=g)))
        th_tau = float(torch.randn(1, generator=g) * 2)
        gamma = float(torch.rand(1, generator=g) * 2)
        met = S.make_metric(st, th_tau, gamma)
        assert met.mu_theta <= met.mu_t * (1 + 1e-12)
        bank = torch.randn(n_b, d, generator=g)
        U = met.U
        qs = bank[torch.randint(n_b, (n_q,), generator=g)] + torch.randn(n_q, d, generator=g) * 0.3
        qs[: n_q // 2] += (torch.randn(n_q // 2, U.shape[1], generator=g, dtype=torch.float64) @ U.T).float() * 3
        dE = torch.cdist(qs.double(), bank.double()).min(1).values
        zq, zb = S.apply_sqrt(qs, met), S.apply_sqrt(bank, met)
        s = torch.cdist(zq, zb).min(1).values
        lo = math.sqrt(met.mu_theta) * dE
        worst_lo = max(worst_lo, float(((lo - s) / dE).max()))
        worst_hi = max(worst_hi, float(((s - dE) / dE).max()))
        # ranking: every pair with dE ratio > kappa(theta) keeps its order
        kap = met.kappa_theta
        a, b = torch.randint(n_q, (4000,), generator=g), torch.randint(n_q, (4000,), generator=g)
        big = dE[a] > kap * dE[b] * (1 + 1e-12)
        viol_rank += int((s[a][big] <= s[b][big]).sum())
    assert worst_lo <= 1e-12 and worst_hi <= 1e-12 and viol_rank == 0


# ------------------------------------------------------------------ T5: exact metric search (P5 route and screen route)

def clustered(g, n_c, per, d, spread=0.05):
    cent = torch.randn(n_c, d, generator=g) * 3
    return (cent.repeat_interleave(per, 0) + torch.randn(n_c * per, d, generator=g) * spread).contiguous()


def test_T5_candidate_route_exact_and_fallback_logged():
    g = torch.Generator().manual_seed(7)
    d = 64
    x = clustered(g, 30, 70, d)
    ids = torch.arange(len(x)) * 3 + 5
    q = x[torch.randint(len(x), (400,), generator=g)] + torch.randn(400, d, generator=g) * 0.02
    total_cert = 0
    for gamma, tht in ((0.2, 2.0), (1.0, 0.0), (1.9, -1.0)):
        st = rnd_state(d, 8, g, scale=1.0)
        met = S.make_metric(st, tht, gamma, r=8)
        d1, i1 = nnx.brute_metric(q, x, ids, met)
        d2, i2, cert = nnx.candidate_route(q, x, ids, met, K=64)
        assert torch.equal(i1, i2) and torch.equal(d1, d2)
        total_cert += int(cert.sum())
    assert total_cert > 0            # the route certifies on clustered data; all others fell back (logged)


@pytest.mark.parametrize("seed", [8, 9])
def test_T5_screen_route_exact_vs_bruteforce_with_ties(seed):
    g = torch.Generator().manual_seed(seed)
    d = 768
    x = clustered(g, 20, 40, d, spread=0.5)
    x[100:110] = x[90:100]                                     # exact duplicate rows -> ties
    ids = torch.arange(len(x)) * 2
    q = torch.cat([x[torch.randint(len(x), (300,), generator=g)] + torch.randn(300, d, generator=g) * 0.1,
                   x[95:105].clone()])
    for gamma, tht, k in ((1.0, 0.0, 32), (1.99, -3.0, 32), (0.05, 3.0, 5)):
        st = rnd_state(d, 32, g, scale=4.0, k=k)
        met = S.make_metric(st, tht, gamma)
        rows = nnx.Rows.transformed(x, ids, met)
        st_ = {}
        d1, i1 = nnx.certified_nn(q, rows, met, q_chunk=128, r_chunk=257, stats=st_)
        d2, i2 = nnx.brute_metric(q, x, ids, met)
        assert torch.equal(i1, i2) and torch.equal(d1, d2)


def test_T5_leave_image_out_exact():
    g = torch.Generator().manual_seed(10)
    d = 768
    x = clustered(g, 10, 50, d, spread=0.4)
    ids = torch.arange(len(x))
    keys = ids // 25
    qi = torch.randint(len(x), (300,), generator=g)
    q = x[qi].clone()
    qk = keys[qi]
    d1, i1 = nnx.certified_nn(q, nnx.Rows(x, ids), None, q_key=qk, row_key=keys, r_chunk=97)
    d2, i2 = nnx.brute_metric(q, x, ids, None, q_key=qk, row_key=keys)
    assert torch.equal(i1, i2) and torch.equal(d1, d2)
    assert bool((keys[i1] != qk).all())


# ------------------------------------------------------------------ T6 (synthetic part): gamma = 0 bitwise canonical

def test_T6_gamma0_bitwise_vs_canonical_oracle():
    g = torch.Generator().manual_seed(11)
    d = 768
    x = clustered(g, 15, 300, d, spread=0.6)
    ids = torch.arange(len(x)) * 7 + 3
    q = x[torch.randint(len(x), (4 * 784,), generator=g)] + torch.randn(4 * 784, d, generator=g) * 0.2
    st = rnd_state(d, 32, g, scale=3.0)
    met = S.make_metric(st, 0.0, 0.0)
    assert met.is_identity and float(met.c.abs().max()) == 0.0
    rows = nnx.Rows.transformed(x, ids, met)
    assert torch.equal(rows.z, x)
    s1, s2 = {}, {}
    d1, i1 = nnx.certified_nn(q, rows, met, stats=s1)
    d0, i0 = X.nn(q, X.RowSet(x, ids), stats=s2)
    assert torch.equal(d1, d0) and torch.equal(i1, i0) and s1["candidates"] == s2["candidates"]
    # same through the Euclidean (metric=None) entry
    d3, i3 = nnx.certified_nn(q, nnx.Rows(x, ids), None)
    assert torch.equal(d3, d0) and torch.equal(i3, i0)
    # readout: native maps and image scores
    hw = (900, 900)
    m1 = F.interpolate(d1.float().reshape(-1, 1, 28, 28), size=hw, mode="bilinear", align_corners=False)
    m0 = F.interpolate(d0.float().reshape(-1, 1, 28, 28), size=hw, mode="bilinear", align_corners=False)
    assert torch.equal(m1, m0) and torch.equal(m1.flatten(1).max(1).values, m0.flatten(1).max(1).values)


def test_P6_readout_nonexpansive():
    g = torch.Generator().manual_seed(12)
    a = torch.rand(8, 784, generator=g, dtype=torch.float64)
    b = a + (torch.rand(8, 784, generator=g, dtype=torch.float64) - 0.5) * 0.01
    ma = F.interpolate(a.reshape(-1, 1, 28, 28), size=(333, 517), mode="bilinear", align_corners=False)
    mb = F.interpolate(b.reshape(-1, 1, 28, 28), size=(333, 517), mode="bilinear", align_corners=False)
    assert float((ma - mb).abs().max()) <= float((a - b).abs().max()) * (1 + 1e-12)
    assert float((ma.flatten(1).max(1).values - mb.flatten(1).max(1).values).abs().max()) <= float((a - b).abs().max()) * (1 + 1e-12)


# ------------------------------------------------------------------ P7: drift bounds

def test_P7_drift_bounds():
    g = torch.Generator().manual_seed(13)
    d, r = 128, 8
    viol = 0
    for k in range(50):
        st = rnd_state(d, r, g, scale=2.0)
        R = rnd_R(300, d, g)
        rho = float(torch.rand(1, generator=g) * 64)
        f = S.novelty_features(st, R, 3)
        Ktil = S.k_tilde_dense(st, R, rho)
        Kp = S.dense_K(st)
        C = R.double().T @ R.double() / len(R)
        # novelty from factors equals dense definition; drift identity
        nu_dense = float((C - Kp).norm() / Kp.norm())
        assert abs(f["nu"] - nu_dense) <= 1e-9 * nu_dense
        assert abs(f["a"] - float(torch.trace(C) / torch.trace(Kp))) <= 1e-12 * f["a"]
        assert abs(float((Ktil - Kp).norm()) - nu_dense * float(Kp.norm()) / (1 + rho)) <= 1e-9 * float(Kp.norm())
        st2, _ = S.prox_update(st, R, rho, r)
        m1 = S.make_metric(st, 0.3, 1.2, r=r)
        m2 = S.make_metric(st2, 0.3, 1.2, r=r)
        M1 = torch.eye(d, dtype=torch.float64) - (m1.U * m1.c[None]) @ m1.U.T
        M2 = torch.eye(d, dtype=torch.float64) - (m2.U * m2.c[None]) @ m2.U.T
        dn = float(torch.linalg.matrix_norm(M1 - M2, 2))
        bank = torch.randn(200, d, generator=g)
        q = torch.randn(500, d, generator=g)
        dE = torch.cdist(q.double(), bank.double()).min(1).values
        s1 = torch.cdist(S.apply_sqrt(q, m1), S.apply_sqrt(bank, m1)).min(1).values
        s2 = torch.cdist(S.apply_sqrt(q, m2), S.apply_sqrt(bank, m2)).min(1).values
        bound = dn * dE / math.sqrt(min(m1.mu_t, m2.mu_t))
        viol += int(((s1 - s2).abs() > bound * (1 + 1e-9) + 1e-12).sum())
    assert viol == 0


# ------------------------------------------------------------------ T7: Level 0 is theta-independent

def test_T7_level0_untouched_by_theta_and_policy():
    w = E.SynthWorld(seed=3, T=3, imgs=4, ppi=30, bank_imgs=3, n_test=60)
    snap = [tuple(t.clone() for t in w.bank_at(t)) for t in range(1, 4)]
    res = w.residuals()
    for th in ([0, 0, 0, 0, 0, 0], [1, -2, 0.5, 3, -1, 2], [-3, 1, 1, -1, 2, -2]):
        E.episode_J(w, res, th)
        E.episode_J(w, res, th, "cumulative")
    for t in range(1, 4):
        assert all(torch.equal(a, b) for a, b in zip(snap[t - 1], w.bank_at(t)))
    # the Level-1/2 APIs take no bank-writing argument and return no bank
    assert "bank" not in inspect.signature(S.prox_update).parameters
    assert "bank" not in inspect.signature(E.run_fast_stream).parameters


# ------------------------------------------------------------------ T8: ES determinism + end-to-end wiring

def test_T8_es_deterministic_and_climbs_known_objective():
    J = lambda th: -float(np.sum((th - np.array([1.0, -0.5, 0.25, 0.0, 2.0, -1.0])) ** 2))
    a, ta = E.es_optimize(J, np.zeros(6), iters=40)
    b, tb = E.es_optimize(J, np.zeros(6), iters=40)
    assert ta == tb and np.array_equal(a, b)
    c, _ = E.es_optimize(J, np.zeros(6), iters=40, seed=7)
    assert not np.array_equal(a, c)
    assert J(a) > J(np.zeros(6)) + 1.0


def test_T8_es_end_to_end_synthetic_deterministic():
    w = E.SynthWorld(seed=5, T=3, imgs=4, ppi=40, bank_imgs=3, n_test=80)
    res = w.residuals()
    J = lambda th: E.episode_J(w, res, th)
    a, ta = E.es_optimize(J, np.zeros(6), iters=2, P=4)
    b, tb = E.es_optimize(J, np.zeros(6), iters=2, P=4)
    assert ta == tb and np.array_equal(a, b)


# ------------------------------------------------------------------ save / reload, determinism, stability

def test_save_reload_bitwise_stream_resume():
    g = torch.Generator().manual_seed(14)
    d, r = 768, 32
    res = [rnd_R(512, d, g, scale=float(1 + t), rank=64) for t in range(6)]
    pol = S.Learned([0.3, 0.2, -0.5, 0.4])
    full = E.run_fast_stream(res, pol)
    b = full[2]["state"].to_bytes()
    assert len(b) > 0
    st = S.FastState.from_bytes(b)
    assert torch.equal(st.U, full[2]["state"].U) and torch.equal(st.lam, full[2]["state"].lam) and st.t == 3
    resumed = E.run_fast_stream(res, pol, state=st, start_t=4)
    for a, c in zip(full[3:], resumed):
        assert torch.equal(a["state"].U, c["state"].U) and torch.equal(a["state"].lam, c["state"].lam)
        assert a["rho"] == c["rho"]
    again = E.run_fast_stream(res, pol)
    assert all(torch.equal(a["state"].U, c["state"].U) for a, c in zip(full, again))
    assert full[0]["state"].nbytes() == L.FAST_BYTES == 98_432


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_numerical_stability_extremes(scale):
    g = torch.Generator().manual_seed(15)
    d, r = 64, 8
    st = S.FastState.empty(d, r)
    for t, rho in enumerate([0.0, 64.0, 1e6, 0.0, 1e-9]):
        R = rnd_R(5 if t == 3 else 100, d, g, scale=scale)       # t=3: m=5 < r (rank-deficient)
        st, dg = S.prox_update(st, R, rho, r)
        assert bool(torch.isfinite(st.U).all()) and bool(torch.isfinite(st.lam).all())
        assert bool((st.lam >= 0).all()) and bool((st.lam[:-1] >= st.lam[1:]).all())
        U, lam = S.factors64(st)
        assert float((U.T @ U - torch.eye(U.shape[1], dtype=torch.float64)).abs().max()) <= 1e-12
        met = S.make_metric(st, 0.0, 1.0, r=r)
        assert 0 < met.mu_t <= 1 and bool((met.c >= 0).all()) and bool((met.c < 1).all())
    with pytest.raises(RuntimeError):
        S.prox_update(st, R, float("nan"), r)
    with pytest.raises(RuntimeError):
        S.prox_update(st, R, -1.0, r)


def test_policies_and_features():
    assert S.Cumulative()(1, None) == 0.0 and S.Cumulative()(5, [1, 1, 1, 1]) == 4.0
    assert S.Latest()(7, [1, 1, 1, 1]) == 0.0
    assert S.Learned([0, 0, 0, 0])(3, [1, 2, 3, 4]) == 32.0     # A5: theta_0 -> rho = 32
    th = S.Theta()
    assert th.gamma == 1.0 and th.nbytes() == 32
    met0 = S.make_metric(S.FastState.empty(10, 4), 0.0, 1.0, r=4)
    assert met0.is_identity                                     # empty fast state -> identity metric
    p = S.CausalPermuted([0.1, 0.2, 0.3, 0.4], key=1)
    hs = {t: [1.0, float(t), 0.5, 0.1] for t in range(2, 9)}
    for t in range(2, 9):
        p(t, hs[t])                                             # never needs a future h
    assert set(p.hist) == set(range(2, 9))


# ------------------------------------------------------------------ T9: byte ledger

def test_T9_ledger():
    assert L.SCRC_OVERHEAD == 98_464
    l15 = L.ledger(15)
    assert l15["A0"]["total"] == 15_359_984 and l15["A0"]["bank_rows"] == 93_680      # canonical C2-DOM B2 final
    for arm, v in l15.items():
        assert v["total"] <= L.B2 and 0 <= v["slack"] < 101 + 64
    assert l15["A4"]["bank_rows"] == 92_705 and l15["A4"]["total"] == 15_359_973
    # the prereg statement "every arm's total equals 15,359,984 B" cannot hold at 101-B row granularity
    assert len({v["total"] for v in l15.values()}) > 1


# ------------------------------------------------------------------ GPU parity (only when CUDA exists)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_gpu_certified_matches_cpu_oracle():
    g = torch.Generator().manual_seed(16)
    d = 768
    x = clustered(g, 12, 400, d, spread=0.6)
    ids = torch.arange(len(x)) * 5
    q = x[torch.randint(len(x), (2000,), generator=g)] + torch.randn(2000, d, generator=g) * 0.2
    st = rnd_state(d, 32, g, scale=3.0)
    for gamma in (0.0, 1.0):
        met = S.make_metric(st, 0.0, gamma)
        dc, ic = nnx.certified_nn(q, nnx.Rows.transformed(x, ids, met), met)
        dg, ig = nnx.certified_nn(q, nnx.Rows.transformed(x, ids, met, device="cuda"), met)
        assert torch.equal(ic, ig.cpu())
        assert float(((dc - dg.cpu()).abs() / dc).max()) <= 1e-13
    assert nnx.screen_error_ratio(q[:500].cuda(), x.cuda()) <= 1.0
