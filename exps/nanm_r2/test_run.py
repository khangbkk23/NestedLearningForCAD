"""Synthetic checks for NANM-R2; canonical Mahalanobis implementations act as oracles (no dataset access)."""
import math

import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r2 import run as G
from exps.nar_lite_01 import run as L


def _spd(d, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * d, d, generator=g, dtype=torch.float64) * torch.linspace(0.1, 3, d, dtype=torch.float64)
    return X.T @ X / len(X)


def test_sqdist_bruteforce():
    g = torch.Generator().manual_seed(0)
    q, m = torch.randn(50, 16, generator=g), torch.randn(30, 16, generator=g)
    assert torch.allclose(G.sqdist(q, m), torch.cdist(q, m) ** 2, atol=1e-4)


def test_whitening_equals_mahalanobis_oracle():
    S = G.shrink(_spd(12, 1))
    W, cond = G.inv_sqrt_psd(S)
    assert torch.allclose(W @ S @ W, torch.eye(12, dtype=torch.float64), atol=1e-8) and cond >= 1
    g = torch.Generator().manual_seed(2)
    z, m = torch.randn(7, 12, generator=g, dtype=torch.float64), torch.randn(5, 12, generator=g, dtype=torch.float64)
    oracle = torch.stack([torch.stack([(zi - mj) @ torch.linalg.solve(S, zi - mj) for mj in m]) for zi in z])
    assert torch.allclose(G.sqdist((z @ W).float(), (m @ W).float()).double(), oracle, rtol=1e-4, atol=1e-4)


def test_shrink_preserves_trace_and_improves_condition():
    C = _spd(10, 3)
    S = G.shrink(C, 0.5)
    assert abs(float(torch.trace(S) - torch.trace(C))) < 1e-9
    assert G.inv_sqrt_psd(S)[1] < G.inv_sqrt_psd(C)[1]


def test_ppca_inverse_sqrt_and_logdet_oracle():
    C = _spd(16, 4)
    V, e, s2 = G.ppca(C, 4)
    Sig = V @ torch.diag(e) @ V.T + s2 * (torch.eye(16, dtype=torch.float64) - V @ V.T)
    W, logdet, cond = G.ppca_inv_sqrt(V, e, s2)
    assert torch.allclose(W @ Sig @ W, torch.eye(16, dtype=torch.float64), atol=1e-8)
    assert abs(logdet - float(torch.linalg.slogdet(Sig)[1])) < 1e-8
    assert abs(s2 - float(torch.linalg.eigvalsh(C)[:-4].mean())) < 1e-10 and cond >= 1


def test_smooth_nc_equals_lite_readout_on_nonnegative():
    g = torch.Generator().manual_seed(5)
    x = torch.rand(3, 28, 28, generator=g)
    assert torch.allclose(G.smooth_nc(x), L.smooth392(x), atol=1e-7)


def test_control_sizes_meet_byte_target():
    n_fit = [99568, 106624, 28224, 184240]
    base = sum(math.ceil(0.01 * n) for n in n_fit)
    r, sizes = G.control_sizes(n_fit, 2_362_376)
    assert sum(sizes) >= base + math.ceil(2_362_376 / 1536)
    assert sum(math.ceil((r - 1e-6) * n) for n in n_fit) < base + math.ceil(2_362_376 / 1536) + 4


def test_cheb_window():
    T = G.cheb_table(3)
    assert bool(T[0, 3 * 28 + 3]) and not bool(T[0, 4]) and bool(T[29, 0]) and T.sum(1).max() == 49


def _mt(m, v=None, W2=None, logdet=0.):
    W2 = torch.eye(m.shape[1]) if W2 is None else W2
    return {'A0': m, 'A0p236': m, 'A0p148': m, 'v': torch.ones(len(m)) if v is None else v, 'W2': W2,
            'logdet': logdet, 'pos': torch.zeros(len(m), dtype=torch.long), 'm_w2': m @ W2}


def test_task_energy_arms_against_oracles():
    g = torch.Generator().manual_seed(6)
    q, m = A.unit(torch.randn(20, 8, generator=g)), A.unit(torch.randn(9, 8, generator=g))
    v = torch.rand(9, generator=g) + .1
    S = _spd(8, 7).float(); W = G.inv_sqrt_psd(S.double())[0].float()
    mt = _mt(m, v, W, logdet=1.5)
    cheb = G.cheb_table()
    pos = torch.zeros(20, dtype=torch.long)
    a0 = G.task_energy(q, pos, mt, cheb, 'A0')
    assert torch.allclose(a0, (1 - (q @ m.T).amax(1)), atol=1e-5)          # unit vectors: 1/2||q-m||^2 = 1-cos
    d2 = torch.cdist(q, m) ** 2
    assert torch.allclose(G.task_energy(q, pos, mt, cheb, 'L1ratio'), (d2 / (2 * v)).min(1).values, atol=1e-5)
    assert torch.allclose(G.task_energy(q, pos, mt, cheb, 'L1nll'),
                          (d2 / (2 * v) + 0.5 * G.D * v.log()).min(1).values, atol=1e-3)
    maha = torch.stack([torch.stack([(qi - mj) @ torch.linalg.solve(S, qi - mj) for mj in m]) for qi in q])
    l2 = G.task_energy(q, pos, mt, cheb, 'L2nolog', q @ W)
    assert torch.allclose(l2, 0.5 * maha.min(1).values, rtol=1e-3, atol=1e-4)
    assert torch.allclose(G.task_energy(q, pos, mt, cheb, 'L2nll', q @ W), l2 + 0.75, atol=1e-5)
    mt['pos'] = torch.tensor([0, 5, 100, 0, 0, 783, 30, 29, 1])            # S1: only items within Chebyshev 3
    s1 = G.task_energy(q, pos, mt, cheb, 'S1')
    allowed = torch.tensor([True, False, False, True, True, False, True, True, True])
    assert torch.allclose(s1, 0.5 * d2[:, allowed].min(1).values, atol=1e-5)
