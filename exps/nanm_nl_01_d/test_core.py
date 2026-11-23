"""CPU synthetic tests for NANM-NL-01D core (no data, no GPU)."""
import itertools
import math

import torch

from exps.nanm_nl_01_d import core as C


def _z(n, d=16, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _brute_ffs(Z, n, start=None, base=None):
    sel = [] if base is None else None
    D = C.sqdist(Z, Z)
    if base is not None:
        mind = C.sqdist(Z, base).min(1).values; cur = int(mind.argmax())
    else:
        mind = torch.full((len(Z),), math.inf, dtype=torch.float64); cur = start
    sel = []
    for _ in range(n):
        sel.append(cur); mind = torch.minimum(mind, D[:, cur]); m = mind.clone(); m[sel] = -1; cur = int(m.argmax())
    return sel


def test_ffs_matches_bruteforce():
    Z = _z(200)
    assert C.ffs(Z, 50, start=7).tolist() == _brute_ffs(Z, 50, start=7)
    B = _z(10, seed=1)
    assert C.ffs(Z, 30, base=B).tolist() == _brute_ffs(Z, 30, base=B)


def test_prefix_min():
    Q, Kz = _z(40, seed=2), _z(25, seed=3)
    pm = C.prefix_min(Q, Kz)
    for j in (0, 5, 24):
        assert torch.allclose(pm[:, j], C.sqdist(Q, Kz[:j + 1]).min(1).values)


def test_round_quota_monotone():
    q = []
    for t, w in enumerate([[1], [1, 1], [1, 1, 1], [1, 2, 3, 4]]):
        nq = C.round_quota(256, w, q)
        assert sum(nq) == 256 and min(nq) >= 1 and all(a <= b for a, b in zip(nq, q)); q = nq
    assert C.round_quota(256, [1, 1, 1], [128, 128]) == [86, 85, 85]


def _synthetic_C(T, nev=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    Cm = [torch.cummin(torch.rand(nev * T, C.UNITS, generator=g, dtype=torch.float64) + 1, 1).values.T.contiguous()
          for _ in range(T)]
    lref = torch.full((T,), 0.5, dtype=torch.float64)
    return Cm, lref


def test_o1_dp_exact_vs_enumeration():
    T = 3; Cm, lref = _synthetic_C(T)
    big = [torch.full((T,), 1e9, dtype=torch.float64)] * T
    path, _, flags = C.o1_dp(Cm, lref, big)
    st = [C.compositions(C.UNITS, t + 1) for t in range(T)]
    ell = [C.lattice_losses(Cm, s, lref) for s in st]
    best = math.inf
    for a in range(len(st[1])):
        for b in range(len(st[2])):
            if all(st[2][b][i] <= st[1][a][i] for i in range(1)) and all(st[2][b][i] <= st[1][a][i] for i in range(2)):
                v = float(ell[1][a, :1].mean() + ell[2][b, :2].mean())
                best = min(best, v)
    got = float(C.lattice_losses(Cm, torch.tensor([path[1]]), lref)[0, :1].mean()
                + C.lattice_losses(Cm, torch.tensor([path[2]]), lref)[0, :2].mean())
    assert abs(got - best) < 1e-12 and not any(flags)
    assert all(path[2][i] <= path[1][i] for i in range(2))


def test_policies_causal_and_budget():
    X = [_z(C.N_FIT, d=8, seed=10 + t) for t in range(4)]
    P = [_z(C.N_FB, d=8, seed=20 + t) for t in range(4)]
    meta = [{'task': f't{t}', 'n_train_images': 30 + t} for t in range(4)]
    sched = {256: [[1.0], [0.5, 0.5], [0.3, 0.3, 0.4], [0.25] * 4]}
    for arm in ('B0', 'B1', 'B2', 'B4', 'B5', 'B3', 'B6'):
        def run(Xs, Ps):
            pol = (C.ContCorePolicy(256, 1, extra=C.b6_extra(256) if arm == 'B6' else None) if arm in ('B3', 'B6')
                   else C.QuotaPolicy(arm, 256, 1, schedule=sched[256]))
            out = []
            for t in range(4):
                out.append(pol.step(t, Xs[t], meta[t], P=Ps[t] if arm == 'B4' else None))
            return pol, out
        pol, a = run(X, P)
        X2 = X[:2] + [_z(C.N_FIT, d=8, seed=99), _z(C.N_FIT, d=8, seed=98)]
        _, b = run(X2, P[:2] + [_z(C.N_FB, d=8, seed=97)] * 2)
        key = lambda bank: sorted((p, int(i)) for p, idx in bank for i in idx)
        assert key(a[0]) == key(b[0]) and key(a[1]) == key(b[1]), arm
        n = sum(len(idx) for _, idx in a[-1])
        assert n == (256 + (C.b6_extra(256)(3) if arm == 'B6' else 0)), (arm, n)
    b4 = C.QuotaPolicy('B4', 256, 1).bytes(3); b6 = C.ContCorePolicy(256, 1, extra=C.b6_extra(256))
    b6.ids = [None] * (256 + C.b6_extra(256)(3))
    assert b6.bytes(3) <= b4
