"""Focused synthetic tests for NANM-GEOM-01 (CPU, no real data)."""
import math

import pytest
import torch

from exps.nanm_geom_01 import geom as G
from exps.nanm_geom_01 import run as GR
from exps.nanm_nl_02 import core as C
from exps.nanm_r2b import run as B
from nanm import g3

F64 = torch.float64


def _pts(n, d, seed, scale=1.0):
    return scale * torch.randn(n, d, generator=torch.Generator().manual_seed(seed), dtype=F64)


def _pd(d, seed):
    X = _pts(4 * d, d, seed)
    return X.T @ X / (4 * d) + 0.1 * torch.eye(d, dtype=F64)


def test_nll_parts_is_core_nll_and_splits():
    Vq, K = _pts(50, 8, 0), _pts(7, 8, 1)
    L = torch.linalg.cholesky(torch.linalg.inv(_pd(8, 2)))
    ld = float(torch.linalg.slogdet(L)[1])
    v, N, F = G.nll_parts(Vq, K, L, ld)
    assert torch.equal(v, C.nll_mixture(Vq, K, L, ld))
    assert torch.allclose(v, N + F, atol=1e-12, rtol=0)
    assert abs(N - (math.log(7) - ld + 4 * C.LOG2PI)) < 1e-12


def test_pertask_mixture_reduces_to_core_with_shared_metric():
    Vq, K1, K2 = _pts(40, 6, 3), _pts(5, 6, 4), _pts(9, 6, 5)
    L = torch.linalg.cholesky(torch.linalg.inv(_pd(6, 6))); ld = float(torch.linalg.slogdet(L)[1])
    a = G.nll_pertask_mixture(Vq, [(K1, L, ld), (K2, L, ld)])
    b = C.nll_mixture(Vq, torch.cat([K1, K2]), L, ld)
    assert torch.allclose(a, b, atol=1e-10, rtol=0)


def test_pertask_mixture_is_a_normalised_density_2d():
    K1, K2 = _pts(3, 2, 7, 0.5), _pts(4, 2, 8, 0.5) + 1.0
    L1 = torch.linalg.cholesky(torch.linalg.inv(torch.tensor([[0.3, 0.1], [0.1, 0.2]], dtype=F64)))
    L2 = torch.linalg.cholesky(torch.linalg.inv(torch.tensor([[0.1, 0.0], [0.0, 0.4]], dtype=F64)))
    g = torch.linspace(-6, 7, 601, dtype=F64); h = float(g[1] - g[0])
    X = torch.cartesian_prod(g, g)
    nll = G.nll_pertask_mixture(X, [(K1, L1, float(torch.linalg.slogdet(L1)[1])), (K2, L2, float(torch.linalg.slogdet(L2)[1]))])
    assert abs(float(torch.exp(-nll).sum()) * h * h - 1.0) < 1e-3


def test_factorial_identity_and_additive_case():
    A, B_, C_, D = 1.0, 1.3, 5.0, 7.5
    e = G.factorial(A, B_, C_, D)
    assert abs(e['shapley_W'] + e['shapley_K'] - e['delta']) < 1e-12 and abs(e['interaction'] - 2.2) < 1e-12
    e = G.factorial(0.0, 0.5, 3.0, 3.5)                                      # additive: no interaction
    assert e['interaction'] == 0 and e['shapley_W'] == 3.0 and e['shapley_K'] == 0.5


def test_keys_only_bound_superset():
    Vq, K = _pts(200, 8, 9), _pts(30, 8, 10)
    L = torch.eye(8, dtype=F64) * 1.7; ld = 8 * math.log(1.7)
    for k_old in (5, 10, 15, 29):
        d = float(C.nll_mixture(Vq, K, L, ld).mean() - C.nll_mixture(Vq, K[:k_old], L, ld).mean())
        assert d <= math.log(30 / k_old) + 1e-12


def test_whiten_alpha_canonical_equals_b_whiten():
    S = _pd(C.D, 11) * 1000; n = 1000
    W, _ = G.whiten_alpha(S, n, 0.5)
    Wb, _ = B.whiten(S, n)
    assert torch.equal(W, Wb.float().double())


def test_energy64_matches_g3_score():
    q16 = torch.nn.functional.normalize(_pts(300, C.D, 12), dim=1).half()
    keys = torch.nn.functional.normalize(_pts(40, C.D, 13), dim=1).half().double()
    W, _ = G.whiten_alpha(_pd(C.D, 14), 1.0, 0.5)
    e32 = g3.score(q16, keys.float(), W.float()).double(); e64 = G.energy64(q16.double(), keys, W)
    assert float(((e32 - e64).abs() / e64).median()) < 1e-3


def test_spearman():
    x = torch.arange(10, dtype=F64)
    assert abs(G.spearman(x, x ** 3) - 1) < 1e-12 and abs(G.spearman(x, -x) + 1) < 1e-12


def test_verdict_mapping():
    U = 100.0
    ok = {'X1': {'H': 6.0, 'H_pos': 9, 'Acq': 1.0}}; no = {'X1': {'H': 4.0, 'H_pos': 9, 'Acq': 1.0}}
    assert G.verdict(False, 20, 15, 5, 1, 9, 9, ok, U)[0] == 'GEOM01_PROTOCOL_BLOCKED'
    assert G.verdict(True, 1.0, 1, 0, 0, 9, 9, ok, U)[0] == 'GEOM01_NO_ACTIONABLE_DRIFT'
    assert G.verdict(True, 20, 15, 5, 1, 9, 9, no, U)[0] == 'GEOM01_NO_ACTIONABLE_DRIFT'
    assert G.verdict(True, 20, 15, 5, 1, 9, 9, {'X1': {'H': 6.0, 'H_pos': 6, 'Acq': 1.0}}, U)[0] == 'GEOM01_NO_ACTIONABLE_DRIFT'
    assert G.verdict(True, 20, 15, 5, 1, 9, 9, {'X1': {'H': 6.0, 'H_pos': 9, 'Acq': 3.0}}, U)[0] == 'GEOM01_NO_ACTIONABLE_DRIFT'
    assert G.verdict(True, 20, 15, 5, 11, 9, 9, ok, U)[0] == 'GEOM01_INTERACTION_DOMINANT'
    assert G.verdict(True, 20, 15, 5, 1, 6, 9, ok, U)[0] == 'GEOM01_INTERACTION_DOMINANT'
    assert G.verdict(True, 20, 15, 5, 1, 9, 9, ok, U)[0] == 'GEOM01_WHITENING_DRIFT_DOMINANT'
    assert G.verdict(True, 20, 5, 15, 1, 0, 9, ok, U)[0] == 'GEOM01_KEY_EXPANSION_DOMINANT'


# ---------------------------------------------------------------- synthetic G3 stream (canonical FIT-only states)
@pytest.fixture(scope='module')
def stream():
    from exps.nanm_nl_02_v2.test_v2 import _synthetic_stream
    return _synthetic_stream()


def test_factorial_pair_endpoints_and_bound(stream):
    I_map = lambda w: (stream.W[w], stream.ldW[w])
    c, e, s = GR.factorial_pair(stream, 0, 0, 1, I_map, True)
    assert c['A']['nll'] == stream.nll(0, 0, stream.W[0], stream.ldW[0])
    assert c['D']['nll'] == stream.nll(1, 0, stream.W[1], stream.ldW[1])
    assert e['nll']['k_only'] <= math.log(len(stream.Kv[1]) / len(stream.Kv[0])) + 1e-12
    assert abs(e['nll']['shapley_W'] + e['nll']['shapley_K'] - e['nll']['delta']) < 1e-9
    assert abs(e['N']['w_only'] - (stream.ldW[0] - stream.ldW[1])) < 1e-9            # W-only normalisation = -dlog|det W|
    assert abs(e['N']['k_only'] - math.log(2)) < 1e-12 and abs(e['N']['interaction']) < 1e-9
    assert abs(e['nll']['delta'] - e['N']['delta'] - e['F']['delta']) < 1e-9
    assert c['A']['energy'] > 0 and -1 <= s['spearman_energy_A_D'] <= 1


def test_controls_at_first_stage_coincide(stream):
    maps = GR.control_maps(stream, 0)
    x0 = GR.control_nll(stream, 0, 0, *maps['X0'])
    assert x0 == stream.nll(0, 0, stream.W[0], stream.ldW[0])
    assert GR.control_nll(stream, 0, 0, *maps['X1']) == x0                         # frozen-first = canonical at t = 0
    assert abs(GR.control_nll(stream, 0, 0, *maps['X4']) - x0) < 1e-9               # per-task mixture = canonical at t = 0
    m1 = GR.control_maps(stream, 1)
    assert torch.equal(m1['X1'][1][0], stream.W[0]) and torch.equal(m1['X0'][1][0], stream.W[1])
    assert len(m1['X4'][1]) == 2


def test_control_bytes():
    b = GR.control_bytes(4, 504)
    assert b['X0'] == b['X1'] == b['X2a'] == 504 * 1536 + 2359304 and b['X3'] < b['X0'] and b['X4'] - b['X0'] == 3 * 2359304
