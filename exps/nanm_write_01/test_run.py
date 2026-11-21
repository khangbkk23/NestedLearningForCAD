"""Unit tests for NANM-WRITE-01 kernels (CPU, synthetic normal-like data; no cache or test data)."""
import math

import numpy as np
import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r2 import run as R2
from exps.nanm_write_01 import run as W

D = 768


def tasks(n_tasks=4, n=1568, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(n_tasks):
        scale = torch.ones(D); scale[:40] = 6.0 + t; scale[40 + 10 * t:60 + 10 * t] = 3.0
        mu = torch.randn(D, generator=g) * 0.5
        out.append(A.unit(mu + torch.randn(n, D, generator=g) * scale * 0.05))
    return out


def test_kcentre_parity_with_canonical():
    x = tasks(1)[0]
    assert torch.equal(W.kcentre(x, 30, 2711), A.greedy_coreset(x, 30, 2711))
    assert torch.equal(W.kcentre(x, 30, 2712), A.greedy_coreset(x, 30, 2712))


def test_two_event_equals_one_shot_under_same_metric():
    x = tasks(1)[0]
    H, _ = W.feat_metric(x)
    one = W.kcentre(x, 31, 2711, H)
    two = W.kcentre(x, 31, 2711, H, chosen=W.kcentre(x, 16, 2711, H))
    assert torch.equal(one, two)
    two_e = W.kcentre(x, 31, 2711, None, chosen=W.kcentre(x, 16, 2711, None))
    assert torch.equal(two_e, A.greedy_coreset(x, 31, 2711))


def test_scale_invariance():
    x = tasks(1)[0]
    assert torch.equal(W.kcentre(x, 25, 2711, 2.0 * torch.eye(D, dtype=torch.float64)), W.kcentre(x, 25, 2711))


def test_metric_changes_selection():
    x = tasks(1)[0]
    H, _ = W.feat_metric(x)
    assert not torch.equal(W.kcentre(x, 25, 2711, H), W.kcentre(x, 25, 2711))


def test_feat_metric_whitens_shrunk_covariance():
    x = tasks(1)[0].double()
    H, cond = W.feat_metric(x.float())
    C = torch.cov(x.T, correction=0)
    T = R2.shrink(C, 0.5)
    assert torch.allclose(H @ T @ H, torch.eye(D, dtype=torch.float64), atol=1e-6) and cond > 1


def _streams(arms, data):
    return {a: W.run_stream(lambda t: data[t], len(data), a, 2711, keep_reader=True) for a in arms}


def test_stream_structure_b5_b0_b6_b4():
    data = tasks()
    r = _streams(('B0', 'B5', 'B6', 'B4', 'B3', 'B2'), data)
    for t in range(4):
        assert torch.equal(r['B5'][t]['idx'], r['B0'][t]['idx'])               # transfer disabled == G3
        assert torch.equal(r['B6'][t]['ref_idx'], r['B0'][t]['idx'])           # reference bank == G3 bank
        assert len(r['B4'][t]['idx']) == len(r['B0'][t]['idx']) == math.ceil(0.01 * len(data[t]))
    for t in (0, 1):                                                           # §2: B4 == B6 at tasks 1, 2
        assert torch.equal(r['B4'][t]['idx'], r['B6'][t]['idx'])
    assert torch.equal(r['B4'][0]['idx'], r['B0'][0]['idx'])                   # H_1 = I
    h = math.ceil(r['B3'][0]['k'] / 2)
    assert torch.equal(r['B3'][0]['idx'][:h], r['B4'][0]['idx'][:h])           # B3 differs only after event 1
    # B6's slow statistic is built on the reference bank (== B0's bank) -> identical reader W at every task
    for t in range(4):
        assert torch.equal(r['B6'][t]['W'], r['B0'][t]['W'])


def test_causality_future_poisoning():
    data = tasks()
    poisoned = data[:2] + [torch.full_like(data[2], float('nan')), A.unit(torch.randn_like(data[3]))]
    calls = []

    def get(t):
        calls.append(t)
        return poisoned[t]
    for arm in ('B4', 'B3', 'B6', 'B2'):
        calls.clear()
        clean = W.run_stream(lambda t: data[t], 2, arm, 2711)
        try:
            pois = W.run_stream(get, 4, arm, 2711)
        except RuntimeError:                                  # NaN task may fail the PD check at t=3: fine
            pois = None
        assert calls == sorted(calls) and len(set(calls)) == len(calls)
        if pois is not None:
            for t in range(2):
                assert torch.equal(clean[t]['idx'], pois[t]['idx'])
        else:
            calls.clear()
            pois = W.run_stream(get, 2, arm, 2711)
            for t in range(2):
                assert torch.equal(clean[t]['idx'], pois[t]['idx'])


def test_energies_matches_bruteforce():
    g = torch.Generator().manual_seed(3)
    q = A.unit(torch.randn(50, D, generator=g)); k = A.unit(torch.randn(20, D, generator=g))
    Wm = torch.eye(D) * 1.5
    e = W.energies(q, k, Wm)
    ref = 0.5 * (((q[:, None] - k[None]) @ Wm) ** 2).sum(-1).min(1).values
    assert torch.allclose(e, ref, rtol=1e-4, atol=1e-5)


def test_macro_log_and_boot():
    per = {'a': np.array([1.0, 3.0]), 'b': np.array([2.0, 2.0])}
    assert abs(W.macro_log(per, ['a', 'b']) - math.log(2.0)) < 1e-12
    bidx = W.boot_index({'a': 2, 'b': 2}, B=50, seed=1)
    reps = W.macro_log(per, ['a', 'b'], bidx)
    assert reps.shape == (50,) and np.isfinite(reps).all()


def test_decide_logic():
    Dn = 0.04
    assert W.decide(0.02, 0.001, [0.01] * 5, 0.0, 0.0, 0.0, Dn)[0] == 'PASS'
    assert W.decide(0.003, 0.001, [0.01] * 5, 0.0, 0.0, 0.0, Dn)[0] == 'FAIL'               # < 0.10 D
    assert W.decide(0.02, 0.001, [0.01] * 5, -0.01, 0.0, 0.0, Dn)[0] == 'FAIL'             # P4
    assert W.decide(0.02, 0.001, [0.01] * 5, 0.0, 0.2, 0.0, Dn)[0] == 'FAIL'               # P5
    assert W.decide(0.006, 0.001, [0.01] * 5, 0.0, 0.0, 0.0, Dn)[0] == 'INCONCLUSIVE'      # between 0.10 D, 0.25 D
    assert W.decide(0.02, -0.001, [0.01] * 5, 0.0, 0.0, 0.0, Dn)[0] == 'INCONCLUSIVE'      # CI
    assert W.decide(0.02, 0.001, [0.01] * 5, 0.0, 0.0, 0.0, -1.0)[0] == 'INCONCLUSIVE'     # anchor invalid
    assert W.contrast_supported(0.005, 0.001, [0.001] * 5, Dn)
    assert not W.contrast_supported(0.003, 0.001, [0.001] * 5, Dn)


def test_byte_accounting():
    assert W.S_BYTES == 768 * 768 * 4 + 8
