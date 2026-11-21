"""Synthetic checks for NANM-R1 (no dataset access)."""
import math

import numpy as np
import torch

from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1


def test_kcentre_prefix_consistent():
    g = torch.Generator().manual_seed(4)
    x = A.unit(torch.randn(500, 32, generator=g))
    big = A.greedy_coreset(x, 60, dim=16)
    for n in (5, 20, 59):
        assert torch.equal(big[:n], A.greedy_coreset(x, n, dim=16))


def test_n_of_matches_r0_formula():
    for n in (99568, 28224, 184240):
        assert R1.n_of(0.01, n) == math.ceil(A.CORESET_RATIO * n)


def test_variants_sizes():
    fit = torch.randn(1000, 8)
    cs = {'kc_idx': torch.randperm(1000)[:40], 'rand_idx': torch.randperm(1000)[:10]}
    v = R1.variants(fit, cs)
    assert len(v['full']) == 1000 and len(v['kc1']) == 10 and len(v['kc4']) == 40 and len(v['kc0.25']) == 3
    assert torch.equal(v['kc2'], fit[cs['kc_idx'][:20]])


def test_boot_pairing_and_identity():
    rng = np.random.default_rng(0)
    h = {}
    for c in ('a', 'b'):
        pos = rng.random((30, 64)).astype(np.float32); tot = pos + rng.random((30, 64)).astype(np.float32) * 5
        h['x', c] = (pos, tot); h['y', c] = (pos.copy(), tot.copy())
    reps = R1.boot_reps(h, ['x', 'y'], ['a', 'b'], B=50)
    c = R1.ci(reps, 'x', 'y')
    assert c['lo'] == 0 and c['hi'] == 0


def test_decide_bands():
    assert R1.decide(0.02, 0.004, 0.001) == 'BYTES_SUFFICE'
    assert R1.decide(0.02, 0.012, 0.001) == 'CONSOLIDATION_HEADROOM'
    assert R1.decide(0.012, 0.012, 0.001) == 'INCONCLUSIVE'
    assert R1.decide(0.02, 0.012, -0.001) == 'INCONCLUSIVE'
    assert R1.decide(0.02, 0.008, 0.001) == 'INCONCLUSIVE'


def test_stats_matches_manual():
    g = torch.Generator().manual_seed(5)
    fe = {c: torch.rand(3, 2, 28, 28, generator=g) for c in ('p', 'q', 'r')}
    ke = {c: {'kc1': torch.rand(3, 2, 28, 28, generator=g)} for c in ('p', 'q', 'r')}
    st = R1.stats(fe, ke, 1, ('p', 'q', 'r'))
    m = torch.cat([A.smooth(fe[c][1]).flatten() for c in ('p', 'q')]).double()
    assert abs(st['F'][0] - float(m.mean())) < 1e-9 and abs(st['F'][1] - float(m.std(unbiased=False))) < 1e-7
