"""CPU tests for BENCH-G3 pure functions: python -B -m pytest -q exps/bench_g3/test_run.py"""
import math

import numpy as np
import torch

from exps.bench_g3 import run as G
from exps.nanm_r0 import audit as A
from exps.nanm_r1 import run as R1


def test_a0b_sizes_byte_parity():
    for n_fit in ([784 * 540, 784 * 330, 784 * 300, 784 * 300], [784 * 160] * 6, [784 * k for k in range(60, 72)]):
        r, s = G.a0b_sizes(n_fit)
        base = [R1.n_of(0.01, n) for n in n_fit]
        g3 = G.KEY * sum(base) + G.EXTRA_G3
        assert 0 < G.KEY * sum(s) - g3 <= G.KEY
        assert all(a >= b for a, b in zip(s, base))


def test_c_sizes_budget():
    n_fit = [784 * k for k in (540, 330, 300, 300, 300, 600, 600, 600, 600, 600, 600, 300)]
    for extra in (0, G.EXTRA_G3):
        r, s = G.c_sizes(n_fit, extra)
        assert sum(s) * G.KEY + extra <= G.C_BYTES
        bump = [R1.n_of(r * 1.001 + 1e-9, n) for n in n_fit]
        assert sum(bump) * G.KEY + extra > G.C_BYTES or bump == s
        assert r < 0.01


def test_chaudhry():
    mat = {(0, 0): .5, (1, 0): .6, (2, 0): .4, (1, 1): .7, (2, 1): .7, (2, 2): .9}
    f = G.chaudhry(mat, 3)
    assert math.isclose(f['per_task'][0], .2) and math.isclose(f['per_task'][1], 0.)
    assert math.isclose(f['fm'], .1)
    assert math.isclose(f['acq_minus_final'], ((.5 - .4) + 0 + 0) / 3)


def test_holm():
    h = G.holm({'a': 0.01, 'b': 0.04})
    assert h['a'][1] and h['b'][1] and math.isclose(h['a'][0], 0.02)
    h = G.holm({'a': 0.03, 'b': 0.04})
    assert not h['a'][1] and not h['b'][1]


def test_boot_p():
    assert G.boot_p(np.ones(100)) == 0.
    assert G.boot_p(np.r_[np.ones(95), -np.ones(5)]) == 0.1


def test_maps224_and_binned_ap():
    g = torch.rand(4, 392, 392)
    m = G.maps224(g)
    assert m.shape == (4, 224, 224) and float(m.min()) >= 0
    y = [np.zeros((224, 224), bool) for _ in range(4)]
    y[1][10:40, 10:40] = True
    s = m.numpy().copy(); s[1][10:40, 10:40] += 1.
    pos, tot = G.hist224(s, y, 0., float(s.max()) + 1e-12)
    ap = A.binned_ap(pos.sum(0), tot.sum(0))
    from sklearn.metrics import average_precision_score
    ref = average_precision_score(np.concatenate([v.ravel() for v in y]), np.concatenate([v.ravel() for v in s]))
    assert abs(ap - ref) < 1e-3


def test_label_of():
    assert G.label_of(.4, .41, .45) == 'ABOVE_PUBLISHED'
    assert G.label_of(.4, .35, .39) == 'BELOW_PUBLISHED'
    assert G.label_of(.4, .38, .42) == 'WITHIN'
