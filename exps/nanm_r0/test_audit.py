"""Synthetic checks for the NANM-R0 audit (no dataset access)."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.nanm_r0 import audit as A


def test_min_cos_dist_matches_bruteforce():
    g = torch.Generator().manual_seed(0)
    q, m = A.unit(torch.randn(300, 64, generator=g)), A.unit(torch.randn(500, 64, generator=g))
    ref = (1 - (q @ m.T).amax(1)).clamp_min(0)
    assert torch.allclose(A.min_cos_dist(q, m, qc=37), ref, atol=1e-6)
    assert A.min_cos_dist(m[:10], m).abs().max() < 1e-5          # self-match -> 0


def test_coreset_distinct_and_seeded():
    g = torch.Generator().manual_seed(1)
    centres = A.unit(torch.randn(20, 64, generator=g))
    lab = torch.arange(400) % 20
    x = A.unit(centres[lab] + .02 * torch.randn(400, 64, generator=g))
    a, b = A.greedy_coreset(x, 20, dim=32), A.greedy_coreset(x, 20, dim=32)
    assert torch.equal(a, b) and len(set(a.tolist())) == 20
    assert len(set(lab[a].tolist())) == 20                       # k-centre hits every well-separated mode


def test_binned_ap_close_to_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(20000) < .05
    s = rng.normal(size=20000) + 2 * y
    lo, hi = s.min(), s.max() + 1e-9
    b = np.clip(((s - lo) / (hi - lo) * 8192).astype(int), 0, 8191)
    ap = A.binned_ap(np.bincount(b, weights=y, minlength=8192), np.bincount(b, minlength=8192).astype(float))
    assert abs(ap - average_precision_score(y, s)) < 2e-3


def test_exact_ap_shift_invariant_and_native():
    g = torch.Generator().manual_seed(2)
    maps = torch.rand(3, 392, 392, generator=g)
    masks = [np.zeros((50, 60), bool) for _ in range(3)]; masks[1][10:20, 10:30] = True
    hws = [(50, 60)] * 3
    a = A.exact_ap(maps, hws, masks); b = A.exact_ap(maps + 3.0, hws, masks)
    assert abs(a - b) < 1e-6


def test_decide_bands():
    assert A.decide(.02, .001, [0.] * 15)[0] == 'H1_SUPPORTED'
    assert A.decide(.02, -.001, [0.] * 15)[0] == 'H1_INCONCLUSIVE'
    assert A.decide(.02, .001, [-.06] + [0.] * 14)[0] == 'H1_INCONCLUSIVE'
    assert A.decide(.02, .001, [-.02] * 4 + [0.] * 11)[0] == 'H1_INCONCLUSIVE'
    assert A.decide(.004, .001, [0.] * 15)[0] == 'H1_NOT_SUPPORTED'
