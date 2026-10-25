# exps/ptag/test_ptag.py
"""CPU correctness tests for PTAG-01 (no detector artifacts, no masks)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import kendalltau

from exps.ptag import analysis as A
from exps.ptag import metrics as M
from exps.ptag import synth as Y
from exps.task_p.p2.metrics import exact_ranking


@pytest.mark.parametrize("levels,buckets", [(7, 8), (1000, 8), (None, 3), (None, 1)])
def test_exact_ap_matches_exact_ranking(levels, buckets):
    g = torch.Generator().manual_seed(levels or 0)
    v = torch.rand(50_000, generator=g)
    if levels:
        v = torch.floor(v * levels) / levels          # heavy ties, also across positive/negative
    y = torch.rand(50_000, generator=g) < 0.07
    assert abs(M.exact_ap(v, y, buckets=buckets) - exact_ranking(v, y)["pixel_ap"]) < 1e-12


def test_exact_ap_constant_scores():
    v, y = torch.ones(1000), torch.zeros(1000, dtype=torch.bool)
    y[:100] = True
    assert abs(M.exact_ap(v, y) - 0.1) < 1e-12


def test_weighted_quantile():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1001)
    assert M.weighted_quantile(x, np.ones_like(x), 0.99) == np.quantile(x, 0.99, method="inverted_cdf")
    w = rng.integers(0, 4, size=1001)
    assert M.weighted_quantile(x, w, 0.99) == np.quantile(np.repeat(x, w), 0.99, method="inverted_cdf")


def test_tau_b_ties_and_constants():
    rng = np.random.default_rng(2)
    x, y = rng.integers(0, 4, 40).astype(float), rng.integers(0, 5, 40).astype(float)
    assert abs(M.tau_b(x, y) - kendalltau(x, y, variant="b").statistic) < 1e-12
    assert abs(M.tau_b(x, y) - A.tau_b_naive(x, y)) < 1e-12
    assert M.tau_b(np.ones(5), np.arange(5)) == 0.0


def test_host_weights_cluster_resample():
    w = M.host_weights(0, 20)
    assert w.shape == (1000, 20) and (w.sum(1) == 20).all()
    assert (w == M.host_weights(0, 20)).all() and not (w == M.host_weights(1, 20)).all()


@pytest.mark.parametrize("hw", [(900, 900), (1024, 1024)])
def test_patch_mask_partitions_native_grid(hw):
    total = sum(Y.patch_mask([p], hw).sum() for p in range(784))
    assert total == hw[0] * hw[1]
    m = Y.patch_mask([0], hw)
    assert m[0, 0] and not m[hw[0] - 1, hw[1] - 1]


def test_blob_connected_and_sized():
    for s in range(200):
        b = Y.blob(np.random.default_rng(s))
        assert 2 <= len(b) <= 6 and len(set(b)) == len(b)
        reached, frontier = {b[0]}, [b[0]]
        while frontier:
            r, c = divmod(frontier.pop(), 28)
            for n in ((r - 1) * 28 + c, (r + 1) * 28 + c, r * 28 + c - 1, r * 28 + c + 1):
                if n in b and n not in reached and abs(n % 28 - c) <= 1:
                    reached.add(n); frontier.append(n)
        assert reached == set(b)


def test_q1_perturbs_only_blob():
    x = torch.randn(784, 768)
    y, m, info = Y.q1(x, (900, 900), 0.5, 0, 3, 1)
    changed = sorted(torch.nonzero((y != x).any(1)).flatten().tolist())
    assert changed == info["blob"] and m.any()
    assert torch.equal(Y.q1(x, (900, 900), 0.5, 0, 3, 1)[0], y)


def test_q3_cutpaste_mask_and_untouched_pixels():
    img = np.random.default_rng(0).integers(0, 256, (900, 900, 3), dtype=np.uint8)
    out, mask, info = Y.q3(img, 0, 2, 4)
    assert mask.any() and (out[~mask] == img[~mask]).all()
    ph, pw = info["size"]
    assert mask.sum() == ph * pw


def test_native_perlin_matches_repo_generator_on_1024():
    gen = Y.NativePerlin()
    ref = Y.PerlinAnomalyGenerator({"dtd_dir": str(Y.DTD)})
    np.random.seed(123); a = gen._perlin_mask(1024, 1024)
    np.random.seed(123); b = ref._perlin_mask(1024, 1024)
    assert np.array_equal(a, b)
    np.random.seed(5); m = gen._perlin_mask(900, 900)
    assert m.shape == (900, 900)


def test_q2_deterministic_and_nonempty():
    gen = Y.NativePerlin()
    img = np.random.default_rng(0).integers(0, 256, (900, 900, 3), dtype=np.uint8)
    a, ma, _ = Y.q2(gen, img, "bottle", 0, 1, 2)
    b, mb, _ = Y.q2(gen, img, "bottle", 0, 1, 2)
    assert np.array_equal(a, b) and np.array_equal(ma, mb) and ma.any() and a.dtype == np.uint8


def test_calm_leave_host_out_never_selects_host_rows():
    from exps.ptag.scoring import calm_scores
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(64, 768, generator=g)
    ids = (torch.arange(64) // 8) * 784 + torch.arange(64) % 8      # row r comes from image r // 8 (serial // 784)
    owners = torch.zeros(64, dtype=torch.int64)
    X = torch.stack([rows[0:8].repeat(98, 1), rows[8:16].repeat(98, 1)])       # each image copies its own rows
    d, _ = calm_scores(rows, owners, ids, X, np.array([0, 1]), [0, 1], "cpu")
    assert (d > 1.0).all()                                                      # exact self-matches excluded
    d_in, _ = calm_scores(rows, owners, ids, X, np.array([0, 1]), [-1, -1], "cpu")
    assert (d_in < 1e-6).all()


class _Stub:
    def __init__(self, real, prox):
        self.r, self.p = real, prox

    def real(self, c, b=None):
        return self.r[c]

    def proxy(self, c, p, b=None):
        return self.p[c]

    delta = A.Data.delta


def test_agreement_sign_rule():
    s = _Stub({"a": 0.5, "b": 0.6, "c": 0.4}, {"a": 1.0, "b": 2.0, "c": 1.0})
    cons = [("F", "x", "k", [("a", "b")]), ("F", "y", "k", [("a", "c")])]
    assert A.agreement(s, "Q1", cons) == [True, False]          # zero proxy delta counts as disagreement
