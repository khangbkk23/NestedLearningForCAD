# exps/ntm/test_stage0.py
"""NTM-00 Stage 0 synthetic tests. No dataset, label, mask or result artifact is read."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from exps.ntm.stage0 import combine, context_distance, context_mask, loo_bank_distances
from exps.task_p.p0_normal_support import distances


def _rand(n, d, seed):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


class TestContext(unittest.TestCase):
    def test_exclusion_window(self):
        allowed = context_mask(4)
        excluded = (~allowed).sum(1).reshape(28, 28)
        self.assertEqual(int(excluded[14, 14]), 81)        # interior: (2R+1)^2
        self.assertEqual(int(excluded[0, 0]), 25)          # corner: (R+1)^2
        self.assertTrue(bool((~allowed).diagonal().all()))  # a patch never explains itself
        self.assertTrue(torch.equal(allowed, allowed.T))

    def test_context_distance_is_brute_force(self):
        x = _rand(784, 16, 0)
        allowed = context_mask(4)
        got = context_distance(x, allowed)
        D = distances(x, x)
        ref = torch.stack([D[i][allowed[i]].min() for i in range(784)])
        self.assertTrue(torch.equal(got, ref))


class TestCombine(unittest.TestCase):
    def test_floor_rule_invariants(self):
        g = np.random.default_rng(1)
        dm, dx, tau = g.random(5000) * 10, g.random(5000) * 10, g.random(5000) * 10
        floor, mn = combine("NTM_FLOOR", dm, dx, tau), combine("NTM_MIN", dm, dx, tau)
        self.assertTrue(np.all(floor <= dm))                          # context can only lower a score
        self.assertTrue(np.all(floor >= np.minimum(dm, tau)))        # never below the category level
        self.assertTrue(np.array_equal(floor[dm <= tau], dm[dm <= tau]))   # inactive while the bank is adequate
        self.assertTrue(np.all(mn <= floor))
        self.assertTrue(np.array_equal(combine("BANK", dm, dx, tau), dm))
        self.assertTrue(np.array_equal(combine("CTX", dm, dx, tau), dx))


class TestLeaveOneImageOut(unittest.TestCase):
    def test_own_image_rows_are_excluded(self):
        feats = _rand(3 * 784, 8, 2).reshape(3, 784, 8)
        rows = torch.cat([feats[0, :50], feats[1, :50], _rand(40, 8, 3)])       # rows copied from images 10 and 11
        row_ids = torch.cat([torch.arange(50) + 10 * 784, torch.arange(50) + 11 * 784, torch.arange(40) + 99 * 784])
        got = loo_bank_distances(feats[:2], [10, 11], rows, row_ids, "cpu")
        for n, g in enumerate((10, 11)):
            keep = (row_ids // 784) != g
            ref = distances(feats[n], rows[keep]).min(1).values
            self.assertTrue(torch.allclose(got[n * 784:(n + 1) * 784], ref, rtol=1e-6, atol=1e-6))
        self.assertTrue(bool((got[:50] > 0).all()))       # own copies excluded: no zero distance


if __name__ == "__main__":
    unittest.main()
