"""CALM-00A synthetic tests. No dataset, no mask, no P1 artifact is touched."""
from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.calm import certify, decompose, stats
from exps.task_p.p0_normal_support import distances


def _bank(n, seed=0, dim=768):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g, dtype=torch.float32)


class TestMonotonicity(unittest.TestCase):
    def test_addition_never_increases_and_removal_never_decreases(self):
        bank = _bank(9)
        q = _bank(5, seed=1)
        full = distances(q, bank).min(1).values
        sub = distances(q, bank[:6]).min(1).values
        self.assertTrue(bool((full <= sub + 1e-12).all()))
        bigger = distances(q, torch.cat((bank, _bank(3, seed=2)))).min(1).values
        self.assertTrue(bool((bigger <= full + 1e-12).all()))
        # exactness: a new row farther than the current NN changes nothing
        far = bank.max(0).values + 1e6
        same = distances(q, torch.cat((bank, far[None]))).min(1).values
        self.assertTrue(bool(torch.equal(same, full)))


class TestCells(unittest.TestCase):
    def test_set_algebra(self):
        acq = {"bank": _bank(3), "ids": torch.tensor([0, 1, 2]), "owners": torch.tensor([0, 0, 0], dtype=torch.int8)}
        fin = {"bank": _bank(4, seed=3), "ids": torch.tensor([1, 2, 3, 4]),
               "owners": torch.tensor([0, 0, 1, 1], dtype=torch.int8)}
        cells = decompose.build_cells(acq, fin, query_cat=0)
        self.assertEqual(cells["00"]["ids"].tolist(), [0, 1, 2])
        self.assertEqual(cells["10"]["ids"].tolist(), [1, 2])
        self.assertEqual(sorted(cells["01"]["ids"].tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(cells["11"]["ids"].tolist(), [1, 2, 3, 4])
        self.assertEqual(cells["oracle"]["ids"].tolist(), [1, 2])


class TestAp(unittest.TestCase):
    def test_pooled_ap_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(0)
        maps = [rng.integers(0, 5, size=(4, 5)).astype(np.float32) for _ in range(3)]
        masks = [rng.random((4, 5)) > 0.7 for _ in range(3)]
        masks[0][0, 0] = True
        got = stats.pooled_ap(maps, masks)["ap"]
        ref = average_precision_score(np.concatenate([m.ravel() for m in masks]),
                                      np.concatenate([m.ravel() for m in maps]))
        self.assertAlmostEqual(got, ref, places=12)

    def test_per_category_ap_is_monotone_invariant_pooled_is_not(self):
        rng = np.random.default_rng(1)
        maps = [rng.random((6, 6)).astype(np.float32) for _ in range(2)]
        masks = [np.zeros((6, 6), bool), np.zeros((6, 6), bool)]
        masks[0][0, :3] = True
        masks[1][3:, 3:] = True
        per_cat = stats.pooled_ap(maps[:1], masks[:1])["ap"]
        self.assertAlmostEqual(stats.pooled_ap([np.exp(maps[0])], masks[:1])["ap"], per_cat, places=10)
        self.assertAlmostEqual(stats.pooled_ap([maps[0] * 7.0], masks[:1])["ap"], per_cat, places=10)
        base = stats.pooled_ap(maps, masks)["ap"]
        t1 = [np.exp(maps[0]), maps[1]]
        self.assertNotAlmostEqual(stats.pooled_ap(t1, masks)["ap"], base, places=6)

    def test_binned_hist_approximates_exact_ap(self):
        rng = np.random.default_rng(2)
        maps = [rng.random((8, 8)).astype(np.float32) for _ in range(4)]
        masks = [rng.random((8, 8)) > 0.8 for _ in range(4)]
        masks[0][0, 0] = True
        exact = stats.pooled_ap(maps, masks)["ap"]
        h = stats.binned_hist(maps, masks)
        self.assertAlmostEqual(stats.ap_from_hist(h[0].sum(0), h[1].sum(0)), exact, delta=2e-3)


class TestShapley(unittest.TestCase):
    def test_identity_and_checksum(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            a = rng.random(4)
            s = stats.shapley(*a)
            self.assertAlmostEqual(s["shapley_sum"], s["total"], places=12)

    def test_interaction_matches_order_dependence(self):
        s = stats.shapley(0.50, 0.55, 0.55, 0.65)
        self.assertAlmostEqual(s["interaction"],
                               (0.65 - 0.55) - (0.55 - 0.50), places=12)
        self.assertNotEqual(s["interaction"], 0.0)

    def test_interaction_is_zero_for_additive_effects(self):
        s = stats.shapley(0.5, 0.6, 0.7, 0.8)  # both effects additive and equal
        self.assertAlmostEqual(s["interaction"], 0.0, places=12)


class TestBootstrap(unittest.TestCase):
    def test_strata_preserved_and_weights_integral(self):
        labels = np.array([0] * 5 + [1] * 5)
        ws = stats.bootstrap_weights(labels, 20, 7)
        for w in ws:
            self.assertEqual(w[:5].sum(), 5.0)
            self.assertEqual(w[5:].sum(), 5.0)
            self.assertTrue(np.all((w == 0) | (w >= 1)))

    def test_weighted_ap_matches_exact_at_unit_weights(self):
        rng = np.random.default_rng(4)
        maps = [rng.random((5, 5)).astype(np.float32) for _ in range(3)]
        masks = [rng.random((5, 5)) > 0.6 for _ in range(3)]
        masks[0][0, 0] = True
        h = stats.binned_hist(maps, masks)
        self.assertAlmostEqual(stats.weighted_ap(h, np.ones(3)),
                               stats.pooled_ap(maps, masks)["ap"], delta=2e-3)


class TestCertificate(unittest.TestCase):
    def test_vir_two_row_counterexample(self):
        # One defect patch sits unsupported close to a foreign row; normals sit inside their radius.
        d1 = np.array([[0.1], [5.0]])        # two images, one patch each: normal, defect
        owner1 = np.array([[1], [1]])        # both nearest to the foreign category (owner 1)
        radii = {0: 0.5, 1: 1.0}
        labels = np.array([0, 1])
        v = certify.violation(d1, owner1, query_cat=0, radii=radii)
        self.assertTrue(bool(v["foreign"].all()))
        self.assertFalse(bool(v["violating"][0, 0]))   # normal: 0.1 <= 1.0
        self.assertTrue(bool(v["violating"][1, 0]))    # defect: 5.0 > 1.0
        s = certify.violation_summary(v, d1, np.full_like(d1, 9.0), labels)
        self.assertEqual(s["affected_defect_patches"], 1)
        self.assertEqual(s["affected_normal_patches"], 0)
        self.assertEqual(s["vir"], float("inf"))

    def test_radius_is_normal_only_quantile(self):
        bank = _bank(4)
        feats = [_bank(784, seed=9)]
        r = certify.category_radius(bank, feats, q=0.5)
        d = distances(feats[0], bank).min(1).values
        self.assertAlmostEqual(r, float(torch.quantile(d, 0.5)), places=10)


if __name__ == "__main__":
    unittest.main()
