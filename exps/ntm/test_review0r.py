# exps/ntm/test_review0r.py
"""NTM-00R synthetic counterexamples: where NTM-floor helps, where it destroys ranking, where it is inert."""
from __future__ import annotations

import unittest

import numpy as np

from exps.ntm.review0r import fixture_aps
from exps.ntm.stage0 import combine


class TestScoreChangeCondition(unittest.TestCase):
    def test_change_iff_both_context_and_tau_below_bank(self):
        g = np.random.default_rng(0)
        dm, dx, tau = g.random(20000) * 4, g.random(20000) * 4, g.random(20000) * 4
        s = combine("NTM_FLOOR", dm, dx, tau)
        self.assertTrue(np.array_equal(s < dm, (dx < dm) & (tau < dm)))
        changed = s < dm
        self.assertTrue(np.allclose((dm - s)[changed], (dm - np.maximum(dx, tau))[changed]))


class TestMaskingFixtures(unittest.TestCase):
    def test_extended_and_repeated_defects_collapse_to_prevalence(self):
        for kind, prevalence in (("large_connected_defect", 100 / 784), ("repeated_distant_defects", 8 / 784)):
            r = fixture_aps(kind)
            self.assertEqual(r["bank_ap"], 1.0)
            self.assertAlmostEqual(r["ntm_floor_ap"], prevalence, places=6)   # every patch tied at tau
            self.assertEqual(r["defect_patches_capped_at_tau"], r["n_defect"])

    def test_small_unique_defect_with_evicted_normal_mode_is_repaired(self):
        r = fixture_aps("small_unique_defect_with_evicted_normal_mode")
        self.assertLess(r["bank_ap"], 0.5)
        self.assertEqual(r["ntm_floor_ap"], 1.0)

    def test_inert_cases(self):
        r = fixture_aps("regular_texture_adequate_bank")
        self.assertEqual((r["bank_ap"], r["ntm_floor_ap"]), (1.0, 1.0))
        r = fixture_aps("unique_normal_part")
        self.assertAlmostEqual(r["bank_ap"], r["ntm_floor_ap"], places=12)  # unique parts are never rescued


if __name__ == "__main__":
    unittest.main()
