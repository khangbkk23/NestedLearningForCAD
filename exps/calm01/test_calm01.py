# exps/calm01/test_calm01.py
"""CALM-01 Phase 1 synthetic tests. No dataset, label, mask or result artifact is touched."""
from __future__ import annotations

import copy
import itertools
import math
import unittest

import numpy as np
import torch

from exps.calm01 import arms
from exps.calm01.calm import CALMMemory
from exps.calm01.closest_pair import ClosestPairMemory
from exps.calm01.storage import quantize_rows
from exps.task_p.p0_normal_support import distances, fingerprint
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1


def _rand(n, dim, seed, loc=0.0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (loc + scale * torch.randn(n, dim, generator=g)).float()


def _stream(memory, plan, dim, patches, seed=0):
    """plan: list of (category, images, loc, scale); calls begin/update/end like a task stream."""
    serial = 0
    for c, images, loc, scale in plan:
        memory.begin_category(c)
        for i in range(images):
            x = _rand(patches, dim, seed + 1000 * c + i, loc, scale)
            memory.update(x, list(range(serial, serial + patches))); serial += patches
        memory.end_category()
    return memory


class TestObjectiveRepair(unittest.TestCase):
    def test_old_utility_ignores_inflation_corrected_harm_does_not(self):
        D, before, after, w = 1.0, 1.0, 5.0, 1.0
        old_value = w * (max(D - before, 0) - max(D - after, 0))
        corrected = w * (max(after - D, 0) - max(before - D, 0))
        self.assertEqual(old_value, 0.0)          # CALM-00 utility: score 1 -> 5 is free
        self.assertEqual(corrected, 4.0)          # repaired deficit charges the inflation

    def test_every_massive_old_row_has_positive_eviction_cost(self):
        m = _stream(CALMMemory(20, storage="fp32", dim=8), [(0, 3, 0.0, 1.0), (1, 2, 4.0, 1.0)], 8, 9)
        m.begin_category(2)
        m.scales.observe(2, _rand(9, 8, 5, -4.0), m.bank)   # update() always observes first
        w = m._row_weights(2, 9.0)
        cost = w * m.mu * m.graph.d
        live = m.bank.occupied & (m.mu > 0) & torch.isfinite(m.graph.d)
        self.assertTrue(bool(live.any()))
        self.assertTrue(bool((cost[live] > 0).all()))


class TestQuantizer(unittest.TestCase):
    def test_matches_csg01_rule(self):
        from exps.csg01.quantize import encode_support
        x = _rand(40, 768, 3, scale=7.0)
        x[5] = 2.5                                  # constant row: step 0
        ours, ref = quantize_rows(x), encode_support([x])
        self.assertTrue(np.array_equal(ours["q"].numpy(), ref["q"]))
        self.assertTrue(np.array_equal(ours["xmin"].numpy(), ref["xmin"]))
        self.assertTrue(np.array_equal(ours["step"].numpy(), ref["step"]))
        self.assertTrue(np.array_equal(ours["decoded"].numpy(), ref["decoded"]))


class TestCadicControl(unittest.TestCase):
    def _batches(self):
        return [_rand(12, 16, 50 + b, loc=float(b % 3), scale=1.0 + 0.2 * b) for b in range(10)]

    def test_exact_parity_with_repository_cadic(self):
        ref = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=20, dim=16))
        ours = ClosestPairMemory(20, mode="cadic", storage="fp32", dim=16, distance="cadic_fp32")
        replaced = 0
        for b, x in enumerate(self._batches()):
            r = ref.update(x)
            s = ours.update(x, list(range(12 * b, 12 * b + 12)))
            self.assertEqual((r["accepted"], r["replaced"]), (s["inserted"], s["replaced"]))
            replaced += r["replaced"]
            n = ref.count                             # CADIC grows; ours has fixed slots
            self.assertEqual(int(ours.bank.occupied.sum()), n)
            self.assertTrue(torch.equal(ref.features, ours.bank.x[:n]))
        self.assertGreater(replaced, 10)             # the fixture exercises replacement

    def test_fp64_kernel_makes_identical_decisions_on_fixture(self):
        a = ClosestPairMemory(20, mode="cadic", storage="fp32", dim=16, distance="cadic_fp32")
        b = ClosestPairMemory(20, mode="cadic", storage="fp32", dim=16, distance="p0_fp64")
        for i, x in enumerate(self._batches()):
            a.update(x, list(range(i * 12, i * 12 + 12))); b.update(x, list(range(i * 12, i * 12 + 12)))
        self.assertTrue(torch.equal(a.bank.ids, b.bank.ids))

    def test_quota_respects_equal_shares(self):
        m = ClosestPairMemory(24, mode="quota", storage="u8", dim=8)
        _stream(m, [(0, 4, 0.0, 1.0), (1, 4, 3.0, 1.0), (2, 4, 6.0, 1.0)], 8, 16)
        self.assertEqual([m.count(k) for k in (0, 1)], [8, 8])
        self.assertEqual(int(m.bank.occupied.sum()), 24)


class TestCalmSwapExactness(unittest.TestCase):
    def _setup(self, seed=0):
        m = CALMMemory(7, storage="u8", dim=6, shortlist=100)
        _stream(m, [(0, 2, 0.0, 1.0 + 0.05 * seed)], 6, 9, seed=7 + seed)
        m.begin_category(1)
        m.update(_rand(9, 6, 77 + seed, 2.0, 1.5), list(range(100, 109)))
        x = _rand(9, 6, 500 + seed, 2.0 - 0.05 * seed, 1.5)
        m.scales.observe(1, x, m.bank)
        enc = m.bank.encode(x); Y = enc["decoded"]
        G = distances(x, Y)
        DI = torch.full((9, m.capacity), torch.inf, dtype=torch.float64)
        DI[:, m.bank.occupied] = distances(x, m.bank.x[m.bank.occupied])
        return m, x, Y, G, DI

    def _brute(self, m, Y, G, DI, c):
        base = m.objective(c, DI)
        protected = m._protected()
        best = (math.inf, None, None)
        for xi, j in itertools.product(range(len(Y)), range(m.capacity)):
            if bool(protected[j]):
                continue
            t = copy.deepcopy(m)
            k = int(t.bank.owner[j])
            survivors = [r for r in range(t.capacity) if r != j and int(t.bank.owner[r]) == k]
            dist = [float(torch.linalg.vector_norm(t.bank.x[j].double() - t.bank.x[r].double())) for r in survivors]
            to_new = float(torch.linalg.vector_norm(t.bank.x[j].double() - Y[xi].double()))
            mu, kap = float(t.mu[j]), float(t.kappa[j])
            if k == c and (not dist or to_new < min(dist)):
                carry = (mu, kap + mu * to_new)
            else:
                r = survivors[int(np.argmin(dist))]
                t.mu[r] += mu; t.kappa[r] += kap + mu * min(dist); carry = (0.0, 0.0)
            t.bank.x[j] = Y[xi]; t.bank.owner[j] = c
            t.mu[j], t.kappa[j] = carry
            DJ = DI.clone(); DJ[:, j] = G[:, xi]
            delta = t.objective(c, DJ) - base
            if delta < best[0] - 1e-12:
                best = (delta, xi, j)
        return best

    def test_best_swap_equals_brute_force_optimum(self):
        """The selected swap attains the brute-force optimal objective change. Both the joint
        (current-category row) and the separable (old-category row) paths must be exercised."""
        paths = set()
        for seed in range(12):
            m, x, Y, G, DI = self._setup(seed)
            delta, xi, j, _ = m.best_swap(DI, G, torch.zeros(9, dtype=torch.bool), 1, m._weight(1, 9), Y)
            ref = self._brute(m, Y, G, DI, 1)
            self.assertAlmostEqual(delta, ref[0], places=12)  # a tie may pick another optimal pair
            paths.add(int(m.bank.owner[j]) == 1)
        self.assertEqual(paths, {True, False})


class TestCalmInvariants(unittest.TestCase):
    PLAN = [(0, 4, 0.0, 0.4), (1, 4, 3.0, 1.5), (2, 4, -3.0, 1.0), (0, 2, 0.0, 0.4)]

    def _run(self, **kw):
        m = CALMMemory(30, storage="u8", dim=8, **kw)
        m.check_terms = True                          # incremental top-2 == full recompute
        return _stream(m, self.PLAN, 8, 16)

    def test_mass_conservation_floor_and_monotone_deficit(self):
        m = self._run()
        for k, images in ((0, 6), (1, 4), (2, 4)):
            self.assertAlmostEqual(float(m.mu[m.bank.owner == k].sum()), images * 16.0, places=9)
            self.assertAlmostEqual(m.mass[k], images * 16.0)
            self.assertGreaterEqual(m.count(k), m.floor)
        self.assertTrue(bool((m.kappa >= 0).all()))
        self.assertTrue(all(d >= -1e-12 for d in m.deficits().values()))

    def test_deterministic_replay(self):
        a, b = self._run(eta=2.0), self._run(eta=2.0)
        self.assertEqual(fingerprint({"x": a.bank.x, "mu": a.mu, "k": a.kappa}),
                         fingerprint({"x": b.bank.x, "mu": b.mu, "k": b.kappa}))

    def test_price_controller_invariants(self):
        m = CALMMemory(8, storage="fp32", dim=4, eta=1.0)
        m.categories, m.price, m.j_acq = [0, 1, 2], {0: 1.0, 1: 1.0, 2: 1.0}, {0: 0, 1: 0, 2: 0}
        m.deficits = lambda: {0: 0.6, 1: 0.0, 2: 0.0}
        m._price_step(1.0)
        self.assertGreater(m.price[0], m.price[1])
        self.assertEqual(m.price[1], m.price[2])
        self.assertTrue(all(.25 <= p <= 4 for p in m.price.values()))
        self.assertAlmostEqual(sum(m.price.values()) / 3, 1.0, places=12)
        m.deficits = lambda: {0: 50.0, 1: 0.0, 2: 0.0}
        for _ in range(5):
            m._price_step(1.0)
        self.assertTrue(all(.25 <= p <= 4 for p in m.price.values()))

    def test_clock_separation(self):
        fixed, nested = self._run(eta=0.0), self._run(eta=2.0)
        self.assertTrue(all(p == 1.0 for p in fixed.price.values()))
        self.assertTrue(all(e["event"] == "boundary" for e in nested.price_trace))
        self.assertNotEqual(nested.price[1], 1.0)
        same = self._run(clock="image", eta_image=0.05)
        self.assertTrue(any(e["event"] == "image" for e in same.price_trace))

    def test_returning_category_resumes(self):
        m = self._run()
        self.assertEqual(m.categories, [0, 1, 2])
        self.assertIn(0, m.j_acq)


class TestScaleDisparityMechanism(unittest.TestCase):
    """Synthetic mechanism fixture (not evidence on real data): a tight first category followed
    by a diffuse one. Closest-pair eviction is not scale-invariant and drains the tight category."""

    PLAN = [(0, 6, 0.0, 0.3), (1, 6, 2.0, 1.5)]

    def _count0(self, m):
        _stream(m, self.PLAN, 16, 16, seed=11)
        return m.count(0)

    def test_scale_aware_arms_retain_more_of_the_tight_category(self):
        cadic = self._count0(ClosestPairMemory(24, mode="cadic", storage="u8", dim=16))
        scaled = self._count0(ClosestPairMemory(24, mode="scaled", storage="u8", dim=16))
        calm = self._count0(CALMMemory(24, storage="u8", dim=16))
        self.assertLess(cadic, scaled)
        self.assertLess(cadic, calm)


class TestGraphIntegrity(unittest.TestCase):
    def test_incremental_graph_equals_rebuild_for_every_arm(self):
        plan = [(0, 3, 0.0, 0.4), (1, 3, 3.0, 1.5), (2, 2, -3.0, 1.0), (0, 1, 0.0, 0.4)]
        for name in arms.ARMS:
            m = arms.build(name, 40 * (8 + 8 + 1 + 16) + 32 * 3, 3, dim=8, n_ref=3)
            _stream(m, plan, 8, 16)
            d, i = m.graph.d.clone(), m.graph.i.clone()
            m.graph.d.fill_(torch.inf); m.graph.i.fill_(-1); m.graph.rebuild()
            # indices exact; values up to FP64 reduction order (d(a,b) row vs column, ~1e-16)
            self.assertTrue(torch.equal(i, m.graph.i), name)
            self.assertTrue(torch.allclose(d, m.graph.d, rtol=1e-12, atol=0), name)


class TestCudaCache(unittest.TestCase):
    """CADIC compares single distances: bitwise-identical decisions. CALM compares sums of hundreds of
    distances, so CPU/GPU reduction order (~1e-15) can flip a near-tie swap: require GPU determinism
    and equivalence of the tracked state instead."""

    PLAN = [(0, 3, 0.0, 0.4), (1, 3, 3.0, 1.5), (2, 2, -3.0, 1.0)]

    def _run(self, name, device):
        return _stream(arms.build(name, 40 * 793 + 96, 3, dim=64, cache_device=device), self.PLAN, 64, 49)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_fp64_cache(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        a, b = self._run("A_CADIC_FP32", "cpu"), self._run("A_CADIC_FP32", "cuda")
        self.assertTrue(torch.equal(a.bank.ids, b.bank.ids) and torch.equal(a.bank.x, b.bank.x))
        a, b, c = self._run("L2_NESTED_U8", "cpu"), self._run("L2_NESTED_U8", "cuda"), self._run("L2_NESTED_U8", "cuda")
        self.assertTrue(torch.equal(b.bank.ids, c.bank.ids) and torch.equal(b.mu, c.mu))     # deterministic
        shared = len(set(a.bank.ids.tolist()) & set(b.bank.ids.tolist())) / a.capacity
        self.assertGreaterEqual(shared, 0.99)
        for k in range(3):
            self.assertEqual(a.count(k), b.count(k))
            self.assertAlmostEqual(a.cost(k), b.cost(k), places=5)


class TestLedgerAndReadout(unittest.TestCase):
    def test_equal_bytes_capacities_and_charged_state(self):
        for name in arms.ARMS:
            for budget in arms.BUDGETS.values():
                k = arms.capacity_for_budget(name, budget, 15)
                self.assertLessEqual(arms.ledger(name, k, 15)["charged_persistent_bytes"], budget)
                self.assertGreater(arms.ledger(name, k + 1, 15)["charged_persistent_bytes"], budget)
        self.assertEqual(arms.capacity_for_budget("A_CADIC_FP32", arms.BUDGETS["cadic10000"], 15), 10_000)

    def test_serialised_state_matches_ledger(self):
        for name in arms.ARMS:
            m = arms.build(name, 30 * (8 + 8 + 1 + 16) + 32 * 3, 3, dim=8, n_ref=4)
            _stream(m, [(0, 2, 0.0, 1.0), (1, 2, 3.0, 1.0), (2, 1, 6.0, 1.0)], 8, 16)
            state = arms.persistent_state(m, name, 3)
            self.assertEqual(arms.state_bytes(state),
                             arms.ledger(name, m.capacity, 3, dim=8)["charged_persistent_bytes"], name)

    def test_export_is_raw_nn_bank_and_readout_is_read_only(self):
        from exps.calm.decompose import score_cell
        m = _stream(CALMMemory(30, storage="u8", dim=768, shortlist=2), [(0, 1, 0.0, 1.0)], 768, 784)
        before = fingerprint({"x": m.bank.x, "o": m.bank.owner, "mu": m.mu})
        cell = m.export()
        q = _rand(784, 768, 99)
        s = score_cell(cell, q)
        ref = distances(q, cell["bank"]).min(1).values
        # different query chunking changes BLAS reduction order only (CALM-00A: <= 2.03e-12)
        self.assertTrue(torch.allclose(s["d1"], ref, rtol=1e-11, atol=1e-9))
        self.assertEqual(before, fingerprint({"x": m.bank.x, "o": m.bank.owner, "mu": m.mu}))
        self.assertEqual(len(cell["bank"]), int(m.bank.occupied.sum()))


if __name__ == "__main__":
    unittest.main()
