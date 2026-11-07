"""Focused tests for NL-SGM-01A-AM02 arm definitions (synthetic data only; no cache, probe or outcome access)."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from exps.csg01.select import random_priorities
from exps.nl_sgm_01a import am02 as A
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import rules as R
from exps.nl_sgm_01a.test_nl_sgm_01a import T2Causality, run, stream

torch.set_num_threads(6)


class AM1Bytes(unittest.TestCase):
    def test_capacities_and_bytes(self):
        self.assertEqual((A.C1_CAPACITY, A.PQ_CAPACITY), (2499, 75121))
        b = A.arm_bytes()
        for k in ("C1_FPS_FP32", "C2_FPS_PQ", "C3_HASH_PQ"):
            self.assertLessEqual(b[k], 7_680_000)
        self.assertGreater(A.C1_CAPACITY + 1, 7_680_000 // 3073)
        self.assertGreater((A.PQ_CAPACITY + 1) * 97 + K.CODEBOOK_BYTES, 7_680_000)    # maximal
        self.assertLess(b["C4_SAMEID_PQ"], b["C1_FPS_FP32"])

    def test_charged_state_matches_formula(self):
        arm = R.CodecArm(A.QuotaHashRule(500), "pq_fixed", budget=500 * 97 + K.CODEBOOK_BYTES)
        run(arm, stream(n_tasks=2, imgs=2, seed=3))
        self.assertEqual(R.state_bytes(arm.persistent_state()), 500 * 97 + K.CODEBOOK_BYTES)


class AM2QuotaHash(unittest.TestCase):
    def test_causal(self):
        T2Causality()._check(lambda: A.QuotaHashRule(300))
        T2Causality()._check(lambda: R.CodecArm(A.QuotaHashRule(400), "pq_fixed", budget=10**6))
        T2Causality()._check(lambda: R.CodecArm(R.FPSRule(400), "pq_fixed", budget=10**6))

    def test_balanced_and_requota_equals_global_bottom_q(self):
        data = stream(n_tasks=3, imgs=2, patches=300, seed=7)
        rule = A.QuotaHashRule(301)
        run(rule, data)
        quotas = R.AP.per_group_quota(301, 3)
        for k, q in enumerate(quotas):
            g = torch.cat([gg for _, gg in data[k]]).numpy()
            p = random_priorities(rule.seed, g.astype(np.int64))
            want = np.sort(g[np.lexsort((g, p))[:q]])
            self.assertTrue(np.array_equal(np.sort(rule.kept[k][1].numpy()), want))

    def test_fps_rule_category_balanced(self):
        rule = R.FPSRule(301)
        run(rule, stream(n_tasks=3, imgs=2, patches=300, seed=8))
        self.assertEqual([len(rule.kept[k][1]) for k in range(3)], R.AP.per_group_quota(301, 3))



# ====================================================================== AM02/v2 pre-LOCK tests (plan v2 §8)
from exps.nl_sgm_01a import am02_eval as E  # noqa: E402
from exps.nl_sgm_01a import am02_run as RUN  # noqa: E402


def _rnd(n, d=768, seed=0, loc=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) + loc


class AM3Metrics(unittest.TestCase):
    def test_zero_drift_when_support_unchanged(self):
        W = _rnd(3000, seed=1); Q = _rnd(2 * 784, seed=2).reshape(2, 784, 768)
        ids = torch.arange(3000)
        sup = E.rowset(W[:500], ids[:500])
        dA, _ = E.probe_nn(Q, sup); dF, _ = E.probe_nn(Q, sup)
        dW, _ = E.probe_nn(Q, E.rowset(W, ids))
        self.assertTrue(np.array_equal(E.retention(dA.numpy(), dF.numpy(), dW.numpy(), 1.7), np.zeros(2)))
        self.assertEqual(E.survival(ids[:500], ids[:500]), 1.0)

    def test_monotone_subset_and_distance(self):
        data = stream(n_tasks=3, imgs=2, patches=300, seed=11)
        probes = _rnd(2 * 300, seed=12).reshape(2, 300, 768)
        for make in (lambda: R.CodecArm(R.FPSRule(401), "pq_fixed", budget=10**6),
                     lambda: R.CodecArm(A.QuotaHashRule(401), "pq_fixed", budget=10**6),
                     lambda: R.CodecArm(R.FPSRule(401), "fp32", budget=10**7)):
            arm = make(); prev, prevd = None, None
            for t, task in enumerate(data):
                arm.begin_task(t)
                for x, g in task:
                    arm.image(x, g, t)
                arm.end_task(t)
                x, g, o = arm.support()
                cur = set(g[o == 0].tolist())
                d, _ = E.probe_nn(probes, E.by_owner(x, g, o)[0])
                if prev is not None:
                    self.assertTrue(cur <= prev)
                    self.assertTrue(bool((d >= prevd).all()))
                prev, prevd = cur, d

    def test_category_restriction_and_frozen_reference(self):
        x = _rnd(400, seed=3); g = torch.arange(400); o = torch.zeros(400, dtype=torch.int64)
        Q = _rnd(784, seed=4).reshape(1, 784, 768)
        d0, _ = E.probe_nn(Q, E.by_owner(x, g, o)[0])
        x2 = torch.cat([x, Q[0, :50].clone()]); g2 = torch.arange(450); o2 = torch.cat([o, torch.ones(50, dtype=torch.int64)])
        d1, _ = E.probe_nn(Q, E.by_owner(x2, g2, o2)[0])                  # poisoned nearer rows of category 1
        dg, _ = E.probe_nn(Q, E.rowset(x2, g2))
        self.assertTrue(torch.equal(d0, d1))
        self.assertTrue(bool((dg[0, :50] == 0).all()))

    def test_c4_identity_and_c0_sample_parity(self):
        self.assertTrue(np.array_equal(E.excess(np.ones((2, 5)), np.ones((2, 5)), 1.0), np.zeros(2)))
        data = stream(n_tasks=1, imgs=2, patches=784, seed=13)
        arm = R.CodecArm(R.FPSRule(300), "pq_fixed", budget=10**6)
        run(arm, data)
        x = torch.cat([a for a, _ in data[0]]); g = torch.cat([b for _, b in data[0]])
        self.assertTrue(torch.equal(K.fit(RUN.c0_sample(x, g), seed=20261106), arm.codebooks[0]))

    def test_spot_check(self):
        x = _rnd(5000, seed=5); q = _rnd(30, seed=6)
        self.assertTrue(E.spot_check(q, E.rowset(x, torch.arange(5000) * 2)))


class AM4Verdict(unittest.TestCase):
    @staticmethod
    def s(lo, hi, lo2=None, hi2=None):
        return {"primary": {"lo": lo, "hi": hi, "point": (lo + hi) / 2},
                "nested": {"lo": lo if lo2 is None else lo2, "hi": hi if hi2 is None else hi2, "point": 0}}

    def st(self, **kw):
        base = {"R_C1": self.s(0.1, 0.3), "A_C1": self.s(0.2, 0.4), "R_C2": self.s(0.05, 0.2),
                "A_C2": self.s(0.06, 0.2), "dR": self.s(-0.05, 0.2), "dA": self.s(-0.05, 0.2)}
        base.update(kw); return base

    def test_truth_table(self):
        s = self.s
        self.assertEqual(E.verdict(self.st(), blocked=True), "AM02_BLOCKED")
        self.assertEqual(E.verdict(self.st(R_C1=s(0, 0.01), A_C1=s(0, 0.04))), "NO_RETENTION_PROBLEM")
        self.assertEqual(E.verdict(self.st(R_C1=s(0, 0.01), A_C1=s(0, 0.04, hi2=0.06))), "HEADROOM_NOT_ESTABLISHED")
        hd = dict(dR=s(0.01, 0.2), dA=s(0.01, 0.2))
        self.assertEqual(E.verdict(self.st(**hd, R_C2=s(0, 0.01), A_C2=s(0, 0.04))), "CONVENTIONAL_PQ_SUFFICIENT")
        self.assertEqual(E.verdict(self.st(**hd, R_C2=s(0.03, 0.1))), "RESIDUAL_GAP_CONSOLIDATION_CANDIDATE")
        self.assertEqual(E.verdict(self.st(**hd, R_C2=s(0.03, 0.1, lo2=0.01), A_C2=s(0.01, 0.2))),
                         "HEADROOM_ESTABLISHED_RESIDUAL_INCONCLUSIVE")
        self.assertEqual(E.verdict(self.st(dR=s(0.01, 0.2, lo2=-0.01), dA=s(0.01, 0.2))), "HEADROOM_NOT_ESTABLISHED")
        self.assertEqual(E.verdict(self.st(dR=s(-0.01, 0.01), dA=s(-0.02, 0.04))), "NO_COMPRESSION_HEADROOM")
        self.assertEqual(E.verdict(self.st(dR=s(-0.01, 0.03), dA=s(-0.02, 0.04))), "HEADROOM_NOT_ESTABLISHED")

    def test_bootstrap_deterministic(self):
        v = np.arange(42, dtype=float) / 7; c = np.repeat(np.arange(7), 6)
        self.assertEqual(E.ci_both(v, c), E.ci_both(v, c))


class AM5AnalyzeSmoke(unittest.TestCase):
    """End-to-end smoke of am02_analyze on synthetic stage outputs (shapes as the runner writes them)."""

    def test_analyze_runs_on_synthetic_outputs(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock
        from exps.nl_sgm_01a import am02_analyze as AN
        from exps.nl_sgm_01a import common as C
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); st = out / "stages"
            for d in ("a0", "C1_FPS_FP32", "C2_FPS_PQ", "C3_HASH_PQ", "c4"):
                (st / d).mkdir(parents=True)
                (st / d / "STAGE_DONE.json").write_text(json.dumps({"wall_s": 10.0, "peak_rss_gb": 1.0, "device": "cpu",
                                                                     "threads": 6, "bytes": 1}))
            for k, c in enumerate(C.ORDER):
                torch.save({"sel_d": torch.tensor(rng.uniform(1, 2, (6, 784))),
                            "eval_d_all": torch.tensor(rng.uniform(1, 2, (90, 784))),
                            "eval_id_all": torch.tensor(rng.integers(0, 10**6, (90, 784)))}, st / "a0" / f"W_{k:02d}_{c}.pt")
            cb = torch.zeros(96, 256, 8, dtype=torch.float16)
            torch.save({"codebook": cb}, st / "c4" / "C0.pt")
            diag = {k: {"recon_norm_median": 0.5, "subspace_mse": torch.zeros(96), "codeword_entropy": torch.ones(96),
                        "frac_resid_gt_bottle_p99": 0.01, "rows": 10} for k in range(15)}
            torch.save(diag, st / "c4" / "codec_diagnostics.pt")
            for a, d in (("C1", "C1_FPS_FP32"), ("C2", "C2_FPS_PQ"), ("C3", "C3_HASH_PQ"), ("C4", "c4")):
                for t in range(15):
                    q = 40 - 2 * t
                    g = torch.cat([torch.arange(k * 1000, k * 1000 + q) for k in range(t + 1)])
                    o = torch.cat([torch.full((q,), k, dtype=torch.int8) for k in range(t + 1)])
                    ck = {"gids": g, "owner": o}
                    if a in ("C2", "C3"):
                        ck["codebooks"] = cb[None]
                    torch.save(ck, st / d / f"ckpt_t{t:02d}.pt")
                    res = {k: {"cat_d": torch.tensor(rng.uniform(1, 2, (6, 784)) + 0.01 * t),
                               "glob_d": torch.tensor(rng.uniform(1, 2, (6, 784))),
                               "glob_owner": torch.tensor(rng.integers(0, t + 1, (6, 784)))} for k in range(t + 1)}
                    torch.save(res, st / d / f"eval_t{t:02d}.pt")
            lockp = out / "LOCK.json"; lockp.write_text("{}")
            with mock.patch.object(AN, "OUT", out), mock.patch.object(AN, "LOCKP", lockp), \
                    mock.patch.object(AN.L, "verify_lock", lambda p: {}):
                AN.main()
            fv = json.loads((out / "FINAL_VERDICT.json").read_text())
            self.assertIn(fv["verdict"], ["NO_RETENTION_PROBLEM", "CONVENTIONAL_PQ_SUFFICIENT",
                                          "RESIDUAL_GAP_CONSOLIDATION_CANDIDATE", "HEADROOM_ESTABLISHED_RESIDUAL_INCONCLUSIVE",
                                          "NO_COMPRESSION_HEADROOM", "HEADROOM_NOT_ESTABLISHED"])
            for f in ("METRICS.json", "PER_CATEGORY_TABLE.csv", "MATCHED_CONTROL_RESULTS.csv", "RESOURCE_LEDGER.json"):
                self.assertTrue((out / f).exists())


if __name__ == "__main__":
    unittest.main()


class AM6PruneRegression(unittest.TestCase):
    """Run r1 failure: CodecArm._prune dropped codes of rows buffered by boundary rules (task > 4*cap+10000 rows)."""

    def test_prune_keeps_buffered_rows(self):
        data = stream(n_tasks=2, imgs=14, patches=784, seed=21)              # 10,976 rows/task > 4*50+10,000
        for make in (lambda: R.CodecArm(A.QuotaHashRule(50), "pq_fixed", budget=10**6),
                     lambda: R.CodecArm(R.FPSRule(50), "pq_fixed", budget=10**6)):
            arm = make()
            run(arm, data)
            st = arm.persistent_state()
            self.assertEqual(st["codes"].shape[0], 50)
