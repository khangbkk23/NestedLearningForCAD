"""NL-SGM-01A focused tests (CPU; synthetic fixtures plus identity lists; no probe or test feature is read).

T1 custody/disjointness  T2 causality  T3 certified exact NN  T4 prefix-min  T5 canonical adapter parity
T6 PQ determinism/ledger  T7 transport decomposition  T8 metric identities/self-exclusion  T9 scorer parity
T10 lock  T11 clustered statistics  T12 B6 provenance (no historical FP32 access)
"""
from __future__ import annotations

import io
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a import lock as L
from exps.nl_sgm_01a import metrics as MT
from exps.nl_sgm_01a import rules as R
from exps.nl_sgm_01a import stats as ST
from exps.task_p.p0_normal_support import Support, distances

torch.set_num_threads(6)


def rnd(n, d=768, seed=0, scale=1.0, loc=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * scale + loc


def stream(n_tasks=3, imgs=3, patches=784, d=768, seed=1):
    out = []
    gid = 0
    for t in range(n_tasks):
        task = []
        for j in range(imgs):
            x = rnd(patches, d, seed=seed + 100 * t + j, scale=1.0 + 0.3 * t, loc=2.0 * t)
            task.append((x, torch.arange(gid, gid + patches)))
            gid += patches
        out.append(task)
    return out


def run(arm, data, upto=None):
    snaps = []
    for t, task in enumerate(data):
        arm.begin_task(t)
        for x, g in task:
            arm.image(x, g, t)
        arm.end_task(t)
        d, g, o = arm.support() if hasattr(arm, "support") else arm.rows()
        snaps.append((d.clone(), g.clone()))
        if upto is not None and t == upto:
            break
    return snaps


class T1Custody(unittest.TestCase):
    def test_cohorts(self):
        co = C.cohorts()
        self.assertEqual(sum(len(v["writers"]) for v in co.values()), 2919)
        self.assertEqual(C.r15_digest()[:8], C.R15_DIGEST)
        for c, v in co.items():
            self.assertEqual((len(v["sel"]), len(v["eval"])), (6, 6))
            self.assertFalse(set(v["writers"]) & (set(v["sel"]) | set(v["eval"])))
            self.assertFalse(set(v["sel"]) & set(v["eval"]))
            self.assertTrue(all("/train/good/" in p for p in v["writers"] + v["sel"] + v["eval"]))

    def test_guard(self):
        for bad in ("bottle/test/broken/000.png", "bottle/ground_truth/x.png", "x/bottle_test.pt", "visa/candle"):
            with self.assertRaises(RuntimeError):
                C.guard_path(bad)


class T2Causality(unittest.TestCase):
    def _check(self, make):
        a, b = stream(seed=1), stream(seed=1)
        b[2] = [(torch.full_like(x, 1e3), g) for x, g in b[2]]          # poisoned future task
        sa, sb = run(make(), a), run(make(), b)
        for t in (0, 1):
            self.assertTrue(torch.equal(sa[t][0], sb[t][0]) and torch.equal(sa[t][1], sb[t][1]))
        self.assertFalse(torch.equal(sa[2][0], sb[2][0]))

    def test_rules(self):
        self._check(lambda: R.ClosestPairRule(300, "cadic"))
        self._check(lambda: R.FPSRule(300))
        self._check(lambda: R.CodecArm(R.ClosestPairRule(400, "cadic"), "pq_refit", budget=10**6))
        self._check(lambda: R.CodecArm(R.ClosestPairRule(400, "cadic"), "pq_append",
                                       budget=3 * K.CODEBOOK_BYTES + 300 * 97))


class T3Exact(unittest.TestCase):
    def _cmp(self, q, x, ids=None, exclude=None, **kw):
        ids = torch.arange(len(x)) * 3 + 1 if ids is None else ids
        d, i = X.nn(q, X.RowSet(x, ids), exclude=exclude, **kw)
        bd, bi = X.brute(q, x, ids, exclude=exclude)
        self.assertTrue(torch.equal(d, bd)); self.assertTrue(torch.equal(i, bi))

    def test_ties_and_hard_cases(self):
        x, q = rnd(2500, seed=3, scale=3), rnd(150, seed=4, scale=3)
        x[5] = q[0]; x[6] = q[0]                                         # exact duplicate tie
        x[7] = q[1] + 1e-7; x[8] = q[1] - 1e-7                          # near tie
        x[9] = q[2].clone(); x[9][0] = torch.nextafter(x[9][0], torch.tensor(1e9))   # 1-ulp neighbour
        x[100:120] *= 60.0                                               # large norm difference
        x[200] = q[3] * 1000; q[4] = x[200] + 1e-3                       # cancellation-sensitive (huge norms)
        self._cmp(q, x)
        self._cmp(q, x, r_chunk=611, q_chunk=7)                          # chunk-boundary invariance

    def test_self_exclusion(self):
        x = rnd(800, seed=5)
        ids = torch.arange(800)
        self._cmp(x[:60].clone(), x, ids=ids, exclude=ids[:60])

    def test_kth_self_excluded(self):
        x = rnd(1200, seed=21); x[7] = x[3]
        ids = torch.arange(1200)
        r = X.kth(x[:30].clone(), X.RowSet(x, ids), 16, exclude=ids[:30], r_chunk=500)
        D = (x[:30].double()[:, None] - x.double()[None]).square().sum(-1).sqrt()
        D[torch.arange(30), torch.arange(30)] = math.inf
        self.assertTrue(torch.equal(r, D.sort(1).values[:, 15]))

    def test_bound_holds(self):
        self.assertLessEqual(X.verify_screen_bound(rnd(64, seed=7, scale=20), rnd(4000, seed=8, scale=20)), 1.0)


class T4Prefix(unittest.TestCase):
    def test_prefix_equals_full(self):
        x, q = rnd(3000, seed=9), rnd(80, seed=10)
        x[2500] = x[10]                                                  # cross-block exact tie
        q[0] = x[10]
        ids = torch.arange(3000)
        full = X.nn(q, X.RowSet(x, ids))
        d, i = X.nn(q, X.RowSet(x[:1000], ids[:1000]))
        for a, b in ((1000, 2200), (2200, 3000)):
            d, i = X.merge(d, i, *X.nn(q, X.RowSet(x[a:b], ids[a:b])))
        self.assertTrue(torch.equal(d, full[0]) and torch.equal(i, full[1]))


class T5Parity(unittest.TestCase):
    def test_closest_pair_is_canonical(self):
        from exps.calm01.closest_pair import ClosestPairMemory
        data = stream(n_tasks=2, imgs=3, patches=64, d=16)
        for mode in ("cadic", "quota"):
            a = R.ClosestPairRule(40, mode, dim=16)
            b = ClosestPairMemory(40, mode=mode, storage="fp32", dim=16)
            for t, task in enumerate(data):
                a.begin_task(t); b.begin_category(t)
                for x, g in task:
                    a.image(x, g, t); b.update(x, g.tolist())
                a.end_task(t); b.end_category()
            self.assertTrue(torch.equal(a.mems[0].bank.ids, b.bank.ids))

    def test_partition_matches_ad02_memory(self):
        """Per-group CADIC decisions equal the canonical AD-02 instrumented memory (CADIC FP32 kernel)."""
        from exps.ad02.hope_cad_ad02_memory import InstrumentedNormalSupportMemory
        assign, fp = R.s4_assignment()
        ours = R.ClosestPairRule(64, "cadic", assignment=assign, dim=16, distance="cadic_fp32")
        ref = InstrumentedNormalSupportMemory(budget=64, grid=4, allocation="spatial", dim=16,
                                              assignment=assign, device="cpu")
        for j in range(6):
            x = rnd(784, 16, seed=40 + j, scale=1 + 0.1 * j)
            ours.image(x, torch.arange(j * 784, j * 784 + 784), 0)
            ref.update(x, task_id=0, image_index=j, global_step=j)
        for g, m in enumerate(ours.mems):
            refb = ref._banks[g]
            n = refb.count
            mine = m.bank.x[m.bank.occupied]
            self.assertEqual(int(m.bank.occupied.sum()), n)
            self.assertTrue(torch.equal(torch.sort(mine, 0).values, torch.sort(refb.features[:n], 0).values))

    def test_partitions_reproducible(self):
        a1, f1 = R.r4_assignment(0); a2, f2 = R.r4_assignment(0)
        self.assertEqual(f1, f2); self.assertTrue(torch.equal(a1, a2))
        self.assertEqual(sorted(torch.bincount(a1).tolist()), [49] * 16)
        s, _ = R.s4_assignment(); self.assertEqual(sorted(torch.bincount(s).tolist()), [49] * 16)
        self.assertEqual(R.AP.per_group_quota(2500, 16)[:5], [157, 157, 157, 157, 156])

    def test_greedy_is_canonical(self):
        data = stream(n_tasks=1, imgs=4, patches=50, d=768)[0]
        a, b = R.GreedyRule(60), Support("STREAM_GREEDY", capacity=60)
        a.begin_task(0)
        for x, g in data:
            a.image(x, g, 0); b.update(x, g)
        self.assertTrue(torch.equal(a.s.ids, b.ids))

    def test_chunked_graph_rebuild(self):
        from exps.calm01.closest_pair import ClosestPairMemory
        m = ClosestPairMemory(300, mode="cadic", storage="fp32", dim=16)
        m.begin_category(0)
        x = rnd(300, 16, seed=31); x[5] = x[9]
        m.update(x, list(range(300)))
        m.graph.rebuild()                                                # canonical (unchunked) reference
        d0, i0 = m.graph.d.clone(), m.graph.i.clone()
        m.graph.d.fill_(0); m.graph.i.fill_(-7)
        R.rebuild_graph(m.graph, chunk=37)
        # identical neighbours; distances may differ by FP64 GEMM blocking (1 ulp), as in canonical block recomputes
        self.assertTrue(torch.equal(m.graph.i, i0))
        self.assertLessEqual(float((m.graph.d - d0).abs().max()), 1e-12)

    def test_calm_l1_capacity(self):
        self.assertEqual(R.CALMRule().capacity, 9684)


class T6PQ(unittest.TestCase):
    def test_fit_encode_deterministic_and_ledger(self):
        x = rnd(3000, seed=11, scale=2)
        cb1, cb2 = K.fit(x, iters=5), K.fit(x, iters=5)
        self.assertTrue(torch.equal(cb1, cb2)); self.assertEqual(cb1.dtype, torch.float16)
        self.assertEqual(cb1.numel() * 2, K.CODEBOOK_BYTES)
        c = K.encode(x[:500], cb1)
        self.assertTrue(torch.equal(c, K.encode(x[:500], cb1)))
        dec = K.decode(c, cb1)
        self.assertTrue(torch.equal(K.encode(dec, cb1), c))             # centroids re-encode to themselves
        self.assertEqual(K.pq_capacity(C.B1, 1), 75121); self.assertEqual(K.pq_capacity(C.B1, 15), 18368)
        self.assertEqual(K.u8_capacity(C.B1), 9884)

    def test_arm_state_bytes(self):
        for kind, cap in (("pq_fixed", 1500), ("u8", 300), ("fp32", 300)):
            arm = R.CodecArm(R.ClosestPairRule(cap, "cadic"), kind, budget=10**7)
            run(arm, stream(n_tasks=2, imgs=2))
            st = arm.persistent_state()
            row = {"pq_fixed": K.PQ_ROW_BYTES, "u8": K.U8_ROW_BYTES, "fp32": 3072}[kind] + 1
            extra = K.CODEBOOK_BYTES if kind.startswith("pq") else 0
            self.assertEqual(R.state_bytes(st), cap * row + extra)
            buf = io.BytesIO(); torch.save(st, buf)
            self.assertLess(buf.tell() - R.state_bytes(st), 4096)      # serialization overhead only

    def test_append_shrinks_and_versions(self):
        budget = 3 * K.CODEBOOK_BYTES + 2000 * 97
        arm = R.CodecArm(R.ClosestPairRule(K.pq_capacity(budget, 1), "cadic"), "pq_append", budget=budget)
        run(arm, stream(n_tasks=3, imgs=2))
        self.assertEqual(arm.inner.capacity, K.pq_capacity(budget, 3))
        self.assertLessEqual(R.state_bytes(arm.persistent_state()), budget)
        self.assertEqual(sorted(arm.codebooks), [0, 1, 2])


class T7Transport(unittest.TestCase):
    def test_decomposition(self):
        x = rnd(2000, seed=12, scale=2)
        a, b = K.fit(x[:1000], iters=3), K.fit(x[1000:], iters=3, seed=7)
        codes = K.encode(x[:300], a)
        new, rec = K.transport(codes, a, b)
        self.assertTrue(torch.allclose(rec["step_sq"], rec["move_sq"] + rec["reassign_sq"], atol=1e-9))
        old = K.decode(codes, a)
        self.assertTrue(torch.allclose(rec["step_sq"], (K.decode(new, b) - old).double().square().sum(1), atol=1e-6))
        self.assertTrue(torch.allclose(rec["frozen_code_drift"], (K.decode(codes, b) - old).double().norm(dim=1)))


class T8Metrics(unittest.TestCase):
    def test_identities(self):
        W = rnd(1500, seed=13); ids = torch.arange(1500)
        q = rnd(100, seed=14)
        S = ids[::7]
        cb = K.fit(W, iters=3)
        dW, _ = X.nn(q, X.RowSet(W, ids)); dS, _ = X.nn(q, X.RowSet(W[S], S))
        dQ, _ = X.nn(q, MT.rowset(K.decode(K.encode(W[S], cb), cb), S))
        parts = MT.decompose(dW, dS, dQ)
        MT.check_identities(parts)
        self.assertTrue(bool((parts["E_sel"] >= 0).all()))

    def test_self_excluded_scale(self):
        W = rnd(500, seed=15); ids = torch.arange(500)
        W[1] = W[0]                                                       # exact duplicate pair
        q50, n, deg = MT.scale_q50(W[:50], ids[:50], X.RowSet(W, ids))
        d_ex, _ = X.nn(W[:50], X.RowSet(W, ids), exclude=ids[:50])
        expect = d_ex[d_ex > 0].median()
        self.assertEqual(q50, float(expect)); self.assertEqual(n, 48); self.assertFalse(deg)
        z = torch.zeros(10, 768); _, _, deg = MT.scale_q50(z[:3], torch.arange(3), X.RowSet(z, torch.arange(10)))
        self.assertTrue(deg)


class T9Scorer(unittest.TestCase):
    def test_reference_matches_p0(self):
        from exps.task_p.p1_readout import native_map
        x, q = rnd(3000, seed=16, scale=2.5), rnd(784, seed=17, scale=2.5)
        d, i = X.nn(q, X.RowSet(x, torch.arange(3000)))
        P = distances(q, x)
        v, j = P.min(1)
        self.assertLess(float((v - d).abs().max()), 1e-9)
        top2 = P.topk(2, largest=False).values
        clear = (top2[:, 1] - top2[:, 0]) > 1e-9
        self.assertTrue(torch.equal(i[clear], j[clear]))
        m = native_map(d, (900, 900))
        self.assertEqual(m.shape, (900, 900))
        self.assertLessEqual(float(m.max()), float(d.max()) + 1e-5)


class T10Lock(unittest.TestCase):
    def test_exclusive_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.txt").write_text("x")
            p = Path(td) / "LOCK.json"
            L.write_lock(p, {"k": 1}, root=td, sources=["a.txt"])
            with self.assertRaises(FileExistsError):
                L.write_lock(p, {"k": 2}, root=td, sources=["a.txt"])
            L.verify_lock(p, root=td)
            (Path(td) / "a.txt").write_text("y")
            with self.assertRaises(RuntimeError):
                L.verify_lock(p, root=td)


class T11Stats(unittest.TestCase):
    def test_cluster_bootstrap(self):
        cats = np.repeat(np.arange(15), 6)
        diff = np.repeat(np.linspace(-1, 1, 15), 6)
        a = ST.clustered_ci(diff, cats, b=300); b = ST.clustered_ci(diff, cats, b=300)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["point"], 0.0)
        self.assertLess(a["lo"], 0); self.assertGreater(a["hi"], 0)
        const = ST.clustered_ci(np.full(90, 0.3), cats, b=200, nested=True)
        self.assertAlmostEqual(const["lo"], 0.3); self.assertAlmostEqual(const["hi"], 0.3)
        h = ST.relative_harm([1.0, 0.002], [0.5, 0.0], tau=0.1)
        self.assertTrue(np.isfinite(h).all()); self.assertAlmostEqual(h[1], 0.02)


class T12Provenance(unittest.TestCase):
    def test_refit_reads_only_decoded_and_current_task(self):
        seen = []
        orig = K.fit

        def spy(sample, **kw):
            seen.append(sample.clone()); return orig(sample, **kw)
        K.fit = spy
        try:
            data = stream(n_tasks=3, imgs=2)
            arm = R.CodecArm(R.ClosestPairRule(400, "cadic"), "pq_refit", budget=10**6)
            for t, task in enumerate(data):
                arm.begin_task(t)
                for x, g in task:
                    arm.image(x, g, t)
                if t >= 1:
                    dec_before, _, _ = arm.inner.rows()
                arm.end_task(t)
                if t >= 1:
                    fitrows = seen[-1]
                    cur = torch.cat([x for x, _ in task])
                    allowed = torch.cat([dec_before, cur])
                    hit = (torch.cdist(fitrows, allowed, compute_mode="donot_use_mm_for_euclid_dist").min(1).values == 0)
                    self.assertTrue(bool(hit.all()))
                    hist = torch.cat([x for x, _ in data[t - 1]])
                    raw_hist = (torch.cdist(fitrows, hist, compute_mode="donot_use_mm_for_euclid_dist").min(1).values == 0)
                    self.assertFalse(bool(raw_hist.any()))                  # no historical FP32 source row
            self.assertEqual(len(arm.drift), 2)
        finally:
            K.fit = orig


if __name__ == "__main__":
    unittest.main()
