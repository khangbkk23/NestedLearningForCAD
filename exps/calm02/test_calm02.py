# exps/calm02/test_calm02.py
"""CALM-02 synthetic tests. No dataset image, label, mask or result artifact is read."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from exps.calm01 import arms as A
from exps.calm01.test_calm01 import _rand, _stream
from exps.calm02.score import Scorer, decode, pixel_map
from exps.task_p.p0_normal_support import distances
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1


class TestDecode(unittest.TestCase):
    def test_charged_state_decodes_to_the_scored_bank(self):
        for name in ("A_CADIC_FP32", "L1_FIXED_U8", "L2_NESTED_U8"):
            m = A.build(name, 30 * (8 + 8 + 1 + 16) + 32 * 3, 3, dim=8)
            _stream(m, [(0, 2, 0.0, 1.0), (1, 2, 3.0, 1.0)], 8, 16)
            state = A.persistent_state(m, name, 3)
            self.assertTrue(torch.equal(decode(state), m.bank.x), name)


class TestScorer(unittest.TestCase):
    def setUp(self):
        self.bank = _rand(300, 768, 1, scale=2.0)
        self.q = _rand(784, 768, 2, scale=2.0)

    def test_patch_is_exact_fp64_nearest_neighbour(self):
        d, idx = Scorer(self.bank, "cpu").patch(self.q)
        ref = distances(self.q, self.bank)
        self.assertTrue(torch.equal(idx, ref.argmin(1)))
        self.assertTrue(torch.allclose(d, ref.min(1).values, rtol=1e-12, atol=1e-10))

    def test_matches_repository_cadic_scorer(self):
        cadic = CADICPatchCoresetV1(CADICPatchCoresetConfig(budget=300, dim=768, image_neighbors=9))
        cadic.update(self.bank)
        self.assertTrue(torch.equal(cadic.features, self.bank))
        image_ref, pixel_ref = cadic.score(self.q[None], b=9)
        sc = Scorer(self.bank, "cpu")
        d, idx = sc.patch(self.q)
        self.assertTrue(torch.allclose(d.float(), pixel_ref[0], rtol=1e-5))
        self.assertAlmostEqual(sc.image_cadic(self.q, d, idx), float(image_ref[0]), delta=1e-4 * float(image_ref[0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_neighbour_search_agrees_with_cpu(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        a, b = Scorer(self.bank, "cpu").patch(self.q), Scorer(self.bank, "cuda").patch(self.q)
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_pixel_map_is_harness_upsampling(self):
        d = torch.rand(784, dtype=torch.float64)
        ref = F.interpolate(d.float().reshape(1, 1, 28, 28), size=(224, 224), mode="bilinear",
                            align_corners=False)[0, 0]
        self.assertTrue(torch.equal(pixel_map(d), ref))


class TestNativeAP(unittest.TestCase):
    def _case(self, seed):
        import numpy as np
        g = np.random.default_rng(seed)
        hws = [(30, 41), (17, 17), (64, 50)]
        patches = [np.round(g.random(784) * 4, 1) for _ in hws]          # coarse values: many ties
        masks = [g.random(hw) > 0.8 for hw in hws]
        return patches, hws, masks

    def test_equals_sklearn_including_ties_across_chunk_boundaries(self):
        import numpy as np
        from unittest import mock
        from sklearn.metrics import average_precision_score
        from exps.calm02 import native
        from exps.task_p.p1_readout import native_map
        for seed in range(3):
            patches, hws, masks = self._case(seed)
            s = np.concatenate([native_map(torch.as_tensor(p), hw).ravel() for p, hw in zip(patches, hws)])
            y = np.concatenate([m.ravel() for m in masks])
            ref = average_precision_score(y, s)
            for chunk in (7, 1000, 1 << 24):
                with mock.patch.object(native, "CHUNK", chunk):
                    got = native.native_ap(patches, hws, masks)
                self.assertAlmostEqual(got["ap"], ref, places=12)
                self.assertEqual(got["n_total"], y.size)

    def test_hist_bins_cover_every_pixel(self):
        from exps.calm02 import native
        patches, hws, masks = self._case(9)
        pos, tot = native.native_hists(patches, hws, masks, bins=64)
        self.assertEqual([int(t) for t in tot.sum(1)], [h * w for h, w in hws])
        self.assertEqual([int(p) for p in pos.sum(1)], [int(m.sum()) for m in masks])


class TestGateRules(unittest.TestCase):
    def test_classification_follows_preregistration(self):
        from exps.calm02.gate import classify, gate_verdict
        self.assertEqual(classify(0.006, [0.001, 0.01], 0.005), "IMPROVEMENT")
        self.assertEqual(classify(0.006, [-0.001, 0.01], 0.005), "NO_DIFFERENCE_ESTABLISHED")   # CI spans 0
        self.assertEqual(classify(0.004, [0.001, 0.008], 0.005), "NO_DIFFERENCE_ESTABLISHED")   # below margin
        self.assertEqual(classify(-0.02, [-0.03, -0.01], 0.005), "DETERIORATION")
        self.assertEqual(classify(-0.02, [-0.03, -0.01], 0.005, lower_is_better=True), "IMPROVEMENT")  # forgetting
        self.assertEqual(classify(0.02, [0.01, 0.03], 0.005, lower_is_better=True), "DETERIORATION")
        n = "NO_DIFFERENCE_ESTABLISHED"
        base = {"macro_final_ap": n, "macro_forgetting": n, "pooled_final_ap": n, "macro_image_auroc": n}
        self.assertEqual(gate_verdict(base), "INCONCLUSIVE")
        self.assertEqual(gate_verdict(dict(base, macro_final_ap="IMPROVEMENT")), "POSITIVE_SIGNAL")
        self.assertEqual(gate_verdict(dict(base, macro_final_ap="IMPROVEMENT", macro_image_auroc="DETERIORATION")), "INCONCLUSIVE")
        self.assertEqual(gate_verdict(dict(base, macro_final_ap="IMPROVEMENT", macro_forgetting="DETERIORATION")), "NEGATIVE_SIGNAL")
        self.assertEqual(gate_verdict(dict(base, macro_forgetting="IMPROVEMENT")), "POSITIVE_SIGNAL")

    def test_bootstrap_weights_are_stratified_and_deterministic(self):
        import numpy as np
        from exps.calm02.gate import CATS, weights
        co = {c: [{"label": int(i >= 10)} for i in range(20)] for c in CATS}
        a, b = weights(co, 5, 770011), weights(co, 5, 770011)
        for wa, wb in zip(a, b):
            for c in CATS:
                self.assertTrue(np.array_equal(wa[c], wb[c]))
                self.assertEqual((wa[c][:10].sum(), wa[c][10:].sum()), (10, 10))


class TestPipelineEndToEnd(unittest.TestCase):
    """train -> score -> lock -> evaluate on a synthetic two-task benchmark in a temporary directory,
    so the one-shot evaluation code path is exercised before any real mask is read."""

    def test_synthetic_benchmark(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        import numpy as np
        from PIL import Image

        from exps.calm02 import run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, out = tmp / "data", tmp / "out"
            tasks = []
            for k, task in enumerate(("a", "b")):
                test = []
                (data / task).mkdir(parents=True, exist_ok=True)
                for i in range(6):
                    label = int(i >= 3)
                    mask = None
                    Image.fromarray(np.zeros((64, 48, 3), dtype=np.uint8)).save(data / f"{task}/test_{i}.png")
                    if label:
                        mask = f"{task}/mask_{i}.png"
                        m = np.zeros((64, 48), dtype=np.uint8); m[10 + 4 * i:30 + 4 * i, 20:40] = 255
                        Image.fromarray(m).save(data / mask)
                    test.append({"relative_path": f"{task}/test_{i}.png", "label": label, "mask_path": mask})
                train = [{"relative_path": f"{task}/train_{i}.png"} for i in range(3)]
                tasks.append({"task": task, "train": train, "test": test})
                (out / "features" / "synthetic").mkdir(parents=True, exist_ok=True)
                for split, rows in (("train", train), ("test", test)):
                    f = _rand(len(rows) * 784, 768, 300 + 10 * k + (split == "test"), loc=3.0 * k).reshape(len(rows), 784, 768)
                    if split == "test":
                        f[3:, 100:140] += 4.0                    # anomalous patches
                    torch.save({"features": f, "rows": rows}, out / "features" / "synthetic" / f"{task}_{split}.pt")
            (out / "features" / "synthetic" / "MANIFEST.json").write_text("{}")
            ids = {"benchmark": "synthetic", "root": str(data), "tasks": tasks}
            frozen = dict(run.FROZEN, bootstrap_reps=40)
            with mock.patch.object(run, "OUT", out), mock.patch.object(run, "FEATURES", out / "features"), \
                    mock.patch.object(run, "identities", lambda b: ids), \
                    mock.patch.object(run, "BUDGETS", {"B1": 120 * 3072}), mock.patch.object(run, "FROZEN", frozen):
                run.freeze("synthetic")
                for arm in run.ARMS:
                    run.train("synthetic", arm, "B1", 2, "cpu")
                    run.score("synthetic", arm, "B1", "cpu")
                run.lock("synthetic")
                with self.assertRaises(RuntimeError):
                    run.lock("synthetic")                    # create-exclusive
                run.evaluate("synthetic")
                with self.assertRaises(RuntimeError):
                    run.evaluate("synthetic")                # one-shot
            res = json.loads((out / "EVAL_synthetic.json").read_text())
            self.assertEqual(set(res["decisions"]), {"B1"})
            self.assertEqual(set(res["decisions"]["B1"]), {"final_ap", "forgetting", "slow_level_supported_on_this_benchmark"})
            self.assertNotIn("vs_cadic_paper", res)          # only for mvtec/visa
            for arm in run.ARMS:
                r = res["arms"][f"{arm}_B1"]
                self.assertTrue(0 < r["primary_native"]["macro_final_ap"] <= 1)
                self.assertEqual(set(r["primary_native"]["forgetting_per_task"]), {"a"})
                s2 = r["secondary_224"]
                self.assertTrue(0 < s2["macro_p_aupr"] <= 1 and 0 <= s2["macro_i_auroc"] <= 1)
                self.assertTrue(np.isnan(np.array(s2["p_aupr_matrix"], dtype=float)[0, 1]))   # future task unscored


if __name__ == "__main__":
    unittest.main()
