"""NORM-00 tests (synthetic only; no dataset, feature or label file is read)."""
from __future__ import annotations

import numpy as np
import torch

from exps.norm_00 import run as N


def test_decision_rule():
    z = {c: 0.0 for c in N.WEAK}
    assert N.verdict(0.012, z) == "NORMALIZATION_MATTERS_GO"
    assert N.verdict(0.0, {**z, "screw": 0.06, "cable": 0.05, "zipper": 0.07}) == "NORMALIZATION_MATTERS_GO"
    assert N.verdict(0.002, {**z, "screw": 0.01}) == "NORMALIZATION_EXONERATED"
    assert N.verdict(-0.02, z) == "NORMALIZATION_HARMFUL"
    assert N.verdict(0.007, z) == "NORMALIZATION_INCONCLUSIVE"


def test_custody():
    ok = [N.FEAT / "screw_train.pt", N.FEAT / "screw_test.pt", N.MVTEC / "screw/train/good/000.png",
          N.MVTEC / "screw/test/thread_top/000.png", N.MVTEC / "screw/ground_truth/thread_top/000_mask.png",
          N.OUT / "RESULT.json", N.ROOT / "checkpoints/cadic/vit_base_patch8_224_augreg_in21k_state_dict.pth"]
    no = [N.ROOT / "data/visa/candle/x.png", N.ROOT / "results/hope_cad/rh01b/PROTOCOL.json",
          N.ROOT / "results/hope_cad/vitb8_nl00/runs/q2off_seed2711/stage05.pt"]
    assert all(N.decide(p) for p in ok) and not any(N.decide(p) for p in no)


def test_norms_differ_only_in_normalisation():
    assert set(N.NORMS) == {"IMN", "INC"} and N.NORMS["INC"] == ((0.5,) * 3, (0.5,) * 3)


def test_metrics_synthetic():
    r = np.random.default_rng(0)
    n = 12
    labels = np.array([0] * 6 + [1] * 6)
    dist = r.random((n, 784)) * 0.1
    masks = [np.zeros((224, 224), dtype=bool) for _ in range(n)]
    for i in range(6, 12):
        dist[i, 100:110] += 1.0
        masks[i][24:32, 128:208] = True
    m = N.metrics(dist, labels, masks, (224, 224))
    assert m["image_auroc"] == 1.0 and m["pixel_ap_native"] > 0.3 and m["pixel_ap_224"] > 0.3


def test_score_exact_small():
    from exps.nl_cad_pilot_00 import nnx  # noqa: F401
    r = torch.Generator().manual_seed(0)
    train = torch.randn(3, 784, 768, generator=r)
    test = torch.randn(2, 784, 768, generator=r)
    d = N.score(train, test, "cpu")
    ref = torch.cdist(test.reshape(-1, 768).double(), train.reshape(-1, 768).double()).min(1).values
    assert np.allclose(d.reshape(-1), ref.numpy(), rtol=1e-12)
