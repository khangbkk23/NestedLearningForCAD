"""SAND-FINAL focused tests (normal-only: no test feature or mask is opened here).

    python -m pytest exps/sand/test_sand.py -q
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from exps.cadfinal import metrics as MX
from exps.nl_sgm_01a import am03_eval as A3
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a import common as C
from exps.nl_sgm_01a import exact as X
from exps.nl_sgm_01a_c2dom import common as D2
from exps.nl_sgm_01a_c2dom import stream as ST
from exps.sand import common as S
from exps.sand import score as SC


def _toy(n=5, seed=0):
    r = np.random.default_rng(seed)
    return r.random((n, 784)).astype(np.float32) * 0.3, r.random((n, 784)) * 7.0


def test_calibrate_matches_numpy():
    a, d = _toy()
    cal = S.calibrate(a, d)
    assert cal["n_patches"] == a.size
    assert abs(cal["mu_theta"] - a.astype(np.float64).mean()) < 1e-15 and abs(cal["sd_m"] - d.std()) < 1e-15


def test_fuse_is_shifted_min_of_z_and_nonnegative():
    a, d = _toy(seed=1)
    cal = S.calibrate(*_toy(seed=2))
    g, frac = S.fuse(a, d, cal)
    zt, zm = S.zscores(a, d, cal)
    assert g.dtype == np.float32 and g.min() >= 0
    assert np.allclose(g, np.minimum(zt, zm) + S.shift(cal), atol=1e-5)
    assert abs(frac - (zt < zm).mean()) < 1e-12
    # zero residual and zero distance are the smallest possible scores and stay non-negative
    g0, _ = S.fuse(np.zeros((1, 784), np.float32), np.zeros((1, 784)), cal)
    assert g0.min() >= 0


def test_fusion_reduces_to_a_channel_when_the_other_is_far_above():
    a, d = _toy(seed=3)
    cal = S.calibrate(a, d)
    g, frac = S.fuse(a, d + 1e6, cal)                  # support channel never the minimum
    assert frac == 1.0 and (np.argsort(g.ravel(), kind="stable") == np.argsort(
        ((a.astype(np.float64) - cal["mu_theta"]) / cal["sd_theta"] + S.shift(cal)).astype(np.float32).ravel(),
        kind="stable")).all()
    g2, frac2 = S.fuse(a + 1e6, d, cal)                # decoder channel never the minimum
    assert frac2 == 0.0 and np.allclose(g2, (d - cal["mu_m"]) / cal["sd_m"] + S.shift(cal), atol=1e-4)


def test_readout_native_is_the_am03_chain():
    g = _toy(seed=4)[0]
    m, img = S.readout_native(g, (64, 80))
    ref = A3.native_maps(torch.from_numpy(g), (64, 80))
    assert np.array_equal(m, ref) and np.array_equal(img, ref.reshape(len(ref), -1).max(1))


def test_readout_dinomaly_reproduces_the_canonical_decoder_scorer():
    from exps.vitb8_q2 import common as V
    torch.manual_seed(0)
    en, de = torch.randn(3, 768, 28, 28), torch.randn(3, 768, 28, 28)
    s256, image = V.score([en], [de])
    grid = V.grid([en], [de]).numpy().reshape(3, 784)
    maps, img = S.readout_dinomaly(grid, (300, 280))
    assert np.allclose(img, image.numpy(), atol=1e-6)
    ref = np.stack([MX.to_native("dinomaly", x, (300, 280)) for x in s256.numpy()])
    assert np.allclose(maps, ref, atol=1e-6)


def test_exact_ap_is_invariant_to_the_fusion_shift_and_equals_sklearn():
    r = np.random.default_rng(5)
    maps = [r.random((40, 40)).astype(np.float32) for _ in range(4)]
    masks = [r.random((40, 40)) > 0.9 for _ in range(4)]
    ap = MX.exact_ap(maps, masks)["ap"]
    ref = average_precision_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([m.ravel() for m in maps]))
    assert abs(ap - ref) < 1e-9
    assert abs(MX.exact_ap([m * 2 for m in maps], masks)["ap"] - ap) < 1e-12           # exact in FP32: identical
    # an additive shift rounds in FP32 and can merge neighbouring scores into ties: bounded, far below any tolerance
    assert abs(MX.exact_ap([m * 2 + 1 for m in maps], masks)["ap"] - ap) < 1e-6


def test_gates():
    a1 = {"final_pixel_ap": 0.5312, "final_image_auroc": 0.9200, "acq_pixel_ap": 0.5585}
    a2 = {"final_pixel_ap": 0.5323, "final_image_auroc": 0.9089, "acq_pixel_ap": 0.5378}
    g = S.gate0(a1, a2, {"final_pixel_ap": 0.5424, "final_image_auroc": 0.9101})
    assert g["pass"] and g["verdict"] == "FUSION_SUPPORTED"
    assert abs(g["final_pixel_ap_gain_over_best_single"] - 0.0101) < 1e-12        # best single AP is A2
    assert abs(g["final_image_auroc_diff_to_best_single"] + 0.0099) < 1e-12       # best single AUROC is A1
    assert not S.gate0(a1, a2, {"final_pixel_ap": 0.5422, "final_image_auroc": 0.9300})["pass"]      # AP margin
    assert not S.gate0(a1, a2, {"final_pixel_ap": 0.5600, "final_image_auroc": 0.9099})["pass"]      # AUROC guard
    a3 = {"final_pixel_ap": 0.55, "acq_pixel_ap": 0.57, "final_image_auroc": 0.93}
    assert S.gate1(a3, {"final_pixel_ap": 0.5551, "acq_pixel_ap": 0.5601, "final_image_auroc": 0.9201})["pass"]
    assert not S.gate1(a3, {"final_pixel_ap": 0.5549, "acq_pixel_ap": 0.57, "final_image_auroc": 0.93})["pass"]
    assert not S.gate1(a3, {"final_pixel_ap": 0.57, "acq_pixel_ap": 0.5599, "final_image_auroc": 0.93})["pass"]
    assert not S.gate1(a3, {"final_pixel_ap": 0.57, "acq_pixel_ap": 0.57, "final_image_auroc": 0.9199})["pass"]


def test_support_loader_equals_the_c2dom_loader():
    a, b = SC.supports(S.MEM_DIR, 3), ST.supports_at(S.MEM_ARM, 3)
    assert list(a) == list(b)
    for c in a:
        assert torch.equal(a[c].ids, b[c].ids) and torch.equal(a[c].x, b[c].x)


def test_quantization_integrity_and_grid_addresses():
    """Stored codes are the category's SEL-pool codes of the same ids; ids address (SEL image, grid position)."""
    from exps.nl_sgm_01a import am02_run as R2
    co, n = D2.setup()
    part = D2.partition(co, n)
    t, c = 14, "grid"
    k = S.ORDER.index(c)
    ck = torch.load(S.MEM_DIR / f"ckpt_t{t:02d}.pt", weights_only=False)
    nr = int(ck["header"][1:].sum())
    m = ck["owner"][:nr].long() == k
    ids, codes = ck["ids"][:nr][m].long(), ck["codes"][:nr][m]
    cb = torch.load(D2.OUT / "codebooks" / f"{k:02d}_{c}.pt", weights_only=False)
    pos = torch.searchsorted(cb["sel_gids"], ids)
    assert torch.equal(cb["sel_gids"][pos], ids) and torch.equal(cb["sel_codes"][pos], codes)
    assert torch.equal(ck["codebooks"][k], cb["codebook"])
    img = (ids - R2.gid_base(n, c)) // 784
    assert set(img.tolist()) <= {j for j, _ in part[c]["sel"]}          # only SEL-pool images, never probes / FIT pool
    assert torch.equal(SC.supports(S.MEM_DIR, t, only=c)[c].x, K.decode(codes, cb["codebook"])[torch.argsort(ids)])


def test_exact_nn_equals_brute_force_fp64():
    rows = SC.supports(S.MEM_DIR, 14, only="bottle")["bottle"]
    F_ = C.Features()
    q = F_.images(C.cohorts()["bottle"]["sel"][:1]).reshape(-1, 768)[:200].clone()
    d, i = X.nn(q, rows)
    ref = torch.cdist(q.double(), rows.x.double()).min(1)
    assert torch.allclose(d, ref.values, rtol=0, atol=1e-9)
    assert torch.equal(rows.ids[ref.indices], i) or torch.allclose(
        (q.double() - rows.x[torch.searchsorted(rows.ids, i)].double()).norm(dim=1), ref.values, atol=1e-9)
