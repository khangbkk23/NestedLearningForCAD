"""CPU tests for NL-ALIGN-02-MVTec evaluation (no GPU, no MVTec label)."""
import hashlib

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from exps.cadfinal import metrics as M
from exps.dinomaly import readout as R
from exps.nl_align_02 import m


def test_exact_ap_ties_match_sklearn():
    rng = np.random.default_rng(0)
    maps = [np.round(rng.random((7, 9)).astype(np.float32), 1) for _ in range(5)]   # heavy ties
    masks = [rng.random((7, 9)) < 0.3 for _ in range(5)]
    ref = average_precision_score(np.concatenate([y.ravel() for y in masks]), np.concatenate([x.ravel() for x in maps]))
    assert M.exact_ap(maps, masks)["ap"] == pytest.approx(ref, abs=1e-12)


def test_degenerate_cohorts_rejected():
    with pytest.warns(Warning):                                # this sklearn returns nan (not an error) ...
        assert np.isnan(roc_auc_score(np.zeros(5), np.arange(5.)))
    with pytest.raises(Exception):                             # ... so m.evaluate guards it explicitly
        m.require(0 < np.zeros(5).sum() < 5, "invalid AUROC cohort")
    with pytest.raises(Exception):
        M.exact_ap([np.ones((2, 2), np.float32)], [np.zeros((2, 2), bool)])   # no positive pixel


def test_mask_alignment_enforced():
    with pytest.raises(Exception):
        M.exact_ap([np.ones((4, 4), np.float32)], [np.ones((4, 5), bool)])


def test_native_upsampling_not_mask_downsampling():
    x = torch.rand(256, 256)
    out = R.native(x, (900, 700))
    assert tuple(out.shape) == (900, 700)


def test_image_score_is_top1pct_mean_of_map():
    torch.manual_seed(0)
    en = [torch.randn(2, 16, 28, 28) for _ in range(2)]
    de = [torch.randn(2, 16, 28, 28) for _ in range(2)]
    s, im = R.dinomaly(en, de)
    flat = s.flatten(1)
    k = int(flat.shape[1] * 0.01)
    assert torch.allclose(im, flat.sort(1, descending=True)[0][:, :k].mean(1))
    assert tuple(s.shape) == (2, 256, 256)


def test_pooled_hist_ap_close_to_exact():
    rng = np.random.default_rng(1)
    maps = [rng.random((30, 30)).astype(np.float32) for _ in range(6)]
    masks = [rng.random((30, 30)) < 0.1 for _ in range(6)]
    pp, pt = np.zeros(65536), np.zeros(65536)
    for x, y in zip(maps, masks):
        a, b = M.hist(x, y, 0.0, 2.0, 65536)
        pp += a; pt += b
    assert M.ap_from_hist(pp, pt) == pytest.approx(M.exact_ap(maps, masks)["ap"], abs=2e-3)


def test_acquired_sets_never_include_future_tasks():
    order = list(m.K.ORDERS["mvtec"])
    for s in range(1, 16):
        assert m.acquired(s) == order[:s]
    for a, s, c in m.cells(2711):
        assert order.index(c) < s, (a, s, c)                       # no pre-acquisition / future cell
    assert ("FS_B", 7, "metal_nut") not in m.cells(2711)
    assert len(m.cells(2711)) == len(set(m.cells(2711)))         # no duplicates
    with pytest.raises(Exception):
        m.acquired(16)


def _fake(names):
    """Synthetic metric dicts with known structure: AP = base(c) - 0.01 * (s - acq) for every arm."""
    ap, iau, q90, q99, boot, pool, acq = {}, {}, {}, {}, {}, {}, {}
    for sd in m.SEEDS:
        for a, s, c in m.cells(sd):
            j = names.index(c) + 1
            v = 0.5 + 0.01 * j - 0.01 * (s - j) + (0.002 if a == "FS_B" else 0.0)
            ap[(a, sd, s, c)] = v
            iau[(a, sd, s, c)] = 0.9
            q90[(a, sd, s, c)], q99[(a, sd, s, c)] = 0.1, 0.2
            boot[(a, sd, s, c)] = np.full(m.REPS, v)
            pool[(a, sd, s)] = [np.array([1., 0.]), np.array([2., 2.])]
        for c in names[:m.START]:
            j = names.index(c) + 1
            acq[(sd, c)] = 0.5 + 0.01 * j
    return ap, dict(ap), iau, q90, q99, boot, pool, acq


def test_summary_macro_forgetting_and_pairing():
    names = list(m.K.ORDERS["mvtec"][:9])
    out = m._summarize(*_fake(names), names)
    # macro over the acquired set only
    exp7 = np.mean([0.5 + 0.01 * j - 0.01 * (7 - j) for j in range(1, 8)])
    assert out["per_stage"]["ID|7"]["macro_ap"] == pytest.approx(exp7)
    assert out["per_stage"]["ID|7"]["macro_ap_per_seed"] == pytest.approx([exp7] * 3)
    # paired contrast: constant +0.002, zero-width CI
    d = out["contrasts"]["FS_B-ID"]
    assert d["delta"] == pytest.approx(0.002) and d["ci95"] == pytest.approx([0.002, 0.002])
    # forgetting F_{c,s} = AP_acq - AP_s = 0.01 (s - a(c)); none at or before acquisition
    f = out["forgetting"]["ID"]["cells"]
    assert f["bottle|9"]["F"] == pytest.approx(0.08)
    assert "pill|9" not in f and "metal_nut|8" not in f and "leather|7" not in f
    assert f["leather|8"]["acq_source"].startswith("ID own stage 7")


def test_checkpoint_not_mutated_by_loading(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"state": {"w": torch.randn(3)}}, p)
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    st = torch.load(p, weights_only=False)["state"]
    st["w"].add_(1.0)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h
