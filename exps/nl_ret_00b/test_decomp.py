"""Focused mechanical tests for NL-RET-00B (CPU). Mechanical correctness only - no scientific claim."""
import json

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from exps.nl_ret_00b import core as X


def _sk(p, n):
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    return average_precision_score(y, np.r_[p, n])


def test_ap_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.integers(0, 20, 50).astype(np.float32) / 7
        n = rng.integers(0, 20, 400).astype(np.float32) / 7          # heavy ties across classes
        a, b = np.sort(n[:150]), np.sort(n[150:])
        assert abs(X.ap_pn(p, [a, b]) - _sk(p, n)) < 1e-12


def test_constant_maps_and_empty_parts():
    p = np.full(10, 0.3, np.float32); n = np.full(90, 0.3, np.float32)
    assert abs(X.ap_pn(p, [np.sort(n), np.empty(0, np.float32)]) - 0.1) < 1e-12   # all tied -> prevalence
    with pytest.raises(ValueError):
        X.ap_pn(np.empty(0, np.float32), [np.sort(n)])


def test_factorial_cells_and_shapley_identity():
    rng = np.random.default_rng(1)
    pa, pf = rng.normal(1, .3, 300).astype(np.float32), rng.normal(.8, .3, 300).astype(np.float32)
    na, nf = np.sort(rng.normal(0, .3, 5000).astype(np.float32)), np.sort(rng.normal(.1, .35, 5000).astype(np.float32))
    aa, ff, fa, af = X.ap_pn(pa, [na]), X.ap_pn(pf, [nf]), X.ap_pn(pf, [na]), X.ap_pn(pa, [nf])
    assert abs(fa - _sk(pf, na)) < 1e-12 and abs(af - _sk(pa, nf)) < 1e-12
    d, dp, dn = X.shapley(aa, fa, af, ff)
    assert abs(dp + dn - d) < 1e-14 and d == aa - ff
    # only positives move -> D_N == 0 exactly
    d2, dp2, dn2 = X.shapley(aa, fa, aa, fa)
    assert dn2 == 0 and abs(dp2 - d2) < 1e-15


def test_order_stat_and_quantiles_union():
    rng = np.random.default_rng(2)
    a, b = np.sort(rng.random(1000).astype(np.float32)), np.sort(rng.random(777).astype(np.float32) * 2)
    u = np.sort(np.r_[a, b])
    for k in (0, 5, 900, 1776):
        assert X.order_stat([a, b], k) == float(u[k])
    q = X.quantiles([a, b], (0.5, 0.99))
    assert q[0.99] == float(u[int(np.floor(0.99 * (u.size - 1)))])
    assert X.count_ge([a, b], u[1000]) == int((u >= u[1000]).sum())


def test_pct_rank_preserves_within_image_order_and_within_ap():
    rng = np.random.default_rng(3)
    img = rng.random((40, 40)).astype(np.float32); img[5:9, 5:9] += 0.5; img[0, :10] = 0.2   # ties
    m = np.zeros_like(img, bool); m[5:9, 5:9] = True
    r = X.pct_rank(img)
    i, j = np.unravel_index(rng.integers(0, 1600, 500), img.shape), np.unravel_index(rng.integers(0, 1600, 500), img.shape)
    assert np.all((img[i] < img[j]) == (r[i] < r[j])) and np.all((img[i] == img[j]) == (r[i] == r[j]))
    assert abs(X.ap_pn(img[m], [np.sort(img[~m])]) - X.ap_pn(r[m], [np.sort(r[~m])])) < 1e-12


def test_streaming_state_vs_in_memory_and_normal_only_images(tmp_path, monkeypatch):
    """decomp.state on a synthetic fixture: pixel partition, identity check, no within-AP for good images, and
    streaming AP equals sklearn on the concatenated native arrays."""
    from exps.cadfinal import metrics as M
    from exps.nl_ret_00b import decomp as D
    rng = np.random.default_rng(4)
    rows, hws, masks = [], [], []
    for k in range(6):
        lab = int(k >= 2)
        hw = (48 + 8 * k, 40)
        m = np.zeros(hw, bool)
        if lab:
            m[10:20, 5:15 + k] = True
        rows.append({"relative_path": f"x/test/{k}.png", "label": lab}); hws.append(hw); masks.append(m)
    s = torch.rand(6, 256, 256) * 0.3
    s[2:, 60:120, 40:110] += 0.4
    p = tmp_path / "f.pt"
    torch.save({"paths": [r["relative_path"] for r in rows], "s256": s, "image_RD": torch.zeros(6)}, p)
    P, NG, NB, img = D.state(p, rows, hws, masks)
    maps = [M.to_native("dinomaly", s[i].numpy(), hw) for i, hw in enumerate(hws)]
    assert P.size == sum(m.sum() for m in masks)
    assert NG.size == sum(m.size for r, m in zip(rows, masks) if r["label"] == 0)
    y = np.concatenate([m.ravel() for m in masks]); sc = np.concatenate([x.ravel() for x in maps])
    assert abs(X.ap_pn(P, [NG, NB]) - average_precision_score(y, sc)) < 1e-12
    assert abs(X.ap_pn(P, [NG, NB]) - M.exact_ap(maps, masks)["ap"]) < 1e-12
    assert all("within_ap" not in r for r in img if r["label"] == 0)
    assert all("within_ap" in r for r in img if r["label"] == 1)
    bad = [dict(r) for r in rows][::-1]
    with pytest.raises(Exception):
        D.state(p, bad, hws[::-1], masks[::-1])                      # identity order enforced


def test_determinism_and_inputs_untouched(tmp_path):
    from exps.cadfinal import common as K
    from exps.nl_ret_00b import decomp as D
    rows = [{"relative_path": "a.png", "label": 1}, {"relative_path": "b.png", "label": 0}]
    hws = [(30, 30), (30, 30)]
    m = np.zeros((30, 30), bool); m[3:9, 3:9] = True
    masks = [m, np.zeros((30, 30), bool)]
    s = torch.rand(2, 256, 256)
    p = tmp_path / "g.pt"; torch.save({"paths": ["a.png", "b.png"], "s256": s, "image_RD": torch.zeros(2)}, p)
    h0 = K.sha(p)
    r1 = D.state(p, rows, hws, masks); r2 = D.state(p, rows, hws, masks)
    assert all(np.array_equal(a, b) for a, b in zip(r1[:3], r2[:3])) and json.dumps(r1[3]) == json.dumps(r2[3])
    assert K.sha(p) == h0
