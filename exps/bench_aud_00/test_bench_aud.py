"""BENCH-AUD-00 tests (synthetic only; no label, mask, feature or test artifact is read).

    PYTHONPATH=. python -m pytest -q -p no:cacheprovider exps/bench_aud_00/test_bench_aud.py
"""
from __future__ import annotations

import numpy as np
import torch

from exps.bench_aud_00 import run as B
from exps.sand_e2a_00 import common as E


def test_custody():
    ok = [B.OUT / "RESULT.json", E.OUT / "CALIB.json", B.STATS, B.STATS_LOCK, B.CACHE_JSON, B.CACHE_DIR / "test_screw.pt",
          E.MEM / "ckpt_t03.pt", E.GRIDS / "stage01_bottle.pt", E.MANIFEST,
          E.MVTEC / "screw/ground_truth/thread_top/000_mask.png", E.ROOT / "exps/x.py"]
    for p in ok:
        assert B.decide(p), p
    no = [B.CACHE_DIR / "train_screw.pt", B.CACHE_DIR / "heldout_bottle.pt", E.MVTEC / "screw/test/good/000.png",
          E.ROOT / "data/visa/x.png", E.ROOT / "results/hope_cad/rh01b/PROTOCOL.json", E.RUN / "stage05_bottle.pt"]
    for p in no:
        assert not B.decide(p), p


def test_route_nearest_centroid():
    r = np.random.default_rng(0)
    cents = {c: r.normal(0, 1, 768) * 5 for c in E.ORDER}
    mv = np.stack([cents["grid"] + r.normal(0, 0.1, 768), cents["zipper"] + r.normal(0, 0.1, 768)])
    assert B.route(mv, cents, list(E.ORDER)) == ["grid", "zipper"]
    assert B.route(mv[:1], cents, ["bottle", "cable"])[0] in ("bottle", "cable")


def test_maps224_matches_cadic_adapter():
    from exps.calm02.score import pixel_map
    g = np.random.default_rng(1).random((3, 784)).astype(np.float32)
    m = B.maps224(g)
    for i in range(3):
        assert np.array_equal(m[i], pixel_map(torch.from_numpy(g[i])).numpy())


def test_mask224_nearest():
    m = np.zeros((900, 900), dtype=bool)
    m[100:300, 400:500] = True
    r = B.mask224(m)
    assert r.shape == (224, 224) and r.any() and abs(r.mean() - m.mean()) < 0.01
    assert not B.mask224(np.zeros((900, 900), dtype=bool)).any()


def test_metrics_and_alt_scores():
    r = np.random.default_rng(2)
    n = 20
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int8)
    grid = r.random((n, 784)).astype(np.float32) * 0.1
    masks = [np.zeros((224, 224), dtype=bool) for _ in range(n)]
    for i in range(10, 20):
        grid[i, 100:110] += 1.0                                      # grid row 3, cols 16-25
        masks[i][24:32, 128:208] = True                              # the same region at 224 x 224
    out = B.metrics224(grid, masks, labels, alt=True)
    assert out["i_auroc"] == 1.0 and out["pixel_aupr"] > 0.3
    assert {f"i_auroc_{a}" for a in B.ALT if a != "max"} <= set(out) and len(out["_image_max"]) == n


def test_size_terciles():
    labels = np.array([0] * 6 + [1] * 6)
    masks = [np.zeros((10, 10), dtype=bool) for _ in range(12)]
    for j, i in enumerate(range(6, 12)):
        masks[i][: j + 1] = True                                     # increasing areas
    scores = np.array([0.0] * 6 + [0.5, 0.5, 2, 2, 3, 3])
    t = B.size_terciles(labels, masks, scores)
    assert t["large"] == 1.0 and t["small"] == 1.0


def test_band():
    assert B.band({"lo": 0.01, "hi": 0.03}) == "AHEAD" and B.band({"lo": -0.03, "hi": -0.01}) == "BEHIND"
    assert B.band({"lo": -0.01, "hi": 0.02}) == "NOT_ESTABLISHED"


def test_summarize_synthetic():
    from exps.nl_sgm_01a import am03_eval as A3
    r = np.random.default_rng(3)
    cells = {f"{t}|{c}": {a: {"pixel_aupr": float(r.uniform(0.2, 0.8)), "i_auroc": float(r.uniform(0.8, 1.0))}
                          for a in ("DEC", "MEM", "FUS", "FUS_w0")} for t, c in E.KEYS}
    routing = {f"{t}|{c}": {"n": 10, "correct": 10, "wrong_to": []} for t, c in E.KEYS}
    pub = B.published()
    assert set(pub) == set(B.PUBLISHED_ROWS) and len(pub["ReplayCAD"]["i_auroc"]) == 15
    s = B.summarize(cells, routing, pub, lambda x, y: A3.bootstrap_delta(x, y, seed=1, B=500))
    assert s["routing_accuracy_by_stage"][15] == 1.0 and set(s["final_macro"]) == {"DEC", "MEM", "FUS", "FUS_w0"}
    assert len(s["vs_published"]) == 4


def test_sources_exist():
    for s in B.SOURCES:
        assert (E.ROOT / s).exists(), s
