"""Pre-lock tests for the C2-DOM sweep (synthetic + small real-feature checks; no test data)."""
import numpy as np
import pytest
import torch

from exps.csg01.select import select_fps
from exps.nl_sgm_01a import codecs as K
from exps.nl_sgm_01a_c2dom import common as D
from exps.nl_sgm_01a_c2dom import fps as FPS
from exps.nl_sgm_01a_c2dom import analyze as AN


def test_ledger_within_budget_and_formula():
    for b in D.BUDGETS.values():
        for codec in ("fp32", "pq_dom"):
            for t in range(15):
                cap = D.capacity(codec, b, t)
                assert D.ledger_bytes(codec, b, t) <= b
                row = D.FP32_ROW_BYTES if codec == "fp32" else D.PQ_ROW_BYTES
                assert b - D.ledger_bytes(codec, b, t) < row          # maximal
    assert D.capacity("fp32", D.BUDGETS["B1"], 0) == 2495
    assert D.capacity("pq_dom", D.BUDGETS["B1"], 14) == 17640


def test_state_bytes_matches_ledger():
    t, B = 3, D.BUDGETS["B1"]
    cap = D.capacity("pq_dom", B, t)
    st = {"codes": torch.zeros(cap, 96, dtype=torch.uint8), "ids": torch.zeros(cap, dtype=torch.int32),
          "owner": torch.zeros(cap, dtype=torch.int8), "codebooks": torch.zeros(t + 1, 96, 256, 8, dtype=torch.float16),
          "header": torch.zeros(16, dtype=torch.int32)}
    assert D.state_bytes(st) == D.ledger_bytes("pq_dom", B, t)


def test_partition_disjoint_and_deterministic():
    co, n = D.setup()
    p1, p2 = D.partition(co, n), D.partition(co, n)
    for c in D.ORDER:
        assert p1[c] == p2[c]
        f, s = D.pool_gids(n, c, p1[c]["fit"]), D.pool_gids(n, c, p1[c]["sel"])
        assert not torch.isin(f, s).any() and len(f) + len(s) == n[c] * 784
        assert abs(len(p1[c]["fit"]) - len(p1[c]["sel"])) <= 1


@pytest.mark.parametrize("seed,n,k", [(0, 3000, 400), (1, 5000, 2500), (2, 2000, 1999)])
def test_fps_cpu_path_equals_canonical(seed, n, k):
    x = torch.randn(n, 768, generator=torch.Generator().manual_seed(seed))
    idx, rec = FPS.select(x, k, allow_gpu=False)
    assert np.array_equal(idx, select_fps(x, k)) and rec["impl"] == "cpu_canonical"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("pq", [False, True])
def test_fps_gpu_equals_canonical_real_features(pq):
    from exps.nl_sgm_01a import common as C
    F = C.Features(); co = C.cohorts()
    x = F.images(co["grid"]["writers"][:40]).reshape(-1, 768).clone()
    if pq:
        cb = K.fit(x[::5].contiguous(), iters=2)
        x = K.decode(K.encode(x, cb), cb)
    old = FPS.GPU_MIN_WORK
    FPS.GPU_MIN_WORK = 0
    try:
        idx, rec = FPS.select(x, 3000, allow_gpu=True)
    finally:
        FPS.GPU_MIN_WORK = old
    assert np.array_equal(idx, select_fps(x, 3000))
    assert rec["impl"] in ("gpu_certified", "gpu_certified_exact_ties", "cpu_canonical_fallback")


def test_fps_gpu_fallback_on_ties():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x = torch.randn(400, 768, generator=torch.Generator().manual_seed(5))
    x = torch.cat([x, x])                                       # k > distinct rows -> uncertified -> fallback
    old = FPS.GPU_MIN_WORK; FPS.GPU_MIN_WORK = 0
    try:
        idx, rec = FPS.select(x, 600, allow_gpu=True)
    finally:
        FPS.GPU_MIN_WORK = old
    assert np.array_equal(idx, select_fps(x, 600))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_fps_gpu_duplicate_ties_real_pq():
    """Amendment 01: decoded PQ rows with exact duplicates (k < distinct rows): certified on GPU, and identical to
    canonical (indices; at minimum identical selected vectors in identical order)."""
    o = torch.load(D.OUT / "codebooks" / "00_bottle.pt", weights_only=False)
    x = K.decode(o["sel_codes"][:12000], o["codebook"])
    g = torch.Generator().manual_seed(7)
    x = torch.cat([x, x[torch.randint(0, 12000, (400,), generator=g)]])          # 400 exact duplicates
    old = FPS.GPU_MIN_WORK; FPS.GPU_MIN_WORK = 0
    try:
        idx, rec = FPS.select(x, 11500, allow_gpu=True)
    finally:
        FPS.GPU_MIN_WORK = old
    can = select_fps(x, 11500)
    assert rec["impl"] in ("gpu_certified", "gpu_certified_exact_ties")
    assert torch.equal(x[torch.as_tensor(idx)], x[torch.as_tensor(can)])
    assert np.array_equal(idx, can)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_fps_gpu_large_duplicate_groups():
    """Amendment 04: groups of >8 identical rows (e.g. repeated background codes) must not force a CPU fallback."""
    o = torch.load(D.OUT / "codebooks" / "01_cable.pt", weights_only=False)
    x = K.decode(o["sel_codes"][:8000], o["codebook"])
    x = torch.cat([x[:3000], x[3000:3001].expand(25, -1), x[3001:], x[5:6].expand(12, -1)]).contiguous()
    old = FPS.GPU_MIN_WORK; FPS.GPU_MIN_WORK = 0
    try:
        idx, rec = FPS.select(x, 7900, allow_gpu=True)
    finally:
        FPS.GPU_MIN_WORK = old
    assert rec["impl"] in ("gpu_certified", "gpu_certified_exact_ties") and not rec.get("fallback")
    assert np.array_equal(idx, select_fps(x, 7900))


def test_pareto_and_fm():
    pts = {"a": {"bytes": 1, "final_pixel_ap": .5, "pixel_fm": .01}, "b": {"bytes": 2, "final_pixel_ap": .4, "pixel_fm": .02},
           "c": {"bytes": 3, "final_pixel_ap": .6, "pixel_fm": .03}}
    assert AN.pareto(pts, [("bytes", -1), ("final_pixel_ap", 1), ("pixel_fm", -1)]) == ["a", "c"]
    cells = {(t, c): {"pixel_ap": 0.5 - 0.01 * t, "image_auroc": 0.9} for t in range(15) for c in D.ORDER[:t + 1]}
    s = AN.summarize(cells)
    assert abs(s["pixel_fm"] - np.mean([0.01 * (14 - i) for i in range(14)])) < 1e-12
