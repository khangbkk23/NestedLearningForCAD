"""G1: deterministic selection, quotas, UINT8 quantization and byte matching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.csg01 import contract as C1
from exps.csg01 import quantize, select


def _toy_bins(n=400, seed=0, dim=C1.DIM):
    rng = np.random.default_rng(seed)
    return [torch.tensor(rng.normal(size=(n, dim)), dtype=torch.float32) for _ in range(C1.BINS)]


def test_quota_vectors_and_extra_bin():
    qb = C1.quota_vector(C1.SELECTION_COUNT)
    qe = C1.quota_vector(C1.SIZE_MATCHED_COUNT)
    assert int(qb.sum()) == 2500 and int(qe.sum()) == 632
    assert qb[0] == 52 and int((qb[1:] == 51).sum()) == 48
    assert int((qe[:44] == 13).sum()) == 44 and int((qe[44:] == 12).sum()) == 5


def test_fps_deterministic_unique_and_medoid_init():
    D = torch.tensor(np.random.default_rng(1).normal(size=(600, C1.DIM)), dtype=torch.float64)
    first = select.select_fps(D, 25)
    second = select.select_fps(D, 25)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 25
    assert first.min() >= 0 and first.max() < D.shape[0]
    mean = D.mean(dim=0)
    assert int(first[0]) == int(((D - mean) ** 2).sum(dim=1).argmin())


def test_fps_covering_improves_with_more_selections():
    D = torch.tensor(np.random.default_rng(2).normal(size=(300, C1.DIM)), dtype=torch.float64)
    for k in (5, 15, 30):
        sel = select.select_fps(D, k)
        chosen = D[torch.as_tensor(sel)]
        d2 = ((D * D).sum(1)[:, None] + (chosen * chosen).sum(1)[None, :]
              - 2 * D @ chosen.T).clamp_min(0)
        radius = float(torch.sqrt(d2.min(1).values.max()))
        assert radius >= 0.0
    # more selections cannot increase the exact covering radius
    def radius_of(k):
        sel = select.select_fps(D, k)
        chosen = D[torch.as_tensor(sel)]
        d2 = ((D * D).sum(1)[:, None] + (chosen * chosen).sum(1)[None, :]
              - 2 * D @ chosen.T).clamp_min(0)
        return float(torch.sqrt(d2.min(1).values.max()))
    assert radius_of(30) <= radius_of(15) + 1e-9 <= radius_of(5) + 2e-9


def test_random_selection_deterministic_and_distinct():
    n = 300
    gids = np.arange(n, dtype=np.int64)
    a = select.select_random(n, 40, C1.RANDOM_SEED, gids)
    b = select.select_random(n, 40, C1.RANDOM_SEED, gids)
    c = select.select_random(n, 40, C1.RANDOM_SEED + 1, gids)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert len(set(a.tolist())) == 40


def test_build_selection_quotas_and_provenance():
    bins = _toy_bins(n=64, seed=3)
    updates = [{"category": "bottle", "relative_path": f"bottle/train/good/{i:03d}.png"}
               for i in range(4)]
    positions = C1.C0.bin_positions()
    sel = select.build_selection(bins, t=4, quotas=C1.quota_vector(2500), policy="fps",
                                 updates=updates, positions=positions)
    assert sel["n_selected"] == 2500
    assert int(sel["quotas"].sum()) == 2500
    assert len({r["global_id"] for r in sel["provenance"]}) == 2500
    assert sel["all_events_causal"] is True
    # 4 events x 16 tokens per bin = 64 candidates; quotas are 51/52
    assert max(r["event"] for r in sel["provenance"]) < 4
    patch_bins = C1.C0.patch_bin_indices()
    for r in sel["provenance"]:
        assert r["global_id"] // C1.PATCHES == r["event"]
        assert int(patch_bins[r["patch"]]) == r["bin"]


def test_c_uses_exact_b_identities():
    bins = _toy_bins(n=64, seed=4)
    positions = C1.C0.bin_positions()
    updates = [{"category": "bottle", "relative_path": f"bottle/train/good/{i:03d}.png"}
               for i in range(4)]
    sel_b = select.build_selection(bins, 4, C1.quota_vector(2500), "fps",
                                   updates=updates, positions=positions)
    support = select.take_selected(bins, sel_b)
    encoded = quantize.encode_support(support)
    state_b = select.pack_support(support)
    state_c = {"counts": state_b["counts"], "offsets": state_b["offsets"],
               **quantize.pack_quantized(encoded)}
    assert np.array_equal(state_c["counts"], state_b["counts"])
    assert np.array_equal(state_c["q"].shape, (2500, C1.DIM))
    assert state_c["q"].dtype == np.uint8
    assert np.array_equal(select.selected_global_ids(sel_b),
                          select.selected_global_ids(sel_b))


def test_uint8_roundtrip_bound_and_constant_vector():
    rng = np.random.default_rng(5)
    X = [torch.tensor(rng.normal(size=(30, C1.DIM)), dtype=torch.float32) for _ in range(3)]
    enc = quantize.encode_support(X)
    assert enc["q"].shape == (90, C1.DIM) and enc["q"].dtype == np.uint8
    assert enc["xmin"].dtype == np.float32 and enc["step"].dtype == np.float32
    assert np.all(enc["error"] <= enc["bound"] + C1.QUANT_TOL)
    decoded = quantize.decode(enc["q"], enc["xmin"], enc["step"])
    assert np.allclose(decoded, enc["decoded"], atol=0, rtol=0)
    # constant vector path
    const = torch.full((2, C1.DIM), 3.25, dtype=torch.float32)
    ec = quantize.encode_support([const])
    assert ec["constant_vectors"] == 2
    assert np.all(ec["step"] == 0.0)
    assert np.all(ec["q"] == 0)
    assert np.allclose(ec["decoded"], 3.25, atol=0, rtol=0)
    assert np.all(ec["error"] == 0.0)


def test_round_half_to_even_and_uint8_bound():
    # step = 1; x=0.5 -> rint=0, x=1.5 -> rint=2; both within h/2
    x = np.zeros((2, C1.DIM), dtype=np.float32)
    x[0, 0] = 0.0
    x[0, 1] = 255.0
    x[0, 2] = 0.5
    x[1, 0] = 0.0
    x[1, 1] = 255.0
    x[1, 2] = 1.5
    X = torch.tensor(x)
    enc = quantize.encode_support([X])
    assert enc["step"][0] == pytest.approx(1.0)
    assert int(enc["q"][0, 2]) == 0
    assert int(enc["q"][1, 2]) == 2
    assert np.all(enc["error"] <= np.sqrt(C1.DIM) * enc["step"].astype(np.float64) / 2 + C1.QUANT_TOL)


def test_byte_matching_dry_run():
    plan = C1.choose_size_matched_count()
    assert plan["chosen_count"] == C1.SIZE_MATCHED_COUNT
    assert abs(plan["ratio"] - 1.0) < 0.05
    c = C1.dry_run_bytes("C")["bytes"]
    d = C1.dry_run_bytes("D")["bytes"]
    e = C1.dry_run_bytes("E")["bytes"]
    assert abs(c / d - 1.0) < 0.05
    assert abs(c / e - 1.0) < 0.05
    b = C1.dry_run_bytes("B")["bytes"]
    assert b > 3 * c  # B is the FP32 selection reference, not a compact arm
