"""G1: causal prefix fixtures, witness eligibility and default-deny data access."""

from __future__ import annotations

import numpy as np
import torch

from exps.csg00 import contract as C, features, fit_modes


class FakeCache:
    def __init__(self, tensors):
        self.tensors = tensors

    def row(self, category, index):
        return self.tensors[category][index]


def _make_stream(events: int):
    updates = []
    for e in range(events):
        category = "bottle" if e < 100 else "carpet"
        index = e if e < 100 else e - 100
        updates.append({
            "event": e + 1, "category": category, "cache_index": index,
            "relative_path": f"{category}/train/good/{index:03d}.png",
            "raw_png_sha256": f"{category}-{index}",
        })
    return updates


def _make_cache(events: int, rng):
    return FakeCache({
        "bottle": torch.tensor(rng.normal(size=(100, C.PATCHES, C.DIM)), dtype=torch.float32),
        "carpet": torch.tensor(rng.normal(size=(100, C.PATCHES, C.DIM)), dtype=torch.float32),
        "hazelnut": torch.tensor(rng.normal(size=(100, C.PATCHES, C.DIM)), dtype=torch.float32),
    })


def test_writer_prefix_is_causal():
    rng = np.random.default_rng(41)
    updates = _make_stream(200)
    cache = _make_cache(200, rng)

    prefix = features.writer_tensor(cache, updates, 100)
    assert prefix.shape == (100, C.PATCHES, C.DIM)
    assert torch.equal(prefix[0], cache.row("bottle", 0))

    # Poison every future (carpet) feature; the t=100 prefix must not change.
    poisoned = FakeCache({
        "bottle": cache.tensors["bottle"],
        "carpet": torch.full_like(cache.tensors["carpet"], 1.0e9),
        "hazelnut": torch.full_like(cache.tensors["hazelnut"], 1.0e9),
    })
    prefix2 = features.writer_tensor(poisoned, updates, 100)
    assert torch.equal(prefix, prefix2)

    positions = C.bin_positions()
    bins = features.descriptor_bins(prefix, positions)
    bins2 = features.descriptor_bins(prefix2, positions)
    category_of_event = [row["category"] for row in updates[:100]]
    fit1 = fit_modes.fit_all_bins(bins, 100, category_of_event)
    fit2 = fit_modes.fit_all_bins(bins2, 100, category_of_event)
    for key in fit1["state"]:
        assert np.array_equal(fit1["state"][key], fit2["state"][key]), key


def test_witness_eligibility_is_boundary_and_category_ordered():
    assert C.ELIGIBLE_PREFIX[100] == 19
    assert C.ELIGIBLE_PREFIX[200] == 39
    assert C.ELIGIBLE_PREFIX[300] == 59
    groups = []
    start = 0
    for cat, count in C.PROBE_BLOCKS:
        groups.append((cat, start, start + count))
        start += count
    assert groups == [("bottle", 0, 19), ("carpet", 19, 39), ("hazelnut", 39, 59)]
    # a future-category witness is never in an earlier prefix
    assert C.ELIGIBLE_PREFIX[100] == groups[0][2]
    assert C.ELIGIBLE_PREFIX[200] == groups[1][2]


def test_stream_verification_rejects_foreign_identities():
    stream = C.load_stream()
    whitelist = set(stream["writer_paths"]) | set(stream["probe_paths"])
    assert len(whitelist) == 409
    for rel in stream["writer_paths"][:5] + stream["probe_paths"][:5]:
        assert rel.split("/")[1:3] == ["train", "good"]
    # default deny
    for bad in ("bottle/test/good/000.png", "bottle/ground_truth/x/000_mask.png"):
        try:
            C.assert_normal_only(bad, whitelist)
        except PermissionError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"guard allowed {bad}")


def test_stream_counts_and_blocks():
    stream = C.load_stream()
    assert len(stream["updates"]) == 350
    assert len(stream["probes"]) == 59
    cats = [row["category"] for row in stream["updates"]]
    assert cats[:100] == ["bottle"] * 100
    assert cats[100:200] == ["carpet"] * 100
    assert cats[200:300] == ["hazelnut"] * 100
    assert cats[300:] == ["bottle"] * 50
    assert len({row["relative_path"] for row in stream["updates"]}) == 350
    assert len({row["raw_png_sha256"] for row in stream["updates"]}) == 350
    assert stream["schedule"][100]["eligible_witness_prefix"] == 19
    assert stream["schedule"][350]["eligible_witness_prefix"] == 59
