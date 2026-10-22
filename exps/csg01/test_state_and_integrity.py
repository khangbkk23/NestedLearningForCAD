"""G1: state serialization, byte accounting, causality and access integrity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from exps.csg00 import contract as C0
from exps.csg00 import features as F0
from exps.csg01 import contract as C1
from exps.csg01 import quantize, scoring, select


def _support_fixture(seed=21, n=80):
    rng = np.random.default_rng(seed)
    bins = [torch.tensor(rng.normal(size=(n, C1.DIM)), dtype=torch.float32)
            for _ in range(C1.BINS)]
    updates = [{"category": "bottle", "relative_path": f"bottle/train/good/{i:03d}.png"}
               for i in range(n // C1.TOKENS_PER_BIN)]
    positions = C0.bin_positions()
    sel = select.build_selection(bins, len(updates), C1.quota_vector(C1.SELECTION_COUNT),
                                 "fps", updates=updates, positions=positions)
    support = select.take_selected(bins, sel)
    return bins, support, sel


def _states(support):
    state_b = select.pack_support(support)
    enc_c = quantize.encode_support(support)
    state_c = {"counts": state_b["counts"], "offsets": state_b["offsets"],
               **quantize.pack_quantized(enc_c)}
    rng = np.random.default_rng(22)
    support_d = [torch.tensor(rng.normal(size=(51 if b else 52, C1.DIM)), dtype=torch.float32)
                 for b in range(C1.BINS)]
    state_d0 = select.pack_support(support_d)
    enc_d = quantize.encode_support(support_d)
    state_d = {"counts": state_d0["counts"], "offsets": state_d0["offsets"],
               **quantize.pack_quantized(enc_d)}
    qe = C1.quota_vector(C1.SIZE_MATCHED_COUNT)
    support_e = [s[: int(qe[b])] for b, s in enumerate(support)]
    state_e = select.pack_support(support_e)
    return state_b, state_c, state_d, state_e, enc_c


def test_state_serialization_reload_and_byte_ratios(tmp_path):
    _, support, _ = _support_fixture()
    state_b, state_c, state_d, state_e, _ = _states(support)
    info = {}
    for arm, state in (("B", state_b), ("C", state_c), ("D", state_d), ("E", state_e)):
        info[arm] = C1.write_state(tmp_path / f"{arm}.csg00", arm, 100,
                                   C1.arms_params()[arm], state, exclusive=True)
        _, reloaded = C1.read_state(tmp_path / f"{arm}.csg00")
        assert set(reloaded) == set(state)
        for key in state:
            assert np.array_equal(np.asarray(reloaded[key]), np.asarray(state[key])), (arm, key)
    assert abs(info["C"]["bytes"] / info["D"]["bytes"] - 1.0) < 0.05
    assert abs(info["C"]["bytes"] / info["E"]["bytes"] - 1.0) < 0.05
    assert info["B"]["bytes"] > 3 * info["C"]["bytes"]
    # create-exclusive
    with pytest.raises(FileExistsError):
        C1.write_state(tmp_path / "B.csg00", "B", 100, C1.arms_params()["B"], state_b)


def test_tamper_detection(tmp_path):
    _, support, _ = _support_fixture()
    state_b, _, _, _, _ = _states(support)
    C1.write_state(tmp_path / "B.csg00", "B", 100, C1.arms_params()["B"], state_b)
    raw = bytearray((tmp_path / "B.csg00").read_bytes())
    raw[-1] ^= 0xFF
    (tmp_path / "tampered.csg00").write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="payload hash"):
        C1.read_state(tmp_path / "tampered.csg00")


def test_causal_prefix_poisoning():
    rng = np.random.default_rng(31)
    updates = []
    for e in range(200):
        cat = "bottle" if e < 100 else "carpet"
        idx = e if e < 100 else e - 100
        updates.append({"event": e + 1, "category": cat, "cache_index": idx,
                        "relative_path": f"{cat}/train/good/{idx:03d}.png",
                        "raw_png_sha256": f"{cat}-{idx}"})
    tensors = {
        "bottle": torch.tensor(rng.normal(size=(100, C1.PATCHES, C1.DIM)), dtype=torch.float32),
        "carpet": torch.tensor(rng.normal(size=(100, C1.PATCHES, C1.DIM)), dtype=torch.float32),
        "hazelnut": torch.zeros((1, C1.PATCHES, C1.DIM)),
    }

    class Cache:
        def row(self, category, index):
            return tensors[category][index]

    prefix = F0.writer_tensor(Cache(), updates, 100)
    poisoned = {
        "bottle": tensors["bottle"],
        "carpet": torch.full_like(tensors["carpet"], 1.0e9),
        "hazelnut": tensors["hazelnut"],
    }

    class Poisoned:
        def row(self, category, index):
            return poisoned[category][index]

    prefix2 = F0.writer_tensor(Poisoned(), updates, 100)
    assert torch.equal(prefix, prefix2)
    positions = C0.bin_positions()
    bins1 = F0.descriptor_bins(prefix, positions)
    bins2 = F0.descriptor_bins(prefix2, positions)
    q = C1.quota_vector(C1.SELECTION_COUNT)
    sel1 = select.build_selection(bins1, 100, q, "fps", updates=updates, positions=positions)
    sel2 = select.build_selection(bins2, 100, q, "fps", updates=updates, positions=positions)
    assert np.array_equal(select.selected_global_ids(sel1), select.selected_global_ids(sel2))
    assert all(r["event"] < 100 for r in sel1["provenance"])


def test_normal_only_guard():
    whitelist = {"bottle/train/good/000.png"}
    C0.assert_normal_only("bottle/train/good/000.png", whitelist)
    for bad in ("bottle/test/crack/000.png", "bottle/ground_truth/x/000_mask.png",
                "carpet/train/good/000.png"):
        with pytest.raises(PermissionError):
            C0.assert_normal_only(bad, whitelist)


def test_selection_and_quantization_determinism():
    bins, support, sel = _support_fixture(seed=41)
    assert sel["n_selected"] == C1.SELECTION_COUNT
    enc1 = quantize.encode_support(support)
    enc2 = quantize.encode_support(support)
    assert np.array_equal(enc1["q"], enc2["q"])
    assert np.array_equal(enc1["xmin"], enc2["xmin"])
    assert np.array_equal(enc1["step"], enc2["step"])
    assert np.array_equal(enc1["decoded"], enc2["decoded"])


def test_covering_radius_and_quantization_bounds():
    bins, support, _ = _support_fixture(seed=51)
    eps = scoring.covering_radius(bins, support)
    assert eps.shape == (C1.BINS,)
    assert np.all(eps >= 0.0)
    enc = quantize.encode_support(support)
    summary = quantize.quantization_summary(enc, select.pack_support(support)["counts"])
    assert summary["bound_violations"] == 0
    assert summary["error_max"] <= summary["bound_max"] + C1.QUANT_TOL
