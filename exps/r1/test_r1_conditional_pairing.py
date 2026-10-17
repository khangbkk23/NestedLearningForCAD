"""Synthetic invariants only. Never open feature caches or image datasets."""
import copy

import pytest
import torch
from torch.nn import functional as F

from exps.r1.run_r1_conditional_pairing import (
    Neural, Mean, Ridge, Reader, configure, derangement, digest, image_loss,
    inference, make_input, optimizer, step, validate_alignment, validate_roles,
    CHECKPOINT_SHA, MANIFEST_DIGEST,
)


@pytest.fixture(autouse=True)
def deterministic():
    configure(); torch.manual_seed(27)


def pair(n=4):
    c = F.normalize(torch.randn(n, 3, 7), dim=-1)
    z = F.normalize(torch.randn(n, 3, 7), dim=-1)
    return c, z


def test_cache_alignment_rejects_independent_row_order():
    entries = [{"relative_path": f"bottle/train/good/{i:03}.png", "raw_png_sha256": str(i)} for i in range(2)]
    base = {"class_name": "bottle", "relative_paths": [e["relative_path"] for e in entries],
            "per_image_source_sha256": [e["raw_png_sha256"] for e in entries],
            "source_manifest_digest": MANIFEST_DIGEST, "patches": torch.zeros(2, 784, 768),
            "metadata": {"checkpoint_sha256": CHECKPOINT_SHA, "block_index": 8, "feature_normalization": "none"}}
    clean = {**base, "view": "clean"}; masked = {**base, "view": "masked_rgb16"}
    validate_alignment(clean, masked, entries, "bottle")
    masked["relative_paths"] = masked["relative_paths"][::-1]
    with pytest.raises(RuntimeError, match="alignment"): validate_alignment(clean, masked, entries, "bottle")


def test_no_clean_target_predictor_input_and_rgb_mask_oracle():
    c, z = pair(); model = Neural(7, 3, 5)
    x = make_input(c); old = inference(model, c)
    z.mul_(100)
    assert torch.equal(x, make_input(c)) and torch.equal(old, inference(model, c))
    assert torch.equal(x[..., :7], c)
    from exps.ol04.hope_outer_learning_pilot_v1 import mask_rgb
    raw = torch.randn(1, 3, 224, 224)
    masked, mask = mask_rgb(raw)
    changed = raw.clone(); changed[:, :, mask] += 100
    assert int(mask.sum()) == 9216
    assert torch.equal(masked, mask_rgb(changed)[0])
    assert torch.equal(masked[:, :, ~mask], raw[:, :, ~mask])


def test_split_identity_and_content_overlap_rejected():
    a = {"relative_path": "bottle/train/good/000.png", "raw_png_sha256": "a"}
    b = {"relative_path": "bottle/train/good/001.png", "raw_png_sha256": "b"}
    validate_roles([a], [b])
    with pytest.raises(RuntimeError, match="identity"): validate_roles([a], [a])
    with pytest.raises(RuntimeError, match="content"): validate_roles([a], [{**b, "raw_png_sha256": "a"}])


def test_whole_image_derangement_preserves_position_and_marginals():
    z = torch.arange(10 * 16 * 7).reshape(10, 16, 7)
    d = derangement(10, 7102)
    assert bool((d != torch.arange(10)).all())
    shuffled = z[d]
    assert torch.equal(z.sort(dim=0).values, shuffled.sort(dim=0).values)
    for i in range(10): assert torch.equal(shuffled[i], z[d[i]])
    assert torch.equal(d, derangement(10, 7102))
    with pytest.raises(RuntimeError): derangement(1, 1)


def test_future_task_and_unauthorized_row_access_rejected():
    row = {"class_name": "hazelnut", "relative_path": "hazelnut/train/good/000.png", "cache_index": 0}
    reader = Reader({"updates": [row], "probes": []})
    with pytest.raises(RuntimeError, match="future"): reader.pair([row], "update")
    reader.arrived.add("hazelnut")
    with pytest.raises(RuntimeError, match="unauthorized"): reader.pair([{**row, "cache_index": 1}], "update")


def test_equal_initialization_schedule_and_real_writes():
    c, z = pair(); a = Neural(7, 3, 5); b = copy.deepcopy(a)
    oa, ob = optimizer(a), optimizer(b)
    assert digest(a.state_dict()) == digest(b.state_dict())
    for _ in range(5):
        da = step(a, oa, c, z); step(b, ob, c, z[derangement(4, 5)])
        assert da["parameter_delta_l2"] > 0 and da["output_delta_rms_l2"] > 0
    assert {int(v["step"]) for v in oa.state.values()} == {5}
    assert {int(v["step"]) for v in ob.state.values()} == {5}


def test_no_write_preserves_parameters_and_optimizer():
    c, z = pair(); m = Neural(7, 3, 5); o = optimizer(m)
    step(m, o, c, z)
    before = digest((m.state_dict(), o.state_dict()))
    d = step(m, o, c, z, write=False)
    assert digest((m.state_dict(), o.state_dict())) == before
    assert d["parameter_delta_l2"] == 0 and d["output_delta_rms_l2"] == 0


def test_read_only_and_deterministic_replay_across_boundaries():
    c, z = pair(); a = Neural(7, 3, 5); b = copy.deepcopy(a)
    oa, ob = optimizer(a), optimizer(b)
    for phase in range(2):
        for _ in range(3):
            step(a, oa, c, z); step(b, ob, c, z)
        before = digest((a.state_dict(), oa.state_dict()))
        assert torch.equal(inference(a, c), inference(b, c))
        assert before == digest((a.state_dict(), oa.state_dict()))
        assert before == digest((b.state_dict(), ob.state_dict()))
        assert {int(v["step"]) for v in oa.state.values()} == {3 * (phase + 1)}


def test_ridge_matches_independent_augmented_design_and_mean():
    c, z = pair(12); r = Ridge(7, 3); m = Mean(7, 3)
    r.update(c[:5], z[:5]); r.update(c[5:], z[5:]); r.fit()
    m.update(z[:5]); m.update(z[5:])
    design = make_input(c).double().flatten(0, 1)
    penalty = torch.diag(torch.tensor([r.lam] * 7 + [0.] * 3, dtype=torch.float64))
    direct = torch.linalg.solve(design.T @ design + penalty, design.T @ z.double().flatten(0, 1))
    expected = (design @ direct).reshape_as(z)
    assert torch.allclose(r.predict(c), expected, atol=1e-10, rtol=1e-9)
    assert torch.allclose(m.predict(c)[0], z.double().mean(0), atol=1e-15, rtol=1e-14)
    before = digest(r.state()); inference(r, c)
    assert before == digest(r.state())


def test_loss_aggregates_channels_then_positions_without_output_normalizing():
    z = torch.zeros(2, 3, 7); pred = torch.ones_like(z) * 2
    assert torch.equal(image_loss(pred, z), torch.full((2,), 28., dtype=torch.float64))
