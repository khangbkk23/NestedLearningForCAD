# tests/test_hope_real_feature_probe.py
"""Focused checks for frozen-feature probe mechanics and diagnostics."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.hope_cad import HopeBlock
from models.hope_cad.state import persistent_states_equal, snapshot_persistent_state
from scripts.hope_cad import probe_real_features as probe


def test_probe_objectives_are_exact_and_locally_detached():
    source = torch.tensor([[[1.0, -2.0]]], requires_grad=True)
    output = torch.tensor([[[3.0, 1.0]]], requires_grad=True)
    preserving = probe.objectives(2, True)[0](0, source, output, None)
    mechanics = probe.objectives(2, False)[0](0, source, output, None)
    torch.testing.assert_close(preserving, 0.5 * ((output - source.detach()) ** 2).mean())
    torch.testing.assert_close(mechanics, 0.5 * output.square().mean())
    torch.autograd.grad(preserving, (output,))
    assert source.grad is None


def test_control_quantiles_and_norm_delta_reference():
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    summary = probe.six_stats(values)
    assert summary["min"] == 0.0 and summary["max"] == 3.0
    assert summary["median"] == pytest.approx(1.5)
    assert summary["mean"] == pytest.approx(1.5)
    absolute, relative = probe.norm_delta(torch.zeros(2), torch.tensor([3.0, 4.0]))
    assert absolute == pytest.approx(5.0)
    assert torch.isfinite(torch.tensor(relative))


def test_geometry_reference_and_zero_cosine_are_finite():
    rows = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    indices = torch.arange(4)
    pairs = torch.tensor([[0, 1], [0, 2], [0, 0]])
    geometry = probe.heavy_geometry(
        {"vit": rows, "smt": rows, "cms_level_0": rows, "hope": rows}, indices, pairs
    )
    # The centered matrix has two equal nonzero singular values.
    assert geometry["vit_effective_rank"] == pytest.approx(2.0, rel=1e-5)
    assert geometry["vit_smt_cosine_mean"] == pytest.approx(1.0)
    zero = probe.cosine_summary(torch.zeros(2, 2), torch.zeros(2, 2))
    assert all(torch.isfinite(torch.tensor(value)) for value in zero.values())


def test_cache_identity_validation_and_parquet_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(probe, "IMAGES_PER_CLASS", 2)
    monkeypatch.setattr(probe, "FEATURE_SHAPE", (3, 2))
    metadata = {"checkpoint_sha256": "locked", "block_index": 8, "preprocessing": {"resize": [224, 224]}}
    payload = probe.cache_payload("bottle", torch.ones(2, 3, 2), ["a.png", "b.png"], metadata)
    probe.validate_cache_payload(payload, "bottle", metadata)
    with pytest.raises(ValueError, match="identity/preprocessing"):
        probe.validate_cache_payload(payload, "bottle", {**metadata, "block_index": 7})
    with pytest.raises(ValueError, match="ordering"):
        probe.validate_cache_payload({**payload, "relative_paths": ["b.png", "a.png"]}, "bottle", metadata)
    path = tmp_path / "events.parquet"
    assert probe.write_event_table(path, [{"event_id": 1, "finite": True}]) == "parquet"
    assert probe.read_event_table(path) == [{"event_id": 1, "finite": True}]


def test_cms_inspection_is_read_only_and_returns_independent_outputs():
    model = HopeBlock(4, cms_update_periods=(1, 2), cms_hidden_dim=4)
    x = torch.randn(1, 5, 4)
    before = snapshot_persistent_state(model.state_dict())
    inspection = model.cms.inspect(x)
    torch.testing.assert_close(inspection.output, model.cms(x))
    assert len(inspection.level_outputs) == 2
    assert persistent_states_equal(before, model.state_dict())
    inspection.level_outputs[0].zero_()
    assert persistent_states_equal(before, model.state_dict())


def test_commit_diagnostic_is_pre_event_and_does_not_change_science():
    torch.manual_seed(6)
    model = HopeBlock(4, cms_update_periods=(1, 2), cms_hidden_dim=4)
    reference = deepcopy(model)
    x = torch.randn(1, 5, 4)
    smt_result = reference.smt(x, update=True)
    cms_expected = reference.cms.inspect(smt_result.memory_prediction)
    objectives = probe.objectives(2, True)
    result = model.commit_image(x, objectives)
    reference.cms.commit_image(smt_result.memory_prediction, objectives)
    torch.testing.assert_close(result.smt_representation, smt_result.memory_prediction)
    torch.testing.assert_close(result.output, cms_expected.output)
    for actual, expected in zip(result.cms_level_outputs, cms_expected.level_outputs):
        torch.testing.assert_close(actual, expected)
    assert persistent_states_equal(model.state_dict(), reference.state_dict())


def test_health_classifier_rejects_mechanical_failure():
    row = {
        "hope_average_patch_norm": 1.0,
        "hope_centered_patch_variance": 1.0,
        "hope_effective_rank": 2.0,
        "hope_cosine_mean": 0.0,
        "hope_cosine_std": 0.5,
        "smt_memory_norm_before": 1.0,
        "smt_memory_norm_after": 1.0,
    }
    continual = {
        "rows": [row], "last_row": row, "mechanical_finite": False,
        "state_bytes_constant": True, "state_keys_constant": True,
        "persistent_graph_free": True, "schedule_valid": True,
    }
    result = probe.classify_health(continual, {"state_equal": True, "output_finite": True}, {"initial_state_equal": True})
    assert result["outcome"] == "DEGENERATE"
    assert not result["mechanical_validity"]


def test_real_extractor_contract_when_local_assets_exist():
    checkpoint = probe.REPO_ROOT / probe.CHECKPOINT_RELATIVE
    image = probe.REPO_ROOT / "data/mvtec/carpet/train/good/000.png"
    if not checkpoint.is_file() or not image.is_file():
        pytest.skip("local checkpoint or MVTec normal image is unavailable")
    protocol = probe.load_protocol()
    row = {"task_id": 3, "task_name": "carpet", "split": "train", "relative_path": "carpet/train/good/000.png"}
    batch = next(iter(probe.make_protocol_loader(protocol, [row], 1)))
    extractor = probe.make_extractor(torch.device("cpu"), checkpoint)
    first = extractor.extract_patch_features(batch["images"])
    second = extractor.extract_patch_features(batch["images"])
    assert first.shape == (1, 784, 768)
    assert first.dtype == torch.float32 and not first.requires_grad
    assert torch.isfinite(first).all()
    assert extractor.block_index == 8 and not extractor.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    extractor.close()
