"""Synthetic P1 checks. No actual anomaly image, mask or confirmation reads."""
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from exps.task_p.p0_normal_support import distances, fingerprint
from exps.task_p.p1_data import (ArrivalReader, CP, CP_SHA, FeatureCache, MEAN, STD,
                                native_mask, sha, source_path, validate_roles)
from exps.task_p.p1_readout import (ARMS, BoundedPixelBootstrap, PixelRanking, bootstrap_weights,
    features_ok, image_metrics, native_map, new_models, readout, restore, score_histogram)


@pytest.fixture(autouse=True)
def cpu_determinism():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def test_geometry_and_exact_nn():
    gen = torch.Generator().manual_seed(4)
    q = torch.randn(784, 768, generator=gen)
    features_ok(q)
    with pytest.raises(RuntimeError, match="784,768"):
        features_ok(q[:16])
    with pytest.raises(RuntimeError):
        features_ok(q.double())
    bank = torch.cat([q[:2], q[:1]])
    actual = distances(q[:5], bank)
    expected = (q[:5, None].double() - bank[None].double()).square().sum(-1).sqrt()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0].argmin().item() == 0 and actual[0, 2] == 0


def test_native_alignment_against_independent_half_pixel_geometry():
    grid = np.arange(784, dtype=np.float32).reshape(28, 28)
    h, w = 83, 139
    actual = native_map(torch.from_numpy(grid.ravel()), (h, w))
    rr = np.clip((np.arange(h) + .5) * 28 / h - .5, 0, 27)
    cc = np.clip((np.arange(w) + .5) * 28 / w - .5, 0, 27)
    # A linear ramp is reproduced exactly except ordinary FP32 rounding.
    np.testing.assert_allclose(actual, rr[:, None] * 28 + cc[None, :], rtol=2e-7, atol=1e-4)
    impulse = torch.zeros(784)
    impulse[7 * 28 + 19] = 1
    mapped = native_map(impulse, (84, 140))
    assert np.unravel_index(mapped.argmax(), mapped.shape) == (22, 97)
    assert mapped[22, 97] == 1 and mapped[97 % 84, 22] == 0


def test_native_mask_polarity_and_annotation_failures(tmp_path):
    values = np.zeros((24, 36), dtype=np.uint8)
    values[4:9, 17:25] = 255
    file = tmp_path / "mask.png"
    Image.fromarray(values).save(file)
    row = {"hw": [24, 36], "label": 1, "mask_path": "mask.png", "mask_sha256": sha(file)}
    assert np.array_equal(native_mask(tmp_path, row), values > 0)
    assert native_mask(tmp_path, dict(row, label=0, mask_path=None)).sum() == 0
    with pytest.raises(RuntimeError, match="geometry"):
        native_mask(tmp_path, dict(row, hw=[36, 24]))
    Image.fromarray(np.zeros_like(values)).save(file)
    with pytest.raises(RuntimeError, match="empty"):
        native_mask(tmp_path, dict(row, mask_sha256=sha(file)))
    Image.fromarray(np.full_like(values, 127)).save(file)
    with pytest.raises(RuntimeError, match="polarity"):
        native_mask(tmp_path, dict(row, mask_sha256=sha(file)))


def test_exact_native_metrics_against_sklearn_with_ties_and_image_weights():
    gen = np.random.default_rng(19)
    maps = [gen.integers(0, 9, size=(11 + i, 15)).astype(np.float32) for i in range(4)]
    masks = [gen.random(m.shape) < .13 for m in maps]
    ranking = PixelRanking([score_histogram(s, y) for s, y in zip(maps, masks)])
    for weights in (np.ones(4, dtype=int), np.array([0, 3, 1, 2])):
        y = np.concatenate([m.ravel() for m in masks])
        s = np.concatenate([m.ravel() for m in maps])
        w = np.concatenate([np.full(m.size, v) for m, v in zip(maps, weights)])
        result = ranking.metrics(weights)
        assert result["ap"] == pytest.approx(average_precision_score(y, s, sample_weight=w), abs=1e-14)
        assert result["auroc"] == pytest.approx(roc_auc_score(y, s, sample_weight=w), abs=1e-14)
        assert result["prevalence"] == pytest.approx(np.average(y, weights=w))


def test_auroc_and_ap_against_pairwise_and_threshold_references():
    y, s = np.array([0, 1, 0, 1, 1, 0]), np.array([.2, .5, .5, .8, .9, .9])
    result = image_metrics(y, s)
    pair = [(a > b) + .5 * (a == b) for a in s[y == 1] for b in s[y == 0]]
    expected_ap, previous_recall = 0., 0.
    for threshold in sorted(set(s), reverse=True):
        selected = s >= threshold
        recall, precision = y[selected].sum() / y.sum(), y[selected].mean()
        expected_ap += (recall - previous_recall) * precision
        previous_recall = recall
    assert result["auroc"] == pytest.approx(np.mean(pair))
    assert result["ap"] == pytest.approx(expected_ap)
    assert image_metrics([0, 0], [1, 2])["auroc"] is None
    assert image_metrics([0, 1], [4, 4])["auroc"] == .5
    assert image_metrics([0, 1], [4, 4])["ap"] == .5
    with pytest.raises(RuntimeError):
        image_metrics([0, -1], [1, 2])


def test_bootstrap_bounds_enclose_exact_ap_for_coarse_bins():
    gen = np.random.default_rng(21)
    scores = [gen.random((9, 13)).astype(np.float32) for _ in range(6)]
    masks = [np.zeros((9, 13), bool) for _ in range(3)] + [gen.random((9, 13)) < .2 for _ in range(3)]
    hist = [score_histogram(s, m) for s, m in zip(scores, masks)]
    exact, bounded = PixelRanking(hist), BoundedPixelBootstrap(hist, bins=3)
    draws = np.stack([bootstrap_weights([0, 0, 0, 1, 1, 1], gen) for _ in range(35)])
    lower, upper = bounded.bounds(draws)
    for w, lo, hi in zip(draws, lower, upper):
        value = exact.metrics(w)["ap"]
        assert lo - 1e-12 <= value <= hi + 1e-12
        assert w[:3].sum() == w[3:].sum() == 3
    fine = BoundedPixelBootstrap(hist, bins=65536)
    lo, hi = fine.bounds(draws[:1])
    assert hi[0] - lo[0] < .01


def test_native_pixels_preserved_and_not_macro_per_image_ap():
    s1, m1 = np.ones((3, 5), np.float32), np.zeros((3, 5), bool)
    s2, m2 = np.full((7, 11), 2., np.float32), np.zeros((7, 11), bool)
    m2[2:4, 5:8] = True
    result = PixelRanking([score_histogram(s1, m1), score_histogram(s2, m2)]).metrics()
    assert result["prevalence"] == pytest.approx(6 / 92)
    assert result["ap"] == pytest.approx(6 / 77)
    with pytest.raises(RuntimeError, match="alignment"):
        score_histogram(s1, m1.T)


def test_all_arms_identical_readout_nonmutation_serialization():
    models = new_models(capacity=6)
    bank = torch.arange(6, dtype=torch.float32)[:, None].expand(6, 768).clone()
    q = torch.zeros(784, 768)
    q[:, 0] = torch.arange(784) / 10
    outputs = []
    for model in models.values():
        model.bank[:] = bank
        model.ids[:] = torch.arange(6)
        model.n = 6
        before = fingerprint(model.state())
        output = readout(model, q, (43, 77))
        assert output["image_score"] == max(output["patch_distances"])
        assert fingerprint(model.state()) == before
        stream = io.BytesIO()
        torch.save(model.state(), stream)
        stream.seek(0)
        reopened = restore(torch.load(stream, weights_only=False))
        assert fingerprint(reopened.state()) == before
        np.testing.assert_array_equal(readout(reopened, q, (43, 77))["map"], output["map"])
        outputs.append(output["map"])
    assert all(np.array_equal(outputs[0], x) for x in outputs)


def test_task_arrival_no_future_or_anomalous_update(monkeypatch):
    train = [{"role": "normal_train", "category": c, "relative_path": f"{c}/train/good/0.png"}
             for c in ("bottle", "carpet")]
    evaluation = {"role": "development_eval", "category": "carpet", "relative_path": "carpet/test/hole/0.png"}
    class FakeCache:
        root = Path("/unused")
        calls = []
        def get(self, row):
            self.calls.append(row)
            return torch.zeros(784, 768)
    cache = FakeCache()
    reader = ArrivalReader({"updates": train, "evaluations": [evaluation], "boundaries": [1, 2]}, cache)
    with pytest.raises(RuntimeError):
        reader.update(train[1])
    with pytest.raises(RuntimeError):
        reader.update(evaluation)
    assert not cache.calls
    reader.update(train[0])
    with pytest.raises(RuntimeError):
        reader.evaluate(evaluation, 1)
    with pytest.raises(RuntimeError):
        reader.update(train[0])
    reader.update(train[1])
    monkeypatch.setattr("exps.task_p.p1_data.native_mask", lambda *_: np.zeros((1, 1), bool))
    reader.evaluate(evaluation, 2)
    assert [r["role"] for r in reader.access] == ["update", "update", "evaluation"]


def test_role_content_overlap_and_source_symlink_rejection(tmp_path):
    train = {"relative_path": "bottle/train/good/0.png", "role": "normal_train", "category": "bottle", "sha256": "one"}
    row = {"relative_path": "bottle/test/good/0.png", "role": "development_eval", "category": "bottle", "sha256": "two", "label": 0, "mask_path": None}
    validate_roles([train], [row])
    with pytest.raises(RuntimeError, match="overlap"):
        validate_roles([train], [dict(row, sha256="one")])
    with pytest.raises(RuntimeError, match="label"):
        validate_roles([], [dict(row, label=1, mask_path="x")])
    (tmp_path / "real").write_bytes(b"x")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(RuntimeError, match="symlink"):
        source_path(tmp_path, "link")


def test_canonical_preprocessing_rgb_and_grayscale(tmp_path):
    from dataset.benchmark_protocol_v1 import _ImageDataset
    for mode in ("RGB", "L"):
        values = np.arange(9 * 17 * (3 if mode == "RGB" else 1), dtype=np.uint8)
        shape = (9, 17, 3) if mode == "RGB" else (9, 17)
        path = tmp_path / f"{mode}.png"
        Image.fromarray(values.reshape(shape)).save(path)
        row = {"relative_path": path.name}
        actual = _ImageDataset(tmp_path.resolve(), [row], 224, MEAN, STD, True)[0]["images"]
        with Image.open(path) as image:
            array = np.array(image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32)
        expected = (torch.from_numpy(array.transpose(2, 0, 1)) / 255 - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        assert torch.equal(actual, expected)


def test_canonical_backbone_synthetic_geometry_capture_and_frozen():
    from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor
    assert sha(CP) == CP_SHA
    extractor = CADICViTFeatureExtractor(CADICViTConfig(checkpoint_path=str(CP), checkpoint_identity=CP_SHA))
    captured = []
    hook = extractor.model.blocks[9].register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    output = extractor(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1, 784, 768) and output.dtype == torch.float32
    assert torch.equal(output, captured[0][:, 1:])
    assert not output.requires_grad and not any(p.requires_grad for p in extractor.parameters())
    extractor.train(True)
    assert not extractor.training
    hook.remove()
    extractor.close()


def test_decision_and_paired_contrasts_are_not_normal_support_gates():
    from exps.task_p.p1_readout import classify
    from exps.task_p.analyze_p1 import contrast
    assert contrast(.04, np.array([.02, .03, .05]), np.array([.021, .031, .051]))["decision"] == "OUTPERFORMS"
    assert contrast(0., np.array([-.01, 0., .005]), np.array([-.009, .001, .006]))["decision"] == "MATCHES_WITHIN_2PP"
    values = {a: {c: {"image_auroc": .8, "pixel_ap": .3, "prevalence": .02}
                        for c in ("bottle", "carpet", "hazelnut")} for a in ARMS}
    ci = {a: {"image_auroc": [.6, .9], "ap_minus_prevalence": [.1, .4]} for a in ARMS}
    assert classify(values, values, {a: 0. for a in ARMS}, ci) == "P1_SIGNAL_VALID"
    assert classify(values, values, {a: .2 for a in ARMS}, ci) == "P1_CONTINUAL_RETENTION_FAILURE"
    bad = {a: {c: {"image_auroc": .5, "pixel_ap": .02, "prevalence": .02}
                     for c in ("bottle", "carpet", "hazelnut")} for a in ARMS}
    assert classify(bad, bad, {a: 0. for a in ARMS}, ci) == "P1_REPRESENTATION_OR_SCORER_FAILURE"


def test_once_only_json_and_missing_protocol_fail_closed(tmp_path, monkeypatch):
    from exps.task_p.p1_data import write_json
    from exps.task_p.p1_protocol import load_protocol
    target = tmp_path / "artifact.json"
    write_json(target, {"first": True})
    with pytest.raises(FileExistsError):
        write_json(target, {"second": True})
    monkeypatch.setattr("exps.task_p.p1_protocol.OUT", tmp_path)
    with pytest.raises(RuntimeError, match="no resolved"):
        load_protocol()
