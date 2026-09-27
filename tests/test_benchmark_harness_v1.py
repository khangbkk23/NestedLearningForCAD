# tests/test_benchmark_harness_v1.py
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from dataset.benchmark_manifest_v1 import build_training_manifest, manifest_digest
from dataset.benchmark_protocol_v1 import MVTecContinualProtocol
from models.fake_benchmark_adapter_v1 import FakeBenchmarkAdapter
from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig, CADICPatchCoresetV1
from training.benchmark_artifacts_v1 import BenchmarkArtifacts
from training.benchmark_engine_v1 import BenchmarkEngineV1
from training.benchmark_metrics_v1 import forgetting_matrix, pixel_aupr

TASKS = ["bottle"]

def make_mvtec(root):
    train = root / "bottle" / "train" / "good"
    test = root / "bottle" / "test" / "good"
    defect = root / "bottle" / "test" / "scratch"
    gt = root / "bottle" / "ground_truth" / "scratch"

    for path in [train, test, defect, gt]:
        path.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        Image.fromarray(
            np.full((12, 12, 3), i * 40 + 30, np.uint8)
        ).save(train / f"{i:03}.png")

    Image.fromarray(np.full((12, 12, 3), 30, np.uint8)).save(test / "000.png")
    Image.fromarray(np.full((12, 12, 3), 240, np.uint8)).save(defect / "000.png")
    Image.fromarray(np.pad(np.ones((4, 4), np.uint8), 4) * 255).save(
        gt / "000_mask.png"
    )

def configs(root):
    protocol = {
        "id": "test_protocol",
        "version": 1,
        "dataset": {"name": "MVTec AD", "root": str(root)},
        "task_order": TASKS,
        "training": {"normal_only": True, "dev_mode": "disabled", "drop_last": False},
        "evaluation": {
            "final_per_task": True,
            "primary": ["i_auroc", "p_aupr"],
            "aggregation": "macro_tasks",
            "pixels": "all",
            "pooled": "secondary_only",
        },
        "forgetting": {"enabled": True, "formula": "mean_prior_max_minus_final"},
        "leakage": {"official_test_feedback": "forbidden"},
    }

    method = {
        "id": "fake",
        "version": 1,
        "adapter": "fake",
        "exact_parity_claim": False,
        "preprocessing": {
            "image_size": 12,
            "resize": "direct_square_bilinear_pil",
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        "runtime": {"batch_size": 2, "dtype": "float32"},
    }
    return protocol, method


def test_manifest_is_portable_and_train_only(tmp_path):
    make_mvtec(tmp_path)
    protocol, _ = configs(tmp_path)
    manifest = build_training_manifest(protocol, 3, "abc")

    assert all(
        "/train/good/" in "/" + entry["relative_path"]
        for entry in manifest["entries"]
    )
    assert manifest["digest"] == manifest_digest(manifest)

    other = tmp_path / "other"
    other.mkdir()
    make_mvtec(other)
    protocol2, _ = configs(other)

    assert manifest["digest"] == build_training_manifest(
        protocol2, 3, "abc"
    )["digest"]


def test_metrics_and_forgetting():
    assert pixel_aupr(
        np.array([[[0.0, 1.0], [0.0, 1.0]]]),
        np.array([[[0, 1], [0, 1]]]),
    ) == 1.0

    result = forgetting_matrix([
        [0.9, None],
        [0.7, 0.8],
    ])
    assert result["fm"] == pytest.approx(0.2)
    assert result["matrix"][0][1] is None


def test_forgetting_ignores_future_task_scores():
    matrix = [
        [0.80, 0.99, 0.99],
        [0.85, 0.70, 0.99],
        [0.80, 0.60, 0.50],
    ]
    result = forgetting_matrix(matrix)

    assert result["per_task_forgetting"][0] == pytest.approx(0.05)
    assert result["per_task_forgetting"][1] == pytest.approx(0.10)
    assert result["fm"] == pytest.approx(0.075)


def test_cadic_closest_pair_blockwise_matches_direct_and_nonzero_row():
    features = torch.tensor([
        [0.0, 0.0],
        [100.0, 100.0],
        [200.0, 200.0],
        [10.0, 10.0],
        [11.0, 11.0],
        [300.0, 300.0],
    ])

    coreset = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(
            budget=6, dim=2, chunk_size=2, pair_chunk_size=2, query_chunk_size=2
        )
    )
    coreset.update(features)

    value, row = coreset._closest_pair()

    direct = torch.cdist(
        features, features, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    direct.fill_diagonal_(float("inf"))

    expected_value, flat = direct.reshape(-1).min(dim=0)
    expected_row = int(flat.item() // features.shape[0])

    assert row == expected_row == 3
    assert value.item() == pytest.approx(expected_value.item())
    assert coreset.distance_profile()["pair_max_shape"] == [2, 2]


def test_cadic_nearest_two_dimensional_chunking_matches_direct_and_ties():
    bank = torch.tensor([
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [10.0, 0.0],
    ])
    query = torch.tensor([
        [1.0, 0.0],
        [8.0, 0.0],
        [0.0, 1.0],
        [9.0, 0.0],
        [1.0, 0.0],
    ])

    coreset = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(
            budget=4, dim=2, chunk_size=2, query_chunk_size=2, pair_chunk_size=2
        )
    )
    coreset.features = bank.clone()

    values, indices = coreset._nearest(query, bank)

    reference = torch.cdist(
        query, bank, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    ref_values, ref_indices = reference.min(dim=1)

    assert torch.equal(indices, ref_indices)
    assert torch.allclose(values, ref_values)
    assert indices[0].item() == 0

    profile = coreset.distance_profile()
    assert profile["nearest_max_shape"] == [2, 2]
    assert profile["nearest_max_elements"] <= 4


def test_cadic_image_score_large_distance_is_finite_and_support_ties_stable():
    coreset = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(
            budget=3,
            dim=2,
            chunk_size=2,
            query_chunk_size=1,
            pair_chunk_size=2,
            image_neighbors=2,
        )
    )

    coreset.features = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])

    pixels, indices, images = coreset.pixel_scores(
        torch.tensor([[[1e10, 0.0]]])
    )

    assert torch.isfinite(pixels).all()
    assert torch.isfinite(images).all()
    assert indices.shape == (1, 1)
    assert coreset._topk_indices(coreset.features[0], 2).tolist() == [0, 1]


def test_cadic_load_state_is_clone_independent():
    source = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(budget=2, dim=2)
    )
    source.update(torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
    ]))

    state = source.state_dict()

    target = CADICPatchCoresetV1(
        CADICPatchCoresetConfig(budget=2, dim=2)
    )
    target.load_state_dict(state)
    target.features[0, 0] = 999.0

    assert state["features"][0, 0].item() == 1.0


def test_cadic_forgetting_uses_historical_maximum_and_separate_metrics():
    image = [
        [0.6, None, None],
        [0.8, 0.7, None],
        [0.7, 0.2, 0.9],
    ]
    pixel = [
        [0.4, None, None],
        [0.5, 0.6, None],
        [0.2, 0.1, 0.8],
    ]

    assert forgetting_matrix(image)["fm"] == pytest.approx(0.3)
    assert forgetting_matrix(pixel)["fm"] == pytest.approx(0.4)


class _ArtifactsForForgettingTest:
    def __init__(self, root):
        self.root = Path(root)
        self.states = self.root / "states"
        self.states.mkdir(parents=True, exist_ok=True)

    def write_json(self, relative_path, value):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False))


class _ForgettingDataset:
    task_names = ["task0", "task1", "task2"]

    def build_test_loader(self, task_id):
        return [{
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "masks": torch.tensor([
                [[0, 0], [0, 0]],
                [[0, 1], [0, 1]],
            ], dtype=torch.long),
            "task_id": torch.tensor([task_id, task_id]),
        }]


class _ForgettingAdapter:
    def __init__(self):
        self.state = {"step": -1}
        self.calls = []

    def state_dict(self):
        return {"step": int(self.state["step"])}

    def load_state_dict(self, state):
        self.state = {"step": int(state["step"])}

    def score_batch(self, batch):
        step = self.state["step"]
        task_id = int(batch["task_id"][0].item())
        self.calls.append((step, task_id))

        return {
            "image_scores": torch.tensor([0.1, 0.9], dtype=torch.float32),
            "anomaly_maps": torch.tensor([
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.9], [0.1, 0.9]],
            ], dtype=torch.float32),
        }


def test_engine_forgetting_is_lower_triangular(tmp_path):
    artifacts = _ArtifactsForForgettingTest(tmp_path / "run")

    for step in range(3):
        torch.save({"step": step}, artifacts.states / f"task_{step:02d}.pt")

    dataset = _ForgettingDataset()
    adapter = _ForgettingAdapter()
    engine = BenchmarkEngineV1(
        protocol={},
        adapter=adapter,
        artifacts=artifacts,
        fail_on_state_mutation=True,
    )

    result = engine.evaluate_forgetting(dataset)

    assert adapter.calls == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    assert result["matrix"][0] == [1.0, None, None]
    assert result["matrix"][1] == [1.0, 1.0, None]
    assert result["matrix"][2] == [1.0, 1.0, 1.0]
    assert result["future_tasks_evaluated_before_learning"] is False


def test_fake_engine_no_training_test_access(tmp_path):
    make_mvtec(tmp_path)
    protocol, method = configs(tmp_path)
    manifest = build_training_manifest(protocol, 0, "abc")

    proto = MVTecContinualProtocol(protocol, manifest, method, 0)
    art = BenchmarkArtifacts(tmp_path / "run")
    adapter = FakeBenchmarkAdapter()

    engine = BenchmarkEngineV1(
        protocol,
        adapter,
        art,
        fail_on_state_mutation=True,
    )

    rows = engine.train(proto)
    assert rows[0]["train_samples"] == 2
    assert (art.states / "final.pt").is_file()

    adapter.load_state_dict(
        torch.load(art.states / "final.pt", weights_only=False)
    )

    per, macro = engine.evaluate_final(proto)

    assert set(per) == {"bottle"}
    assert macro["task_count"] == 1

    fm = engine.evaluate_forgetting(proto)
    assert fm["matrix"]


def test_setup_and_run_scripts_end_to_end(tmp_path):
    make_mvtec(tmp_path)

    protocol_path = tmp_path / "p.yaml"
    method_path = tmp_path / "m.yaml"
    protocol, method = configs(tmp_path)

    import yaml

    protocol_path.write_text(yaml.safe_dump(protocol))
    method_path.write_text(yaml.safe_dump(method))

    env = dict(__import__("os").environ)
    env["MVTEC_ROOT"] = str(tmp_path)

    out = tmp_path / "out"

    cmd = [
        sys.executable,
        "scripts/benchmarks/0_setup_benchmark.py",
        "--protocol",
        str(protocol_path),
        "--method",
        str(method_path),
        "--seed",
        "0",
        "--output-root",
        str(out),
        "--smoke",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr

    run = Path(result.stdout.strip())

    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/1_run_benchmark.py",
            "--run-dir",
            str(run),
            "--phase",
            "all",
            "--max-tasks",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (run / "summary.json").is_file()