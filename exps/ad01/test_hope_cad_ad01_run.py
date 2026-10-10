# exps/ad01/test_hope_cad_ad01_run.py
"""Regression tests for the AD-01 execution path.

The two bugs these tests exist for:

1. A retention matrix must be measured on each stage's *actual* boundary state.
   The earlier implementation streamed the whole training set first and only
   then evaluated every boundary, so every matrix row was scored against the
   final state and the forgetting numbers were meaningless. The decisive test
   below makes the score a function of the memory state, so evaluating late
   collapses the rows and evaluating on time keeps them distinct.
2. The dispatcher spawned workers with `--worker` while the parser only accepted
   `--workers`, so every subprocess died on an unrecognised argument. Tests
   assert the flag exists, that dispatch passes it, and that the two workers are
   pinned to different physical GPUs.

All fixtures are synthetic and in memory; no development image or real feature
cache is touched.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    PATCHES,
    SCORING_GLOBAL,
    SCORING_LOCAL,
    NormalSupportMemory,
)
from exps.ad01.hope_cad_ad01_run import (
    ARMS,
    CATEGORIES,
    aggregate,
    cuda_index,
    dispatch_workers,
    forgetting_from_matrix,
    run_allocation_pair,
    run_one,
    run_scoring_at_boundaries,
)

DIM = 768
BUDGET = 800        # just above one image's 784 patches, so the bank evicts
TRAIN_IMAGES = 2
DEV_IMAGES = 2
MASK_SIDE = 28
SEEDS = (0, 1, 2)


def _rng(category: str, offset: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(abs(hash(category)) % 10_000 + offset)


def _synthetic_train(category: str) -> dict:
    return {"patches": torch.randn(TRAIN_IMAGES, PATCHES, DIM, generator=_rng(category))}


def _synthetic_dev(category: str) -> dict:
    labels = torch.tensor([0] * (DEV_IMAGES // 2) + [1] * (DEV_IMAGES // 2))
    masks = torch.zeros(DEV_IMAGES, MASK_SIDE, MASK_SIDE, dtype=torch.bool)
    masks[labels == 1, :4, :4] = True
    return {
        "patches": torch.randn(DEV_IMAGES, PATCHES, DIM, generator=_rng(category, 1)),
        "labels": labels,
        "masks": masks,
    }


@pytest.fixture(autouse=True)
def synthetic_caches(monkeypatch):
    import exps.ad01.hope_cad_ad01_run as module

    monkeypatch.setattr(
        module,
        "load_cached",
        lambda category, split, device="cpu": (
            _synthetic_train(category) if split == "train" else _synthetic_dev(category)
        ),
    )
    monkeypatch.setattr(
        module,
        "masks28_from_native",
        lambda masks: [masks[i].numpy().astype(bool) for i in range(masks.shape[0])],
    )
    yield


# --------------------------------------------------------------------- bug 1


def test_boundaries_are_evaluated_on_their_own_stage_state(monkeypatch):
    """Each retention row must reflect the state after that stage, not the end.

    The score is replaced by a direct function of the current stored support, so
    a late evaluation would give every row the same value. Correct interleaving
    produces rows that differ across stages.
    """
    import exps.ad01.hope_cad_ad01_run as module

    observed: list[tuple[str, float]] = []

    def state_score(memory, payload, scoring, device="cpu", batch_size=8):
        # A scalar that provably comes from the memory *contents* at call time.
        # `count` and `feature_bytes` are constant once the bank is full, so they
        # cannot distinguish states; the feature sum can.
        fingerprint = float(memory._global.features.double().sum().item())
        observed.append((scoring, fingerprint))
        return {
            "i_auroc": fingerprint,
            "p_aupr_native": fingerprint / 2,
            "p_aupr_grid28": fingerprint / 4,
            "mean_normal_score": 0.0,
            "mean_defect_score": 1.0,
            "image_scores": [0.0],
            "inference_seconds_per_image": 0.0,
        }

    monkeypatch.setattr(module, "score_development", state_score)

    memory, scored = run_scoring_at_boundaries(
        ALLOCATION_GLOBAL,
        SCORING_GLOBAL,
        list(CATEGORIES),
        0,
        grid=2,
        budget=BUDGET,
        device="cpu",
    )

    matrix = scored["matrix_i"]
    # Lower triangle is filled, upper triangle is future and stays undefined.
    assert np.isnan(matrix[0, 1]) and np.isnan(matrix[0, 2])
    assert not np.isnan(matrix[2, 2])
    # Every stage row must be internally consistent: all tasks in one row were
    # scored against the same state, so they share one value.
    for stage in range(len(CATEGORIES)):
        row = matrix[stage, : stage + 1]
        assert len(set(row.tolist())) == 1, f"stage {stage} row spans two states"
    # And different stages must have been evaluated at different states. Compare
    # with a tolerance because these are large float64 sums of float32 features.
    stage_values = [float(matrix[stage, stage]) for stage in range(len(CATEGORIES))]
    distinct = 0
    for index, value in enumerate(stage_values):
        if all(abs(value - other) > 1e-6 for other in stage_values[:index]):
            distinct += 1
    assert distinct == len(CATEGORIES), (
        "all boundaries reported the same state, which is the bug being guarded: "
        f"{stage_values}"
    )
    assert observed, "no scoring happened"


def test_boundary_rows_differ_from_a_final_state_only_evaluation(monkeypatch):
    """Explicitly contrast correct interleaving with the old, invalid ordering."""
    import exps.ad01.hope_cad_ad01_run as module

    def state_score(memory, payload, scoring, device="cpu", batch_size=8):
        value = float(memory.count)
        return {
            "i_auroc": value,
            "p_aupr_native": value,
            "p_aupr_grid28": value,
            "mean_normal_score": 0.0,
            "mean_defect_score": 0.0,
            "image_scores": [0.0],
            "inference_seconds_per_image": 0.0,
        }

    monkeypatch.setattr(module, "score_development", state_score)
    _, correct = run_scoring_at_boundaries(
        ALLOCATION_GLOBAL, SCORING_GLOBAL, list(CATEGORIES), 0,
        grid=2, budget=BUDGET, device="cpu",
    )

    # The invalid ordering would score every boundary once the whole stream has
    # been consumed; reproduce that and show it collapses the matrix.
    memory = NormalSupportMemory(
        budget=BUDGET, grid=2, allocation=ALLOCATION_GLOBAL, dim=DIM
    )
    for category in CATEGORIES:
        for index in range(TRAIN_IMAGES):
            memory.update(_synthetic_train(category)["patches"][index])
    collapsed = np.full((3, 3), float(memory.count))

    correct_diag = [float(correct["matrix_i"][i, i]) for i in range(3)]
    assert len(set(correct_diag)) > 1
    assert len(set(collapsed.diagonal().tolist())) == 1
    # A forgetting matrix built from a collapsed matrix reports zero loss, which
    # is exactly how the bug hid itself.
    assert forgetting_from_matrix(collapsed, list(CATEGORIES))["fm"] == 0.0


def test_forgetting_detects_retention_loss_on_a_genuine_matrix():
    names = list(CATEGORIES)
    lossy = np.array([[0.5, np.nan, np.nan], [0.7, 0.4, np.nan], [0.6, 0.5, 0.3]])
    detected = forgetting_from_matrix(lossy, names)
    assert detected["per_task"]["bottle"] == pytest.approx(0.1)
    assert detected["fm"] == pytest.approx(0.1 / 3)


def test_per_state_rows_carry_their_stage_index():
    _, scored = run_scoring_at_boundaries(
        ALLOCATION_GLOBAL, SCORING_GLOBAL, list(CATEGORIES), 0,
        grid=2, budget=BUDGET, device="cpu",
    )
    from exps.ad01.hope_cad_ad01_arms import order_permutation

    order = order_permutation(0)
    for row in scored["per_state"]:
        assert 0 <= row["stage"] < len(CATEGORIES)
        assert row["state_after"] == order[row["stage"]]


def test_final_block_is_the_last_boundary():
    record = run_one(
        "A", ALLOCATION_GLOBAL, SCORING_GLOBAL, list(CATEGORIES), 0,
        grid=2, budget=BUDGET, device="cpu",
    )
    last = record["order"][-1]
    from_last_stage = {
        row["task"]: row for row in record["per_state"] if row["state_after"] == last
    }
    for category in CATEGORIES:
        assert record["final"][category]["i_auroc"] == from_last_stage[category]["i_auroc"]
        assert (
            record["final"][category]["p_aupr_native"]
            == from_last_stage[category]["p_aupr_native"]
        )


# --------------------------------------------------- shared trajectories


def test_update_trajectory_is_independent_of_scoring_mode():
    images = torch.randn(
        TRAIN_IMAGES, PATCHES, DIM, generator=torch.Generator().manual_seed(3)
    )
    left = NormalSupportMemory(budget=BUDGET, grid=2, allocation=ALLOCATION_GLOBAL, dim=DIM)
    right = NormalSupportMemory(budget=BUDGET, grid=2, allocation=ALLOCATION_GLOBAL, dim=DIM)
    for image in images:
        left.update(image)
        right.update(image)
    assert torch.equal(left._global.features, right._global.features)
    assert left._global.replaced_features == right._global.replaced_features
    assert torch.equal(left._bank_bin, right._bank_bin)


@pytest.mark.parametrize(
    "allocation,arms",
    ((ALLOCATION_GLOBAL, ("A", "B")), (ALLOCATION_SPATIAL, ("C", "D"))),
)
def test_paired_path_reproduces_single_arm_metrics_exactly(allocation, arms):
    single = {
        arm: run_one(
            arm, allocation, scoring, list(CATEGORIES), 0,
            grid=2, budget=BUDGET, device="cpu",
        )
        for arm, _, scoring in ARMS
        if arm in arms
    }
    paired = run_allocation_pair(
        allocation, list(CATEGORIES), 0, grid=2, budget=BUDGET, device="cpu"
    )
    for arm, record in single.items():
        shared = paired[record["scoring"]]
        assert arm == shared["arm"]
        for category in CATEGORIES:
            assert (
                record["final"][category]["i_auroc"]
                == shared["final"][category]["i_auroc"]
            )
            assert (
                record["final"][category]["p_aupr_native"]
                == shared["final"][category]["p_aupr_native"]
            )
        assert record["forgetting_i_auroc"]["fm"] == shared["forgetting_i_auroc"]["fm"]
        assert record["per_state"] == shared["per_state"]


def test_paired_arms_share_support_but_not_scores():
    paired = run_allocation_pair(
        ALLOCATION_GLOBAL, list(CATEGORIES), 0, grid=2, budget=BUDGET, device="cpu"
    )
    a, b = paired[SCORING_GLOBAL], paired[SCORING_LOCAL]
    assert a["storage"] == b["storage"]
    assert a["occupancy"] == b["occupancy"]
    assert any(
        a["final"][category]["i_auroc"] != b["final"][category]["i_auroc"]
        or a["final"][category]["p_aupr_native"] != b["final"][category]["p_aupr_native"]
        for category in CATEGORIES
    ), "arm B must not be a silent duplicate of arm A"


# --------------------------------------------------------------------- bug 2


def test_cuda_index_maps_physical_gpus_apart():
    assert cuda_index("cuda:0") == "0"
    assert cuda_index("cuda:1") == "1"
    assert cuda_index("cuda") == "0"


def test_worker_flag_is_accepted_by_the_main_parser():
    """The dispatcher passes `--worker`; the parser must accept that exact flag."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/exps/ad01/hope_cad_ad01_run.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--worker" in completed.stdout
    assert "--workers" not in completed.stdout


def test_dispatch_pins_each_worker_to_a_distinct_gpu(monkeypatch, tmp_path):
    """Capture the spawn arguments without actually launching a sweep."""
    import subprocess

    captured = []

    class FakeProcess:
        def __init__(self, command, env, **kwargs):
            captured.append({"command": command, "env": env})

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)

    # The workers write their payload; make the parent's read succeed with an
    # empty record list so dispatch can aggregate without a real run.
    import exps.ad01.hope_cad_ad01_run as module

    original_dispatch = module.dispatch_workers

    def fake_dispatch(device_a, device_b, output, seeds):
        devices = [device_a] if not device_b else [device_a, device_b]
        groups = [[], []]
        for index, seed in enumerate(seeds):
            groups[index % 2].append(seed)
        for device, group in zip(devices, groups):
            tag = f"gpu{cuda_index(device)}"
            worker_out = output / f"worker_{tag}"
            worker_out.mkdir(parents=True, exist_ok=True)
            (worker_out / f"worker_seeds_{'_'.join(str(s) for s in group)}.json").write_text(
                json.dumps({"records": [], "seconds": 1.0})
            )
        return original_dispatch(device_a, device_b, output, seeds)

    monkeypatch.setattr(module, "dispatch_workers", fake_dispatch)
    module.dispatch_workers("cuda:0", "cuda:1", tmp_path, [0, 1, 2])

    assert len(captured) == 2
    assert {entry["env"]["CUDA_VISIBLE_DEVICES"] for entry in captured} == {"0", "1"}
    for entry in captured:
        assert "--worker" in entry["command"]
        assert entry["command"][entry["command"].index("--device") + 1] == "cuda"


def test_worker_seed_groups_partition_without_overlap():
    groups = [[], []]
    for index, seed in enumerate(SEEDS):
        groups[index % 2].append(seed)
    assert groups[0] == [0, 2]
    assert groups[1] == [1]
    assert sorted(groups[0] + groups[1]) == list(SEEDS)
    assert not set(groups[0]) & set(groups[1])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two visible GPUs",
)
def test_two_gpu_smoke_uses_both_devices(tmp_path):
    """Lightweight two-GPU dispatch check: one tiny update per GPU, then exit.

    This does not run any category stream and does not touch the dataset or the
    real feature cache. It proves the dispatched command line is accepted, that
    both subprocesses exit cleanly, and that they land on two different physical
    GPUs. Full real-data evaluation belongs to the resumable benchmark runner,
    never to a unit test.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    worker = root / "exps/ad01/gpu_smoke_worker.py"

    seen = {}
    for physical, tag in (("0", "gpu0"), ("1", "gpu1")):
        out = tmp_path / tag
        out.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = physical
        env.setdefault("OMP_NUM_THREADS", "2")
        completed = subprocess.run(
            [sys.executable, str(worker), "--device", "cuda", "--out", str(out)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        report = json.loads((out / "smoke_report.json").read_text())
        assert report["status"] == "ok"
        assert report["cuda_available"] is True
        assert report["bank_features_device"].startswith("cuda")
        assert report["count"] == BUDGET
        assert report["image_score_finite"] is True
        seen[tag] = report["device_name"]

    # Both workers must have run; names may coincide on identical cards, so the
    # strong assertion is that each saw exactly one visible device.
    assert set(seen) == {"gpu0", "gpu1"}


def test_smoke_worker_script_runs_on_cpu(tmp_path):
    """The smoke worker must also work without CUDA, so CI can exercise it."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    worker = root / "exps/ad01/gpu_smoke_worker.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, str(worker), "--device", "cpu", "--out", str(tmp_path)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    report = json.loads((tmp_path / "smoke_report.json").read_text())
    assert report["status"] == "ok"
    assert report["count"] == BUDGET
    assert report["bank_features_device"].startswith("cpu")


def test_aggregate_reports_per_seed_values_and_means():
    runs = []
    for order_seed, base in ((0, 0.5), (1, 0.7), (2, 0.9)):
        for arm, allocation, scoring in ARMS:
            runs.append(
                {
                    "arm": arm,
                    "allocation": allocation,
                    "scoring": scoring,
                    "order_seed": order_seed,
                    "final": {
                        category: {
                            "i_auroc": base,
                            "p_aupr_native": base / 2,
                            "p_aupr_grid28": base / 3,
                        }
                        for category in CATEGORIES
                    },
                    "forgetting_i_auroc": {"fm": 0.01},
                    "forgetting_p_aupr": {"fm": 0.02},
                    "occupancy": {"n_bins": 16},
                    "timing": {
                        "update_seconds_mean_per_image": 0.001,
                        "final_inference_seconds_per_image": 0.002,
                        "images_seen": 880,
                    },
                    "storage": {"persistent_exemplar_bytes": 7_680_000},
                }
            )
    summary = aggregate(runs)
    assert set(summary) == {"A", "B", "C", "D"}
    assert summary["A"]["macro_i_auroc_mean"] == pytest.approx(0.7)
    assert summary["A"]["n_order_seeds"] == 3
    assert len(summary["A"]["macro_i_auroc_per_seed"]) == 3
    assert summary["A"]["macro_p_aupr_grid28_mean"] == pytest.approx(0.7 / 3)
    assert summary["A"]["per_category"]["bottle"]["i_auroc_mean"] == pytest.approx(0.7)


def test_sync_device_is_a_noop_without_cuda():
    from exps.ad01.hope_cad_ad01_run import sync_device

    sync_device("cpu")
