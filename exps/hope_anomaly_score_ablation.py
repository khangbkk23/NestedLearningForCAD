# exps/hope_anomaly_score_ablation.py
"""Immutable checkpoint verification and controlled associative score readouts."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.nn import functional as F
import yaml

from exps.hope_anomaly_signal import (
    CATEGORIES, CHECKPOINTS, METHOD_NAMES, PooledCovariance,
    bootstrap_image_counts, native_mask, pixel_map, sha256,
)
from exps.hope_image_synchronous_memory import fingerprint
from models.cadic_patch_coreset_v1 import CADICPatchCoresetV1
from scripts.exps import hope_anomaly_signal as original


COSINE_EPS = 1e-8
METRICS = ("image_AUROC", "image_AP", "pixel_AUPR", "pixel_AUROC")


def file_manifest(root: Path) -> list[dict[str, Any]]:
    """Identify original artifacts without copying or modifying their contents."""
    return [{"relative_path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha256(p)} for p in sorted(root.rglob("*")) if p.is_file()]


def assert_files_unchanged(root: Path, records: list[dict[str, Any]]) -> None:
    if file_manifest(root) != records:
        raise ValueError("original artifacts changed during read-only continuation")


def original_process_status(pid_file: Path) -> dict[str, Any]:
    """Only the exact original Python runner is considered a live source job."""
    pid = int(pid_file.read_text().strip())
    cmd_file = Path(f"/proc/{pid}/cmdline")
    if not cmd_file.exists():
        return {"pid": pid, "alive": False, "identity": "STALE_PID", "command": None}
    parts = cmd_file.read_bytes().split(b"\0")
    parts = [p.decode(errors="replace") for p in parts if p]
    match = bool(parts and Path(parts[0]).name.startswith("python")) and any(
        Path(p).as_posix().endswith("scripts/exps/hope_anomaly_signal.py") for p in parts)
    return {"pid": pid, "alive": match, "identity": "RUNNING" if match else "STALE_PID",
            "command": parts}


@torch.no_grad()
def angular_components(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    if a.shape != b.shape or a.ndim != 2 or not len(a):
        raise ValueError("paired nonempty patch vectors required")
    if not a.is_floating_point() or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("matching floating dtype/device required")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite paired vectors")
    angle = 1 - F.cosine_similarity(a, b, dim=-1, eps=COSINE_EPS)
    # FP64 reporting avoids cancellation in the L2 decomposition; the score
    # itself remains the explicitly prescribed FP32 cosine implementation.
    ad, bd = a.detach().double(), b.detach().double()
    norm_a, norm_b = ad.norm(dim=-1), bd.norm(dim=-1)
    raw = (ad - bd).square().sum(-1)
    radial = (norm_a - norm_b).square()
    angular = 2 * norm_a * norm_b * angle.double()
    well_defined = (norm_a >= COSINE_EPS) & (norm_b >= COSINE_EPS)
    error = (raw - radial - angular).abs()
    tolerance = 32 * torch.finfo(a.dtype).eps * raw.clamp_min(1)
    if bool((error[well_defined] > tolerance[well_defined]).any()):
        raise ValueError("well-defined angular/radial decomposition differs")
    return {"angle": angle.detach().clone(), "norm_a": norm_a, "norm_b": norm_b,
            "raw": raw, "radial": radial, "angular": angular,
            "well_defined": well_defined, "decomposition_error": error}


@torch.no_grad()
def angular_scores(model, features: torch.Tensor, *, initial_model=None, capture_components=False) -> dict[str, Any]:
    before = model.state_fingerprint()
    snapshot = model.snapshot_state()
    quantities = model.generate_update_quantities(features, snapshot)
    prediction = F.linear(quantities.keys, snapshot.weights["memory"])
    components = angular_components(prediction, quantities.values)
    scores = components["angle"]
    result = {
        "patch_scores": scores.detach().cpu().clone(), "image_score": float(scores.max()),
        "snapshot_hash": snapshot.identity,
        "near_zero_prediction_fraction": float((components["norm_a"] < COSINE_EPS).double().mean()),
        "near_zero_value_fraction": float((components["norm_b"] < COSINE_EPS).double().mean()),
        "prediction_norm_median": float(components["norm_a"].median()),
        "value_norm_median": float(components["norm_b"].median()),
        "raw_component_mean": float(components["raw"].mean()),
        "radial_component_mean": float(components["radial"].mean()),
        "angular_component_mean": float(components["angular"].mean()),
        "radial_fraction": float(components["radial"].sum() / components["raw"].sum().clamp_min(1e-12)),
        "decomposition_max_abs_error": float(components["decomposition_error"].max()),
        "memory_readout_rms": float(model.read_from_snapshot(quantities, snapshot).double().square().mean().sqrt()),
    }
    if initial_model is not None:
        initial_before = initial_model.state_fingerprint()
        initial = initial_model.snapshot_state()
        old_quantities = initial_model.generate_update_quantities(features, initial)
        old_prediction = F.linear(old_quantities.keys, initial.weights["memory"])
        old_scores = angular_components(old_prediction, old_quantities.values)["angle"]
        delta = scores.double() - old_scores.double()
        result.update({"reset_score_relative_l2": float(delta.norm() / old_scores.double().norm().clamp_min(1e-12)),
                       "history_angle_mean_delta": float(delta.mean()),
                       "history_angle_max_abs_delta": float(delta.abs().max()),
                       "history_angle_delta_std": float(delta.std(unbiased=False))})
        if initial_model.state_fingerprint() != initial_before:
            raise ValueError("initial reference mutated during scoring")
    if not torch.isfinite(scores).all() or model.state_fingerprint() != before:
        raise ValueError("angular scoring is non-finite or mutated source state")
    if capture_components:
        result["patch_components"] = {key: value.detach().cpu().clone().numpy()
                                      for key, value in components.items()}
    return result


@torch.no_grad()
def affinity_components(features: torch.Tensor, readout: torch.Tensor, grid_shape=(28, 28)) -> dict[str, torch.Tensor]:
    """Compare frozen and mutable cosine relations on the original spatial grid."""
    height, width = grid_shape
    if features.ndim != 2 or features.shape != readout.shape or height <= 0 or width <= 0 or \
            height * width != len(features) or len(features) < 2:
        raise ValueError("matching patch vectors and a nontrivial spatial grid required")
    if not features.is_floating_point() or features.dtype != readout.dtype or features.device != readout.device:
        raise ValueError("matching floating dtype/device required")
    if not torch.isfinite(features).all() or not torch.isfinite(readout).all():
        raise ValueError("non-finite spatial vectors")
    x, m = (value.reshape(height, width, -1) for value in (features, readout))
    scores = features.new_zeros(height, width)
    feature_sum, memory_sum = torch.zeros_like(scores), torch.zeros_like(scores)
    counts = torch.zeros_like(scores)
    # Each offset contributes once to each valid source coordinate. Ordered
    # slice additions avoid duplicated-index accelerator reductions.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not (dy or dx):
                continue
            y0, y1 = max(0, -dy), min(height, height - dy)
            x0, x1 = max(0, -dx), min(width, width - dx)
            if y1 <= y0 or x1 <= x0:
                continue
            source = (slice(y0, y1), slice(x0, x1))
            neighbor = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
            feature_cosine = F.cosine_similarity(x[source], x[neighbor], dim=-1, eps=COSINE_EPS)
            memory_cosine = F.cosine_similarity(m[source], m[neighbor], dim=-1, eps=COSINE_EPS)
            scores[source].add_((feature_cosine - memory_cosine).square())
            feature_sum[source].add_(feature_cosine)
            memory_sum[source].add_(memory_cosine)
            counts[source].add_(1)
    if bool((counts == 0).any()):
        raise ValueError("spatial patch has no valid neighbors")
    return {"affinity": (scores / counts).flatten().clone(),
            "feature_affinity_mean": (feature_sum / counts).flatten(),
            "memory_affinity_mean": (memory_sum / counts).flatten(),
            "neighbor_count": counts.flatten(),
            "norm_m": readout.double().norm(dim=-1), "norm_x": features.double().norm(dim=-1)}


@torch.no_grad()
def affinity_scores(model, features: torch.Tensor, *, initial_model=None, capture_components=False,
                    grid_shape=(28, 28)) -> dict[str, Any]:
    before = model.state_fingerprint()
    snapshot = model.snapshot_state()
    quantities = model.generate_update_quantities(features, snapshot)
    readout = model.read_from_snapshot(quantities, snapshot)
    components = affinity_components(features[0], readout, grid_shape)
    scores = components["affinity"]
    result = {"patch_scores": scores.cpu().clone(), "image_score": float(scores.max()),
              "snapshot_hash": snapshot.identity,
              "near_zero_readout_fraction": float((components["norm_m"] < COSINE_EPS).double().mean()),
              "near_zero_feature_fraction": float((components["norm_x"] < COSINE_EPS).double().mean()),
              "query_readout_norm_median": float(components["norm_m"].median()),
              "feature_norm_median": float(components["norm_x"].median()),
              "memory_readout_rms": float(readout.double().square().mean().sqrt())}
    if initial_model is not None:
        initial_before = initial_model.state_fingerprint()
        initial = initial_model.snapshot_state()
        old_quantities = initial_model.generate_update_quantities(features, initial)
        old_readout = initial_model.read_from_snapshot(old_quantities, initial)
        old_scores = affinity_components(features[0], old_readout, grid_shape)["affinity"]
        delta = scores.double() - old_scores.double()
        result.update({"memory_reset_relative_l2": float((readout.double() - old_readout.double()).norm() /
                                                        old_readout.double().norm().clamp_min(1e-12)),
                       "memory_reset_cosine": float(F.cosine_similarity(readout, old_readout, dim=-1, eps=COSINE_EPS).mean()),
                       "history_affinity_relative_l2": float(delta.norm() / old_scores.double().norm().clamp_min(1e-12)),
                       "history_affinity_mean_delta": float(delta.mean()),
                       "history_affinity_max_abs_delta": float(delta.abs().max()),
                       "history_affinity_delta_std": float(delta.std(unbiased=False))})
        if initial_model.state_fingerprint() != initial_before:
            raise ValueError("initial reference mutated during spatial scoring")
    if not torch.isfinite(scores).all() or model.state_fingerprint() != before:
        raise ValueError("spatial scoring is non-finite or mutated source state")
    if capture_components:
        result["patch_components"] = {key: value.detach().cpu().clone().numpy() for key, value in components.items()}
    return result


def verify_unit_arrays(payload: dict, arrays: dict, subset: list[dict], *, native_image_score=False) -> None:
    ids = [row["image_id"] for row in subset]
    if arrays["image_ids"].tolist() != ids or len(payload["images"]) != len(ids):
        raise ValueError("evaluation image identity/order differs")
    if arrays["patch_scores"].shape != (len(ids), 784) or arrays["image_scores"].shape != (len(ids),):
        raise ValueError("evaluation patch geometry differs")
    if not np.isfinite(arrays["patch_scores"]).all() or not np.isfinite(arrays["image_scores"]).all():
        raise ValueError("non-finite stored evaluation scores")
    for actual, expected, score in zip(payload["images"], subset, arrays["image_scores"]):
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError("stored image label, mask or source identity differs")
        if actual["image_score"] != score:
            raise ValueError("stored scalar image score differs")
    if not native_image_score:
        np.testing.assert_array_equal(arrays["image_scores"], arrays["patch_scores"].max(axis=1))
    for metric in ("image_AUROC", "pixel_AUPR"):
        if arrays[f"bootstrap_{metric}"].shape != (400,) or not np.isfinite(arrays[f"bootstrap_{metric}"]).all():
            raise ValueError("paired image-bootstrap artifact differs")
    labels = np.asarray([r["label"] for r in subset])
    context = payload["context"]
    counts = bootstrap_image_counts(labels, seed=1000 * context["seed"] + CATEGORIES.index(context["category"]))
    pos, neg = labels == 1, labels == 0
    scores = arrays["image_scores"]
    greater = (scores[pos, None] > scores[None, neg]).astype(float)
    greater += .5 * (scores[pos, None] == scores[None, neg])
    exact_boot = ((counts[:, pos] @ greater) * counts[:, neg]).sum(1) / (pos.sum() * neg.sum())
    np.testing.assert_allclose(exact_boot, arrays["bootstrap_image_AUROC"], atol=1e-12, rtol=0)
    if payload["metrics"]["pixel_bootstrap_point_error"] > 2e-4:
        raise ValueError("original pixel interval approximation exceeds its declared bound")


def _verify_point_metrics(job: tuple[str, str]) -> dict[str, Any]:
    torch.set_num_threads(1)
    stem, data_root = map(Path, job)
    payload = original.read_json(stem.with_suffix(".json"))
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        patches = archive["patch_scores"]
        scores = archive["image_scores"]
    rows = payload["images"]
    labels = np.asarray([row["label"] for row in rows])
    masks = [native_mask(data_root, row) for row in rows]
    maps = [pixel_map(torch.from_numpy(array), mask.shape) for array, mask in zip(patches, masks)]
    target = np.concatenate([m.ravel() for m in masks]).astype(np.int8)
    values = np.concatenate([m.ravel() for m in maps]).astype(np.float32)
    exact = {"image_AUROC": float(roc_auc_score(labels, scores)),
             "image_AP": float(average_precision_score(labels, scores)),
             "pixel_AUPR": float(average_precision_score(target, values)),
             "pixel_AUROC": float(roc_auc_score(target, values)),
             "pixel_prevalence": float(target.mean())}
    for key, value in exact.items():
        if abs(value - payload["metrics"][key]) > 1e-12:
            raise ValueError(f"exact point metric mismatch: {stem.name} {key}")
    context = payload["context"]
    counts = bootstrap_image_counts(labels, seed=1000 * context["seed"] + CATEGORIES.index(context["category"]))
    image_boot = np.array([roc_auc_score(labels, scores, sample_weight=c) for c in counts])
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
        np.testing.assert_allclose(image_boot, archive["bootstrap_image_AUROC"], rtol=0, atol=1e-12)
    return {"unit": str(stem), "passed": True, "exact_points": exact}


def verify_original(source: Path, device: torch.device, *, workers=2, notify=print) -> dict[str, Any]:
    """Reopen the complete result grid and independently verify its evidence."""
    status = original_process_status(original.ROOT / "logs/hope_cad/anomaly_signal_gate.pid")
    if status["alive"]:
        raise RuntimeError("original experiment is still running")
    summary = original.read_json(source / "summary.json")
    config = yaml.safe_load((source / "config_resolved.yaml").read_text())
    if summary["status"] != "COMPLETED" or original.read_json(source / "progress.json")["phase"] != "completed":
        raise ValueError("original experiment lacks terminal completion")
    original.validate_core()
    records = file_manifest(source)
    dev = original.table_records(source / "manifests/anomaly_dev_manifest.parquet")
    confirmation = original.table_records(source / "manifests/anomaly_confirmation_manifest.parquet")
    dev_ids, sealed_ids = {r["image_id"] for r in dev}, {r["image_id"] for r in confirmation}
    if len(dev_ids) != 60 or dev_ids & sealed_ids or summary["confirmation_evaluated"] or config["confirmation_evaluated"]:
        raise ValueError("development/confirmation isolation differs")
    for name, rows in (("development", dev), ("confirmation", confirmation)):
        if original.manifest_identity(rows) != config[f"{name}_manifest_identity"]:
            raise ValueError("predeclared split identity differs")
        if original.table_records(source / f"anomaly_{'dev' if name == 'development' else 'confirmation'}_manifest.parquet") != rows:
            raise ValueError("duplicate manifest copies disagree")
    for category in CATEGORIES:
        subset = [r for r in dev if r["category"] == category]
        if sorted(r["label"] for r in subset) != [0] * 10 + [1] * 10:
            raise ValueError("development class/label counts differ")
    for row in dev:
        pieces = Path(row["relative_path"]).parts
        if pieces[0] != row["category"] or pieces[1] != "test" or int(pieces[2] != "good") != row["label"]:
            raise ValueError("development path/label mapping differs")
        if sha256(original.DATA_ROOT / row["relative_path"]) != row["image_sha256"]:
            raise ValueError("development image identity changed")
        with Image.open(original.DATA_ROOT / row["relative_path"]) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("development image geometry changed")
        mask = native_mask(original.DATA_ROOT, row)
        if row["label"]:
            if sha256(original.DATA_ROOT / row["mask_path"]) != row["mask_sha256"] or not mask.any():
                raise ValueError("development mask identity/labels differ")
        elif mask.any():
            raise ValueError("normal image acquired a defect mask")
    rows, patches = original.evaluation_data(source)
    cache_check = original.read_json(source / "feature_cache_validation.json")
    if not cache_check["passed"] or sha256(source / "features/development.pt") != cache_check["sha256"]:
        raise ValueError("original feature cache identity differs")
    if config["core_hashes"] != original.CORE_HASHES or config["checkpoints"] != list(CHECKPOINTS):
        raise ValueError("source scientific configuration differs")
    if config["CADIC"]["source_sha256"] != sha256(original.ROOT / "models/cadic_patch_coreset_v1.py"):
        raise ValueError("original CADIC source changed")
    if config["CADIC"]["config"] != original.CADIC_CONFIG.__dict__:
        raise ValueError("original CADIC budget/scoring configuration differs")
    if config["CADIC"]["config_sha256"] != sha256(original.ROOT / config["CADIC"]["config_path"]):
        raise ValueError("original CADIC configuration file changed")
    fixture = original.load_fixture()
    if fixture["initialization_hash"] != config["initialization_hash"]:
        raise ValueError("original initial state differs")
    state_manifest = pd.read_parquet(source / "states/smt_state_manifest.parquet")
    if len(state_manifest) != 48 or state_manifest.duplicated(["seed", "method", "checkpoint"]).any():
        raise ValueError("source checkpoint manifest is incomplete/duplicated")
    for row in state_manifest.to_dict("records"):
        if sha256(Path(row["source_path"])) != row["source_sha256"]:
            raise ValueError("original checkpoint file identity changed")
    metrics = pd.read_parquet(source / "metrics_by_checkpoint.parquet")
    images = pd.read_parquet(source / "image_scores.parquet")
    expected = {(s, m, e, c) for s in range(3) for m in (*METHOD_NAMES.values(), "COVARIANCE", "CADIC")
                for e in CHECKPOINTS for c in CATEGORIES}
    keys = ["seed", "method", "checkpoint", "category"]
    if len(metrics) != 216 or metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].values)) != expected:
        raise ValueError("original metric grid is incomplete/duplicated")
    if len(images) != 3960 or images.duplicated(keys + ["image_id"]).any() or set(images.image_id) != dev_ids:
        raise ValueError("original scored identities are incomplete/duplicated or leaked")
    if not np.isfinite(images.image_score).all():
        raise ValueError("non-finite original image score")
    valid = metrics[metrics.status == "EVALUATED"]
    if len(valid) != 198 or not np.isfinite(valid[list(METRICS)]).all().all():
        raise ValueError("original evaluated metric values differ")
    not_fitted = metrics[metrics.status != "EVALUATED"]
    if len(not_fitted) != 18 or not not_fitted.method.isin(["COVARIANCE", "CADIC"]).all() or not not_fitted.checkpoint.eq(0).all():
        raise ValueError("statistical initial-state handling differs")
    basis = original.smt_model("FROZEN", 0, 0, fixture, device)
    state_checks, point_jobs, aliases = [], {}, {}
    max_replay_abs = 0.0
    for seed in range(3):
        stream = original.table_records(source / f"seed{seed}/stream_manifest.parquet")
        reference_stream = pd.read_parquet(original.MEMORY_ROOT / f"seed{seed}/stream_manifest.parquet")
        reference_stream = reference_stream[(reference_stream.role == "update") & (reference_stream.event_id <= 300)]
        reference_stream = reference_stream.astype(object).where(pd.notna(reference_stream), None).to_dict("records")
        if stream != reference_stream or len(stream) != 300 or [r["class_name"] for r in stream] != [c for c in CATEGORIES for _ in range(100)]:
            raise ValueError("normal stream/order pairing differs")
        if any("/train/good/" not in r["relative_path"] for r in stream):
            raise ValueError("non-normal source in training stream")
        stream_id = original.manifest_identity(stream)
        for method in (*METHOD_NAMES.values(), "COVARIANCE", "CADIC"):
            for event in CHECKPOINTS:
                if event == 0 and method in ("COVARIANCE", "CADIC"):
                    continue
                if method in METHOD_NAMES.values():
                    raw = next(k for k, v in METHOD_NAMES.items() if v == method)
                    model = original.smt_model(raw, seed, event, fixture, device)
                    state_before = model.state_fingerprint()
                    stats = model.memory_stats()
                    if not stats["finite"] or stats["persistent_graph"]:
                        raise ValueError("source memory is degenerate or graph-bearing")
                    updates = event * (49 if raw == "P0" else 1) if raw != "FROZEN" else 0
                    if int(model.smt.online_update_count) != 2 * updates:
                        raise ValueError("source total update counter differs")
                    static = model.smt.state_dict()
                    mutable = {f"memories.{name}.weight" for name in model.smt.memories} | {
                        "memory_update_count", "auxiliary_update_count", "online_update_count"}
                    for key, value in static.items():
                        if key not in mutable:
                            reference = fixture["smt"][key]
                            if isinstance(value, torch.Tensor):
                                if not torch.equal(value.detach().cpu(), reference):
                                    raise ValueError("static/reset source changed in original training")
                            elif value != reference:
                                raise ValueError("source initialization schema changed")
                    scorer = lambda image: original.residual_scores(model, image.to(device))
                    state_values = lambda: model.state_fingerprint()
                else:
                    saved = torch.load(source / f"states/seed{seed}/{method}_event{event}.pt", map_location="cpu", weights_only=False)
                    if saved["stream_identity"] != stream_id or saved["event"] != event:
                        raise ValueError("statistical training checkpoint identity differs")
                    if method == "COVARIANCE":
                        model = PooledCovariance(768, device)
                        model.load_state_dict(saved["state"])
                        if int(model.count) != event * 784 or int(model.completed_events) != event:
                            raise ValueError("pooled covariance counts differ")
                        covariance = model.scatter / (int(model.count) - 1)
                        expected_cov = .9 * covariance + (.1 * covariance.trace() / 768 + 1e-6) * torch.eye(768, dtype=torch.float64, device=device)
                        torch.testing.assert_close(model.factor @ model.factor.T, expected_cov, rtol=1e-10, atol=1e-10)
                        def scorer(image):
                            q = basis.generate_update_quantities(image.to(device), basis.snapshot_state()).queries
                            score = model.score(q)
                            return {"patch_scores": score, "image_score": float(score.max())}
                    else:
                        model = CADICPatchCoresetV1(original.CADIC_CONFIG, device=device)
                        model.load_state_dict(saved["state"])
                        if model.seen_features != event * 784 or model.count > original.CADIC_CONFIG.budget:
                            raise ValueError("CADIC feature count/budget differs")
                        def scorer(image):
                            scalar, score = model.score(image.to(device))
                            return {"patch_scores": score.detach().cpu(), "image_score": float(scalar[0])}
                    state_values = lambda: fingerprint(model.state_dict())
                    state_before = state_values()
                for category in CATEGORIES:
                    subset = [r for r in rows if r["category"] == category]
                    stem = source / f"seed{seed}/units/{method}_event{event}_{category}"
                    payload = original.read_json(stem.with_suffix(".json"))
                    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
                        arrays = {k: archive[k] for k in archive.files}
                    verify_unit_arrays(payload, arrays, subset, native_image_score=method == "CADIC")
                    group = valid[(valid.seed == seed) & (valid.method == method) & (valid.checkpoint == event) & (valid.category == category)].iloc[0]
                    for metric in METRICS:
                        if payload["metrics"][metric] != group[metric]:
                            raise ValueError("unit and consolidated metric disagree")
                    for index, row in enumerate(subset):
                        feature_id = next(i for i, r in enumerate(rows) if r["image_id"] == row["image_id"])
                        result = scorer(patches[feature_id:feature_id + 1])
                        actual = result["patch_scores"].reshape(784).numpy()
                        reference = arrays["patch_scores"][index]
                        np.testing.assert_allclose(actual, reference, atol=2e-4, rtol=3.1e-5)
                        if not np.isclose(result["image_score"], arrays["image_scores"][index], atol=2e-4, rtol=3.1e-5):
                            raise ValueError("source checkpoint does not reproduce original image score")
                        if method in METHOD_NAMES.values() and result["snapshot_hash"] != payload["images"][index]["snapshot_hash"]:
                            raise ValueError("original evaluated source state differs")
                        max_replay_abs = max(max_replay_abs, float(np.abs(actual - reference).max()))
                    signature = hashlib.sha256(category.encode() + arrays["patch_scores"].tobytes() + arrays["image_scores"].tobytes()).hexdigest()
                    if signature not in point_jobs:
                        point_jobs[signature] = (str(stem), str(original.DATA_ROOT))
                    aliases[str(stem)] = signature
                if state_values() != state_before:
                    raise ValueError("checkpoint verification mutated persistent state")
                state_checks.append({"seed": seed, "method": method, "checkpoint": event,
                                     "state_unchanged": True, "state_fingerprint": state_before})
                notify({"phase": "checkpoint_verified", "seed": seed, "method": method, "checkpoint": event})
                del model
    exact = {}
    jobs = list(point_jobs.items())
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = pool.map(_verify_point_metrics, [job for _, job in jobs])
            for (signature, _), result in zip(jobs, results):
                exact[signature] = result
                notify({"phase": "exact_points_verified", "completed": len(exact), "total_unique": len(jobs)})
    else:
        for signature, job in jobs:
            exact[signature] = _verify_point_metrics(job)
            notify({"phase": "exact_points_verified", "completed": len(exact), "total_unique": len(jobs)})
    for stem, signature in aliases.items():
        payload = original.read_json(Path(stem).with_suffix(".json"))
        for key, value in exact[signature]["exact_points"].items():
            if abs(value - payload["metrics"][key]) > 1e-12:
                raise ValueError("reused exact-score metric differs")
    # Units retain paired resamples; verify the saved learned-minus-frozen
    # intervals directly, without regenerating pixel-bootstrap approximations.
    deltas = pd.read_parquet(source / "method_deltas.parquet")
    for row in deltas.to_dict("records"):
        stem = source / f"seed{row['seed']}/units/{row['method']}_event{row['checkpoint']}_{row['category']}"
        frozen = stem.with_name(f"FROZEN_event{row['checkpoint']}_{row['category']}")
        with np.load(stem.with_suffix(".npz")) as own, np.load(frozen.with_suffix(".npz")) as base:
            for metric in ("image_AUROC", "pixel_AUPR"):
                low, high = np.quantile(own[f"bootstrap_{metric}"] - base[f"bootstrap_{metric}"], [.025, .975])
                if not np.isclose(low, row[f"delta_{metric}_ci_low"], atol=1e-12) or not np.isclose(high, row[f"delta_{metric}_ci_high"], atol=1e-12):
                    raise ValueError("saved paired bootstrap interval differs")
    original.validate_core()
    assert_files_unchanged(source, records)
    return {"status": "COMPLETED", "execution_valid": True, "process": status,
            "development_images": len(dev), "confirmation_images": len(confirmation),
            "confirmation_untouched": True, "evaluated_units": 198, "metric_grid_rows": 216,
            "scored_images": 3960, "exact_unique_point_verifications": len(exact),
            "checkpoint_checks": state_checks, "score_replay_max_abs_difference": max_replay_abs,
            "score_replay_tolerance": {"atol": 2e-4, "rtol": 3.1e-5},
            "CADIC_image_score_exception": "predeclared native support-neighbor weighting, not a causal readout ablation",
            "covariance_completed": True, "CADIC_completed": True, "original_files": records,
            "sealed_manifest_identity": config["confirmation_manifest_identity"],
            "development_manifest_identity": config["development_manifest_identity"],
            "core_hashes": original.CORE_HASHES}
