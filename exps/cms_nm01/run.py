# exps/cms_nm01/run.py
"""Normal-only CMS-NM-01 runner and pre-label gate evaluator.

The runner deliberately never loads the development anomaly manifest.  The
separate locked anomaly evaluator is invoked only after ``gates.json`` records
G1 and G2 PASS.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .cms_nd import CMSNormalDensityMemory
from .gaussian_control import StreamingGaussian64
from .audit import isolated_resource_audit, projection_audit


ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT / "results/hope_cad/ad01_phase0/arms_gpu/features"
RESULT_ROOT = ROOT / "results/hope_cad/cms_nm01"
CATEGORIES = ("bottle", "carpet", "hazelnut")
UPDATE_LIMITS = {"bottle": 150, "carpet": 100, "hazelnut": 100}
PROBE_RANGES = {"bottle": (150, 170), "carpet": (100, 120), "hazelnut": (100, 120)}
ORDER_SEEDS = (0, 1, 2)
CALIBRATION_COUNT = 20
DIAGNOSTIC_IMAGES_PER_CATEGORY = 4


def set_runtime() -> None:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.set_num_interop_threads(1)


def order_for_seed(seed: int) -> tuple[str, ...]:
    orders = tuple(itertools.permutations(CATEGORIES))
    return tuple(orders[int(seed) % len(orders)])


def load_train_features() -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for category in CATEGORIES:
        path = FEATURE_ROOT / f"{category}_train.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        patches = payload["patches"].to(dtype=torch.float32).contiguous()
        if patches.ndim != 3 or patches.shape[1:] != (784, 768):
            raise ValueError(f"unexpected feature shape for {category}: {tuple(patches.shape)}")
        values[category] = patches
    return values


def probe_features(train: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        category: train[category][start:end].contiguous()
        for category, (start, end) in PROBE_RANGES.items()
    }


def state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def finite_state(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(value).all().item() for value in model.state_dict().values() if torch.is_floating_point(value))


def make_model(arm: str, *, seed: int) -> CMSNormalDensityMemory | StreamingGaussian64:
    if arm == "CMS-ND-1":
        return CMSNormalDensityMemory(levels=(1,), learning_rates=(1e-3,), retention=(1e-3,), seed=seed)
    if arm in {"CMS-ND-3", "CMS-ND-FROZEN"}:
        return CMSNormalDensityMemory(levels=(1, 8, 64), learning_rates=(1e-3, 1e-3, 1e-3), retention=(1e-3, 1e-3, 1e-3), seed=seed)
    if arm == "GAUSS-64":
        return StreamingGaussian64()
    raise ValueError(f"unknown arm {arm}")


def consume_normal_event(model: CMSNormalDensityMemory | StreamingGaussian64, image: torch.Tensor) -> dict[str, Any]:
    if isinstance(model, CMSNormalDensityMemory):
        return dataclasses.asdict(model.commit_image(image.unsqueeze(0)))
    else:
        model.update_image(image.unsqueeze(0))
        return {"patch_count": int(model.count)}


def score(model: CMSNormalDensityMemory | StreamingGaussian64, images: torch.Tensor) -> dict[str, torch.Tensor]:
    return model.score_patches(images)


def normal_metrics(
    model: CMSNormalDensityMemory | StreamingGaussian64,
    probes: Mapping[str, torch.Tensor],
    diagnostic_patches: torch.Tensor,
) -> dict[str, Any]:
    per_category: dict[str, Any] = {}
    all_mixture: list[torch.Tensor] = []
    all_levels: list[list[torch.Tensor]] = []
    inference_elapsed = 0.0
    for category in CATEGORIES:
        inference_start = time.perf_counter()
        result = score(model, probes[category])
        inference_elapsed += time.perf_counter() - inference_start
        mixture = result["mixture_nll"].detach().cpu()
        levels = [value.detach().cpu() for value in result["level_nll"]]
        all_mixture.append(mixture.reshape(-1))
        all_levels.append(levels)
        per_category[category] = {
            "image_nll": mixture.mean(dim=1).tolist(),
            "mixture_mean": float(mixture.mean().item()),
            "mixture_q50": float(torch.quantile(mixture.reshape(-1), 0.50).item()),
            "mixture_q95": float(torch.quantile(mixture.reshape(-1), 0.95).item()),
            "mixture_q99": float(torch.quantile(mixture.reshape(-1), 0.99).item()),
            "mixture_max": float(mixture.max().item()),
            "level_mean": [float(level.mean().item()) for level in levels],
            "level_q95": [float(torch.quantile(level.reshape(-1), 0.95).item()) for level in levels],
            "level_q99": [float(torch.quantile(level.reshape(-1), 0.99).item()) for level in levels],
        }
    metrics: dict[str, Any] = {
        "per_category": per_category,
        "batched_inference_seconds_per_image": float(inference_elapsed / sum(value.shape[0] for value in probes.values())),
        "persistent_bytes": int(model.persistent_bytes()),
        "state_digest": state_digest(model),
        "finite_state": finite_state(model),
    }
    if isinstance(model, CMSNormalDensityMemory):
        metrics["projection_digest"] = model.projection.digest()
        metrics["projection_orthonormal_error"] = model.projection.orthonormal_error()
        metrics["representation"] = model.saturation_and_rank(diagnostic_patches)
        metrics["online_counters"] = {
            "events": int(model.completed_events),
            "updates": [int(level.update_count) for level in model.levels],
            "pending": [int(level.pending_count) for level in model.levels],
            "accumulated_gradient_norms": [float(torch.sqrt(sum(value.square().sum() for value in level.accumulators()))) for level in model.levels],
            "parameter_anchor_distances": [float(torch.sqrt(sum((parameter.detach() - anchor).square().sum() for parameter, anchor in zip(level.current_parameters(), level.anchors())))) for level in model.levels],
        }
    else:
        z = model.project(diagnostic_patches).reshape(-1, model.mean.numel())
        centered = z - z.mean(0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        metrics["representation"] = {
            "projection_rank": int((singular > max(float(singular.max()) * 1e-6, 1e-12)).sum().item()),
            "projection_top1_energy": float((singular[0].square() / singular.square().sum().clamp_min(1e-12)).item()),
            "level_scale_saturation": [],
            "finite": bool(torch.isfinite(z).all().item()),
        }
    return metrics


def bootstrap_difference(
    frozen: Sequence[float],
    online: Sequence[float],
    *,
    seed: int,
    samples: int = 4000,
) -> dict[str, float]:
    a = np.asarray(frozen, dtype=np.float64)
    b = np.asarray(online, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("bootstrap arrays must be matching vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(samples, len(a)))
    delta = (a[indices] - b[indices]).mean(axis=1)
    return {"mean_frozen_minus_online": float((a - b).mean()), "ci_low": float(np.quantile(delta, 0.025)), "ci_high": float(np.quantile(delta, 0.975))}


def run_arm(
    arm: str,
    order_seed: int,
    train: Mapping[str, torch.Tensor],
    probes: Mapping[str, torch.Tensor],
    diagnostic_patches: torch.Tensor,
    output: Path = RESULT_ROOT,
) -> dict[str, Any]:
    order = order_for_seed(order_seed)
    model = make_model(arm, seed=7101)
    calibration_category = order[0]
    update_seconds: list[float] = []
    initial_metrics = normal_metrics(model, probes, diagnostic_patches)
    event_trace = []
    for index in range(CALIBRATION_COUNT):
        event_trace.append({"category": calibration_category, "image_index": index, "phase": "calibration", **consume_normal_event(model, train[calibration_category][index])})
    calibration_metrics = normal_metrics(model, probes, diagnostic_patches)
    frozen_metrics = calibration_metrics
    stage_records: list[dict[str, Any]] = [{"stage": 0, "category": "calibration", "metrics": calibration_metrics}]
    checkpoint_dir = output / "checkpoints" / f"seed{order_seed}_{arm}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "boundary0.pt")

    for stage, category in enumerate(order, start=1):
        start = CALIBRATION_COUNT if stage == 1 else 0
        stop = UPDATE_LIMITS[category]
        if arm == "CMS-ND-FROZEN":
            start = stop
        stage_start = time.perf_counter()
        for index in range(start, stop):
            event_start = time.perf_counter()
            event_trace.append({"category": category, "image_index": index, "phase": "online", **consume_normal_event(model, train[category][index])})
            update_seconds.append(time.perf_counter() - event_start)
        stage_elapsed = time.perf_counter() - stage_start
        metrics = normal_metrics(model, probes, diagnostic_patches)
        if isinstance(model, CMSNormalDensityMemory):
            if not metrics["finite_state"] or any(value > 0.10 for value in metrics["representation"]["level_scale_saturation"]):
                raise FloatingPointError(f"normal stability stop: seed={order_seed}, arm={arm}, stage={stage}")
        metrics["stage_update_seconds"] = float(stage_elapsed)
        metrics["stage_update_seconds_per_image"] = float(stage_elapsed / max(stop - start, 1))
        stage_records.append({"stage": stage, "category": category, "metrics": metrics})
        torch.save(model.state_dict(), checkpoint_dir / f"boundary{stage}.pt")
    record = {
        "arm": arm,
        "order_seed": int(order_seed),
        "order": list(order),
        "calibration_category": calibration_category,
        "initial_metrics": initial_metrics,
        "event_trace": event_trace,
        "calibration_state": {"path": str(checkpoint_dir / "boundary0.pt"), "digest": calibration_metrics["state_digest"]},
        "calibration_metrics": frozen_metrics,
        "stages": stage_records,
        "update_seconds_per_image_mean": float(np.mean(update_seconds)) if update_seconds else 0.0,
        "update_seconds_per_image_median": float(np.median(update_seconds)) if update_seconds else 0.0,
        "final_metrics": stage_records[-1]["metrics"],
        "checkpoint_dir": str(checkpoint_dir),
        "resource_audit": isolated_resource_audit(model.clone_read_only() if isinstance(model, CMSNormalDensityMemory) else make_gaussian_clone(model), probes[order[-1]][:1]),
    }
    return record


def make_gaussian_clone(model: StreamingGaussian64) -> StreamingGaussian64:
    clone = StreamingGaussian64()
    clone.load_state_dict(model.state_dict())
    return clone


def data_contract() -> dict:
    identities = {}
    for category in CATEGORIES:
        payload = torch.load(FEATURE_ROOT / f"{category}_train.pt", map_location="cpu", weights_only=False)
        paths = [str(path) for path in payload["paths"]]
        update = paths[:UPDATE_LIMITS[category]]
        start, end = PROBE_RANGES[category]
        heldout = paths[start:end]
        if set(update) & set(heldout) or any("/train/good/" not in path for path in update + heldout):
            raise ValueError("normal identity contract violated")
        if any(int(Path(path).stem) != index for index, path in enumerate(paths[:end])):
            raise ValueError("cache identity order differs from locked numeric normal identities")
        identities[category] = {"update": update, "heldout": heldout, "intersection": []}
    return {"normal_only": True, "identities": identities, "development_labels_accessed": False, "sealed_confirmation_accessed": False}


def evaluate_gates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(row["arm"], int(row["order_seed"])): row for row in records}
    g1_rows = []
    bootstrap_rows = []
    categories_pass = 0
    nonzero_ci = 0
    worst_worsening = -math.inf
    for seed in ORDER_SEEDS:
        online = by_key[("CMS-ND-3", seed)]["final_metrics"]["per_category"]
        frozen = by_key[("CMS-ND-FROZEN", seed)]["final_metrics"]["per_category"]
        for category in CATEGORIES:
            frozen_images = frozen[category]["image_nll"]
            online_images = online[category]["image_nll"]
            baseline = max(abs(float(np.mean(frozen_images))), 1e-6)
            relative_improvement = (float(np.mean(frozen_images)) - float(np.mean(online_images))) / baseline
            worst_worsening = max(worst_worsening, -relative_improvement)
            boot = bootstrap_difference(frozen_images, online_images, seed=9000 + seed * 10 + CATEGORIES.index(category))
            row = {"seed": seed, "category": category, "relative_improvement": relative_improvement, **boot}
            g1_rows.append(row)
            bootstrap_rows.append(row)
            if relative_improvement >= 0.01:
                categories_pass += 1
            if boot["ci_low"] > 0:
                nonzero_ci += 1
    # G1 is evaluated per order seed: at least two category gains in at least one
    # paired seed, while no category may worsen by more than five percent.
    seed_pass = {
        seed: sum(row["relative_improvement"] >= 0.01 for row in g1_rows if row["seed"] == seed) >= 2
        for seed in ORDER_SEEDS
    }
    g1_pass = any(seed_pass.values()) and worst_worsening <= 0.05 and nonzero_ci >= 1
    gaussian_rows = []
    gaussian_seed_pass = {}
    for seed in ORDER_SEEDS:
        category_passes = 0
        for category in CATEGORIES:
            neural = by_key[("CMS-ND-3", seed)]["final_metrics"]["per_category"][category]["mixture_mean"]
            gaussian = by_key[("GAUSS-64", seed)]["final_metrics"]["per_category"][category]["mixture_mean"]
            gain = (gaussian - neural) / max(abs(gaussian), 1e-6)
            gaussian_rows.append({"seed": seed, "category": category, "relative_neural_gain": gain})
            category_passes += gain >= 0.01
        gaussian_seed_pass[seed] = category_passes >= 2
    parsimony_pass = any(gaussian_seed_pass.values())
    g2_rows = []
    g2_pass = True
    rank_initial = None
    for seed in ORDER_SEEDS:
        online_row = by_key[("CMS-ND-3", seed)]
        frozen_row = by_key[("CMS-ND-FROZEN", seed)]
        online_final = online_row["final_metrics"]["per_category"]
        frozen_final = frozen_row["final_metrics"]["per_category"]
        calibration = online_row["calibration_metrics"]["per_category"]
        representation = online_row["final_metrics"]["representation"]
        rank_initial = float(online_row["calibration_metrics"]["representation"]["projection_effective_rank"])
        for category in CATEGORIES:
            final_value = float(online_final[category]["mixture_mean"])
            frozen_value = float(frozen_final[category]["mixture_mean"])
            calibration_value = float(calibration[category]["mixture_mean"])
            vs_frozen = (final_value - frozen_value) / max(abs(frozen_value), 1e-6)
            vs_calibration = (final_value - calibration_value) / max(abs(calibration_value), 1e-6)
            acquired_stage = online_row["order"].index(category) + 1
            acquired_value = float(online_row["stages"][acquired_stage]["metrics"]["per_category"][category]["mixture_mean"])
            vs_acquired = (final_value - acquired_value) / max(abs(acquired_value), 1e-6)
            row = {"seed": seed, "category": category, "vs_frozen_relative": vs_frozen, "vs_calibration_relative": vs_calibration, "vs_post_acquisition_relative": vs_acquired}
            g2_rows.append(row)
            # The resolved protocol declares frozen and post-calibration
            # comparisons only. Post-acquisition loss is essential scientific
            # evidence, but was not declared as a blocking numeric gate.
            if vs_frozen > 0.10 or vs_calibration > 0.20:
                g2_pass = False
        for boundary in online_row["stages"]:
            boundary_metrics = boundary["metrics"]
            representation = boundary_metrics["representation"]
            if not boundary_metrics["finite_state"] or not representation["finite"]:
                g2_pass = False
            if representation["projection_effective_rank"] < 0.50 * rank_initial:
                g2_pass = False
            if any(value > 0.10 for value in representation["level_scale_saturation"]):
                g2_pass = False
    return {
        "g1": {"pass": bool(g1_pass), "acquisition_pass": bool(g1_pass), "parsimony_pass": bool(parsimony_pass), "gaussian_rows": gaussian_rows, "gaussian_seed_pass": gaussian_seed_pass, "rows": g1_rows, "seed_pass": seed_pass, "categories_pass": categories_pass, "bootstrap_ci_excluding_zero_count": nonzero_ci, "worst_relative_worsening": worst_worsening},
        "g2": {"pass": bool(g2_pass), "rows": g2_rows, "initial_projection_rank": rank_initial,
               "definition": "original written protocol: final vs frozen <=10%, vs calibration <=20%, finite/rank/scale checks",
               "post_acquisition_diagnostic_pass_at_20pct": all(row["vs_post_acquisition_relative"] <= 0.20 for row in g2_rows)},
        "scientific_gate_pass": bool(g1_pass and g2_pass),
        "anomaly_evaluation_authorized": bool(g1_pass and g2_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    set_runtime()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "normal_identity_contract.json").write_text(json.dumps(data_contract(), indent=2, sort_keys=True))
    train = load_train_features()
    probes = probe_features(train)
    diagnostic_patches = torch.cat([probes[category][:DIAGNOSTIC_IMAGES_PER_CATEGORY] for category in CATEGORIES], dim=0)
    (args.output / "projection_audit.json").write_text(json.dumps(projection_audit(probes), indent=2, sort_keys=True))
    records: list[dict[str, Any]] = []
    for seed in ORDER_SEEDS:
        for arm in ("CMS-ND-3", "CMS-ND-1", "CMS-ND-FROZEN", "GAUSS-64"):
            print(f"[cms-nm01] START seed={seed} arm={arm}", flush=True)
            record = run_arm(arm, seed, train, probes, diagnostic_patches, args.output)
            records.append(record)
            (args.output / "normal_only_records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
            print(f"[cms-nm01] DONE seed={seed} arm={arm} final_bytes={record['final_metrics']['persistent_bytes']}", flush=True)
    (args.output / "normal_only_records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    gates = evaluate_gates(records)
    (args.output / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True))
    print(json.dumps({"g1": gates["g1"]["pass"], "g2": gates["g2"]["pass"], "authorized": gates["anomaly_evaluation_authorized"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

