# exps/ad01/hope_cad_ad01_run.py
"""Driver for the AD-01 four-arm comparison on the permitted development split.

Runs the predeclared two-factor design (allocation x scoring) over the frozen
bottle/carpet/hazelnut development identities, at three paired category-order
seeds, and reports image AUROC, native-resolution pixel AUPR, the secondary
28x28 pixel AUPR, task-boundary forgetting, support occupancy, update time,
inference latency and persistent bytes.

Nothing here reads the sealed confirmation manifest, an anomaly label inside a
fitting path, or any future task. CPU execution only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from exps.ad01.hope_cad_ad01_arms import (
    CATEGORIES,
    CHECKPOINT,
    DEV_MANIFEST,
    DIAGNOSTIC_GRIDS,
    IMAGE_NEIGHBORS,
    METHOD_CONFIG,
    ORDER_SEEDS,
    PRIMARY_GRID,
    PROTOCOL_CONFIG,
    TOTAL_BUDGET,
    average_precision,
    auroc,
    dev_rows_for,
    evaluate_task,
    forgetting_from_matrix,
    load_dev_manifest,
    order_permutation,
)
from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    PATCHES,
    SCORING_GLOBAL,
    SCORING_LOCAL,
    NormalSupportMemory,
)
from models.feature_extractors.cadic_vit_v1 import CADICViTConfig, CADICViTFeatureExtractor

ROOT = Path(__file__).resolve().parents[2]

# Distance-kernel choice for the frozen CADIC replacement rule. The
# matrix-multiply variant is numerically validated against the reference by
# `test_fast_coreset_matches_reference_on_identical_input`, so enabling it
# cannot silently change the experiment. Measured effect: see
# `results/hope_cad/ad01_phase0/update_profile.json`.
FAST_CORESET = True

# CPU and GPU namespaces are kept strictly separate. cuBLAS/cuDNN reductions are
# not bit-identical to CPU ones, so a GPU run must not reuse CPU feature caches
# or CPU arm checkpoints: mixing them would silently blend two numerical
# provenances into one reported comparison.
BASE = ROOT / "results/hope_cad/ad01_phase0"
NAMESPACE = {"cpu": BASE / "arms_cpu", "cuda": BASE / "arms_gpu"}
FEATURE_ROOT = {"cpu": BASE / "arms_cpu/features", "cuda": BASE / "arms_gpu/features"}

ARMS = (
    ("A", ALLOCATION_GLOBAL, SCORING_GLOBAL),
    ("B", ALLOCATION_GLOBAL, SCORING_LOCAL),
    ("C", ALLOCATION_SPATIAL, SCORING_GLOBAL),
    ("D", ALLOCATION_SPATIAL, SCORING_LOCAL),
)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_tag(device: str) -> str:
    return f"GPU{cuda_index(device)}" if str(device).startswith("cuda") else "CPU"


def log(message: str, device: str = "cpu", tag: str = "") -> None:
    """Flushed, timestamped progress line.

    Flushing matters: when a worker's stdout is redirected to a file Python
    block-buffers it, so an unflushed worker log can stay empty for the entire
    run and make a healthy sweep look hung.
    """
    stamp = time.strftime("%H:%M:%S")
    prefix = f"[{stamp}][{gpu_tag(device)}]"
    if tag:
        prefix += f"[{tag}]"
    print(f"{prefix} {message}", flush=True)


def gpu_memory_note(device: str) -> str:
    """Cheap GPU memory readout. Does not synchronise the device."""
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return ""
    try:
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        free, total = torch.cuda.mem_get_info()
        return (
            f"gpu_mem alloc={allocated:.0f}MiB reserved={reserved:.0f}MiB "
            f"free={free / (1024 ** 2):.0f}MiB total={total / (1024 ** 2):.0f}MiB"
        )
    except Exception as error:  # memory reporting must never break a run
        return f"gpu_mem unavailable ({error})"



def load_protocol_and_method() -> tuple[dict, dict]:
    import os

    raw = PROTOCOL_CONFIG.read_text()
    protocol = yaml.safe_load(raw.replace("${MVTEC_ROOT}", os.environ.get("MVTEC_ROOT", str(ROOT / "data/mvtec"))))
    method = yaml.safe_load(METHOD_CONFIG.read_text())
    method = dict(method)
    method["extractor"] = dict(method["extractor"])
    method["extractor"]["checkpoint_path"] = str(CHECKPOINT)
    method["extractor"]["checkpoint_identity"] = "vit_b8_imagenet21k_local"
    return protocol, method


def build_extractor(method: dict, device: str = "cpu") -> CADICViTFeatureExtractor:
    config = dict(method["extractor"])
    config.update(
        image_size=method["preprocessing"]["image_size"],
        mean=tuple(method["preprocessing"]["mean"]),
        std=tuple(method["preprocessing"]["std"]),
    )
    allowed = {k: v for k, v in config.items() if k in CADICViTConfig.__dataclass_fields__}
    return CADICViTFeatureExtractor(CADICViTConfig(**allowed), torch.device(device))


def build_manifest(protocol: dict, seed: int) -> dict:
    from dataset.benchmark_manifest_v1 import build_training_manifest

    return build_training_manifest(protocol, seed, "ad01")


def device_key(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def feature_root(device: str) -> Path:
    return FEATURE_ROOT[device_key(device)]


def namespace_root(device: str) -> Path:
    return NAMESPACE[device_key(device)]


def cache_path(category: str, split: str, device: str = "cpu") -> Path:
    return feature_root(device) / f"{category}_{split}.pt"


def ensure_feature_cache(
    protocol: dict, method: dict, force: bool = False, device: str = "cpu"
) -> dict[str, Any]:
    """Extract and cache frozen features once for every arm and seed.

    Caching is a determinism device as much as a speed device: every arm then
    reads byte-identical inputs, so no arm can differ from another through a
    repeated extraction. The extractor, checkpoint and preprocessing are those
    of the frozen CADIC reference.
    """
    from dataset.benchmark_protocol_v1 import MVTecContinualProtocol

    dev = load_dev_manifest()
    manifest = build_manifest(protocol, 0)
    protocol_obj = MVTecContinualProtocol(protocol, manifest, method, 0, 0)
    extractor = build_extractor(method, device)

    feature_root(device).mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"categories": {}, "checkpoint": str(CHECKPOINT)}

    for category in CATEGORIES:
        task_id = protocol["task_order"].index(category)

        train_target = cache_path(category, "train", device)
        if force or not train_target.is_file():
            loader = protocol_obj.build_train_loader(task_id)
            chunks, paths = [], []
            with torch.no_grad():
                for batch in loader:
                    chunks.append(
                        extractor.extract_patch_features(batch["images"].to(device)).cpu()
                    )
                    paths.extend(batch["relative_path"])
            torch.save({"patches": torch.cat(chunks, 0), "paths": paths}, train_target)
        payload = torch.load(train_target, map_location="cpu", weights_only=False)
        report["categories"][f"{category}_train"] = {
            "images": int(payload["patches"].shape[0]),
            "path": str(train_target.relative_to(ROOT)),
        }

        test_target = cache_path(category, "dev", device)
        if force or not test_target.is_file():
            rows = dev_rows_for(dev, category)
            wanted = {row["relative_path"]: row for row in rows}
            loader = protocol_obj.build_test_loader(task_id)
            patches, masks, paths, labels = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    keep = [
                        index
                        for index, path in enumerate(batch["relative_path"])
                        if path in wanted
                    ]
                    if not keep:
                        continue
                    selector = torch.tensor(keep, dtype=torch.long)
                    patches.append(
                        extractor.extract_patch_features(
                            batch["images"].index_select(0, selector).to(device)
                        ).cpu()
                    )
                    masks.append(batch["masks"].index_select(0, selector))
                    labels.extend(int(batch["labels"][index]) for index in keep)
                    paths.extend(batch["relative_path"][index] for index in keep)
            if sorted(paths) != sorted(wanted):
                raise ValueError(
                    f"development identity mismatch for {category}: "
                    f"{len(paths)} found vs {len(wanted)} expected"
                )
            torch.save(
                {
                    "patches": torch.cat(patches, 0),
                    "masks": torch.cat(masks, 0),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "paths": paths,
                },
                test_target,
            )
        payload = torch.load(test_target, map_location="cpu", weights_only=False)
        report["categories"][f"{category}_dev"] = {
            "images": int(payload["patches"].shape[0]),
            "normal": int((payload["labels"] == 0).sum()),
            "defect": int((payload["labels"] == 1).sum()),
            "path": str(test_target.relative_to(ROOT)),
        }
    return report


def load_cached(category: str, split: str, device: str = "cpu") -> dict[str, Any]:
    """Load a cached feature payload and place it on the working device.

    Placement is a throughput requirement, not a convenience. Once the bank is
    full, the CADIC update rule forms an O(M^2) distance matrix against the
    stored bank roughly five hundred times per image. If the incoming patches
    sit on the host while the bank is on the GPU, every one of those passes
    copies the whole bank across PCIe, which dominated the measured update cost.
    """
    path = cache_path(category, split, device)
    if not path.is_file():
        raise FileNotFoundError(f"missing feature cache {path}; run --cache-only first")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    moved: dict[str, Any] = {}
    # Only the feature tensors travel to the working device. Labels and masks
    # are consumed as numpy for metric computation and must stay on the host.
    host_keys = {"masks", "labels"}
    for key, value in payload.items():
        if torch.is_tensor(value) and key not in host_keys:
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def masks28_from_native(masks: torch.Tensor) -> list[np.ndarray]:
    down = torch.nn.functional.interpolate(
        masks.float().unsqueeze(1), size=(28, 28), mode="nearest"
    ).squeeze(1)
    return [down[index].numpy().astype(bool) for index in range(down.shape[0])]


def sync_device(device: str) -> None:
    """Block until queued GPU work finishes.

    Without this, a wall-clock measurement around a CUDA call records only the
    asynchronous kernel-launch time, which understates real latency and is not
    reproducible across devices or load conditions.
    """
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def score_development(
    memory: NormalSupportMemory,
    dev_payload: dict[str, Any],
    scoring: str,
    device: str = "cpu",
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score one development set and return image and pixel metrics."""
    patches = dev_payload["patches"]
    masks = dev_payload["masks"]
    labels = dev_payload["labels"].numpy().astype(np.int64)
    # Masks are used at native resolution through numpy, so keep them on host.
    masks = masks.cpu() if torch.is_tensor(masks) else masks
    masks28 = masks28_from_native(masks)

    image_chunks, pixel_chunks = [], []
    latency = None
    for start in range(0, patches.shape[0], batch_size):
        stop = min(start + batch_size, patches.shape[0])
        sync_device(device)
        begin = time.perf_counter()
        images, pixel = memory.score(patches[start:stop], scoring=scoring)
        sync_device(device)
        elapsed = time.perf_counter() - begin
        if latency is None:
            latency = elapsed / (stop - start)
        image_chunks.append(images)
        pixel_chunks.append(pixel)

    image_scores = torch.cat(image_chunks).detach().cpu().numpy().astype(np.float64)
    pixel_scores = torch.cat(pixel_chunks).detach().cpu().numpy().astype(np.float64)

    native_scores, native_masks = [], []
    for index in range(pixel_scores.shape[0]):
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(pixel_scores[index]).reshape(1, 1, 28, 28),
            size=masks[index].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        native_scores.append(resized.reshape(-1))
        native_masks.append(masks[index].numpy().reshape(-1))

    return {
        "i_auroc": auroc(labels, image_scores),
        "p_aupr_native": average_precision(
            np.concatenate(native_masks), np.concatenate(native_scores)
        ),
        "p_aupr_grid28": average_precision(
            np.concatenate([m.reshape(-1) for m in masks28]), pixel_scores.reshape(-1)
        ),
        "mean_normal_score": float(image_scores[labels == 0].mean()),
        "mean_defect_score": float(image_scores[labels == 1].mean()),
        "image_scores": image_scores.tolist(),
        "inference_seconds_per_image": latency,
    }


def run_stage_updates(
    allocation: str,
    order: Sequence[str],
    order_seed: int,
    grid: int,
    budget: int,
    device: str = "cpu",
) -> NormalSupportMemory:
    """Build one memory and stream the training images in the given task order."""
    set_determinism(1000 + order_seed)
    memory = NormalSupportMemory(
        budget=budget,
        grid=grid,
        allocation=allocation,
        image_neighbors=IMAGE_NEIGHBORS,
        device=device,
        fast=FAST_CORESET,
    )
    train = {
        category: load_cached(category, "train", device) for category in CATEGORIES
    }
    for category in order:
        patches = train[category]["patches"]
        for index in range(patches.shape[0]):
            memory.update(patches[index])
    return memory


def run_scoring_at_boundaries(
    allocation: str,
    scoring: str,
    order: Sequence[str],
    order_seed: int,
    grid: int,
    budget: int,
    device: str = "cpu",
    freeze: list[dict[str, Any]] | None = None,
    replay: list[dict[str, Any]] | None = None,
) -> tuple[NormalSupportMemory, dict[str, Any]]:
    """Interleave updates with boundary evaluation, for one scoring mode.

    When `freeze` is given, the bank contents are captured after each stage's
    updates. When `replay` is given, each stage's frozen bank is restored instead
    of re-streaming the category. Both modes leave the stored support identical,
    because the CADIC rules are deterministic and the update is independent of
    the scoring mode, so a second scoring pass can reuse the first pass's
    trajectory. That roughly halves the sweep's dominant cost.

    This is the correctness-critical routine. A stage's retention row must be
    measured on the memory state *as it was after that stage*, so the evaluation
    has to happen immediately after that stage's updates, before the next stage
    mutates the state further. Evaluating every boundary only once the whole
    stream has been consumed would score the final state under the label of
    earlier stages and invalidate the forgetting matrix entirely.

    The update stream itself is deterministic and depends only on
    `(allocation, order, seed, grid, budget)`, never on the scoring mode, so the
    A/B and C/D pairs still traverse an identical trajectory even though each is
    evaluated in its own pass.
    """
    set_determinism(1000 + order_seed)
    memory = NormalSupportMemory(
        budget=budget,
        grid=grid,
        allocation=allocation,
        image_neighbors=IMAGE_NEIGHBORS,
        device=device,
        fast=FAST_CORESET,
    )
    train = {
        category: load_cached(category, "train", device) for category in CATEGORIES
    }
    dev = {category: load_cached(category, "dev", device) for category in CATEGORIES}

    names = list(order)
    matrix_i = np.full((len(names), len(names)), np.nan)
    matrix_p = np.full((len(names), len(names)), np.nan)
    per_state: list[dict[str, Any]] = []
    last_stage_results: dict[str, dict[str, Any]] = {}
    update_seconds: list[float] = []

    tag_prefix = f"Seed{order_seed}"
    if replay is not None and len(replay) != len(names):
        raise ValueError("replay snapshot count does not match the task order")
    for stage, category in enumerate(names):
        patches = train[category]["patches"]
        total = int(patches.shape[0])
        if replay is not None:
            memory.restore_banks(replay[stage])
            log(
                f"[{allocation}/{scoring}] {category} restored frozen bank "
                f"({memory.count}/{budget}) instead of re-streaming",
                device,
                tag_prefix,
            )
        log(
            f"[{allocation}/{scoring}] {category} update START "
            f"({total} images, stage {stage + 1}/{len(names)})",
            device,
            tag_prefix,
        )
        stage_started = time.perf_counter()
        last_report = stage_started
        # Synchronising around every single update serialises the GPU and was
        # itself a measurable cost. Progress is therefore reported on a coarse
        # cadence, and the stage is timed as a whole.
        batch_started = time.perf_counter()
        for index in range(0 if replay is not None else patches.shape[0]):
            memory.update(patches[index])

            # Progress every 25 images or 30 seconds, matching the requested
            # cadence. Reading the clock does not synchronise the device, so
            # these lines do not distort the measured throughput.
            if (index + 1) % 25 == 0 or (time.perf_counter() - last_report) > 30:
                elapsed = time.perf_counter() - batch_started
                rate = (index + 1) / elapsed if elapsed > 0 else 0.0
                remaining = (total - index - 1) / rate if rate > 0 else 0.0
                log(
                    f"[{allocation}/{scoring}] {category} update "
                    f"{index + 1}/{total} | {rate:.1f} img/s | "
                    f"eta {remaining:.0f}s | bank {memory.count}/{budget}",
                    device,
                    tag_prefix,
                )
                last_report = time.perf_counter()
        sync_device(device)
        per_image = (time.perf_counter() - batch_started) / max(1, patches.shape[0])
        update_seconds.extend([per_image] * patches.shape[0])
        if freeze is not None:
            # Freeze immediately after this stage's updates: this is exactly the
            # state the retention row for this stage must be measured against.
            freeze.append(memory.clone_banks())
        stage_elapsed = time.perf_counter() - stage_started
        note = gpu_memory_note(device)
        log(
            f"[{allocation}/{scoring}] {category} update COMPLETE in "
            f"{stage_elapsed:.1f}s" + (f" | {note}" if note else ""),
            device,
            tag_prefix,
        )

        # Evaluate every task learned so far against the state that exists
        # right now, i.e. after exactly this stage's updates.
        for task_index, evaluated in enumerate(names[: stage + 1]):
            log(
                f"[{allocation}/{scoring}] scoring {evaluated} START "
                f"(boundary {stage + 1}/{len(names)})",
                device,
                tag_prefix,
            )
            score_started = time.perf_counter()
            result = score_development(memory, dev[evaluated], scoring, device=device)
            log(
                f"[{allocation}/{scoring}] scoring {evaluated} COMPLETE in "
                f"{time.perf_counter() - score_started:.1f}s | "
                f"I-AUROC={result['i_auroc']:.4f} "
                f"P-AUPR(native)={result['p_aupr_native']:.4f} "
                f"P-AUPR(28)={result['p_aupr_grid28']:.4f}",
                device,
                tag_prefix,
            )
            matrix_i[stage, task_index] = result["i_auroc"]
            matrix_p[stage, task_index] = result["p_aupr_native"]
            per_state.append(
                {
                    "state_after": category,
                    "stage": stage,
                    "task": evaluated,
                    "i_auroc": result["i_auroc"],
                    "p_aupr_native": result["p_aupr_native"],
                    "p_aupr_grid28": result["p_aupr_grid28"],
                    "mean_normal_score": result["mean_normal_score"],
                    "mean_defect_score": result["mean_defect_score"],
                }
            )
            if stage == len(names) - 1 and evaluated in CATEGORIES:
                last_stage_results[evaluated] = result

    timing = {
        "update_seconds_total": float(sum(update_seconds)),
        "update_seconds_mean_per_image": (
            float(np.mean(update_seconds)) if update_seconds else None
        ),
        "images_seen": len(update_seconds),
    }
    return memory, {
        "per_state": per_state,
        "final": last_stage_results,
        "matrix_i": matrix_i,
        "matrix_p": matrix_p,
        "order": names,
        "timing": timing,
    }


def assemble_record(
    arm: str,
    allocation: str,
    scoring: str,
    memory: NormalSupportMemory,
    scored: dict[str, Any],
    update_timing: dict[str, Any],
    grid: int,
    budget: int,
    order_seed: int,
    device: str,
) -> dict[str, Any]:
    """Assemble one arm's record from a shared state and one scoring pass."""
    final_results = scored["final"]
    return {
        "arm": arm,
        "device": device,
        "allocation": allocation,
        "scoring": scoring,
        "grid": grid,
        "budget": budget,
        "order_seed": order_seed,
        "order": scored["order"],
        "per_state": scored["per_state"],
        "final": final_results,
        "forgetting_i_auroc": forgetting_from_matrix(scored["matrix_i"], scored["order"]),
        "forgetting_p_aupr": forgetting_from_matrix(scored["matrix_p"], scored["order"]),
        "occupancy": memory.occupancy(),
        "timing": {
            **update_timing,
            "final_inference_seconds_per_image": float(
                np.mean(
                    [
                        final_results[c]["inference_seconds_per_image"]
                        for c in CATEGORIES
                    ]
                )
            ),
        },
        "storage": {
            "persistent_exemplar_bytes": int(memory.feature_bytes),
            "persistent_memory_bytes": int(memory.memory_bytes),
            "exemplar_vectors": int(memory.count),
            "extractor_parameter_bytes": None,  # corpus-fixed; reported once
        },
    }


def run_one(
    arm: str,
    allocation: str,
    scoring: str,
    order: Sequence[str],
    order_seed: int,
    grid: int,
    budget: int,
    device: str = "cpu",
) -> dict[str, Any]:
    """Single-arm entry point, retained for direct calls and parity tests."""
    memory, scored = run_scoring_at_boundaries(
        allocation, scoring, order, order_seed, grid, budget, device=device
    )
    update_timing = scored.pop("timing")
    return assemble_record(
        arm,
        allocation,
        scoring,
        memory,
        scored,
        update_timing,
        grid,
        budget,
        order_seed,
        device,
    )


def run_allocation_pair(
    allocation: str,
    order: Sequence[str],
    order_seed: int,
    grid: int,
    budget: int,
    device: str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Run both scoring arms that share one allocation, from twin trajectories.

    Returns `{scoring_mode: record}`. Each scoring mode gets its own pass so that
    every retention row is measured on that stage's actual boundary state; the
    two passes traverse an identical trajectory because the update rule does not
    depend on the scoring mode. The second pass therefore reproduces the first
    pass's support exactly, which the regression tests assert.
    """
    records: dict[str, dict[str, Any]] = {}
    frozen: list[dict[str, Any]] = []
    measured_update_timing: dict[str, Any] | None = None
    for index, scoring in enumerate((SCORING_GLOBAL, SCORING_LOCAL)):
        memory, scored = run_scoring_at_boundaries(
            allocation,
            scoring,
            order,
            order_seed,
            grid,
            budget,
            device=device,
            freeze=frozen if index == 0 else None,
            replay=None if index == 0 else frozen,
        )
        update_timing = scored.pop("timing")
        if index == 0:
            measured_update_timing = update_timing
        else:
            # The second pass performed no updates, so its timing is not a
            # measurement. Report the one real measurement for both arms.
            update_timing = dict(measured_update_timing or update_timing)
        arm = "A" if allocation == ALLOCATION_GLOBAL else "C"
        if scoring == SCORING_LOCAL:
            arm = "B" if allocation == ALLOCATION_GLOBAL else "D"
        records[scoring] = assemble_record(
            arm,
            allocation,
            scoring,
            memory,
            scored,
            update_timing,
            grid,
            budget,
            order_seed,
            device,
        )
    return records


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean and spread over the three paired order seeds, per arm."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_arm.setdefault(record["arm"], []).append(record)

    summary: dict[str, Any] = {}
    for arm, runs in sorted(by_arm.items()):
        def collect(key: str, sub: str | None = None) -> list[float]:
            values = []
            for run in runs:
                for category in CATEGORIES:
                    target = run["final"][category]
                    values.append(target[key] if sub is None else target[key][sub])
            return values

        i_auroc = collect("i_auroc")
        native = collect("p_aupr_native")
        grid28 = collect("p_aupr_grid28")
        summary[arm] = {
            "allocation": runs[0]["allocation"],
            "scoring": runs[0]["scoring"],
            "n_order_seeds": len(runs),
            "macro_i_auroc_mean": float(np.mean(i_auroc)),
            "macro_i_auroc_per_seed": [
                float(np.mean([run["final"][c]["i_auroc"] for c in CATEGORIES]))
                for run in runs
            ],
            "macro_p_aupr_native_mean": float(np.mean(native)),
            "macro_p_aupr_native_per_seed": [
                float(np.mean([run["final"][c]["p_aupr_native"] for c in CATEGORIES]))
                for run in runs
            ],
            "macro_p_aupr_grid28_mean": float(np.mean(grid28)),
            "per_category": {
                category: {
                    "i_auroc_mean": float(
                        np.mean([run["final"][category]["i_auroc"] for run in runs])
                    ),
                    "p_aupr_native_mean": float(
                        np.mean([run["final"][category]["p_aupr_native"] for run in runs])
                    ),
                    "p_aupr_grid28_mean": float(
                        np.mean([run["final"][category]["p_aupr_grid28"] for run in runs])
                    ),
                }
                for category in CATEGORIES
            },
            "forgetting_i_auroc_mean": float(
                np.mean(
                    [
                        run["forgetting_i_auroc"]["fm"]
                        for run in runs
                        if run["forgetting_i_auroc"]["fm"] is not None
                    ]
                )
            ),
            "forgetting_p_aupr_mean": float(
                np.mean(
                    [
                        run["forgetting_p_aupr"]["fm"]
                        for run in runs
                        if run["forgetting_p_aupr"]["fm"] is not None
                    ]
                )
            ),
            "occupancy": runs[0]["occupancy"],
            "timing": {
                "update_seconds_mean_per_image": float(
                    np.mean([run["timing"]["update_seconds_mean_per_image"] for run in runs])
                ),
                "final_inference_seconds_per_image": float(
                    np.mean(
                        [
                            run["timing"]["final_inference_seconds_per_image"]
                            for run in runs
                        ]
                    )
                ),
                "images_seen": runs[0]["timing"]["images_seen"],
            },
            "storage": runs[0]["storage"],
        }
    return summary


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """The single definition of the worker CLI, shared by both entry points.

    The dispatcher spawns `this_file --worker ...` and the parent parses its own
    arguments with the same parser, so the flag must exist in both places and
    must have the same name. They previously disagreed (`--worker` versus
    `--workers`), which made every dispatched subprocess die immediately on an
    unrecognised argument.
    """
    parser.add_argument("--worker", action="store_true", help="run as a sweep worker")
    parser.add_argument("--seeds", default=None, help="comma-separated order seeds")
    parser.add_argument("--out", default=None, help="worker output directory")


def worker_main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Worker process: owns one device and a disjoint set of order seeds."""
    parser = argparse.ArgumentParser(description="AD-01 sweep worker")
    add_worker_arguments(parser)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if not args.seeds:
        raise SystemExit("--seeds is required for a worker")
    seeds = [int(value) for value in args.seeds.split(",") if value != ""]
    output = Path(args.out)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log(
        f"worker START pid={os.getpid()} device={args.device} seeds={seeds} "
        f"namespace={output} | {gpu_memory_note(args.device)}",
        args.device,
    )
    payload = worker_entry(seeds, args.device, output)
    log(
        f"worker DONE seeds={seeds} records={len(payload['records'])} "
        f"seconds={payload['seconds']:.1f}",
        args.device,
    )
    return payload


def cuda_index(device: str) -> str:
    """Physical GPU index for a device string like `cuda`, `cuda:0`, `cuda:1`."""
    if ":" in device:
        return device.rsplit(":", 1)[1]
    return "0"


def dispatch_workers(
    device_a: str, device_b: str | None, output: Path, seeds: Sequence[int]
) -> dict[str, Any]:
    """Fan order seeds out over the available devices and aggregate centrally.

    Only this parent process writes `arms_results.json`. Each worker writes its
    own `arm_runs/` files and its own per-worker summary inside a private
    directory, so there is never more than one writer for a given path.

    Each worker is pinned with `CUDA_VISIBLE_DEVICES=<physical index>` and is then
    told to use plain `cuda`, so the two workers cannot collide on one device.
    Passing a `cuda:N` string directly in `CUDA_VISIBLE_DEVICES` is invalid and
    silently falls back to device 0, which put both workers on GPU 0.
    """
    import subprocess
    import sys

    requested = [device_a] if not device_b else [device_a, device_b]
    devices: list[str] = []
    for device in requested:
        if device not in devices:
            devices.append(device)
    groups: list[list[int]] = [[] for _ in devices]
    for index, seed in enumerate(seeds):
        groups[index % len(devices)].append(seed)
    groups = [group for group in groups if group]
    devices = devices[: len(groups)]
    # A single worker keeps the original in-process behaviour and its exact
    # checkpoint location, which is what the CPU namespace expects.
    if len(groups) == 1:
        return worker_entry(groups[0], devices[0], output)

    procs = []
    for device, group in zip(devices, groups):
        tag = f"gpu{cuda_index(device)}"
        worker_out = output / f"worker_{tag}"
        worker_out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--seeds",
            ",".join(str(s) for s in group),
            "--device",
            "cuda",
            "--out",
            str(worker_out),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = cuda_index(device)
        # Keep BLAS from spawning one thread per core in every worker; the work
        # is on the GPU and CPU oversubscription only adds contention.
        env.setdefault("OMP_NUM_THREADS", "2")
        env.setdefault("MKL_NUM_THREADS", "2")
        log = (worker_out / "worker.log").open("w")
        procs.append(
            (
                device,
                group,
                tag,
                subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT),
                log,
            )
        )

    records: list[dict[str, Any]] = []
    for device, group, tag, process, log in procs:
        code = process.wait()
        log.close()
        if code != 0:
            raise RuntimeError(f"worker on {device} seeds={group} exited {code}")
        payload = json.loads(
            (
                output
                / f"worker_{tag}"
                / f"worker_seeds_{'_'.join(str(s) for s in group)}.json"
            ).read_text()
        )
        records.extend(payload["records"])
    records.sort(key=lambda record: (record["order_seed"], record["arm"]))
    return {
        "records": records,
        "seconds": None,
        "workers": {devices[i]: groups[i] for i in range(len(groups))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AD-01 four-arm endpoint evaluation")
    parser.add_argument("--stage", choices=["cache-only", "arms", "all"], default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for extraction and scoring; 'cuda' selects the GPU namespace",
    )
    parser.add_argument(
        "--device-b",
        default=None,
        help="second device for a parallel worker, e.g. cuda:1; omit for single-device",
    )
    add_worker_arguments(parser)
    args = parser.parse_args()

    if args.worker:
        # Dispatch straight to the worker path using the already-parsed values.
        # Neither branch checks CUDA availability the same way: a worker is
        # pinned to one physical GPU by the parent through CUDA_VISIBLE_DEVICES
        # and then uses plain `cuda`, so `cuda` is always valid inside it.
        worker_main(
            [
                "--worker",
                "--seeds",
                args.seeds or "",
                "--device",
                "cuda",
                "--out",
                args.out or "",
            ]
        )
        return

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but torch.cuda.is_available() is False. "
            "This shell is likely inside the restricted sandbox, whose /dev has "
            "no nvidia device nodes; run from the host environment instead."
        )
    output = Path(args.out) if args.out else namespace_root(device)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"[AD-01] device={device} device_b={args.device_b} "
        f"namespace={output.relative_to(ROOT)}",
        flush=True,
    )

    protocol, method = load_protocol_and_method()

    if args.stage in ("cache-only", "all"):
        begin = time.perf_counter()
        report = ensure_feature_cache(
            protocol, method, force=args.force_cache, device=device
        )
        report["seconds"] = time.perf_counter() - begin
        report["device"] = device
        (output / "feature_cache.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    if args.stage == "cache-only":
        return

    begin = time.perf_counter()
    dispatched = dispatch_workers(device, args.device_b, output, list(ORDER_SEEDS))
    records = dispatched["records"]
    summary = aggregate(records)
    payload = {
        "summary": summary,
        "runs": records,
        "config": {
            "device": device,
            "device_b": args.device_b,
            "workers": dispatched.get("workers"),
            "feature_cache_root": str(feature_root(device).relative_to(ROOT)),
            "categories": list(CATEGORIES),
            "order_seeds": list(ORDER_SEEDS),
            "primary_grid": PRIMARY_GRID,
            "diagnostic_grids": list(DIAGNOSTIC_GRIDS),
            "budget": TOTAL_BUDGET,
            "image_neighbors": IMAGE_NEIGHBORS,
            "dev_manifest": str(DEV_MANIFEST.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "shared_update_trajectory": (
                "arms sharing an allocation are evaluated from one update "
                "trajectory; update() does not depend on the scoring mode"
            ),
            "total_wall_seconds": time.perf_counter() - begin,
        },
    }
    (output / "arms_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
def sweep_seed(
    order_seed: int, device: str, run_dir: Path, budget: int, grid: int
) -> list[dict[str, Any]]:
    """Run all four arms for one order seed, writing one file per arm.

    Each seed is owned by exactly one worker, and each worker owns its own
    output directory, so no two processes ever write the same file. The
    allocation pairs share a single update trajectory: the global pair supplies
    arm A and arm B, the spatial pair supplies arm C and arm D.
    """
    order = order_permutation(order_seed)
    records: list[dict[str, Any]] = []
    for allocation in (ALLOCATION_GLOBAL, ALLOCATION_SPATIAL):
        expected = [
            arm
            for arm, alloc, _ in ARMS
            if alloc == allocation
        ]
        checkpoints = {arm: run_dir / f"seed{order_seed}_{arm}.json" for arm in expected}
        if all(path.is_file() for path in checkpoints.values()):
            for arm in expected:
                record = json.loads(checkpoints[arm].read_text())
                records.append(record)
                log(f"arm {arm} resumed from checkpoint {checkpoints[arm].name}", device)
            continue

        log(
            f"START allocation={allocation} arms={'/'.join(expected)} "
            f"order={'->'.join(order)} budget={budget} grid={grid}",
            device,
        )
        started = time.perf_counter()
        pair = run_allocation_pair(
            allocation, order, order_seed, grid, budget, device=device
        )
        elapsed = time.perf_counter() - started
        for index, (scoring, record) in enumerate(pair.items()):
            # Attribute the shared sweep's wall time to the first arm only, so
            # the reported per-arm cost is not double counted.
            record["wall_seconds"] = elapsed if index == 0 else 0.0
            record["shared_sweep_seconds"] = elapsed
            checkpoints[record["arm"]].write_text(json.dumps(record, indent=2) + "\n")
            log(
                f"checkpoint saved {checkpoints[record['arm']].name}",
                device,
            )
            records.append(record)
            final = record["final"]
            macro_i = float(np.mean([final[c]["i_auroc"] for c in CATEGORIES]))
            macro_p = float(np.mean([final[c]["p_aupr_native"] for c in CATEGORIES]))
            print(
                f"[seed {order_seed}] arm {record['arm']} "
                f"({allocation}/{scoring}) I-AUROC={macro_i:.4f} "
                f"P-AUPR={macro_p:.4f} fm_i={record['forgetting_i_auroc']['fm']} "
                f"sweep={elapsed:.1f}s",
                flush=True,
            )
    return records


def worker_entry(order_seeds: Sequence[int], device: str, output: Path) -> dict[str, Any]:
    """Independent worker: one device, a disjoint set of seeds, one output dir."""
    run_dir = output / "arm_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    begin = time.perf_counter()
    for order_seed in order_seeds:
        records.extend(
            sweep_seed(order_seed, device, run_dir, TOTAL_BUDGET, PRIMARY_GRID)
        )
    payload = {
        "device": device,
        "order_seeds": list(order_seeds),
        "records": records,
        "seconds": time.perf_counter() - begin,
    }
    (output / f"worker_seeds_{'_'.join(str(s) for s in order_seeds)}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return payload



if __name__ == "__main__":
    main()
