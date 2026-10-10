# exps/ad02/hope_cad_ad02_run.py
"""AD-02 runner: capacity-allocation mechanism validation.

Runs five allocation arms at a matched 2500-vector budget with **global scoring**
everywhere, and records, at every task boundary, the provenance of every stored
exemplar, the replacement events of that stage, per-group occupancy, old-task
normal-support coverage, false-positive distributions, update cost, inference
latency, persistent bytes and transient working memory.

Arms:

    G   global allocation                 (matched CADIC-compatible reference)
    S2  spatial 2x2                       (coarse fineness)
    S4  spatial 4x4                       (AD-01 arm C; replay check)
    S7  spatial 7x7                       (fine fineness)
    R4  balanced non-spatial random x3    (same groups/capacities as S4)

The G and S4 arms must reproduce the archived AD-01 arm A and arm C records
exactly; a mismatch is an integrity blocker. Everything is checkpointed per run
and the runner resumes from existing checkpoints.

Local scoring is deliberately absent: AD-01 showed it is dominated on accuracy,
inference latency and forgetting, and AD-02 does not re-test it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from exps.ad01.hope_cad_ad01_arms import (
    CATEGORIES,
    ORDER_SEEDS,
    TOTAL_BUDGET,
    auroc,
    average_precision,
    forgetting_from_matrix,
    load_dev_manifest,
    order_permutation,
)
from exps.ad01.hope_cad_ad01_fast_coreset import FastCADICPatchCoresetV1
from exps.ad01.hope_cad_ad01_normal_support import (
    ALLOCATION_GLOBAL,
    ALLOCATION_SPATIAL,
    PATCHES,
)
from exps.ad01.hope_cad_ad01_run import load_cached, masks28_from_native
from exps.ad02.hope_cad_ad02_memory import (
    InstrumentedNormalSupportMemory,
    summarise,
)
from exps.ad02.hope_cad_ad02_partition import (
    build_partition,
    per_group_quota,
)

from models.cadic_patch_coreset_v1 import CADICPatchCoresetConfig

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/hope_cad/ad02"
FEATURE_ROOT = ROOT / "results/hope_cad/ad01_phase0/arms_gpu/features"
AD01_RESULTS = ROOT / "results/hope_cad/ad01_phase0/arms_gpu_v5/arms_results.json"
DEV_MANIFEST = (
    ROOT / "results/hope_cad/anomaly_signal_gate/manifests/anomaly_dev_manifest.parquet"
)

IMAGE_NEIGHBORS = 9
COVERAGE_IMAGES = 6
AD02_SCHEMA = "ad02_capacity_allocation_v1"

ARMS: tuple[dict[str, Any], ...] = (
    {"arm": "G", "kind": "global", "grid": 4, "draw": None,
     "role": "global allocation, matched CADIC-compatible reference"},
    {"arm": "S2", "kind": "spatial", "grid": 2, "draw": None,
     "role": "spatial 2x2, coarse partition point"},
    {"arm": "S4", "kind": "spatial", "grid": 4, "draw": None,
     "role": "spatial 4x4, AD-01 arm C replication"},
    {"arm": "S7", "kind": "spatial", "grid": 7, "draw": None,
     "role": "spatial 7x7, fine partition point"},
    {"arm": "R4", "kind": "random", "grid": 4, "draw": 0,
     "role": "balanced non-spatial random partition, draw 0"},
    {"arm": "R4", "kind": "random", "grid": 4, "draw": 1,
     "role": "balanced non-spatial random partition, draw 1"},
    {"arm": "R4", "kind": "random", "grid": 4, "draw": 2,
     "role": "balanced non-spatial random partition, draw 2"},
)


# --------------------------------------------------------------------- utils


def log(message: str, tag: str = "") -> None:
    stamp = time.strftime("%H:%M:%S")
    prefix = f"[{stamp}]" + (f"[{tag}]" if tag else "")
    print(f"{prefix} {message}", flush=True)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_id(arm: dict[str, Any], order_seed: int) -> str:
    suffix = f"_draw{arm['draw']}" if arm.get("draw") is not None else ""
    return f"{arm['arm']}_seed{order_seed}{suffix}"


def unit_key(arm: dict[str, Any], order_seed: int) -> str:
    return run_id(arm, order_seed)


# ---------------------------------------------------------------- partitions


def resolve_partition(
    arm: dict[str, Any], partition_dir: Path, budget: int = TOTAL_BUDGET
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Build or load the arm's partition, persisted with its fingerprint."""
    if arm["kind"] == "global":
        return None, {
            "kind": "global",
            "label": "global",
            "n_groups": 1,
            "grid": 4,
            "assignment": None,
            "diagnostics": None,
            "quotas": [budget],
        }

    label = (
        f"spatial{arm['grid']}x{arm['grid']}"
        if arm["kind"] == "spatial"
        else f"random{arm['grid'] * arm['grid']}_draw{arm['draw']}"
    )
    target = partition_dir / f"{label}.json"
    if target.is_file():
        payload = json.loads(target.read_text())
    else:
        kwargs = {"grid": arm["grid"]}
        if arm["kind"] == "random":
            kwargs["seed"] = 10_000 + int(arm["draw"])
        assignment, diagnostics, meta = build_partition(arm["kind"], **kwargs)
        payload = {
            "kind": arm["kind"],
            "label": label,
            "n_groups": int(diagnostics.n_groups),
            "grid": arm["grid"],
            "assignment": assignment.to(torch.long).tolist(),
            "diagnostics": diagnostics.as_dict(),
            "meta": meta,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")

    assignment = torch.tensor(payload["assignment"], dtype=torch.long)
    payload["quotas"] = per_group_quota(budget, int(payload["n_groups"]))
    return assignment, payload


# ------------------------------------------------------------------- metrics


@torch.no_grad()
def score_development(
    memory: InstrumentedNormalSupportMemory,
    dev_payload: dict[str, Any],
    device: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Score one development set with the global scorer and report distributions.

    Metric definitions are the AD-01 definitions, recomputed here so that the
    extra false-positive distributions come from the same pass rather than from a
    second, differently-chunked evaluation.
    """
    patches = dev_payload["patches"]
    masks = dev_payload["masks"]
    labels = dev_payload["labels"].numpy().astype(np.int64)
    masks = masks.cpu() if torch.is_tensor(masks) else masks
    masks28 = masks28_from_native(masks)

    image_chunks, pixel_chunks = [], []
    latency = None
    for start in range(0, patches.shape[0], batch_size):
        stop = min(start + batch_size, patches.shape[0])
        sync_device(device)
        begin = time.perf_counter()
        images, pixel = memory.score(patches[start:stop], scoring="global")
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
    native_scores = np.concatenate(native_scores)
    native_masks = np.concatenate(native_masks)

    normal = labels == 0
    defect = labels == 1
    normal_image = image_scores[normal]
    defect_image = image_scores[defect]
    normal_patch = pixel_scores[normal].reshape(-1)
    defect_patch = pixel_scores[defect].reshape(-1)
    # Native-resolution pixel split by image label, for the false-positive view.
    pixels_per_image = native_scores.size // labels.size
    native_by_image = native_scores.reshape(labels.size, pixels_per_image)
    normal_native = native_by_image[normal].reshape(-1)
    defect_native = native_by_image[defect].reshape(-1)

    return {
        "i_auroc": auroc(labels, image_scores),
        "p_aupr_native": average_precision(native_masks, native_scores),
        "p_aupr_grid28": average_precision(
            np.concatenate([m.reshape(-1) for m in masks28]), pixel_scores.reshape(-1)
        ),
        "mean_normal_score": float(normal_image.mean()),
        "mean_defect_score": float(defect_image.mean()),
        "normal_image_scores": normal_image.tolist(),
        "defect_image_scores": defect_image.tolist(),
        "normal_image_summary": summarise(normal_image),
        "defect_image_summary": summarise(defect_image),
        "normal_patch_summary": summarise(normal_patch.astype(np.float64)),
        "defect_patch_summary": summarise(defect_patch.astype(np.float64)),
        "normal_native_pixel_summary": summarise(normal_native.astype(np.float64)),
        "defect_native_pixel_summary": summarise(defect_native.astype(np.float64)),
        "separation_margin": float(defect_image.min() - normal_image.max()),
        "inference_seconds_per_image": latency,
    }


# -------------------------------------------------------------------- oracle


@torch.no_grad()
def oracle_scores(
    bank_features: torch.Tensor,
    query_patches: torch.Tensor,
    device: str,
    query_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Budget-free nearest-neighbour readout: every seen normal patch is stored.

    Diagnostic only. It bounds how much of any arm's result is a property of the
    frozen features rather than of the bounded memory, and it is independent of
    the allocation arms.
    """
    config = CADICPatchCoresetConfig(
        budget=int(bank_features.shape[0]),
        dim=int(bank_features.shape[1]),
        dtype="float32",
        distance="euclidean",
        chunk_size=8192,
        image_neighbors=IMAGE_NEIGHBORS,
        query_chunk_size=query_chunk,
        pair_chunk_size=1024,
    )
    kernel = FastCADICPatchCoresetV1(config, device=device)
    kernel.features = bank_features

    images, pixels = [], []
    for image in query_patches:
        pixel, indices = kernel._nearest(image, bank_features)
        star_row = int(torch.argmax(pixel).item())
        star_score = pixel[star_row]
        c_star = bank_features[int(indices[star_row].item())]
        support_indices = kernel._topk_indices(
            c_star, min(IMAGE_NEIGHBORS, kernel.count)
        )
        support = bank_features[support_indices]
        support_dist = torch.linalg.vector_norm(support - image[star_row], dim=1)
        weight = 1.0 - torch.exp(star_score - torch.logsumexp(support_dist, dim=0))
        images.append(weight * star_score)
        pixels.append(pixel)
    return torch.stack(images), torch.stack(pixels)


def run_oracle(seed: int, device: str, out_dir: Path, force: bool = False) -> dict:
    target = out_dir / f"oracle_seed{seed}.json"
    if target.is_file() and not force:
        return json.loads(target.read_text())

    order = order_permutation(seed)
    train = {category: load_cached(category, "train", device) for category in CATEGORIES}
    dev = {category: load_cached(category, "dev", device) for category in CATEGORIES}
    task_ids = {category: index for index, category in enumerate(order)}

    per_state: list[dict[str, Any]] = []
    matrix_i = np.full((len(order), len(order)), np.nan)
    matrix_p = np.full((len(order), len(order)), np.nan)
    seen: list[str] = []
    for stage, category in enumerate(order):
        seen.append(category)
        bank = torch.cat(
            [train[c]["patches"].reshape(-1, train[c]["patches"].shape[-1]) for c in seen],
            dim=0,
        )
        log(
            f"oracle seed={seed} boundary={stage + 1} bank={tuple(bank.shape)}",
            "oracle",
        )
        for task_index, evaluated in enumerate(order[: stage + 1]):
            started = time.perf_counter()
            images, pixels = oracle_scores(bank, dev[evaluated]["patches"], device)
            metrics = score_from_pixel_maps(images, pixels, dev[evaluated], device)
            matrix_i[stage, task_index] = metrics["i_auroc"]
            matrix_p[stage, task_index] = metrics["p_aupr_native"]
            per_state.append(
                {
                    "state_after": category,
                    "stage": stage,
                    "task": evaluated,
                    "task_id": task_ids[evaluated],
                    "i_auroc": metrics["i_auroc"],
                    "p_aupr_native": metrics["p_aupr_native"],
                    "p_aupr_grid28": metrics["p_aupr_grid28"],
                    "seconds": time.perf_counter() - started,
                }
            )
            log(
                f"oracle seed={seed} {evaluated} after {category} "
                f"I-AUROC={metrics['i_auroc']:.4f} "
                f"P-AUPR={metrics['p_aupr_native']:.4f}",
                "oracle",
            )
        del bank

    payload = {
        "schema": AD02_SCHEMA,
        "kind": "budget_free_oracle",
        "order_seed": seed,
        "order": list(order),
        "per_state": per_state,
        "matrix_i": matrix_i.tolist(),
        "matrix_p": matrix_p.tolist(),
        "forgetting_i_auroc": forgetting_from_matrix(matrix_i, order),
        "forgetting_p_aupr": forgetting_from_matrix(matrix_p, order),
        "note": (
            "diagnostic bound only: every normal training patch of every task seen "
            "so far is stored, so this is not a matched-memory arm"
        ),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def score_from_pixel_maps(
    image_scores: torch.Tensor,
    pixel_scores: torch.Tensor,
    dev_payload: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """Metrics from precomputed score maps, using the AD-01 metric path."""
    masks = dev_payload["masks"]
    masks = masks.cpu() if torch.is_tensor(masks) else masks
    labels = dev_payload["labels"].numpy().astype(np.int64)
    masks28 = masks28_from_native(masks)
    images = image_scores.detach().cpu().numpy().astype(np.float64)
    pixels = pixel_scores.detach().cpu().numpy().astype(np.float64)

    native_scores, native_masks = [], []
    for index in range(pixels.shape[0]):
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(pixels[index]).reshape(1, 1, 28, 28),
            size=masks[index].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        native_scores.append(resized.reshape(-1))
        native_masks.append(masks[index].numpy().reshape(-1))
    return {
        "i_auroc": auroc(labels, images),
        "p_aupr_native": average_precision(
            np.concatenate(native_masks), np.concatenate(native_scores)
        ),
        "p_aupr_grid28": average_precision(
            np.concatenate([m.reshape(-1) for m in masks28]), pixels.reshape(-1)
        ),
    }


# --------------------------------------------------------------- one run unit


def run_unit(
    arm: dict[str, Any],
    order_seed: int,
    *,
    device: str,
    out_dir: Path,
    partition_dir: Path,
    force: bool = False,
    budget: int = TOTAL_BUDGET,
    image_limit: int = 0,
) -> tuple[dict[str, Any], bool]:
    """Run one (arm, order seed) unit. Returns `(record, resumed)`."""
    identifier = run_id(arm, order_seed)
    target = out_dir / "runs" / f"{identifier}.json"
    if target.is_file() and not force:
        return json.loads(target.read_text()), True

    assignment, partition = resolve_partition(arm, partition_dir, budget)
    order = order_permutation(order_seed)
    task_ids = {category: index for index, category in enumerate(order)}

    set_determinism(1000 + order_seed)
    allocation = (
        ALLOCATION_GLOBAL if arm["kind"] == "global" else ALLOCATION_SPATIAL
    )
    memory = InstrumentedNormalSupportMemory(
        budget=budget,
        grid=arm["grid"],
        allocation=allocation,
        image_neighbors=IMAGE_NEIGHBORS,
        device=device,
        fast=True,
        assignment=assignment,
        arm_label=identifier,
        task_names=order,
        partition_kind=arm["kind"],
    )

    train = {category: load_cached(category, "train", device) for category in CATEGORIES}
    dev = {category: load_cached(category, "dev", device) for category in CATEGORIES}
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = int(torch.cuda.memory_allocated())

    log(
        f"START {identifier} partition={partition['label']} "
        f"groups={partition['n_groups']} order={'->'.join(order)} "
        f"budget={budget}",
        identifier,
    )

    matrix_i = np.full((len(order), len(order)), np.nan)
    matrix_p = np.full((len(order), len(order)), np.nan)
    per_state: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    update_seconds: list[float] = []
    update_seconds_full: list[float] = []
    update_seconds_fill: list[float] = []
    coverage_baseline: dict[int, np.ndarray] = {}
    sampled = (
        min(COVERAGE_IMAGES, int(image_limit)) if image_limit else COVERAGE_IMAGES
    )
    coverage_queries = {
        category: train[category]["patches"][:sampled].reshape(
            -1, train[category]["patches"].shape[-1]
        )
        for category in CATEGORIES
    }

    global_step = 0
    for stage, category in enumerate(order):
        patches = train[category]["patches"]
        total = int(patches.shape[0])
        if image_limit:
            total = min(total, int(image_limit))
        memory.begin_stage(stage)
        log(f"{identifier} {category} update START ({total} images)", identifier)
        stage_started = time.perf_counter()
        for index in range(total):
            was_full = memory.is_full()
            sync_device(device)
            begin = time.perf_counter()
            memory.update(
                patches[index],
                task_id=task_ids[category],
                image_index=index,
                global_step=global_step,
            )
            sync_device(device)
            elapsed = time.perf_counter() - begin
            update_seconds.append(elapsed)
            (update_seconds_full if was_full else update_seconds_fill).append(elapsed)
            global_step += 1
            if (index + 1) % 100 == 0 or index + 1 == total:
                log(
                    f"{identifier} {category} {index + 1}/{total} "
                    f"bank={memory.count}/{budget} "
                    f"last_update={elapsed * 1000:.1f}ms",
                    identifier,
                )
        events = memory.end_stage()
        log(
            f"{identifier} {category} update COMPLETE "
            f"({time.perf_counter() - stage_started:.1f}s) "
            f"insert={events['insertions_total']} evict={events['evictions_total']}",
            identifier,
        )

        metrics: dict[str, Any] = {}
        coverage: dict[str, Any] = {}
        for task_index, evaluated in enumerate(order[: stage + 1]):
            began = time.perf_counter()
            result = score_development(memory, dev[evaluated], device)
            matrix_i[stage, task_index] = result["i_auroc"]
            matrix_p[stage, task_index] = result["p_aupr_native"]
            metrics[evaluated] = result
            per_state.append(
                {
                    "state_after": category,
                    "stage": stage,
                    "task": evaluated,
                    "i_auroc": result["i_auroc"],
                    "p_aupr_native": result["p_aupr_native"],
                    "p_aupr_grid28": result["p_aupr_grid28"],
                }
            )
            log(
                f"{identifier} scoring {evaluated} @{stage + 1} "
                f"I-AUROC={result['i_auroc']:.4f} "
                f"P-AUPR(native)={result['p_aupr_native']:.4f} "
                f"({time.perf_counter() - began:.1f}s)",
                identifier,
            )

            d_any, d_own, own_slots = memory.coverage_raw(
                coverage_queries[evaluated], task_id=task_ids[evaluated]
            )
            baseline = coverage_baseline.get(task_ids[evaluated])
            if baseline is None:
                coverage_baseline[task_ids[evaluated]] = d_own
                ratio = np.ones_like(d_own)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(
                        np.isfinite(baseline) & (baseline > 0),
                        d_own / baseline,
                        np.where(np.isfinite(d_own), np.inf, 1.0),
                    )
            coverage[evaluated] = {
                "d_any": summarise(d_any),
                "d_own": summarise(d_own),
                "ratio_to_learned_boundary": summarise(ratio),
                "own_slots": own_slots,
                "images_sampled": sampled,
                "queries": int(d_any.size),
            }

        occupancy = memory.occupancy()
        peak = (
            int(torch.cuda.max_memory_allocated())
            if str(device).startswith("cuda")
            else None
        )
        boundaries.append(
            {
                "stage": stage,
                "state_after": category,
                "task_id": task_ids[category],
                "global_step": global_step,
                "count": int(memory.count),
                "capacity": budget,
                "capacity_fraction": int(memory.count) / budget,
                "events": events,
                "slots_by_origin": {
                    str(k): int(v)
                    for k, v in memory.slot_counts_by_origin().items()
                },
                "slots_by_group_and_origin": [
                    {str(k): int(v) for k, v in counts.items()}
                    for counts in memory.slot_counts_by_group_and_origin()
                ],
                "bin_counts": [row["count"] for row in occupancy["bins"]],
                "bin_capacity": [row["capacity"] for row in occupancy["bins"]],
                "bin_fill_ratio": [row["fill_ratio"] for row in occupancy["bins"]],
                "age_by_origin": {
                    str(k): v for k, v in memory.age_stats_by_origin(global_step).items()
                },
                "coverage": coverage,
                "metrics": metrics,
                "transient": memory.transient_bytes(),
                "peak_allocated_bytes": peak,
                "baseline_allocated_bytes": baseline_allocated
                if str(device).startswith("cuda")
                else None,
            }
        )

    # Bitwise provenance audit: every surviving row must equal the exact source
    # patch its recorded (task, image, position) triple points at.
    origin_cache = {}
    for category in CATEGORIES:
        available = int(train[category]["patches"].shape[0])
        if image_limit:
            available = min(available, int(image_limit))
        for index in range(available):
            origin_cache[(task_ids[category], index)] = train[category]["patches"][
                index
            ].reshape(PATCHES, -1)
    origin_check_full = memory.verify_origins(origin_cache)
    final = {category: boundaries[-1]["metrics"][category] for category in CATEGORIES}
    occupancy = memory.occupancy()
    record = {
        "schema": AD02_SCHEMA,
        "run_id": identifier,
        "arm": arm["arm"],
        "role": arm["role"],
        "kind": arm["kind"],
        "draw": arm.get("draw"),
        "grid": arm["grid"],
        "device": device,
        "allocation": allocation,
        "scoring": "global",
        "budget": budget,
        "order_seed": order_seed,
        "order": order,
        "partition": {k: v for k, v in partition.items() if k != "assignment"},
        "quotas": partition["quotas"],
        "per_state": per_state,
        "final": final,
        "matrices": {"i_auroc": matrix_i.tolist(), "p_aupr_native": matrix_p.tolist()},
        "forgetting_i_auroc": forgetting_from_matrix(matrix_i, order),
        "forgetting_p_aupr": forgetting_from_matrix(matrix_p, order),
        "boundaries": boundaries,
        "occupancy": occupancy,
        "origin_check": origin_check_full,
        "timing": {
            "update_seconds_total": float(np.sum(update_seconds)),
            "update_seconds_mean_per_image": float(np.mean(update_seconds)),
            "update_seconds_mean_full_bank": (
                float(np.mean(update_seconds_full)) if update_seconds_full else None
            ),
            "update_seconds_mean_fill": (
                float(np.mean(update_seconds_fill)) if update_seconds_fill else None
            ),
            "updates_full_bank": len(update_seconds_full),
            "updates_fill": len(update_seconds_fill),
            "inference_seconds_per_image": float(
                np.mean(
                    [
                        final[category]["inference_seconds_per_image"]
                        for category in CATEGORIES
                    ]
                )
            ),
            "images_seen": len(update_seconds),
        },
        "storage": {
            "persistent_exemplar_bytes": int(memory.feature_bytes),
            "persistent_memory_bytes": int(memory.memory_bytes),
            "persistent_partition_metadata_bytes": int(
                memory.persistent_metadata_bytes()
            ),
            "exemplar_vectors": int(memory.count),
            "partition_fingerprint": partition.get("diagnostics", {}).get(
                "fingerprint"
            )
            if partition.get("diagnostics")
            else None,
        },
        "transient": memory.transient_bytes(),
    }

    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n")
    log(
        f"DONE {identifier} I-AUROC={np.mean([final[c]['i_auroc'] for c in CATEGORIES]):.4f} "
        f"P-AUPR={np.mean([final[c]['p_aupr_native'] for c in CATEGORIES]):.4f} "
        f"fm_i={record['forgetting_i_auroc']['fm']} "
        f"origin_exact={origin_check_full['exact']}",
        identifier,
    )
    return record, False


# ---------------------------------------------------------- protocol / replay


def ad01_reference() -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(AD01_RESULTS.read_text())
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for record in payload["runs"]:
        if record["arm"] in ("A", "C"):
            lookup[(record["allocation"], record["order_seed"])] = record
    return lookup


def compare_with_ad01(record: dict[str, Any], reference: dict[str, Any]) -> dict:
    """Exact numeric comparison of an AD-02 arm against its archived AD-01 arm."""
    fields = ("i_auroc", "p_aupr_native", "p_aupr_grid28")
    final_mismatch = []
    for category in CATEGORIES:
        for field in fields:
            mine = record["final"][category][field]
            theirs = reference["final"][category][field]
            if mine != theirs:
                final_mismatch.append(
                    {
                        "category": category,
                        "field": field,
                        "ad02": mine,
                        "ad01": theirs,
                        "abs_diff": abs(mine - theirs),
                    }
                )
    per_state_mismatch = []
    theirs_states = {
        (row["state_after"], row["task"]): row for row in reference["per_state"]
    }
    for row in record["per_state"]:
        other = theirs_states.get((row["state_after"], row["task"]))
        if other is None:
            per_state_mismatch.append({"row": row, "reason": "missing in AD-01"})
            continue
        for field in fields:
            if row[field] != other[field]:
                per_state_mismatch.append(
                    {
                        "state_after": row["state_after"],
                        "task": row["task"],
                        "field": field,
                        "ad02": row[field],
                        "ad01": other[field],
                    }
                )
    fm_mine = record["forgetting_i_auroc"]["fm"]
    fm_theirs = reference["forgetting_i_auroc"]["fm"]
    return {
        "ad01_arm": reference["arm"],
        "ad01_allocation": reference["allocation"],
        "final_cells_compared": len(CATEGORIES) * len(fields),
        "final_mismatches": final_mismatch,
        "per_state_cells_compared": len(record["per_state"]) * len(fields),
        "per_state_mismatches": per_state_mismatch,
        "forgetting_i_auroc_match": fm_mine == fm_theirs,
        "exact": bool(not final_mismatch and not per_state_mismatch and fm_mine == fm_theirs),
    }


def write_protocol(device: str, out_dir: Path) -> dict[str, Any]:
    """Freeze the resolved protocol, hashes and partitions before endpoint work.

    AD-01's independent review recorded that the sweep was not reproducible from a
    stored config alone. This manifest pins the code, the feature cache, the
    development manifest and every partition.
    """
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    partition_dir = out_dir / "partitions"
    code_files = [
        "exps/ad02/hope_cad_ad02_partition.py",
        "exps/ad02/hope_cad_ad02_tracked_coreset.py",
        "exps/ad02/hope_cad_ad02_memory.py",
        "exps/ad02/hope_cad_ad02_run.py",
        "exps/ad01/hope_cad_ad01_normal_support.py",
        "exps/ad01/hope_cad_ad01_fast_coreset.py",
        "exps/ad01/hope_cad_ad01_arms.py",
        "exps/ad01/hope_cad_ad01_run.py",
    ]
    payload: dict[str, Any] = {
        "schema": AD02_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": device,
        "branch": subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "code_sha256": {
            path: sha256_file(ROOT / path) for path in code_files
        },
        "feature_cache": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(FEATURE_ROOT.glob("*.pt"))
        },
        "feature_cache_root": str(FEATURE_ROOT.relative_to(ROOT)),
        "dev_manifest": {
            "path": str(DEV_MANIFEST.relative_to(ROOT)),
            "sha256": sha256_file(DEV_MANIFEST),
            "rows": int(len(load_dev_manifest())),
        },
        "ad01_results": {
            "path": str(AD01_RESULTS.relative_to(ROOT)),
            "sha256": sha256_file(AD01_RESULTS),
        },
        "arms": [dict(arm) for arm in ARMS],
        "order_seeds": list(ORDER_SEEDS),
        "budget": TOTAL_BUDGET,
        "image_neighbors": IMAGE_NEIGHBORS,
        "scoring": "global (all arms)",
        "partitions": {},
        "local_scoring_retested": False,
    }
    for arm in ARMS:
        _, partition = resolve_partition(arm, partition_dir)
        key = partition["label"]
        entry = {k: v for k, v in partition.items() if k != "assignment"}
        entry["quotas"] = partition["quotas"]
        payload["partitions"][key] = entry
    (out_dir / "protocol_resolved.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


# ----------------------------------------------------------------------- main


def build_units() -> list[tuple[dict[str, Any], int]]:
    units: list[tuple[dict[str, Any], int]] = []
    for arm in ARMS:
        for seed in ORDER_SEEDS:
            units.append((arm, seed))
    return units


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AD-02 allocation-mechanism sweep")
    parser.add_argument(
        "--stage",
        choices=["protocol", "oracle", "arms", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None, help="comma-separated run ids")
    parser.add_argument(
        "--budget",
        type=int,
        default=TOTAL_BUDGET,
        help="exemplar budget; production value is 2500, small values are smoke only",
    )
    parser.add_argument(
        "--images-per-task",
        type=int,
        default=0,
        help="truncate each task's stream; 0 means the full task (smoke only)",
    )
    args = parser.parse_args(argv)

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable in this shell")
    out_dir = Path(args.out) if args.out else BASE
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    partition_dir = out_dir / "partitions"

    log(
        f"AD-02 device={device} out={out_dir.relative_to(ROOT)} "
        f"shard={args.shard}/{args.shards}",
        "ad02",
    )

    if args.stage in ("protocol", "all") and args.shard == 0:
        protocol = write_protocol(device, out_dir)
        log(
            f"protocol frozen: {len(protocol['partitions'])} partitions, "
            f"{len(protocol['code_sha256'])} code files hashed",
            "ad02",
        )
    if args.stage == "protocol":
        return

    if args.stage in ("oracle", "all"):
        for seed in ORDER_SEEDS:
            if ORDER_SEEDS.index(seed) % args.shards != args.shard:
                continue
            run_oracle(seed, device, out_dir, force=args.force)
        if args.stage == "oracle":
            return

    units = build_units()
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        units = [unit for unit in units if unit_key(*unit) in wanted]
    selected = [
        (arm, seed)
        for index, (arm, seed) in enumerate(units)
        if index % args.shards == args.shard
    ]
    log(f"{len(selected)} of {len(units)} units on this shard", "ad02")

    reference = ad01_reference()
    replay_checks: dict[str, Any] = {}
    started = time.perf_counter()
    for arm, seed in selected:
        record, resumed = run_unit(
            arm,
            seed,
            device=device,
            out_dir=out_dir,
            partition_dir=partition_dir,
            force=args.force,
            budget=args.budget,
            image_limit=args.images_per_task,
        )
        if resumed:
            log(f"resumed {record['run_id']}", "ad02")
        if args.budget != TOTAL_BUDGET or args.images_per_task:
            continue
        if arm["kind"] == "global":
            replay_checks[record["run_id"]] = compare_with_ad01(
                record, reference[("global", seed)]
            )
        elif arm["kind"] == "spatial" and arm["grid"] == 4:
            replay_checks[record["run_id"]] = compare_with_ad01(
                record, reference[("spatial", seed)]
            )

    if replay_checks:
        path = out_dir / f"replay_check_shard{args.shard}.json"
        path.write_text(json.dumps(replay_checks, indent=2) + "\n")
        for key, value in replay_checks.items():
            log(
                f"replay {key}: exact={value['exact']} "
                f"final_mismatch={len(value['final_mismatches'])} "
                f"per_state_mismatch={len(value['per_state_mismatches'])}",
                "ad02",
            )
    log(f"shard {args.shard} finished in {time.perf_counter() - started:.1f}s", "ad02")


if __name__ == "__main__":
    main()
