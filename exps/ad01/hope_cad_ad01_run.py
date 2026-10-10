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
FEATURE_ROOT = ROOT / "results/hope_cad/ad01_phase0/features"
OUTPUT_ROOT = ROOT / "results/hope_cad/ad01_phase0"

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


def cache_path(category: str, split: str) -> Path:
    return FEATURE_ROOT / f"{category}_{split}.pt"


def ensure_feature_cache(protocol: dict, method: dict, force: bool = False) -> dict[str, Any]:
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
    extractor = build_extractor(method)

    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"categories": {}, "checkpoint": str(CHECKPOINT)}

    for category in CATEGORIES:
        task_id = protocol["task_order"].index(category)

        train_target = cache_path(category, "train")
        if force or not train_target.is_file():
            loader = protocol_obj.build_train_loader(task_id)
            chunks, paths = [], []
            with torch.no_grad():
                for batch in loader:
                    chunks.append(
                        extractor.extract_patch_features(batch["images"]).cpu()
                    )
                    paths.extend(batch["relative_path"])
            torch.save({"patches": torch.cat(chunks, 0), "paths": paths}, train_target)
        payload = torch.load(train_target, map_location="cpu", weights_only=False)
        report["categories"][f"{category}_train"] = {
            "images": int(payload["patches"].shape[0]),
            "path": str(train_target.relative_to(ROOT)),
        }

        test_target = cache_path(category, "dev")
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
                            batch["images"].index_select(0, selector)
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


def load_cached(category: str, split: str) -> dict[str, Any]:
    path = cache_path(category, split)
    if not path.is_file():
        raise FileNotFoundError(f"missing feature cache {path}; run --cache-only first")
    return torch.load(path, map_location="cpu", weights_only=False)


def masks28_from_native(masks: torch.Tensor) -> list[np.ndarray]:
    down = torch.nn.functional.interpolate(
        masks.float().unsqueeze(1), size=(28, 28), mode="nearest"
    ).squeeze(1)
    return [down[index].numpy().astype(bool) for index in range(down.shape[0])]


@torch.no_grad()
def score_development(
    memory: NormalSupportMemory,
    dev_payload: dict[str, Any],
    scoring: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    patches = dev_payload["patches"]
    masks = dev_payload["masks"]
    labels = dev_payload["labels"].numpy().astype(np.int64)
    masks28 = masks28_from_native(masks)

    image_chunks, pixel_chunks = [], []
    latency = None
    for start in range(0, patches.shape[0], batch_size):
        stop = min(start + batch_size, patches.shape[0])
        begin = time.perf_counter()
        images, pixel = memory.score(patches[start:stop], scoring=scoring)
        elapsed = time.perf_counter() - begin
        if latency is None:
            latency = elapsed / (stop - start)
        image_chunks.append(images)
        pixel_chunks.append(pixel)

    image_scores = torch.cat(image_chunks).numpy().astype(np.float64)
    pixel_scores = torch.cat(pixel_chunks).numpy().astype(np.float64)

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


def run_one(
    arm: str,
    allocation: str,
    scoring: str,
    order: Sequence[str],
    order_seed: int,
    grid: int,
    budget: int,
) -> dict[str, Any]:
    set_determinism(1000 + order_seed)
    memory = NormalSupportMemory(
        budget=budget,
        grid=grid,
        allocation=allocation,
        image_neighbors=IMAGE_NEIGHBORS,
        device="cpu",
    )

    dev = {category: load_cached(category, "dev") for category in CATEGORIES}
    train = {category: load_cached(category, "train") for category in CATEGORIES}

    names = list(order)
    matrix_i = np.full((len(names), len(names)), np.nan)
    matrix_p = np.full((len(names), len(names)), np.nan)
    per_state: list[dict[str, Any]] = []
    update_seconds: list[float] = []

    for stage, category in enumerate(names):
        patches = train[category]["patches"]
        for index in range(patches.shape[0]):
            begin = time.perf_counter()
            memory.update(patches[index])
            update_seconds.append(time.perf_counter() - begin)

        for task_index, evaluated in enumerate(names[: stage + 1]):
            result = score_development(memory, dev[evaluated], scoring)
            matrix_i[stage, task_index] = result["i_auroc"]
            matrix_p[stage, task_index] = result["p_aupr_native"]
            per_state.append(
                {
                    "state_after": category,
                    "task": evaluated,
                    "i_auroc": result["i_auroc"],
                    "p_aupr_native": result["p_aupr_native"],
                    "p_aupr_grid28": result["p_aupr_grid28"],
                    "mean_normal_score": result["mean_normal_score"],
                    "mean_defect_score": result["mean_defect_score"],
                }
            )

    final_results = {
        category: score_development(memory, dev[category], scoring)
        for category in CATEGORIES
    }
    occupancy = memory.occupancy()
    return {
        "arm": arm,
        "allocation": allocation,
        "scoring": scoring,
        "grid": grid,
        "budget": budget,
        "order_seed": order_seed,
        "order": names,
        "per_state": per_state,
        "final": final_results,
        "forgetting_i_auroc": forgetting_from_matrix(matrix_i, names),
        "forgetting_p_aupr": forgetting_from_matrix(matrix_p, names),
        "occupancy": occupancy,
        "timing": {
            "update_seconds_total": float(sum(update_seconds)),
            "update_seconds_mean_per_image": (
                float(np.mean(update_seconds)) if update_seconds else None
            ),
            "images_seen": len(update_seconds),
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
            "extractor_parameter_bytes": None,  # filled by the caller; shared
        },
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="AD-01 four-arm endpoint evaluation")
    parser.add_argument("--stage", choices=["cache-only", "arms", "all"], default="all")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--out", default=str(OUTPUT_ROOT))
    args = parser.parse_args()

    protocol, method = load_protocol_and_method()
    output = Path(args.out)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.stage in ("cache-only", "all"):
        begin = time.perf_counter()
        report = ensure_feature_cache(protocol, method, force=args.force_cache)
        report["seconds"] = time.perf_counter() - begin
        (output / "feature_cache.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))

    if args.stage == "cache-only":
        return

    records: list[dict[str, Any]] = []
    begin = time.perf_counter()
    for order_seed in ORDER_SEEDS:
        order = order_permutation(order_seed)
        for arm, allocation, scoring in ARMS:
            started = time.perf_counter()
            record = run_one(
                arm, allocation, scoring, order, order_seed, PRIMARY_GRID, TOTAL_BUDGET
            )
            record["wall_seconds"] = time.perf_counter() - started
            records.append(record)
            final = record["final"]
            macro_i = float(np.mean([final[c]["i_auroc"] for c in CATEGORIES]))
            macro_p = float(np.mean([final[c]["p_aupr_native"] for c in CATEGORIES]))
            print(
                f"[seed {order_seed}] arm {arm} ({allocation}/{scoring}) "
                f"order={'-'.join(order)} I-AUROC={macro_i:.4f} "
                f"P-AUPR={macro_p:.4f} fm_i={record['forgetting_i_auroc']['fm']} "
                f"{record['wall_seconds']:.1f}s",
                flush=True,
            )

    summary = aggregate(records)
    payload = {
        "summary": summary,
        "runs": records,
        "config": {
            "categories": list(CATEGORIES),
            "order_seeds": list(ORDER_SEEDS),
            "primary_grid": PRIMARY_GRID,
            "diagnostic_grids": list(DIAGNOSTIC_GRIDS),
            "budget": TOTAL_BUDGET,
            "image_neighbors": IMAGE_NEIGHBORS,
            "dev_manifest": str(DEV_MANIFEST.relative_to(ROOT)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "total_wall_seconds": time.perf_counter() - begin,
        },
    }
    (output / "arms_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
