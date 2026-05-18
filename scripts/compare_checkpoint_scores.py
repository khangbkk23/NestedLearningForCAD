import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from training.meta_nath_engine import MetaNATHEngine
from training.run_experiment import build_model
from utils.global_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare image/pixel score distributions between two checkpoints."
    )
    parser.add_argument("--config", type=str, default="conf/config_phase3.yaml")
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument(
        "--task_ids",
        type=str,
        default="",
        help="Comma-separated task ids to inspect. Default: all tasks up to max_tasks.",
    )
    parser.add_argument("--pixel_sample_limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def _truncate_tasks_if_needed(config: Dict[str, Any], max_tasks: int | None) -> None:
    if max_tasks is None:
        return
    dataset_cfg = config.setdefault("dataset", {})
    class_order = dataset_cfg.get("class_order", [])
    if isinstance(class_order, list) and class_order:
        dataset_cfg["class_order"] = class_order[:max_tasks]


def _parse_task_ids(raw: str, max_tasks: int | None) -> set[int] | None:
    if raw.strip():
        return {int(part.strip()) for part in raw.split(",") if part.strip()}
    if max_tasks is None:
        return None
    return set(range(max_tasks))


def _stats(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _safe_auc(labels: List[int], scores: List[float]) -> float:
    return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else 0.0


def _safe_ap(labels: List[int], scores: List[float]) -> float:
    return float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else 0.0


def _load_engine(config: Dict[str, Any], checkpoint_path: str, device: str) -> MetaNATHEngine:
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_full_state_dict(checkpoint["model_state_dict"])

    evaluation_cfg = config.get("evaluation", {})
    memory_cfg = config.get("memory", {})
    return MetaNATHEngine(
        model=model,
        device=device,
        nearest_neighbors=int(memory_cfg.get("nearest_neighbors", 2)),
        pixel_score_norm=str(evaluation_cfg.get("pixel_score_norm", "none")).lower(),
        gaussian_smoothing_sigma=float(evaluation_cfg.get("gaussian_smoothing_sigma", 0.0)),
    )


def _collect_task_stats(
    engine: MetaNATHEngine,
    loader,
    task_id: int,
    pixel_sample_limit: int,
) -> Dict[str, Any]:
    image_scores: List[float] = []
    image_labels: List[int] = []
    normal_image_scores: List[float] = []
    anomaly_image_scores: List[float] = []
    normal_pixel_scores: List[float] = []
    anomaly_pixel_scores: List[float] = []
    all_pixel_scores: List[float] = []
    all_pixel_labels: List[int] = []

    engine.model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(engine.device)
            labels = batch["anomaly"]
            masks = batch.get("img_mask")
            out = engine.model.score_image(images, b=engine.nearest_neighbors)
            results = out["batch"] if "batch" in out else [out]

            for i, res in enumerate(results):
                label = int(labels[i].item())
                score = float(res["s_img"])
                image_scores.append(score)
                image_labels.append(label)
                if label == 0:
                    normal_image_scores.append(score)
                else:
                    anomaly_image_scores.append(score)

                if masks is None:
                    continue
                mask_flat = (masks[i].cpu().numpy() > 0.5).astype(np.uint8).flatten()
                map_flat = engine._postprocess_anomaly_map(res["anomaly_map"]).numpy().flatten()
                if len(mask_flat) > pixel_sample_limit:
                    indices = np.random.choice(len(mask_flat), pixel_sample_limit, replace=False)
                    mask_flat = mask_flat[indices]
                    map_flat = map_flat[indices]
                normal_pixel_scores.extend(map_flat[mask_flat == 0].tolist())
                anomaly_pixel_scores.extend(map_flat[mask_flat == 1].tolist())
                all_pixel_scores.extend(map_flat.tolist())
                all_pixel_labels.extend(mask_flat.tolist())

    return {
        "task_id": task_id,
        "image_auroc": _safe_auc(image_labels, image_scores),
        "image_ap": _safe_ap(image_labels, image_scores),
        "pixel_auroc": _safe_auc(all_pixel_labels, all_pixel_scores),
        "pixel_aupr": _safe_ap(all_pixel_labels, all_pixel_scores),
        "image_scores": {
            "normal": _stats(normal_image_scores),
            "anomaly": _stats(anomaly_image_scores),
        },
        "pixel_scores": {
            "normal": _stats(normal_pixel_scores),
            "anomaly": _stats(anomaly_pixel_scores),
        },
    }


def _collect_selected_tasks(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: str,
    task_filter: set[int] | None,
    pixel_sample_limit: int,
) -> Dict[int, Dict[str, Any]]:
    engine = _load_engine(config, checkpoint_path, device)
    stream_manager = ContinualStreamingManager(config)
    task_stats: Dict[int, Dict[str, Any]] = {}
    ckpt_label = Path(checkpoint_path).parent.name

    total_tasks = len(stream_manager.categories)
    pbar = tqdm(total=total_tasks, desc=f"Scoring ({ckpt_label})", unit="task")
    while True:
        _, test_loader, task_info = stream_manager.get_next_task()
        if test_loader is None:
            break
        task_id = int(task_info["task_id"])
        category = str(task_info["category"])
        pbar.set_description(f"Scoring {task_id}: {category}")
        if task_filter is not None and task_id not in task_filter:
            pbar.update(1)
            continue
        stats = _collect_task_stats(engine, test_loader, task_id, pixel_sample_limit)
        stats["category"] = category
        task_stats[task_id] = stats
        pbar.update(1)

    pbar.close()
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return task_stats


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any], path: List[str]) -> float | None:
    left: Any = before
    right: Any = after
    for key in path:
        if key not in left or key not in right:
            return None
        left = left[key]
        right = right[key]
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(right) - float(left)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _truncate_tasks_if_needed(config, args.max_tasks)
    set_seed(int(config.get("training", {}).get("seed", 42)))

    task_filter = _parse_task_ids(args.task_ids, args.max_tasks)
    evaluation_cfg = config.get("evaluation", {})
    pixel_sample_limit = int(args.pixel_sample_limit or evaluation_cfg.get("pixel_sample_limit", 10000))
    requested_device = str(config.get("training", {}).get("device", "cuda"))
    device = requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu"

    results: Dict[str, Any] = {
        "before_checkpoint": args.before,
        "after_checkpoint": args.after,
        "pixel_sample_limit": pixel_sample_limit,
        "tasks": [],
    }

    before_by_task = _collect_selected_tasks(
        config=config,
        checkpoint_path=args.before,
        device=device,
        task_filter=task_filter,
        pixel_sample_limit=pixel_sample_limit,
    )
    after_by_task = _collect_selected_tasks(
        config=config,
        checkpoint_path=args.after,
        device=device,
        task_filter=task_filter,
        pixel_sample_limit=pixel_sample_limit,
    )

    for task_id in sorted(before_by_task):
        if task_id not in after_by_task:
            continue
        before_stats = before_by_task[task_id]
        after_stats = after_by_task[task_id]
        task_result = {
            "task_id": task_id,
            "category": str(before_stats["category"]),
            "before": before_stats,
            "after": after_stats,
            "delta": {
                "image_auroc": _metric_delta(before_stats, after_stats, ["image_auroc"]),
                "pixel_auroc": _metric_delta(before_stats, after_stats, ["pixel_auroc"]),
                "pixel_aupr": _metric_delta(before_stats, after_stats, ["pixel_aupr"]),
                "normal_pixel_p99": _metric_delta(before_stats, after_stats, ["pixel_scores", "normal", "p99"]),
                "anomaly_pixel_p50": _metric_delta(before_stats, after_stats, ["pixel_scores", "anomaly", "p50"]),
                "anomaly_pixel_p90": _metric_delta(before_stats, after_stats, ["pixel_scores", "anomaly", "p90"]),
            },
        }
        results["tasks"].append(task_result)
        delta = task_result["delta"]
        print(
            f"{task_id}:{before_stats['category']} "
            f"pixel_aupr={delta['pixel_aupr']:+.6f} "
            f"normal_p99={delta['normal_pixel_p99']:+.6f} "
            f"anom_p50={delta['anomaly_pixel_p50']:+.6f} "
            f"anom_p90={delta['anomaly_pixel_p90']:+.6f}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
