import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
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
    parser = argparse.ArgumentParser(description="Evaluate a saved Meta-NATH checkpoint.")
    parser.add_argument("--config", type=str, default="conf/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--run_suffix", type=str, default="checkpoint_eval")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _truncate_tasks_if_needed(config: Dict[str, Any], max_tasks: Optional[int]) -> None:
    if max_tasks is None:
        return
    dataset_cfg = config.setdefault("dataset", {})
    class_order = dataset_cfg.get("class_order", [])
    if isinstance(class_order, list) and class_order:
        dataset_cfg["class_order"] = class_order[:max_tasks]


def _make_output_dir(config: Dict[str, Any], output_dir: str | None, suffix: str) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        logging_cfg = config.get("logging", {})
        root = Path(logging_cfg.get("results_dir", "results"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = root / f"MetaNATH_Eval_{timestamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compute_forgetting_metrics(
    forgetting_matrix: Dict[str, Dict[str, float]],
    final_task_id: int,
) -> Dict[str, Any]:
    if not forgetting_matrix:
        return {"forgetting_measure": 0.0, "per_task_forgetting": {}}

    final_key = str(final_task_id)
    if final_key not in forgetting_matrix:
        return {"forgetting_measure": 0.0, "per_task_forgetting": {}}

    per_task = {}
    for task_j in range(final_task_id):
        eval_key = str(task_j)
        history_scores = []
        for task_i in range(task_j, final_task_id + 1):
            score = forgetting_matrix.get(str(task_i), {}).get(eval_key)
            if score is not None:
                history_scores.append(float(score))

        final_score = forgetting_matrix[final_key].get(eval_key)
        if not history_scores or final_score is None:
            continue
        per_task[eval_key] = max(0.0, max(history_scores) - float(final_score))

    return {
        "forgetting_measure": float(np.mean(list(per_task.values()))) if per_task else 0.0,
        "per_task_forgetting": per_task,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _truncate_tasks_if_needed(config, args.max_tasks)
    set_seed(int(config.get("training", {}).get("seed", 42)))

    output_dir = _make_output_dir(config, args.output_dir, args.run_suffix)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    training_cfg = config.get("training", {})
    evaluation_cfg = config.get("evaluation", {})
    memory_cfg = config.get("memory", {})

    requested_device = str(training_cfg.get("device", "cuda"))
    device = requested_device if torch.cuda.is_available() or requested_device == "cpu" else "cpu"

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_full_state_dict(checkpoint["model_state_dict"])

    engine = MetaNATHEngine(
        model=model,
        device=device,
        nearest_neighbors=int(memory_cfg.get("nearest_neighbors", 2)),
        pixel_score_norm=str(evaluation_cfg.get("pixel_score_norm", "none")).lower(),
        gaussian_smoothing_sigma=float(evaluation_cfg.get("gaussian_smoothing_sigma", 0.0)),
    )
    stream_manager = ContinualStreamingManager(config)

    pixel_sample_limit = int(evaluation_cfg.get("pixel_sample_limit", 10000))
    forgetting_matrix_enabled = bool(evaluation_cfg.get("forgetting_matrix", False))
    forgetting_metric = str(evaluation_cfg.get("forgetting_metric", "image_auroc"))

    records = []
    forgetting_matrix: Dict[str, Dict[str, float]] = {}

    total_tasks = len(stream_manager.categories)
    task_pbar = tqdm(total=total_tasks, desc="Eval tasks", unit="task", disable=args.quiet)
    while True:
        _, test_loader, task_info = stream_manager.get_next_task()
        if test_loader is None:
            break

        task_id = int(task_info["task_id"])
        category = str(task_info["category"])
        task_pbar.set_description(f"Eval {task_id}: {category}")
        eval_metrics = engine.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=not args.quiet,
            pixel_sample_limit=pixel_sample_limit,
        )

        record = {
            "task_id": task_id,
            "category": category,
            "eval": eval_metrics,
        }

        if forgetting_matrix_enabled:
            row_scores: Dict[str, float] = {}
            row_metrics: Dict[str, Dict[str, Any]] = {}
            n_prev = len(stream_manager.test_datasets_history)
            fm_pbar = tqdm(
                range(n_prev),
                desc=f"  Forgetting eval (task {task_id})",
                unit="eval",
                leave=False,
                disable=args.quiet,
            )
            for prev_task_id in fm_pbar:
                prev_loader = stream_manager.get_test_loader_for_task(prev_task_id)
                if prev_loader is None:
                    continue
                prev_metrics = engine.evaluate_task(
                    test_loader=prev_loader,
                    task_id=prev_task_id,
                    verbose=False,
                    pixel_sample_limit=pixel_sample_limit,
                )
                row_scores[str(prev_task_id)] = float(prev_metrics.get(forgetting_metric, 0.0))
                row_metrics[str(prev_task_id)] = prev_metrics
            record["forgetting_eval"] = row_metrics
            forgetting_matrix[str(task_id)] = row_scores

        records.append(record)
        task_pbar.update(1)

    task_pbar.close()

    final_cumulative_metrics = None
    if records:
        if not args.quiet:
            print("\n===== Final Cumulative Evaluation =====")
        cumulative_loader = stream_manager.get_cumulative_test_loader()
        if cumulative_loader is not None:
            final_cumulative_metrics = engine.evaluate_task(
                test_loader=cumulative_loader,
                task_id=int(records[-1]["task_id"]),
                verbose=not args.quiet,
                pixel_sample_limit=pixel_sample_limit,
            )
            with open(output_dir / "final_cumulative_metrics.json", "w", encoding="utf-8") as f:
                json.dump(final_cumulative_metrics, f, indent=2)

    image_aurocs = [float(rec["eval"].get("image_auroc", 0.0)) for rec in records]
    pixel_aurocs = [float(rec["eval"].get("pixel_auroc", 0.0)) for rec in records]
    pixel_auprs = [float(rec["eval"].get("pixel_aupr", 0.0)) for rec in records]
    image_aps = [float(rec["eval"].get("image_ap", 0.0)) for rec in records]

    summary = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(output_dir),
        "tasks_completed": len(records),
        "nearest_neighbors": int(memory_cfg.get("nearest_neighbors", 2)),
        "pixel_score_norm": str(evaluation_cfg.get("pixel_score_norm", "none")).lower(),
        "gaussian_smoothing_sigma": float(evaluation_cfg.get("gaussian_smoothing_sigma", 0.0)),
        "avg_eval_image_auroc": float(np.mean(image_aurocs)) if image_aurocs else 0.0,
        "avg_eval_pixel_auroc": float(np.mean(pixel_aurocs)) if pixel_aurocs else 0.0,
        "avg_eval_pixel_aupr": float(np.mean(pixel_auprs)) if pixel_auprs else 0.0,
        "avg_eval_image_ap": float(np.mean(image_aps)) if image_aps else 0.0,
    }
    if final_cumulative_metrics is not None:
        summary["final_cumulative_image_auroc"] = float(final_cumulative_metrics.get("image_auroc", 0.0))
        summary["final_cumulative_pixel_aupr"] = float(final_cumulative_metrics.get("pixel_aupr", 0.0))

    if forgetting_matrix_enabled and records:
        final_task_id = int(records[-1]["task_id"])
        forgetting_metrics = _compute_forgetting_metrics(forgetting_matrix, final_task_id)
        summary["forgetting_metric"] = forgetting_metric
        summary["forgetting_measure"] = float(forgetting_metrics["forgetting_measure"])
        with open(output_dir / "forgetting_matrix.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": forgetting_metric,
                    "matrix": forgetting_matrix,
                    **forgetting_metrics,
                },
                f,
                indent=2,
            )

    with open(output_dir / "checkpoint_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "checkpoint_eval_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Checkpoint evaluation completed. Summary: {summary}")


if __name__ == "__main__":
    main()
