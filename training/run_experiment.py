import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from models import ViT_CMS, ViT_Replay
from training import Evaluator, Trainer
from utils.global_seed import set_seed


def apply_profile(config: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    if not profile:
        return config

    profile = profile.lower().strip()
    if profile in ("default", "full"):
        return config

    if profile == "tiny":
        config.setdefault("dataset", {})["batch_size"] = 4
        config.setdefault("dataset", {})["num_workers"] = 0
        config.setdefault("training", {})["epochs_per_task"] = 1
        return config

    if profile == "quick":
        config.setdefault("dataset", {})["batch_size"] = 8
        config.setdefault("dataset", {})["num_workers"] = 2
        config.setdefault("training", {})["epochs_per_task"] = min(
            int(config.get("training", {}).get("epochs_per_task", 2)),
            2,
        )
        return config

    raise ValueError(f"Unsupported profile: {profile}")


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return "cpu"
    return requested_device


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    use_replay = bool(training_cfg.get("use_replay", False))
    model_cls = ViT_Replay if use_replay else ViT_CMS

    common_kwargs = {
        "model_name": model_cfg.get("backbone", "vit_base_patch16_224"),
        "pretrained": bool(model_cfg.get("pretrained", True)),
        "cms_levels": int(model_cfg.get("cms_levels", 3)),
        "k": int(model_cfg.get("k", 2)),
        "extract_layers": model_cfg.get("extract_layers", [3, 6, 9]),
        "img_size": int(config.get("dataset", {}).get("img_size", 256)),
        "use_spatial_gate": bool(model_cfg.get("use_spatial_gate", True)),
        "freeze_backbone": bool(model_cfg.get("freeze_backbone", False)),
        "freeze_patch_embed": bool(model_cfg.get("freeze_patch_embed", False)),
        "reduced_dim": int(model_cfg.get("reduced_dim", 128)),
    }

    if use_replay:
        return model_cls(
            num_classes=int(model_cfg.get("num_classes", 2)),
            buffer_size=int(training_cfg.get("replay_buffer_size", 500)),
            **common_kwargs,
        )

    return model_cls(**common_kwargs)


def maybe_init_wandb(
    config: Dict[str, Any],
    run_dir: Path,
    disable_wandb: bool,
):
    logging_cfg = config.get("logging", {})
    use_wandb = bool(logging_cfg.get("use_wandb", False)) and not disable_wandb
    if not use_wandb:
        return None

    try:
        import wandb

        return wandb.init(
            project=logging_cfg.get("wandb_project", "nested-learning-for-cad"),
            entity=logging_cfg.get("wandb_entity", None),
            name=run_dir.name,
            config=config,
            dir=str(run_dir),
        )
    except Exception as exc:
        print(f"W&B initialization failed, continuing without W&B: {exc}")
        return None


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def _mean_metric(task_metrics: list, key: str) -> float:
    values = []
    for item in task_metrics:
        eval_metrics = item.get("eval", {})
        if key in eval_metrics:
            values.append(float(eval_metrics[key]))
    return float(np.mean(values)) if values else 0.0


def run_experiment(
    config: Dict[str, Any],
    *,
    config_path: str,
    profile: Optional[str] = None,
    run_name: Optional[str] = None,
    max_tasks: Optional[int] = None,
    epochs_override: Optional[int] = None,
    device_override: Optional[str] = None,
    seed_override: Optional[int] = None,
    disable_wandb: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    cfg = apply_profile(cfg, profile)

    if epochs_override is not None:
        cfg.setdefault("training", {})["epochs_per_task"] = int(epochs_override)
    if device_override is not None:
        cfg.setdefault("training", {})["device"] = device_override
    if seed_override is not None:
        cfg.setdefault("training", {})["seed"] = int(seed_override)

    training_cfg = cfg.setdefault("training", {})
    logging_cfg = cfg.setdefault("logging", {})

    requested_device = str(training_cfg.get("device", "cuda"))
    device = resolve_device(requested_device)
    training_cfg["device"] = device

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

    results_root = Path(logging_cfg.get("results_dir", "results"))
    runs_subdir = str(logging_cfg.get("runs_subdir", "runs"))
    run_root = results_root / runs_subdir
    run_root.mkdir(parents=True, exist_ok=True)

    experiment_name = run_name or str(logging_cfg.get("experiment_name", "experiment"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / f"{experiment_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    wandb_run = maybe_init_wandb(cfg, run_dir, disable_wandb)

    model = build_model(cfg)
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        use_replay=bool(training_cfg.get("use_replay", False)),
        replay_batch_size=int(training_cfg.get("replay_batch_size", 32)),
        task_type=str(training_cfg.get("task_type", "auto")),
        pixel_loss_weight=float(training_cfg.get("pixel_loss_weight", 0.2)),
    )
    evaluator = Evaluator(model=model, device=device)

    manager = ContinualStreamingManager(cfg)
    epochs_per_task = int(training_cfg.get("epochs_per_task", 1))
    save_models = bool(logging_cfg.get("save_models", True))

    checkpoints_dir = run_dir / "checkpoints"
    if save_models:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    task_metrics = []
    completed_tasks = 0

    while True:
        if max_tasks is not None and completed_tasks >= int(max_tasks):
            break

        train_loader, test_loader, task_info = manager.get_next_task()
        if train_loader is None:
            break

        task_id = int(task_info["task_id"])
        category = task_info.get("category", f"task_{task_id}")

        if verbose:
            print(f"\n=== Task {task_id}: {category} ===")

        train_metrics = trainer.train_task(
            train_loader=train_loader,
            task_id=task_id,
            epochs=epochs_per_task,
            verbose=verbose,
        )
        eval_metrics = evaluator.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=verbose,
        )

        current_task_metrics = {
            "task_id": task_id,
            "category": category,
            "train": _to_builtin(train_metrics),
            "eval": _to_builtin(eval_metrics),
        }
        task_metrics.append(current_task_metrics)

        if wandb_run is not None:
            wandb_payload = {
                "task_id": task_id,
                "train/loss": float(train_metrics.get("loss", 0.0)),
                "train/accuracy": float(train_metrics.get("accuracy", 0.0)),
                "train/image_loss": float(train_metrics.get("image_loss", 0.0)),
                "train/pixel_loss": float(train_metrics.get("pixel_loss", 0.0)),
            }
            for metric_name, metric_value in eval_metrics.items():
                wandb_payload[f"eval/{metric_name}"] = float(metric_value)
            wandb_run.log(wandb_payload)

        if save_models:
            checkpoint_path = checkpoints_dir / f"task_{task_id:02d}.pt"
            torch.save(
                {
                    "task_id": task_id,
                    "category": category,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": cfg,
                },
                checkpoint_path,
            )

        completed_tasks += 1

    with open(run_dir / "task_metrics.json", "w", encoding="utf-8") as f:
        json.dump(task_metrics, f, indent=2)

    summary = {
        "experiment_name": experiment_name,
        "profile": profile,
        "config_path": config_path,
        "run_dir": run_dir.as_posix(),
        "device": device,
        "seed": seed,
        "num_tasks": completed_tasks,
        "aggregate": {
            "avg_accuracy": _mean_metric(task_metrics, "accuracy"),
            "avg_f1": _mean_metric(task_metrics, "f1"),
            "avg_image_auroc": _mean_metric(task_metrics, "image_auroc"),
            "avg_image_ap": _mean_metric(task_metrics, "image_ap"),
            "avg_pixel_ap": _mean_metric(task_metrics, "pixel_ap"),
        },
    }

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    print("\nRun complete")
    print(f"Run directory: {run_dir.as_posix()}")
    print(json.dumps(summary["aggregate"], indent=2))

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a continual anomaly experiment")
    parser.add_argument("--config", default="conf/config.yaml", help="Path to YAML config")
    parser.add_argument("--profile", default=None, choices=["tiny", "quick", "default", "full"], help="Quick profile override")
    parser.add_argument("--run_name", default=None, help="Optional run name prefix")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per task")
    parser.add_argument("--device", default=None, help="Override training device")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B even if enabled in config")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars and per-step logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    config = load_config(config_path)

    run_experiment(
        config=config,
        config_path=config_path,
        profile=args.profile,
        run_name=args.run_name,
        max_tasks=args.max_tasks,
        epochs_override=args.epochs,
        device_override=args.device,
        seed_override=args.seed,
        disable_wandb=args.disable_wandb,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
