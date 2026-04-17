import argparse
import copy
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from models.vit_cms import ViT_CMS
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.global_seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run continual anomaly experiment with ViT-CMS backbone')
    parser.add_argument('--config', type=str, default='./conf/config.yaml', help='Path to YAML config')
    parser.add_argument(
        '--profile',
        type=str,
        default='default',
        choices=['default', 'tiny', 'small'],
        help='Runtime profile for fast experimentation',
    )
    parser.add_argument('--max_tasks', type=int, default=None, help='Optional cap on number of tasks to run')
    parser.add_argument('--run_suffix', type=str, default='', help='Optional suffix appended to run name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging even if enabled in config')
    parser.add_argument('--quiet', action='store_true', help='Reduce training/evaluation progress logs')
    return parser.parse_args()


def apply_profile(config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)

    dataset_cfg = cfg.setdefault('dataset', {})
    training_cfg = cfg.setdefault('training', {})
    model_cfg = cfg.setdefault('model', {})

    if profile == 'tiny':
        dataset_cfg['batch_size'] = min(int(dataset_cfg.get('batch_size', 32)), 8)
        dataset_cfg['img_size'] = min(int(dataset_cfg.get('img_size', 256)), 128)
        dataset_cfg['num_workers'] = 0
        training_cfg['epochs_per_task'] = 1
        if isinstance(dataset_cfg.get('class_order'), list) and dataset_cfg['class_order']:
            dataset_cfg['class_order'] = dataset_cfg['class_order'][:2]
        if isinstance(model_cfg.get('extract_layers'), list) and model_cfg['extract_layers']:
            model_cfg['extract_layers'] = model_cfg['extract_layers'][:2]

    if profile == 'small':
        dataset_cfg['batch_size'] = min(int(dataset_cfg.get('batch_size', 32)), 16)
        dataset_cfg['img_size'] = min(int(dataset_cfg.get('img_size', 256)), 192)
        dataset_cfg['num_workers'] = min(int(dataset_cfg.get('num_workers', 4)), 2)
        training_cfg['epochs_per_task'] = min(int(training_cfg.get('epochs_per_task', 10)), 2)
        if isinstance(dataset_cfg.get('class_order'), list) and dataset_cfg['class_order']:
            dataset_cfg['class_order'] = dataset_cfg['class_order'][:4]

    return cfg


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get('model', {})
    dataset_cfg = config.get('dataset', {})

    backbone = model_cfg.get('backbone', 'vit_base_patch16_224')
    if 'vit' not in str(backbone).lower():
        raise ValueError(f"Unsupported backbone '{backbone}'. Current runner supports ViT models only.")

    return ViT_CMS(
        model_name=backbone,
        pretrained=bool(model_cfg.get('pretrained', True)),
        cms_levels=int(model_cfg.get('cms_levels', 3)),
        k=int(model_cfg.get('k', 2)),
        extract_layers=list(model_cfg.get('extract_layers', [3, 6, 9])),
        img_size=int(dataset_cfg.get('img_size', 256)),
        use_spatial_gate=bool(model_cfg.get('use_spatial_gate', True)),
        freeze_backbone=bool(model_cfg.get('freeze_backbone', False)),
        freeze_patch_embed=bool(model_cfg.get('freeze_patch_embed', False)),
        reduced_dim=int(model_cfg.get('reduced_dim', 128)),
    )


def maybe_init_wandb(config: Dict[str, Any], run_name: str, run_dir: str, disable_wandb: bool):
    logging_cfg = config.get('logging', {})
    if disable_wandb or not bool(logging_cfg.get('use_wandb', False)):
        return None

    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"[W&B] Disabled because import failed: {exc}")
        return None

    project = logging_cfg.get('wandb_project', 'nested-learning-for-cad')
    entity = logging_cfg.get('wandb_entity', None)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        dir=run_dir,
        reinit=True,
    )
    return run


def _truncate_tasks_if_needed(config: Dict[str, Any], max_tasks: Optional[int]) -> None:
    if max_tasks is None:
        return

    dataset_cfg = config.setdefault('dataset', {})
    class_order = dataset_cfg.get('class_order', [])
    if isinstance(class_order, list) and class_order:
        dataset_cfg['class_order'] = class_order[:max_tasks]


def run_experiment(config: Dict[str, Any], run_suffix: str = '', disable_wandb: bool = False, quiet: bool = False) -> Dict[str, Any]:
    training_cfg = config.get('training', {})
    logging_cfg = config.get('logging', {})

    seed = int(training_cfg.get('seed', 42))
    set_seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = logging_cfg.get('experiment_name', 'vit_cms_hybrid_cad')
    run_name = f"{exp_name}_{timestamp}"
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"

    results_dir = logging_cfg.get('results_dir', 'results')
    run_dir = os.path.join(results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'resolved_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    model = build_model(config)

    requested_device = str(training_cfg.get('device', 'cuda'))
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        print("[Device] CUDA requested but unavailable. Falling back to CPU.")
        device = 'cpu'
    else:
        device = requested_device
    learning_rate = float(training_cfg.get('learning_rate', 1e-4))
    use_replay = bool(training_cfg.get('use_replay', False))
    replay_batch_size = int(training_cfg.get('replay_batch_size', 32))
    task_type = training_cfg.get('task_type', 'anomaly')
    pixel_loss_weight = float(training_cfg.get('pixel_loss_weight', 0.2))

    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_replay=use_replay,
        replay_batch_size=replay_batch_size,
        task_type=task_type,
        pixel_loss_weight=pixel_loss_weight,
    )
    evaluator = Evaluator(model=model, device=device, task_type=task_type)
    stream_manager = ContinualStreamingManager(config)

    wandb_run = maybe_init_wandb(config, run_name=run_name, run_dir=run_dir, disable_wandb=disable_wandb)

    epochs_per_task = int(training_cfg.get('epochs_per_task', 10))
    save_models = bool(logging_cfg.get('save_models', True))

    task_records = []

    while True:
        train_loader, test_loader, task_info = stream_manager.get_next_task()
        if train_loader is None:
            break

        task_id = int(task_info['task_id'])
        category = str(task_info['category'])

        if not quiet:
            print(f"\n===== Task {task_id}: {category} =====")

        train_metrics = trainer.train_task(
            train_loader=train_loader,
            task_id=task_id,
            epochs=epochs_per_task,
            verbose=not quiet,
        )
        eval_metrics = evaluator.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=not quiet,
        )

        record = {
            'task_id': task_id,
            'category': category,
            'train': train_metrics,
            'eval': eval_metrics,
        }
        task_records.append(record)

        record_path = os.path.join(run_dir, f'task_{task_id:02d}_metrics.json')
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)

        if save_models:
            ckpt_path = os.path.join(run_dir, f'task_{task_id:02d}_checkpoint.pt')
            torch.save(
                {
                    'task_id': task_id,
                    'category': category,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config,
                },
                ckpt_path,
            )

        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.log(
                {
                    'task_id': task_id,
                    'train/loss': float(train_metrics.get('loss', 0.0)),
                    'train/accuracy': float(train_metrics.get('accuracy', 0.0)),
                    'train/image_loss': float(train_metrics.get('image_loss', 0.0)),
                    'train/pixel_loss': float(train_metrics.get('pixel_loss', 0.0)),
                    'eval/loss': float(eval_metrics.get('loss', 0.0)),
                    'eval/accuracy': float(eval_metrics.get('accuracy', 0.0)),
                    'eval/f1': float(eval_metrics.get('f1', 0.0)),
                    'eval/auroc': float(eval_metrics.get('auroc', 0.0)),
                    'eval/image_ap': float(eval_metrics.get('image_ap', 0.0)),
                    'eval/pixel_f1': float(eval_metrics.get('pixel_f1', 0.0)),
                },
                step=task_id,
            )

    eval_acc = [float(rec['eval'].get('accuracy', 0.0)) for rec in task_records]
    eval_f1 = [float(rec['eval'].get('f1', 0.0)) for rec in task_records]
    eval_auroc = [float(rec['eval'].get('auroc', 0.0)) for rec in task_records]

    summary = {
        'run_name': run_name,
        'run_dir': run_dir,
        'tasks_completed': len(task_records),
        'avg_eval_accuracy': float(np.mean(eval_acc)) if eval_acc else 0.0,
        'avg_eval_f1': float(np.mean(eval_f1)) if eval_f1 else 0.0,
        'avg_eval_auroc': float(np.mean(eval_auroc)) if eval_auroc else 0.0,
    }

    with open(os.path.join(run_dir, 'run_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(run_dir, 'task_records.json'), 'w', encoding='utf-8') as f:
        json.dump(task_records, f, indent=2)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    print(f"Run completed. Summary: {summary}")
    return summary


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config = apply_profile(base_config, args.profile)
    _truncate_tasks_if_needed(config, args.max_tasks)

    run_suffix = args.run_suffix if args.run_suffix else args.profile
    run_experiment(
        config=config,
        run_suffix=run_suffix,
        disable_wandb=args.disable_wandb,
        quiet=args.quiet,
    )


if __name__ == '__main__':
    main()
