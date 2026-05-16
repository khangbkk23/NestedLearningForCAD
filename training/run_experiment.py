import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from conf.config import load_config
from dataset.load_dataset import ContinualStreamingManager
from models.meta_nath_core import MetaNATHCore
from training.checkpointing import CheckpointManager, resolve_checkpoint_policy
from training.meta_nath_engine import MetaNATHEngine
from utils.global_seed import set_seed


def _model_geometry(model: MetaNATHCore) -> Dict[str, Any]:
    patch_grid = model.patch_grid or model.coreset.patch_grid
    return {
        'patch_grid': list(patch_grid) if patch_grid is not None else None,
        'n_patch': int(model.coreset.n_patch) if model.coreset.n_patch is not None else None,
    }


def _compute_forgetting_metrics(
    forgetting_matrix: Dict[str, Dict[str, float]],
    final_task_id: int,
) -> Dict[str, Any]:
    if not forgetting_matrix:
        return {'forgetting_measure': 0.0, 'per_task_forgetting': {}}

    final_key = str(final_task_id)
    if final_key not in forgetting_matrix:
        return {'forgetting_measure': 0.0, 'per_task_forgetting': {}}

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

    fm = float(np.mean(list(per_task.values()))) if per_task else 0.0
    return {
        'forgetting_measure': fm,
        'per_task_forgetting': per_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Meta-NATH CAD Phase 1-2 experiments')
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
        training_cfg['stream_repeats'] = 1
        if isinstance(dataset_cfg.get('class_order'), list) and dataset_cfg['class_order']:
            dataset_cfg['class_order'] = dataset_cfg['class_order'][:2]
        if isinstance(model_cfg.get('extract_layers'), list) and model_cfg['extract_layers']:
            model_cfg['extract_layers'] = model_cfg['extract_layers'][:2]

    if profile == 'small':
        dataset_cfg['batch_size'] = min(int(dataset_cfg.get('batch_size', 32)), 16)
        dataset_cfg['img_size'] = min(int(dataset_cfg.get('img_size', 256)), 192)
        dataset_cfg['num_workers'] = min(int(dataset_cfg.get('num_workers', 4)), 2)
        training_cfg['epochs_per_task'] = min(int(training_cfg.get('epochs_per_task', 10)), 2)
        training_cfg['stream_repeats'] = min(int(training_cfg.get('stream_repeats', training_cfg['epochs_per_task'])), 2)
        if isinstance(dataset_cfg.get('class_order'), list) and dataset_cfg['class_order']:
            dataset_cfg['class_order'] = dataset_cfg['class_order'][:4]

    return cfg


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get('model', {})
    training_cfg = config.get('training', {})
    n_patch_cfg = model_cfg.get('n_patch', None)
    n_patch = None if n_patch_cfg is None else int(n_patch_cfg)

    return MetaNATHCore(
        d=int(model_cfg.get('embed_dim', 768)),
        tau_acc=float(model_cfg.get('tau_acc', 0.25)),
        max_coreset_size=int(model_cfg.get('max_coreset_size', 1000)),
        n_patch=n_patch,
        store_images=bool(model_cfg.get('store_images', False)),
        device=str(training_cfg.get('device', 'cuda')),
        backbone_name=model_cfg.get('backbone', 'facebook/dinov3-vitb14-pretrain'),
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
    evaluation_cfg = config.get('evaluation', {})
    memory_cfg = config.get('memory', {})

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
    nearest_neighbors = int(memory_cfg.get('nearest_neighbors', 2))
    pixel_score_norm = str(evaluation_cfg.get('pixel_score_norm', 'none')).lower()
    gaussian_smoothing_sigma = float(evaluation_cfg.get('gaussian_smoothing_sigma', 0.0))
    engine = MetaNATHEngine(
        model=model,
        device=device,
        nearest_neighbors=nearest_neighbors,
        pixel_score_norm=pixel_score_norm,
        gaussian_smoothing_sigma=gaussian_smoothing_sigma,
    )
    stream_manager = ContinualStreamingManager(config)

    wandb_run = maybe_init_wandb(config, run_name=run_name, run_dir=run_dir, disable_wandb=disable_wandb)

    epochs_per_task = int(training_cfg.get('stream_repeats', training_cfg.get('epochs_per_task', 1)))
    pixel_sample_limit = int(evaluation_cfg.get('pixel_sample_limit', 10000))
    eval_mode = str(evaluation_cfg.get('mode', 'current')).lower()
    cumulative_frequency = int(evaluation_cfg.get('cumulative_frequency', 5))
    final_cumulative = bool(evaluation_cfg.get('final_cumulative', True))
    forgetting_matrix_enabled = bool(evaluation_cfg.get('forgetting_matrix', False))
    forgetting_metric = str(evaluation_cfg.get('forgetting_metric', 'image_auroc'))
    save_models = bool(logging_cfg.get('save_models', True))
    checkpoint_mode = str(logging_cfg.get('checkpoint_mode', 'phase12_light')).lower()
    checkpoint_policy = resolve_checkpoint_policy(logging_cfg.get('checkpoint_policy', 'last_only'))
    checkpoint_manager = CheckpointManager(
        run_dir=run_dir,
        checkpoint_mode=checkpoint_mode,
        checkpoint_policy=checkpoint_policy,
        save_models=save_models,
    )

    task_records = []
    forgetting_matrix: Dict[str, Dict[str, float]] = {}

    while True:
        train_loader, test_loader, task_info = stream_manager.get_next_task()
        if train_loader is None:
            break

        task_id = int(task_info['task_id'])
        category = str(task_info['category'])

        if not quiet:
            print(f"\n===== Task {task_id}: {category} =====")

        train_metrics = engine.train_task(
            train_loader=train_loader,
            task_id=task_id,
            epochs=epochs_per_task,
            verbose=not quiet,
        )
        eval_metrics = engine.evaluate_task(
            test_loader=test_loader,
            task_id=task_id,
            verbose=not quiet,
            pixel_sample_limit=pixel_sample_limit,
        )

        cumulative_eval_metrics = None
        should_cumulative_eval = (
            eval_mode != 'cumulative'
            and cumulative_frequency > 0
            and (task_id + 1) % cumulative_frequency == 0
        )
        if should_cumulative_eval:
            cumulative_loader = stream_manager.get_cumulative_test_loader()
            if cumulative_loader is not None:
                cumulative_eval_metrics = engine.evaluate_task(
                    test_loader=cumulative_loader,
                    task_id=task_id,
                    verbose=not quiet,
                    pixel_sample_limit=pixel_sample_limit,
                )

        record = {
            'task_id': task_id,
            'category': category,
            'model': _model_geometry(model),
            'train': train_metrics,
            'eval': eval_metrics,
        }
        if cumulative_eval_metrics is not None:
            record['cumulative_eval'] = cumulative_eval_metrics

        if forgetting_matrix_enabled:
            row_metrics: Dict[str, Dict[str, Any]] = {}
            row_scores: Dict[str, float] = {}
            for prev_task_id in range(len(stream_manager.test_datasets_history)):
                if prev_task_id == task_id and eval_mode != 'cumulative':
                    prev_metrics = eval_metrics
                else:
                    prev_loader = stream_manager.get_test_loader_for_task(prev_task_id)
                    if prev_loader is None:
                        continue
                    prev_metrics = engine.evaluate_task(
                        test_loader=prev_loader,
                        task_id=prev_task_id,
                        verbose=False,
                        pixel_sample_limit=pixel_sample_limit,
                    )

                row_metrics[str(prev_task_id)] = prev_metrics
                row_scores[str(prev_task_id)] = float(prev_metrics.get(forgetting_metric, 0.0))

            record['forgetting_eval'] = row_metrics
            forgetting_matrix[str(task_id)] = row_scores
        task_records.append(record)

        record_path = os.path.join(run_dir, f'task_{task_id:02d}_metrics.json')
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)

        checkpoint_manager.save_task(
            model=model,
            config=config,
            task_id=task_id,
            category=category,
            eval_metrics=eval_metrics,
        )

        if wandb_run is not None:
            import wandb  # type: ignore

            wandb.log(
                {
                    'task_id': task_id,
                    'train/surprise_score': float(train_metrics.get('avg_surprise', 0.0)),
                    'train/acc_gating': float(train_metrics.get('avg_acc', 0.0)),
                    'train/acc_min': float(train_metrics.get('acc_min', 0.0)),
                    'train/acc_p10': float(train_metrics.get('acc_p10', 0.0)),
                    'train/acc_p50': float(train_metrics.get('acc_p50', 0.0)),
                    'train/acc_p90': float(train_metrics.get('acc_p90', 0.0)),
                    'train/acc_max': float(train_metrics.get('acc_max', 0.0)),
                    'train/acc_approval_rate': float(train_metrics.get('acc_approval_rate', 0.0)),
                    'train/coreset_size': float(train_metrics.get('coreset_size', 0.0)),
                    'train/coreset_update_count': float(train_metrics.get('coreset_update_count', 0.0)),
                    'train/normal_update_samples': float(train_metrics.get('normal_update_samples', 0.0)),
                    'train/skipped_anomaly_count': float(train_metrics.get('skipped_anomaly_count', 0.0)),
                    'eval/image_auroc': float(eval_metrics.get('image_auroc', 0.0)),
                    'eval/image_ap': float(eval_metrics.get('image_ap', 0.0)),
                    'eval/pixel_auroc': float(eval_metrics.get('pixel_auroc', 0.0)),
                    'eval/pixel_aupr': float(eval_metrics.get('pixel_aupr', 0.0)),
                    'eval/eval_num_images': float(eval_metrics.get('eval_num_images', 0.0)),
                    'eval/eval_seconds': float(eval_metrics.get('eval_seconds', 0.0)),
                },
                step=task_id,
            )

    final_cumulative_metrics = None
    if task_records and final_cumulative and eval_mode != 'cumulative':
        if len(task_records) == 1:
            final_cumulative_metrics = task_records[-1]['eval']
        elif 'cumulative_eval' in task_records[-1]:
            final_cumulative_metrics = task_records[-1]['cumulative_eval']
        else:
            cumulative_loader = stream_manager.get_cumulative_test_loader()
            if cumulative_loader is not None:
                final_cumulative_metrics = engine.evaluate_task(
                    test_loader=cumulative_loader,
                    task_id=int(task_records[-1]['task_id']),
                    verbose=not quiet,
                    pixel_sample_limit=pixel_sample_limit,
                )

        if final_cumulative_metrics is not None:
            with open(os.path.join(run_dir, 'final_cumulative_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(final_cumulative_metrics, f, indent=2)

    eval_image_aurocs = [float(rec['eval'].get('image_auroc', 0.0)) for rec in task_records]
    eval_pixel_aurocs = [float(rec['eval'].get('pixel_auroc', 0.0)) for rec in task_records]
    eval_pixel_auprs = [float(rec['eval'].get('pixel_aupr', 0.0)) for rec in task_records]
    eval_image_aps = [float(rec['eval'].get('image_ap', 0.0)) for rec in task_records]

    avg_eval_image_auroc = float(np.mean(eval_image_aurocs)) if eval_image_aurocs else 0.0
    summary = {
        'run_name': run_name,
        'run_dir': run_dir,
        'tasks_completed': len(task_records),
        'nearest_neighbors': nearest_neighbors,
        'pixel_score_norm': pixel_score_norm,
        'gaussian_smoothing_sigma': gaussian_smoothing_sigma,
        'checkpoint_policy': checkpoint_policy,
        **_model_geometry(model),
        'avg_eval_image_auroc': avg_eval_image_auroc,
        'avg_eval_auroc': avg_eval_image_auroc,
        'avg_eval_pixel_auroc': float(np.mean(eval_pixel_aurocs)) if eval_pixel_aurocs else 0.0,
        'avg_eval_pixel_aupr': float(np.mean(eval_pixel_auprs)) if eval_pixel_auprs else 0.0,
        'avg_eval_image_ap': float(np.mean(eval_image_aps)) if eval_image_aps else 0.0,
    }
    if final_cumulative_metrics is not None:
        summary['final_cumulative_image_auroc'] = float(final_cumulative_metrics.get('image_auroc', 0.0))
        summary['final_cumulative_pixel_aupr'] = float(final_cumulative_metrics.get('pixel_aupr', 0.0))

    if forgetting_matrix_enabled:
        final_task_id = int(task_records[-1]['task_id']) if task_records else 0
        forgetting_metrics = _compute_forgetting_metrics(forgetting_matrix, final_task_id)
        summary['forgetting_metric'] = forgetting_metric
        summary['forgetting_measure'] = float(forgetting_metrics['forgetting_measure'])
        with open(os.path.join(run_dir, 'forgetting_matrix.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'metric': forgetting_metric,
                    'matrix': forgetting_matrix,
                    **forgetting_metrics,
                },
                f,
                indent=2,
            )

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
