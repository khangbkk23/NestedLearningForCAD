import argparse
import copy
import itertools
import json
import os
import sys
from datetime import datetime
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from conf.config import load_config
from training.run_experiment import apply_profile, run_experiment


def _parse_list_str(value: str) -> List[str]:
    return [v.strip() for v in value.split(',') if v.strip()]


def _parse_list_int(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simple ablation sweep runner for ViT-CMS anomaly experiments')
    parser.add_argument('--config', type=str, default='./conf/config.yaml', help='Path to YAML config')
    parser.add_argument('--profile', type=str, default='tiny', choices=['default', 'tiny', 'small'])
    parser.add_argument('--max_tasks', type=int, default=2, help='Task cap per run for quick sweeps')

    parser.add_argument('--generators', type=str, default='superpixel,perlin,mixed')
    parser.add_argument('--cms_levels', type=str, default='2,3')
    parser.add_argument('--k_values', type=str, default='1,2')
    parser.add_argument('--seeds', type=str, default='42')

    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--quiet', action='store_true', help='Reduce train/eval logs from inner runs')
    parser.add_argument('--max_runs', type=int, default=0, help='Optional hard cap on total runs (0 means no cap)')
    return parser.parse_args()


def _truncate_tasks(config, max_tasks: int) -> None:
    if max_tasks is None or max_tasks <= 0:
        return

    dataset_cfg = config.setdefault('dataset', {})
    class_order = dataset_cfg.get('class_order', [])
    if isinstance(class_order, list) and class_order:
        dataset_cfg['class_order'] = class_order[:max_tasks]


def main() -> None:
    args = parse_args()

    generators = _parse_list_str(args.generators)
    cms_levels = _parse_list_int(args.cms_levels)
    k_values = _parse_list_int(args.k_values)
    seeds = _parse_list_int(args.seeds)

    base_config = load_config(args.config)

    combinations = list(itertools.product(generators, cms_levels, k_values, seeds))
    if args.max_runs and args.max_runs > 0:
        combinations = combinations[:args.max_runs]

    summaries = []

    for run_idx, (generator, level, k, seed) in enumerate(combinations, start=1):
        cfg = copy.deepcopy(base_config)
        cfg = apply_profile(cfg, args.profile)
        _truncate_tasks(cfg, args.max_tasks)

        cfg.setdefault('dataset', {})['anomaly_generator'] = generator
        cfg.setdefault('model', {})['cms_levels'] = int(level)
        cfg.setdefault('model', {})['k'] = int(k)
        cfg.setdefault('training', {})['seed'] = int(seed)

        suffix = f"{args.profile}_g-{generator}_l-{level}_k-{k}_s-{seed}"

        print(f"\n[SWEEP] Run {run_idx}/{len(combinations)}: {suffix}")
        summary = run_experiment(
            config=cfg,
            run_suffix=suffix,
            disable_wandb=args.disable_wandb,
            quiet=args.quiet,
        )
        summary['sweep_generator'] = generator
        summary['sweep_cms_levels'] = int(level)
        summary['sweep_k'] = int(k)
        summary['sweep_seed'] = int(seed)
        summaries.append(summary)

    results_root = base_config.get('logging', {}).get('results_dir', 'results')
    sweep_dir = os.path.join(results_root, 'sweeps')
    os.makedirs(sweep_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(sweep_dir, f'sweep_summary_{timestamp}.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)

    print(f"Sweep completed. Summary saved to: {out_path}")


if __name__ == '__main__':
    main()
